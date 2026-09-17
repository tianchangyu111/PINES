import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, laplace, sobel, uniform_filter
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transfer_learning import (
    DEVICE,
    build_led_map,
    compute_mask,
    load_pairs,
    plot_grid,
    plot_score_dist,
    resolve_data_root,
)


CFG = {
    "img_size": 128,
    "mask_thr": 15,
    "mask_erode": 8,
    "led_subdir": "led_field",
    "pbs_subdir": "pbs",
    "analyte_subdir": "analyte",
    "substrate_subdir": "susbtrat",
    "led_bright_thr": 0.5,
    "led_smooth_sigma": 15,
    "feature_bank_mode": "extended48",
    "score_primary": "top3_weighted",
    "support_z_thr": 2.5,
    "n_splits": 3,
    "random_state": 42,
    "refine_batch_size": 8,
    "refine_epochs": 8,
    "refine_lr": 1.5e-4,
    "refine_wd": 1e-5,
    "refine_patience": 4,
    "tv_weight": 0.0002,
    "cam_sigma": 3.0,
    "dpi": 180,
}


def _extract_numeric_id(name):
    import re

    nums = re.findall(r"\d+", Path(name).name)
    return int(nums[0]) if nums else None


def _extract_coarse_batch_id(name):
    import re

    nums = re.findall(r"\d+", Path(name).name)
    if not nums:
        return None
    token = nums[0]
    return int(token[0]) if token else None


def make_balanced_folds(names, labels, n_splits, random_state=42):
    labels = np.asarray(labels).astype(int)
    buckets = {}
    for idx, (nm, lbl) in enumerate(zip(names, labels)):
        coarse = _extract_coarse_batch_id(nm)
        key = (int(lbl), int(coarse) if coarse is not None else -1)
        buckets.setdefault(key, []).append(idx)
    rng = np.random.RandomState(random_state)
    fold_bins = [[] for _ in range(n_splits)]
    for key in sorted(buckets):
        idxs = list(buckets[key])
        idxs.sort(key=lambda i: str(names[i]))
        rng.shuffle(idxs)
        for offset, idx in enumerate(idxs):
            fold_bins[offset % n_splits].append(idx)
    folds = []
    all_idx = np.arange(len(labels))
    for fold_id in range(n_splits):
        va_idx = np.array(sorted(fold_bins[fold_id]), dtype=int)
        tr_mask = np.ones(len(labels), dtype=bool)
        tr_mask[va_idx] = False
        tr_idx = all_idx[tr_mask]
        folds.append((tr_idx, va_idx))
    return folds


def _local_stats(x, w):
    mu = uniform_filter(x, size=w)
    mu2 = uniform_filter(x * x, size=w)
    var = np.maximum(mu2 - mu * mu, 0)
    return mu, np.sqrt(var)


def _dog(x, s1, s2):
    return gaussian_filter(x, sigma=s1) - gaussian_filter(x, sigma=s2)


def _speckle_contrast(x, w):
    mu, std = _local_stats(x, w)
    return std / (mu + 1e-6)


def _high_low_freq(x):
    low = gaussian_filter(x, sigma=6)
    high = (x - low) ** 2
    return low, high


def _grad_mag(x):
    gx = sobel(x, axis=1)
    gy = sobel(x, axis=0)
    return np.sqrt(gx * gx + gy * gy)


def _local_corr_drop(a, b, w):
    ma, sa = _local_stats(a, w)
    mb, sb = _local_stats(b, w)
    cab = uniform_filter(a * b, size=w) - ma * mb
    corr = cab / (np.sqrt(np.maximum(sa * sa, 0)) * np.sqrt(np.maximum(sb * sb, 0)) + 1e-6)
    return 1.0 - np.clip(corr, -1, 1)


def _ssim_like_map(a, b, w):
    c1, c2 = 1e-4, 9e-4
    mu_a = uniform_filter(a, size=w)
    mu_b = uniform_filter(b, size=w)
    var_a = uniform_filter(a * a, size=w) - mu_a * mu_a
    var_b = uniform_filter(b * b, size=w) - mu_b * mu_b
    cov = uniform_filter(a * b, size=w) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    return 1.0 - num / (den + 1e-6)


def _local_percentile_shift(a, b, w):
    ga = gaussian_filter(a, sigma=max(1, w / 6))
    gb = gaussian_filter(b, sigma=max(1, w / 6))
    return gb - ga


def _rank_diff(a, b, w):
    ma, sa = _local_stats(a, w)
    mb, sb = _local_stats(b, w)
    za = (a - ma) / (sa + 1e-6)
    zb = (b - mb) / (sb + 1e-6)
    return zb - za


def _anscombe(x):
    return 2.0 * np.sqrt(np.maximum(x, 0) + 3 / 8)


def _pixel_features_core30(before, after):
    before = before.astype(np.float32)
    after = after.astype(np.float32)
    diff = after - before
    log_r = np.log(after + 1.0) - np.log(before + 1.0)
    feats = [diff, log_r]
    for s in [1, 2, 4, 8, 12]:
        feats.append(gaussian_filter(diff, sigma=s))
        feats.append(gaussian_filter(log_r, sigma=s))
    for s1, s2 in [(1, 2), (2, 4), (4, 8), (1, 4)]:
        feats.append(_dog(diff, s1, s2))
    for w in [7, 13, 21]:
        mu, std = _local_stats(diff, w)
        feats += [mu, std, diff / (std + 1e-6)]
    for w in [7, 13]:
        feats.append(_speckle_contrast(after, w) - _speckle_contrast(before, w))
    lf, hf = _high_low_freq(diff)
    feats += [lf, hf]
    return np.stack(feats, 0).astype(np.float32)


def _pixel_features_extended48(before, after):
    core = _pixel_features_core30(before, after)
    before = before.astype(np.float32)
    after = after.astype(np.float32)
    diff = after - before
    ext = [
        diff / (before + 1e-3),
        diff / (after + before + 1e-3),
        diff / (_local_stats(before, 7)[1] + 1e-6),
        _anscombe(after) - _anscombe(before),
    ]
    for w in [7, 13]:
        ext.append(_local_corr_drop(before, after, w))
        ext.append(_ssim_like_map(before, after, w))
    for s in [1, 2]:
        ext.append(gaussian_filter(_grad_mag(after), sigma=s) - gaussian_filter(_grad_mag(before), sigma=s))
    for s in [1, 2]:
        ext.append(gaussian_filter(laplace(after), sigma=s) - gaussian_filter(laplace(before), sigma=s))
    ext.append(laplace(after) - laplace(before))
    ext.append(_grad_mag(after) - _grad_mag(before))
    for w in [7, 13]:
        ext.append(_local_percentile_shift(before, after, w))
        ext.append(_rank_diff(before, after, w))
    return np.concatenate([core, np.stack(ext, 0).astype(np.float32)], 0)


def _pixel_features(before, after, mode="extended48"):
    return _pixel_features_extended48(before, after) if mode == "extended48" else _pixel_features_core30(before, after)


def build_pbs_pixel_baseline(after_list, before_list, mask, mode="extended48"):
    stk = np.stack([_pixel_features(bef, aft, mode=mode) for aft, bef in zip(after_list, before_list)], 0)
    mu = stk.mean(0)
    std_raw = stk.std(0, ddof=1)
    floor = np.percentile(std_raw[:, mask], 5, axis=1)[:, None, None]
    return mu.astype(np.float32), np.maximum(std_raw, floor).astype(np.float32)


def build_score_maps(before, after, pbs_mu, pbs_std, mask, cfg):
    feats = _pixel_features(before, after, mode=cfg["feature_bank_mode"])
    z = (feats - pbs_mu) / (pbs_std + 1e-6)
    z = np.where(mask[None], z, 0.0)
    z_sorted = np.sort(z, axis=0)[::-1]
    maps = {
        "top1": z_sorted[0].astype(np.float32),
        "top3_mean": z_sorted[:3].mean(0).astype(np.float32),
        "top3_weighted": (0.6 * z_sorted[0] + 0.3 * z_sorted[1] + 0.1 * z_sorted[2]).astype(np.float32),
        "top5_rms": np.sqrt((z_sorted[:5] ** 2).mean(0)).astype(np.float32),
        "support_count": (z > cfg["support_z_thr"]).sum(0).astype(np.float32),
        "support_frac": ((z > cfg["support_z_thr"]).sum(0) / z.shape[0]).astype(np.float32),
        "z_stack": z.astype(np.float32),
        "feature_stack": feats.astype(np.float32),
    }
    maps["primary"] = maps[cfg["score_primary"]]
    return maps


def summarize_image_features(score_maps, mask):
    z = score_maps["z_stack"]
    out = {}
    for k in ["top1", "top3_mean", "top3_weighted", "top5_rms", "support_count", "support_frac"]:
        m = score_maps[k][mask]
        out[f"{k}_mean"] = float(m.mean())
        out[f"{k}_std"] = float(m.std())
        out[f"{k}_max"] = float(m.max())
        out[f"{k}_q95"] = float(np.quantile(m, 0.95))
        out[f"{k}_sigpct"] = float((m > np.quantile(m, 0.95)).mean())
    for i in range(z.shape[0]):
        zi = z[i][mask]
        out[f"z{i:02d}_mean"] = float(zi.mean())
        out[f"z{i:02d}_std"] = float(zi.std())
        out[f"z{i:02d}_q95"] = float(np.quantile(zi, 0.95))
    return out


class _ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class ResNet18_29(nn.Module):
    def __init__(self, in_ch=29):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(_ResBlock(64), _ResBlock(64))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), _ResBlock(128), _ResBlock(128))
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), _ResBlock(256))
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = _ResBlock(128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = _ResBlock(64)
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        s = self.stem(x)
        l1 = self.layer1(s)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        d3 = self.dec3(self.up3(l3) + l2)
        d2 = self.dec2(self.up2(d3) + l1)
        return self.head(d2)


def extract_feature_bank_29(bef, aft):
    def _to01(x):
        x = x.astype(np.float32)
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    def _lstd(x, w):
        mu = gaussian_filter(x, w / 2)
        mu2 = gaussian_filter(x**2, w / 2)
        return np.sqrt(np.maximum(mu2 - mu**2, 0))

    def _sc(x, w):
        return _lstd(x, w) / (gaussian_filter(x, w / 2) + 1e-6)

    b = _to01(bef)
    a = _to01(aft)
    d = a - b
    eps = 1e-6
    lb = gaussian_filter(b, 4)
    hb = np.abs(b - lb)
    la = gaussian_filter(a, 4)
    ha = np.abs(a - la)
    feats = [
        d, np.abs(d), np.log1p(a + eps) - np.log1p(b + eps), a / (b + eps) - 1, np.clip(d / (b + eps), -5, 5),
        gaussian_filter(d, 1), gaussian_filter(d, 3), gaussian_filter(d, 6),
        _lstd(a, 3) - _lstd(b, 3), _lstd(a, 7) - _lstd(b, 7), _lstd(a, 15) - _lstd(b, 15),
        _sc(a, 5) - _sc(b, 5), _sc(a, 9) - _sc(b, 9), la - lb, ha - hb, ha**2 - hb**2,
    ]
    for s in [1, 2, 4]:
        feats.append((a - gaussian_filter(a, s * 2)) - (b - gaussian_filter(b, s * 2)))
    feats += [b, a, gaussian_filter(b, 2), gaussian_filter(a, 2), _lstd(d, 3), _lstd(d, 7), np.abs(gaussian_filter(d, 2) - gaussian_filter(d, 6)), a**2 - b**2, gaussian_filter(np.abs(d), 3), np.maximum(d, 0)]
    return np.stack(feats, 0).astype(np.float32)


def build_pbs_baseline_29(pbs_after_list, pbs_before_list, mask):
    stk = np.stack([extract_feature_bank_29(b, a) for a, b in zip(pbs_after_list, pbs_before_list)], 0)
    mu = stk.mean(axis=0)
    std_raw = stk.std(axis=0, ddof=1) if stk.shape[0] > 1 else np.zeros_like(mu)
    floor = np.percentile(std_raw[:, mask], 5, axis=1)[:, None, None]
    return mu.astype(np.float32), np.maximum(std_raw, floor).astype(np.float32)


def _make_soft_target(bef, aft, pbs_mu29, pbs_std29, mask):
    feat = extract_feature_bank_29(bef, aft)
    z = np.where(mask[None], (feat - pbs_mu29) / (pbs_std29 + 1e-6), 0.0)
    z_sorted = np.sort(z, axis=0)[::-1]
    top1 = np.clip(z_sorted[0], 0, None)
    top3 = np.clip(z_sorted[:3].mean(0), 0, None)
    top5 = np.clip(z_sorted[:5].mean(0), 0, None)
    support = (z > 2.0).sum(0).astype(np.float32)

    def _n01(x):
        mn, mx = float(x.min()), float(x.max())
        return np.zeros_like(x, np.float32) if mx - mn < 1e-8 else ((x - mn) / (mx - mn)).astype(np.float32)

    return _n01((0.2 * _n01(top1) + 0.5 * _n01(top3) + 0.3 * _n01(top5)) * (0.65 + 0.35 * _n01(support)))


class Refine29Dataset(Dataset):
    def __init__(self, after_list, before_list, labels, pbs_mu29, pbs_std29, mask, cm, cs):
        self.after_list = after_list
        self.before_list = before_list
        self.labels = labels
        self.pbs_mu29 = pbs_mu29
        self.pbs_std29 = pbs_std29
        self.mask = mask
        self.cm = cm
        self.cs = cs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        aft = self.after_list[idx]
        bef = self.before_list[idx]
        feat = extract_feature_bank_29(bef, aft)
        feat = (feat - self.cm[:, None, None]) / (self.cs[:, None, None] + 1e-6)
        if self.labels[idx] == 1:
            soft = _make_soft_target(bef, aft, self.pbs_mu29, self.pbs_std29, self.mask)
        else:
            soft = np.zeros(feat.shape[1:], np.float32)
        return torch.from_numpy(feat), torch.from_numpy(soft), torch.tensor(self.labels[idx], dtype=torch.long)


def _compute_channel_stats(after_list, before_list):
    stk = np.stack([extract_feature_bank_29(b, a) for a, b in zip(after_list, before_list)], 0)
    return stk.mean(axis=(0, 2, 3)).astype(np.float32), stk.std(axis=(0, 2, 3)).astype(np.float32)


def _tv_loss(x):
    return torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean() + torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()


def _refine_loss(logit, target, tv_w):
    return F.smooth_l1_loss(torch.sigmoid(logit).squeeze(1), target) + tv_w * _tv_loss(logit)


def compute_metrics(y_true, y_pred, y_prob):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["roc_auc"] = np.nan
    return out


def find_best_threshold(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    candidates = np.unique(np.concatenate([[0.5], y_prob]))
    best_thr, best_acc, best_bal = 0.5, -1.0, -1.0
    for thr in candidates:
        pred = (y_prob >= thr).astype(int)
        acc = accuracy_score(y_true, pred)
        bal = balanced_accuracy_score(y_true, pred)
        if (acc > best_acc + 1e-12) or (abs(acc - best_acc) <= 1e-12 and bal > best_bal):
            best_thr, best_acc, best_bal = float(thr), float(acc), float(bal)
    return best_thr, best_acc, best_bal


def scan_thresholds(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    candidates = np.unique(np.concatenate([np.linspace(0.5, 0.9999, 200), y_prob]))
    rows = []
    for thr in candidates:
        pred = (y_prob >= thr).astype(int)
        row = {"threshold": float(thr)}
        row.update(compute_metrics(y_true, pred, y_prob))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "accuracy", "mcc", "f1", "threshold"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def plot_classifier_cv_curves(cv_df, save_path):
    if cv_df is None or len(cv_df) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor="#111111")
    ax.set_facecolor("#1a1a2e")
    x = cv_df["fold"].astype(int).values
    for col, color in [
        ("accuracy", "#4a90d9"),
        ("balanced_accuracy", "#f5a623"),
        ("f1", "#e74c3c"),
        ("roc_auc", "#2ecc71"),
    ]:
        if col in cv_df.columns:
            ax.plot(x, cv_df[col].values, marker="o", linewidth=2, label=col, color=color)
    ax.set_xlabel("Fold", color="w")
    ax.set_ylabel("Score", color="w")
    ax.set_ylim(0, 1.05)
    ax.set_title("Classifier CV Metrics", color="w", fontweight="bold")
    ax.tick_params(colors="w")
    ax.grid(alpha=0.2, color="w")
    [s.set_color("w") for s in ax.spines.values()]
    ax.legend(frameon=False, labelcolor="white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="#111111")
    plt.close()


def plot_refinement_curve(hist_df, save_path):
    if hist_df is None or len(hist_df) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor="#111111")
    ax.set_facecolor("#1a1a2e")
    x = hist_df["epoch"].astype(int).values
    for col, color in [("ana_score", "#e74c3c"), ("pbs_score", "#4a90d9"), ("sep", "#2ecc71")]:
        if col in hist_df.columns:
            ax.plot(x, hist_df[col].values, marker="o", linewidth=2, label=col, color=color)
    ax.set_xlabel("Epoch", color="w")
    ax.set_ylabel("Score", color="w")
    ax.set_title("Refinement Training Curve", color="w", fontweight="bold")
    ax.tick_params(colors="w")
    ax.grid(alpha=0.2, color="w")
    [s.set_color("w") for s in ax.spines.values()]
    ax.legend(frameon=False, labelcolor="white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="#111111")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Gas-domain hybrid transfer: retrain handcrafted classifier + fine-tune refinement model.")
    parser.add_argument("--data-dir", default=r"data/gas")
    parser.add_argument("--output-dir", default="gas_hybrid_transfer_results")
    parser.add_argument("--base-output-dir", default=r"outputs/hybrid")
    parser.add_argument("--refine-epochs", type=int, default=8)
    parser.add_argument("--refined-thr-percentile", type=float, default=95.0)
    parser.add_argument("--refine-overlay-sigma", type=float, default=3.5)
    parser.add_argument("--feature-bank-mode", choices=["core30", "extended48"], default="core30")
    parser.add_argument("--shap-top-n", type=int, default=0)
    parser.add_argument("--reverse-analyte", action="store_true", default=True)
    parser.add_argument("--no-reverse-analyte", dest="reverse_analyte", action="store_false")
    return parser.parse_args()


def build_classifier(base_clf):
    clf_params = base_clf.get_params()
    clf_params["n_jobs"] = 1
    return lgb.LGBMClassifier(**clf_params)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(args.base_output_dir)
    data_root = resolve_data_root(args.data_dir, out_dir)
    final_clf_dir = base_dir / "final_classifier"
    base_refine_path = base_dir / "stage2a_refine.pt"

    base_meta = json.loads((final_clf_dir / "best_classifier_meta.json").read_text(encoding="utf-8"))
    base_clf = joblib.load(final_clf_dir / "best_classifier.joblib")
    refine_ckpt = torch.load(base_refine_path, map_location=DEVICE, weights_only=False)

    cfg = dict(CFG)
    cfg.update({k: refine_ckpt["cfg"][k] for k in ["img_size", "mask_thr", "mask_erode", "feature_bank_mode", "score_primary", "support_z_thr"] if k in refine_ckpt["cfg"]})
    cfg["refine_epochs"] = args.refine_epochs
    cfg["feature_bank_mode"] = args.feature_bank_mode

    size = (cfg["img_size"], cfg["img_size"])
    led_map = build_led_map(data_root / cfg["led_subdir"], size, cfg["led_bright_thr"], cfg["led_smooth_sigma"])

    pbs_a, pbs_b, pbs_n = load_pairs(data_root / cfg["pbs_subdir"], data_root / cfg["substrate_subdir"], "pbs", size, led_map=led_map)
    ana_a, ana_b, ana_n = load_pairs(data_root / cfg["analyte_subdir"], data_root / cfg["substrate_subdir"], "analyte", size, led_map=led_map)
    if args.reverse_analyte:
        ana_a, ana_b = ana_b, ana_a
        print("OK Signal-decrease adaptation enabled: reverse analyte pairs only; keep PBS unchanged")
    all_after = pbs_a + ana_a
    all_before = pbs_b + ana_b
    all_labels = np.array([0] * len(pbs_a) + [1] * len(ana_a), dtype=int)
    all_names = pbs_n + ana_n
    feature_cols = list(base_meta["feature_cols"])
    shap_csv = base_dir / "shap_top20.csv"
    if args.shap_top_n > 0 and shap_csv.exists():
        shap_features = pd.read_csv(shap_csv)["feature"].astype(str).tolist()[: args.shap_top_n]
        feature_cols = [c for c in feature_cols if c in set(shap_features)]
        print(f"OK Using source-model SHAP top-{len(feature_cols)} features for classifier transfer")

    def _subset(seq, idxs):
        return [seq[int(i)] for i in idxs]

    def _build_feature_table(eval_after, eval_before, eval_labels, eval_names, train_after, train_before, train_labels):
        mask_local = compute_mask(train_after, thr=cfg["mask_thr"], erode=cfg["mask_erode"])
        train_pbs_after = [img for img, lbl in zip(train_after, train_labels) if lbl == 0]
        train_pbs_before = [img for img, lbl in zip(train_before, train_labels) if lbl == 0]
        pbs_mu_px_local, pbs_std_px_local = build_pbs_pixel_baseline(
            train_pbs_after,
            train_pbs_before,
            mask_local,
            mode=cfg["feature_bank_mode"],
        )
        maps_local = [
            build_score_maps(bef, aft, pbs_mu_px_local, pbs_std_px_local, mask_local, cfg)
            for aft, bef in zip(eval_after, eval_before)
        ]
        rows_local = []
        for nm, lbl, maps in zip(eval_names, eval_labels, maps_local):
            row = summarize_image_features(maps, mask_local)
            row.update({"name": nm, "label": int(lbl)})
            rows_local.append(row)
        return pd.DataFrame(rows_local), maps_local, mask_local, pbs_mu_px_local, pbs_std_px_local

    base_idx = np.arange(len(all_labels))
    n_splits_eff = min(cfg["n_splits"], max(2, len(all_labels)))
    fold_splits = make_balanced_folds(all_names, all_labels, n_splits_eff, random_state=cfg["random_state"])
    tr_idx_eval, va_idx_eval = fold_splits[0]
    print(f"OK Evaluation split: train={len(tr_idx_eval)}  val={len(va_idx_eval)}")

    cv_rows = []
    cv_detail_rows = []
    if len(fold_splits) >= 2 and len(np.unique(all_labels)) == 2:
        for fold, (tr_idx, va_idx) in enumerate(fold_splits, 1):
            tr_after = _subset(all_after, tr_idx)
            tr_before = _subset(all_before, tr_idx)
            tr_labels = all_labels[tr_idx]
            tr_names = _subset(all_names, tr_idx)
            va_after = _subset(all_after, va_idx)
            va_before = _subset(all_before, va_idx)
            va_labels = all_labels[va_idx]
            va_names = _subset(all_names, va_idx)

            tr_df_fold, _, _, _, _ = _build_feature_table(tr_after, tr_before, tr_labels, tr_names, tr_after, tr_before, tr_labels)
            va_df_fold, _, _, _, _ = _build_feature_table(va_after, va_before, va_labels, va_names, tr_after, tr_before, tr_labels)
            fold_feature_cols = feature_cols if cfg["feature_bank_mode"] == "extended48" else [c for c in tr_df_fold.columns if c not in ["name", "label"]]

            clf = build_classifier(base_clf)
            clf.fit(tr_df_fold[fold_feature_cols].values.astype(np.float32), tr_labels)
            tr_prob = clf.predict_proba(tr_df_fold[fold_feature_cols].values.astype(np.float32))[:, 1]
            thr, tr_acc, tr_bal = find_best_threshold(tr_labels, tr_prob)
            prob = clf.predict_proba(va_df_fold[fold_feature_cols].values.astype(np.float32))[:, 1]
            pred = (prob >= thr).astype(int)
            for nm, y_true_i, prob_i, pred_i in zip(va_names, va_labels, prob, pred):
                true_label = "Analyte" if int(y_true_i) == 1 else "PBS"
                pred_label = "Analyte" if int(pred_i) == 1 else "PBS"
                cv_detail_rows.append(
                    {
                        "fold": fold,
                        "name": nm,
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "prob_analyte": float(prob_i),
                        "confidence": float(max(prob_i, 1.0 - prob_i)),
                        "correct": bool(int(pred_i) == int(y_true_i)),
                        "decision_threshold": float(thr),
                    }
                )
            row = {"fold": fold, "decision_threshold": thr, "train_accuracy_at_threshold": tr_acc, "train_balanced_accuracy_at_threshold": tr_bal}
            row.update(compute_metrics(va_labels, pred, prob))
            cv_rows.append(row)
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(out_dir / "classifier_cv_results.csv", index=False)
    plot_classifier_cv_curves(cv_df, out_dir / "classifier_cv_metrics.png")
    cv_detail_df = pd.DataFrame(cv_detail_rows)
    if len(cv_detail_df):
        cv_detail_df = cv_detail_df.sort_values(["fold", "true_label", "name"]).reset_index(drop=True)
        cv_detail_df.to_csv(out_dir / "classifier_cv_fold_details.csv", index=False)
        cv_detail_df[cv_detail_df["correct"]].to_csv(out_dir / "classifier_cv_fold_details_correct.csv", index=False)
        cv_detail_df[~cv_detail_df["correct"]].to_csv(out_dir / "classifier_cv_fold_details_errors.csv", index=False)
        fold_label_summary = (
            cv_detail_df.groupby(["fold", "true_label"])
            .agg(
                total=("name", "count"),
                correct=("correct", "sum"),
                mean_prob_analyte=("prob_analyte", "mean"),
                mean_confidence=("confidence", "mean"),
            )
            .reset_index()
        )
        fold_label_summary["accuracy_within_label"] = fold_label_summary["correct"] / fold_label_summary["total"]
        fold_label_summary.to_csv(out_dir / "classifier_cv_fold_label_summary.csv", index=False)

    full_df, full_maps, mask, pbs_mu_px, pbs_std_px = _build_feature_table(
        all_after,
        all_before,
        all_labels,
        all_names,
        all_after,
        all_before,
        all_labels,
    )
    if cfg["feature_bank_mode"] != "extended48":
        feature_cols = [c for c in full_df.columns if c not in ["name", "label"]]
        print(f"OK Classifier transfer uses {cfg['feature_bank_mode']} feature bank; {len(feature_cols)} features")
    x = full_df[feature_cols].values.astype(np.float32)
    y = full_df["label"].values.astype(int)
    clf = build_classifier(base_clf)
    clf.fit(x, y)
    full_prob = clf.predict_proba(x)[:, 1]
    final_thr, final_thr_acc, final_thr_bal = find_best_threshold(y, full_prob)
    joblib.dump(clf, out_dir / "transferred_classifier.joblib")
    (out_dir / "transferred_classifier_meta.json").write_text(
        json.dumps(
            {
                "feature_cols": feature_cols,
                "base_model_name": base_meta["best_model_name"],
                "classifier_model": "lightgbm",
                "decision_threshold": final_thr,
                "train_accuracy_at_threshold": final_thr_acc,
                "train_balanced_accuracy_at_threshold": final_thr_bal,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    train_after_ref = _subset(all_after, tr_idx_eval)
    train_before_ref = _subset(all_before, tr_idx_eval)
    train_labels_ref = all_labels[tr_idx_eval]
    val_after_ref = _subset(all_after, va_idx_eval)
    val_before_ref = _subset(all_before, va_idx_eval)
    val_labels_ref = all_labels[va_idx_eval]
    mask_ref = compute_mask(train_after_ref, thr=cfg["mask_thr"], erode=cfg["mask_erode"])
    train_pbs_after_ref = [img for img, lbl in zip(train_after_ref, train_labels_ref) if lbl == 0]
    train_pbs_before_ref = [img for img, lbl in zip(train_before_ref, train_labels_ref) if lbl == 0]
    pbs_mu29, pbs_std29 = build_pbs_baseline_29(train_pbs_after_ref, train_pbs_before_ref, mask_ref)
    cm, cs = _compute_channel_stats(train_after_ref, train_before_ref)
    tr_ds = Refine29Dataset(train_after_ref, train_before_ref, train_labels_ref, pbs_mu29, pbs_std29, mask_ref, cm, cs)
    va_ds = Refine29Dataset(val_after_ref, val_before_ref, val_labels_ref, pbs_mu29, pbs_std29, mask_ref, cm, cs)
    tr_ld = DataLoader(tr_ds, batch_size=cfg["refine_batch_size"], shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    va_ld = DataLoader(va_ds, batch_size=cfg["refine_batch_size"], shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    model = ResNet18_29(29).to(DEVICE)
    model.load_state_dict(refine_ckpt["model_state"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["refine_lr"], weight_decay=cfg["refine_wd"])
    best_sep, wait, best_state = -1e9, 0, None
    refine_hist = []
    for ep in range(1, cfg["refine_epochs"] + 1):
        model.train()
        for feat, soft, _ in tr_ld:
            feat, soft = feat.to(DEVICE), soft.to(DEVICE)
            opt.zero_grad()
            _refine_loss(model(feat), soft, cfg["tv_weight"]).backward()
            opt.step()
        model.eval()
        scores_ana, scores_pbs = [], []
        with torch.no_grad():
            for feat, _, lbl in va_ld:
                hm = torch.sigmoid(model(feat.to(DEVICE))).squeeze(1).cpu().numpy()
                for ib, lb in enumerate(lbl.tolist()):
                    flat = hm[ib].ravel()
                    k = max(1, int(len(flat) * 0.2))
                    sc = np.partition(flat, -k)[-k:].mean()
                    (scores_ana if lb == 1 else scores_pbs).append(sc)
        sep = (np.mean(scores_ana) if scores_ana else 0.0) - 0.5 * (np.mean(scores_pbs) if scores_pbs else 1.0)
        refine_hist.append(
            {
                "epoch": ep,
                "ana_score": float(np.mean(scores_ana) if scores_ana else 0.0),
                "pbs_score": float(np.mean(scores_pbs) if scores_pbs else 0.0),
                "sep": float(sep),
            }
        )
        print(f"refine ep{ep:02d}: ana={np.mean(scores_ana) if scores_ana else 0:.4f} pbs={np.mean(scores_pbs) if scores_pbs else 0:.4f} sep={sep:.4f}")
        if sep > best_sep:
            best_sep, wait = sep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg["refine_patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    refine_hist_df = pd.DataFrame(refine_hist)
    refine_hist_df.to_csv(out_dir / "refinement_training_curve.csv", index=False)
    plot_refinement_curve(refine_hist_df, out_dir / "refinement_training_curve.png")

    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": cfg,
            "channel_mean": cm.tolist(),
            "channel_std": cs.tolist(),
            "pbs_mu_px": pbs_mu_px,
            "pbs_std_px": pbs_std_px,
            "pbs_mu29": pbs_mu29,
            "pbs_std29": pbs_std29,
        },
        out_dir / "transferred_stage2a_refine.pt",
    )

    pbs_maps = full_maps[: len(pbs_n)]
    ana_maps = full_maps[len(pbs_n) :]
    records = []
    for nm, lbl, aft, bef, maps in zip(all_names, all_labels, all_after, all_before, pbs_maps + ana_maps):
        row = summarize_image_features(maps, mask)
        prob = float(clf.predict_proba(pd.DataFrame([row])[feature_cols].values.astype(np.float32))[:, 1][0])
        feat29 = extract_feature_bank_29(bef, aft)
        feat29 = (feat29 - cm[:, None, None]) / (cs[:, None, None] + 1e-6)
        with torch.no_grad():
            refined = torch.sigmoid(model(torch.from_numpy(feat29[None]).to(DEVICE)))[0, 0].cpu().numpy().astype(np.float32)
        refined = (refined - refined.min()) / (refined.max() - refined.min() + 1e-8)
        records.append(
            {
                "name": nm,
                "label": "Analyte" if lbl == 1 else "PBS",
                "prob": prob,
                "pred_label": "Analyte" if prob >= final_thr else "PBS",
                "score_main": refined,
                "score": refined,
                "after": aft,
                "before": bef,
                "coarse": maps["primary"],
                "refined": refined,
            }
        )

    ana_rec = [r for r in records if r["label"] == "Analyte"]
    pbs_rec = [r for r in records if r["label"] == "PBS"]
    pbs_vals = np.concatenate([r["refined"][mask] for r in pbs_rec]) if pbs_rec else np.array([0.0], np.float32)
    cfg_plot = dict(
        cfg,
        cam_thr=float(np.percentile(pbs_vals, args.refined_thr_percentile)),
        cam_sigma=args.refine_overlay_sigma,
    )
    plot_grid(ana_rec, mask, "Analyte - transferred classifier + refinement", out_dir / "analyte_overlay.png", cfg_plot)
    plot_grid(pbs_rec, mask, "PBS - transferred classifier + refinement", out_dir / "pbs_control.png", cfg_plot)
    plot_score_dist(ana_rec, pbs_rec, out_dir / "score_distribution.png")

    result_df = pd.DataFrame(
        [
            {
                "name": r["name"],
                "label": r["label"],
                "prob_analyte": r["prob"],
                "predicted_label": r["pred_label"],
                "confidence": float(max(r["prob"], 1.0 - r["prob"])),
                "correct": bool(r["pred_label"] == r["label"]),
                "coarse_max": float(r["coarse"].max()),
                "refined_max": float(r["refined"].max()),
                "refined_q95": float(np.quantile(r["refined"][mask], 0.95)),
            }
            for r in records
        ]
    ).sort_values(["label", "name"])
    result_df.to_csv(out_dir / "final_prediction_summary.csv", index=False)
    result_df.to_csv(out_dir / "sample_classification_details.csv", index=False)
    result_df[result_df["label"] == "PBS"].to_csv(out_dir / "pbs_classification_details.csv", index=False)
    result_df[result_df["label"] == "Analyte"].to_csv(out_dir / "analyte_classification_details.csv", index=False)
    label_summary = (
        result_df.groupby("label")
        .agg(
            total=("name", "count"),
            correct=("correct", "sum"),
            mean_prob_analyte=("prob_analyte", "mean"),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    label_summary["accuracy_within_label"] = label_summary["correct"] / label_summary["total"]
    label_summary.to_csv(out_dir / "label_classification_summary.csv", index=False)

    zip_path = shutil.make_archive(str(out_dir), "zip", str(out_dir))
    print(f"OK Transferred classifier: {out_dir / 'transferred_classifier.joblib'}")
    print(f"OK Transferred localization model: {out_dir / 'transferred_stage2a_refine.pt'}")
    print(f"OK Results summary: {out_dir / 'final_prediction_summary.csv'}")
    print(f"OK Output archive: {zip_path}")


if __name__ == "__main__":
    main()
