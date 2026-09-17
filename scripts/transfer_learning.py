import argparse
import os
import shutil
import tempfile
import warnings
import zipfile
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, uniform_filter, sobel
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THERMAL = LinearSegmentedColormap.from_list(
    "thermal",
    [
        (0.00, "#000010"),
        (0.05, "#00003a"),
        (0.15, "#0000cd"),
        (0.30, "#005f8f"),
        (0.45, "#00c8c8"),
        (0.58, "#00c800"),
        (0.70, "#c8c800"),
        (0.82, "#ff6400"),
        (0.91, "#ff0000"),
        (1.00, "#ffffff"),
    ],
    N=512,
)

CFG = {
    "data_dir": "data",
    "led_subdir": "led_field",
    "analyte_subdir": "analyte",
    "pbs_subdir": "pbs",
    "substrate_subdir": "susbtrat",
    "analyte_prefix": "ab",
    "pbs_prefix": "ap",
    "img_size": 128,
    "mask_thr": 15,
    "mask_erode": 8,
    "led_bright_thr": 0.5,
    "led_smooth_sigma": 15,
    "epochs": 150,
    "batch_size": 8,
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "dropout": 0.7,
    "pretrained": True,
    "val_ratio": 0.25,
    "freeze_backbone": True,
    "early_stop_patience": 20,
    "cam_thr": 0.30,
    "cam_sigma": 3.5,
    "feature_bank_mode": "extended48",
    "score_primary": "top3_weighted",
    "response_topk": 5,
    "support_z_thr": 2.5,
    "output_dir": "results",
    "model_save_path": "saved_model.pt",
    "dpi": 200,
}

CORE30_NAMES = (
    ["diff_raw", "log_ratio"]
    + [f"gauss_diff_s{s}" for s in [1, 2, 4, 8, 12]]
    + [f"gauss_logr_s{s}" for s in [1, 2, 4, 8, 12]]
    + [f"DoG_{a}_{b}" for a, b in [(1, 2), (2, 4), (4, 8), (1, 4)]]
    + [f"local_mean_w{w}" for w in [7, 13, 21]]
    + [f"local_std_w{w}" for w in [7, 13, 21]]
    + [f"local_snr_w{w}" for w in [7, 13, 21]]
    + [f"deltaK_w{w}" for w in [7, 13]]
    + ["low_freq", "high_freq_sq"]
)

EXT18_NAMES = [
    "relative_diff",
    "sym_ratio",
    "anscombe_diff",
    "local_normdiff_w7",
    "local_normdiff_w13",
    "corr_drop_w7",
    "corr_drop_w13",
    "gradmag_diff_s1",
    "gradmag_diff_s2",
    "laplacian_diff",
    "log_absdiff_s1",
    "log_absdiff_s2",
    "absdiff_mean_w7",
    "absdiff_mean_w13",
    "absdiff_std_w7",
    "absdiff_std_w13",
    "energy_ratio_w7",
    "energy_ratio_w13",
]

FEAT_NAMES = list(CORE30_NAMES) + list(EXT18_NAMES)


def build_led_map(led_dir, target_size, bright_thr=0.5, sigma=15):
    led_dir = Path(led_dir)
    if not led_dir.exists():
        print("  [LED] led_field directory not found; skipping flat-field correction")
        return None
    from skimage.filters import threshold_otsu

    bright_imgs = []
    for f in sorted(led_dir.iterdir()):
        if f.suffix.lower() not in [".png", ".tif", ".tiff", ".jpg"]:
            continue
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img_f = img.astype(np.float32)
        if img_f.max() > bright_thr * 255:
            img_f = cv2.resize(img_f, target_size, interpolation=cv2.INTER_LINEAR)
            bright_imgs.append(img_f)

    if not bright_imgs:
        print("  [LED] No bright frames found; skipping correction")
        return None

    print(f"  [LED] Total {len(bright_imgs)} bright frames; computing illumination map...")
    mean_led = np.stack(bright_imgs).mean(0)
    norm = mean_led / (mean_led.max() + 1e-8)
    try:
        thr = threshold_otsu(norm)
    except Exception:
        thr = 0.3
    binary = (norm > thr).astype(np.uint8)
    rows = np.where(binary.any(axis=1))[0]
    cols = np.where(binary.any(axis=0))[0]
    if len(rows) > 0 and len(cols) > 0:
        m = 5
        r0, r1 = max(rows[0] + m, 0), min(rows[-1] - m, mean_led.shape[0])
        c0, c1 = max(cols[0] + m, 0), min(cols[-1] - m, mean_led.shape[1])
        led_crop = mean_led[r0:r1, c0:c1]
        print(f"  [LED] Bright region: {r1-r0}x{c1-c0}px -> resize to {target_size[1]}x{target_size[0]}")
    else:
        led_crop = mean_led
    ksize = int(6 * sigma + 1) | 1
    led_s = cv2.GaussianBlur(led_crop, (ksize, ksize), sigma)
    led_r = cv2.resize(led_s, target_size, interpolation=cv2.INTER_LINEAR)
    led_r /= np.percentile(led_r, 99) + 1e-8
    led_map = np.clip(led_r, 0.05, None)
    print(f"  [LED] LED map range: [{led_map.min():.3f}, {led_map.max():.3f}]")
    return led_map.astype(np.float32)


def apply_flatfield(img, led_map):
    if led_map is None:
        return img
    corrected = img / (led_map + 1e-8)
    scale = img.mean() / (corrected.mean() + 1e-8)
    return np.clip(corrected * scale, 0, 255).astype(np.float32)


def load_img(path, size):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)


def load_pairs(after_dir, sub_dir, prefix, size, led_map=None):
    a_list, b_list, names = [], [], []
    for f in sorted(Path(after_dir).iterdir()):
        if f.suffix.lower() not in [".png", ".tif", ".tiff", ".jpg"]:
            continue
        bp = Path(sub_dir) / f.name
        if not bp.exists():
            continue
        a = load_img(f, size)
        b = load_img(bp, size)
        if a is None or b is None:
            continue
        if led_map is not None:
            a = apply_flatfield(a, led_map)
            b = apply_flatfield(b, led_map)
        a_list.append(a)
        b_list.append(b)
        names.append(f.name)
    return a_list, b_list, names


def compute_mask(imgs, thr=15, erode=8):
    m = (np.stack(imgs).mean(0) > thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.erode(m, k, iterations=erode).astype(bool)


class BioDataset(Dataset):
    def __init__(self, pbs_a, pbs_b, ana_a, ana_b, img_size=128, augment=False):
        self.samples = []

        def rs(imgs):
            return [cv2.resize(im, (img_size, img_size), interpolation=cv2.INTER_LINEAR) for im in imgs]

        for a, b in zip(rs(pbs_a), rs(pbs_b)):
            self.samples.append((b, a, 0))
        for a, b in zip(rs(ana_a), rs(ana_b)):
            self.samples.append((b, a, 1))
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        before, after, label = self.samples[idx]
        b01 = before / 255.0
        a01 = after / 255.0
        d01 = a01 - b01
        img = np.stack([b01, a01, d01], 0).astype(np.float32)
        if self.augment:
            if np.random.rand() > 0.5:
                img = np.flip(img, 2).copy()
            if np.random.rand() > 0.5:
                img = np.flip(img, 1).copy()
            img = np.rot90(img, np.random.randint(0, 4), (1, 2)).copy()
            h, w = img.shape[1], img.shape[2]
            scale = np.random.uniform(0.8, 1.0)
            ch, cw = int(h * scale), int(w * scale)
            y0 = np.random.randint(0, h - ch + 1)
            x0 = np.random.randint(0, w - cw + 1)
            crop = img[:, y0 : y0 + ch, x0 : x0 + cw]
            img = np.stack([cv2.resize(crop[c], (w, h)) for c in range(3)])
            if np.random.rand() > 0.5:
                dx = int(np.random.uniform(-0.1, 0.1) * w)
                dy = int(np.random.uniform(-0.1, 0.1) * h)
                m = np.float32([[1, 0, dx], [0, 1, dy]])
                img = np.stack([cv2.warpAffine(img[c], m, (w, h)) for c in range(3)])
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.85, 1.15)
                beta = np.random.uniform(-0.05, 0.05)
                img = np.clip(img * alpha + beta, -1, 2).astype(np.float32)
            img = np.clip(img + np.random.normal(0, 0.01, img.shape).astype(np.float32), -1, 2)
        return torch.FloatTensor(img), torch.tensor(label, dtype=torch.long)


class BiosensorNet(nn.Module):
    def __init__(self, pretrained=True, dropout=0.5, freeze=False):
        super().__init__()
        bb = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        if freeze:
            for name, param in bb.named_parameters():
                if not name.startswith("layer4") and not name.startswith("fc"):
                    param.requires_grad = False
        in_f = bb.fc.in_features
        bb.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_f, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 2),
        )
        self.bb = bb
        self._grads = None
        self._acts = None
        self.bb.layer4.register_forward_hook(self._save_act)
        self.bb.layer4.register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o):
        self._acts = o.detach()

    def _save_grad(self, m, gi, go):
        self._grads = go[0].detach()

    def forward(self, x):
        return self.bb(x)

    def grad_cam(self, x_tensor, target_class=1):
        self.eval()
        x = x_tensor.unsqueeze(0).to(DEVICE)
        x.requires_grad_(True)
        logits = self(x)
        self.zero_grad()
        logits[0, target_class].backward()
        w = self._grads[0].mean(dim=(1, 2))
        cam = F.relu((w[:, None, None] * self._acts[0]).sum(0))
        cam = cam.cpu().numpy()
        return cam / cam.max() if cam.max() > 0 else cam


def train_model(model, tr_loader, val_loader, cfg):
    ds = tr_loader.dataset
    if hasattr(ds, "dataset"):
        ds = ds.dataset
    labels_all = [s[2] for s in ds.samples]
    n_pos = sum(labels_all)
    n_neg = len(labels_all) - n_pos
    w = torch.FloatTensor([n_pos / len(labels_all), n_neg / len(labels_all)]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_auc": [], "lr": []}
    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience = cfg.get("early_stop_patience", 20)

    for ep in range(cfg["epochs"]):
        model.train()
        tl = tc = tt = 0
        for imgs, lbs in tr_loader:
            imgs, lbs = imgs.to(DEVICE), lbs.to(DEVICE)
            opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
            tc += (out.argmax(1) == lbs).sum().item()
            tt += len(lbs)
        sched.step()

        model.eval()
        vl = vc = vt = 0
        probs, lbls = [], []
        with torch.no_grad():
            for imgs, lbs in val_loader:
                imgs, lbs = imgs.to(DEVICE), lbs.to(DEVICE)
                out = model(imgs)
                loss = criterion(out, lbs)
                vl += loss.item()
                vc += (out.argmax(1) == lbs).sum().item()
                vt += len(lbs)
                probs += F.softmax(out, 1)[:, 1].cpu().tolist()
                lbls += lbs.cpu().tolist()

        tl /= len(tr_loader)
        vl /= len(val_loader)
        ta = tc / tt if tt > 0 else 0
        va = vc / vt if vt > 0 else 0
        try:
            auc = roc_auc_score(lbls, probs)
        except Exception:
            auc = 0.5

        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)
        hist["train_acc"].append(ta)
        hist["val_acc"].append(va)
        hist["val_auc"].append(auc)
        hist["lr"].append(opt.param_groups[0]["lr"])

        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"  Epoch {ep+1:3d}/{cfg['epochs']}  "
                f"train={tl:.4f}/{ta:.2%}  val={vl:.4f}/{va:.2%}  AUC={auc:.3f}"
            )

        if patience_counter >= patience:
            print(f"  STOP Early stopping at epoch {ep+1} (no val_loss improvement for {patience} epochs)")
            break

    if best_state:
        model.load_state_dict(best_state)
    print(f"  OK Complete  best_val_loss={best_loss:.4f}")
    return hist


def _local_mean_std(x, w):
    mu = uniform_filter(x, size=w)
    mu2 = uniform_filter(x * x, size=w)
    var = np.clip(mu2 - mu * mu, 0, None)
    return mu, np.sqrt(var)


def _grad_mag(x, sigma=1):
    xs = gaussian_filter(x, sigma=sigma)
    gx = sobel(xs, axis=1)
    gy = sobel(xs, axis=0)
    return np.sqrt(gx * gx + gy * gy)


def _local_corr_drop(a, b, w):
    ma = uniform_filter(a, size=w)
    mb = uniform_filter(b, size=w)
    va = np.clip(uniform_filter(a * a, size=w) - ma * ma, 0, None)
    vb = np.clip(uniform_filter(b * b, size=w) - mb * mb, 0, None)
    cov = uniform_filter(a * b, size=w) - ma * mb
    corr = cov / (np.sqrt(va * vb) + 1e-6)
    return 1.0 - np.clip(corr, -1, 1)


def _feature_names_by_cfg(cfg):
    mode = cfg.get("feature_bank_mode", "extended48") if isinstance(cfg, dict) else "extended48"
    return list(CORE30_NAMES) if mode == "core30" else list(FEAT_NAMES)


def _pixel_features(before, after, cfg=None):
    mode = cfg.get("feature_bank_mode", "extended48") if isinstance(cfg, dict) else "extended48"
    before = before.astype(np.float32)
    after = after.astype(np.float32)
    diff = after - before
    log_r = np.log(after + 1.0) - np.log(before + 1.0)
    feats = [diff, log_r]
    for s in [1, 2, 4, 8, 12]:
        feats.append(gaussian_filter(diff, sigma=s))
        feats.append(gaussian_filter(log_r, sigma=s))
    for s1, s2 in [(1, 2), (2, 4), (4, 8), (1, 4)]:
        feats.append(gaussian_filter(diff, s1) - gaussian_filter(diff, s2))
    for w in [7, 13, 21]:
        dsm = gaussian_filter(diff, sigma=w / 5)
        mu, sd = _local_mean_std(dsm, w)
        feats += [mu, sd, mu / (sd + 1e-4)]
    for w in [7, 13]:
        ma, sda = _local_mean_std(after, w)
        mb, sdb = _local_mean_std(before, w)
        ka = sda / (ma + 1e-4)
        kb = sdb / (mb + 1e-4)
        feats.append(ka - kb)
    lf = gaussian_filter(diff, sigma=6)
    feats += [lf, (diff - lf) ** 2]

    if mode != "core30":
        rel_diff = diff / (before + 5.0)
        sym_ratio = diff / (after + before + 5.0)
        anscombe_b = 2.0 * np.sqrt(np.clip(before, 0, None) + 3.0 / 8.0)
        anscombe_a = 2.0 * np.sqrt(np.clip(after, 0, None) + 3.0 / 8.0)
        anscombe_diff = anscombe_a - anscombe_b
        _, sd7 = _local_mean_std(before, 7)
        _, sd13 = _local_mean_std(before, 13)
        local_norm7 = diff / (sd7 + 1e-4)
        local_norm13 = diff / (sd13 + 1e-4)
        corr7 = _local_corr_drop(after, before, 7)
        corr13 = _local_corr_drop(after, before, 13)
        g1 = _grad_mag(after, 1) - _grad_mag(before, 1)
        g2 = _grad_mag(after, 2) - _grad_mag(before, 2)
        lap = cv2.Laplacian(after, cv2.CV_32F) - cv2.Laplacian(before, cv2.CV_32F)
        log_abs1 = np.abs(gaussian_filter(lap, 1))
        log_abs2 = np.abs(gaussian_filter(lap, 2))
        absd = np.abs(diff)
        abs_mu7, abs_sd7 = _local_mean_std(absd, 7)
        abs_mu13, abs_sd13 = _local_mean_std(absd, 13)
        e_hi = (diff - gaussian_filter(diff, sigma=2)) ** 2
        e_lo = gaussian_filter(diff, sigma=4) ** 2
        er7 = uniform_filter(e_hi, 7) / (uniform_filter(e_lo, 7) + 1e-4)
        er13 = uniform_filter(e_hi, 13) / (uniform_filter(e_lo, 13) + 1e-4)
        feats += [
            rel_diff,
            sym_ratio,
            anscombe_diff,
            local_norm7,
            local_norm13,
            corr7,
            corr13,
            g1,
            g2,
            lap,
            log_abs1,
            log_abs2,
            abs_mu7,
            abs_mu13,
            abs_sd7,
            abs_sd13,
            er7,
            er13,
        ]
    return np.stack(feats, axis=0)


def build_pbs_pixel_baseline(pbs_after_list, pbs_before_list, mask, cfg=None):
    stack = np.stack([_pixel_features(b, a, cfg) for a, b in zip(pbs_after_list, pbs_before_list)])
    mu = stack.mean(0)
    std = stack.std(0, ddof=1)
    for f in range(std.shape[0]):
        floor = np.percentile(std[f][mask], 10) if mask.any() else 1e-4
        std[f] = np.clip(std[f], max(floor, 1e-4), None)
    return mu, std


def _aggregate_response_maps(z_abs, mask, cfg):
    top1 = z_abs.max(0)
    top3 = np.sort(z_abs, axis=0)[-3:]
    top5 = np.sort(z_abs, axis=0)[-5:]
    top3_mean = top3.mean(0)
    top3_weighted = 0.6 * top3[-1] + 0.3 * top3[-2] + 0.1 * top3[-3]
    top5_rms = np.sqrt(np.mean(top5**2, axis=0))
    support_thr = cfg.get("support_z_thr", 2.5) if isinstance(cfg, dict) else 2.5
    support_count = (z_abs > support_thr).sum(0).astype(np.float32)
    support_frac = support_count / float(z_abs.shape[0])
    maps = {
        "top1": np.where(mask, top1, 0),
        "top3_mean": np.where(mask, top3_mean, 0),
        "top3_weighted": np.where(mask, top3_weighted, 0),
        "top5_rms": np.where(mask, top5_rms, 0),
        "support_count": np.where(mask, support_count, 0),
        "support_frac": np.where(mask, support_frac, 0),
    }
    primary = cfg.get("score_primary", "top3_weighted") if isinstance(cfg, dict) else "top3_weighted"
    maps["primary"] = maps.get(primary, maps["top3_weighted"])
    return maps


def _pixel_score(after, before, pbs_mu, pbs_std, mask, cfg=None):
    fm = _pixel_features(before, after, cfg)
    z = np.where(mask[None, :, :], (fm - pbs_mu) / (pbs_std + 1e-8), 0)
    z_abs = np.abs(z)
    maps = _aggregate_response_maps(z_abs, mask, cfg if cfg is not None else {})
    maps["z_abs_mean"] = np.where(mask, z_abs.mean(0), 0)
    maps["feature_stack"] = fm
    return maps


def infer_all(model, after_list, before_list, names, mask, cfg, pbs_mu=None, pbs_std=None, pbs_null_mu=None, pbs_null_std=None):
    model.eval()
    s = cfg["img_size"]
    records = []
    for after, before, name in zip(after_list, before_list, names):
        b01 = cv2.resize(before / 255.0, (s, s))
        a01 = cv2.resize(after / 255.0, (s, s))
        t = torch.FloatTensor(np.stack([b01, a01, a01 - b01])[None])
        with torch.no_grad():
            prob = float(F.softmax(model(t.to(DEVICE)), 1)[0, 1].cpu())

        if pbs_mu is not None:
            maps_raw = _pixel_score(after, before, pbs_mu, pbs_std, mask, cfg)
            maps_norm = {}
            for k, arr in maps_raw.items():
                if k == "feature_stack":
                    continue
                if pbs_null_mu is not None and k == "primary":
                    z2 = np.where(mask, (arr - pbs_null_mu) / (pbs_null_std + 1e-8), 0)
                    z_pos = np.clip(z2, 0, None)
                    mx = z_pos[mask].max() if mask.any() else 1.0
                    maps_norm[k] = np.log1p(z_pos) / (np.log1p(mx) + 1e-8) if mx > 0 else z_pos
                else:
                    cur = np.where(mask, arr, 0)
                    mx = cur[mask].max() if mask.any() else 1.0
                    maps_norm[k] = cur / (mx + 1e-8) if mx > 0 else cur
            score_primary = maps_norm["primary"]
        else:
            cam_s = model.grad_cam(t[0], target_class=1)
            cam_h = cv2.resize(cam_s, (after.shape[1], after.shape[0]), interpolation=cv2.INTER_CUBIC)
            maps_raw = {
                "top1": cam_h,
                "top3_mean": cam_h,
                "top3_weighted": cam_h,
                "top5_rms": cam_h,
                "support_count": cam_h,
                "support_frac": cam_h,
                "primary": cam_h,
            }
            maps_norm = maps_raw.copy()
            score_primary = np.clip(cam_h, 0, 1)

        records.append(
            {
                "name": name,
                "prob": prob,
                "score": np.where(mask, score_primary, 0),
                "score_main": np.where(mask, score_primary, 0),
                "score_top1": np.where(mask, maps_norm["top1"], 0),
                "score_top3mean": np.where(mask, maps_norm["top3_mean"], 0),
                "score_top3weighted": np.where(mask, maps_norm["top3_weighted"], 0),
                "score_top5rms": np.where(mask, maps_norm["top5_rms"], 0),
                "score_support_count": np.where(mask, maps_norm["support_count"], 0),
                "score_support_frac": np.where(mask, maps_norm["support_frac"], 0),
                "score_raw_primary": np.where(mask, maps_raw["primary"], 0),
                "feature_stack": maps_raw.get("feature_stack", None),
                "after": after,
                "before": before,
            }
        )
    return records


def make_overlay(after_raw, score_map, mask, thr, sigma):
    gray_01 = np.clip(after_raw / 255.0, 0, 1)
    base = np.stack([gray_01] * 3, -1)
    sm = gaussian_filter(np.where(mask, score_map, 0), sigma=sigma)
    pos = np.where(mask & (sm > thr), sm - thr, 0)
    norm = np.log1p(pos) / np.log1p(pos).max() if pos.max() > 0 else pos
    norm = np.power(norm, 0.5)
    color = THERMAL(norm)[:, :, :3]
    alpha = np.where(norm > 0, np.clip(norm**0.5, 0, 1) * 0.88, 0)
    alpha = gaussian_filter(alpha, sigma=1.5)
    return np.clip(base * (1 - alpha[:, :, None]) + color * alpha[:, :, None], 0, 1)


def plot_grid(records, mask, title, path, cfg, cols=5):
    n = len(records)
    rows = (n - 1) // cols + 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.4, rows * 4.8), facecolor="#111111")
    axes = np.array(axes).reshape(rows, cols)
    for i, r in enumerate(records):
        ax = axes[i // cols][i % cols]
        ax.set_facecolor("#111111")
        ov = make_overlay(r["after"], r.get("score_main", r["score"]), mask, cfg["cam_thr"], cfg["cam_sigma"])
        ax.imshow(ov, interpolation="bilinear")
        ax.axis("off")
        p = r["prob"]
        ct, st = ("#ff3333", "RESPONSE") if p > 0.7 else (("#ff9900", "WEAK") if p > 0.5 else ("#6699bb", "no signal"))
        ax.set_title(f'{r["name"]}\n{st}  P={p:.3f}', color=ct, fontsize=9, fontweight="bold", pad=5)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    cax = fig.add_axes([0.15, -0.005, 0.70, 0.013])
    cax.imshow(np.linspace(0, 1, 512).reshape(1, -1), aspect="auto", cmap=THERMAL)
    cax.set_xticks([0, 128, 256, 384, 511])
    cax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color="w", fontsize=9)
    cax.set_yticks([])
    cax.set_xlabel("Response Confidence", color="w", fontsize=10)
    [s.set_color("w") for s in cax.spines.values()]
    plt.suptitle(title, color="white", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.5)
    plt.savefig(path, dpi=cfg["dpi"], bbox_inches="tight", facecolor="#111111")
    plt.close()
    print(f"  -> {path}")


def plot_score_dist(ana_rec, pbs_rec, path, dpi=150):
    from scipy.stats import ttest_ind

    ap = [r["prob"] for r in ana_rec]
    pp = [r["prob"] for r in pbs_rec]
    _, p = ttest_ind(ap, pp)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#111111")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="w")
        ax.grid(alpha=0.2, color="w")
        [s.set_color("w") for s in ax.spines.values()]
    ax = axes[0]
    bp = ax.boxplot([pp, ap], positions=[0, 1], patch_artist=True, widths=0.4, medianprops=dict(color="white", lw=2))
    bp["boxes"][0].set_facecolor("#4a90d9")
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("#e74c3c")
    bp["boxes"][1].set_alpha(0.7)
    np.random.seed(42)
    for vals, pos, c in [(pp, 0, "#4a90d9"), (ap, 1, "#e74c3c")]:
        jit = [pos + np.random.uniform(-0.08, 0.08) for _ in vals]
        ax.scatter(jit, vals, c=c, s=60, alpha=0.9, edgecolors="w", lw=0.5, zorder=5)
    ymax = max(max(ap), max(pp))
    ax.plot([0, 0, 1, 1], [ymax + 0.03] * 4, "w-", lw=1)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    ax.text(0.5, ymax + 0.04, f"{sig}  p={p:.3f}", ha="center", color="w", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"PBS\n(n={len(pp)})", f"Analyte\n(n={len(ap)})"], color="w", fontsize=11)
    ax.set_ylabel("P(Analyte)", color="w")
    ax.set_title("Classification Score Distribution", color="w", fontweight="bold")

    ax = axes[1]
    all_n = [r["name"] for r in pbs_rec + ana_rec]
    all_p = pp + ap
    clrs = ["#4a90d9"] * len(pp) + ["#e74c3c"] * len(ap)
    x = np.arange(len(all_p))
    ax.bar(x, all_p, color=clrs, alpha=0.8, edgecolor="w", lw=0.5)
    ax.axhline(0.5, color="yellow", ls="--", lw=1.5, label="threshold=0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(all_n, rotation=45, ha="right", color="w", fontsize=7)
    ax.set_ylabel("P(Analyte)", color="w")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-image Score  (blue=PBS  red=Analyte)", color="w", fontweight="bold")
    ax.legend(fontsize=9)
    plt.suptitle("Classification Result Summary", color="w", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="#111111")
    plt.close()
    print(f"  -> {path}")


def save_csv(ana_rec, pbs_rec, mask, path):
    rows = []
    for recs, label in [(pbs_rec, "PBS"), (ana_rec, "Analyte")]:
        for r in recs:
            sm = r.get("score_main", r["score"])
            mv = sm[mask] if mask is not None else sm.ravel()
            rows.append(
                {
                    "filename": r["name"],
                    "label": label,
                    "prob_analyte": round(r["prob"], 4),
                    "score_main_max": round(float(sm.max()), 4),
                    "score_main_mean_masked": round(float(mv.mean()), 4),
                    "score_top1_max": round(float(r.get("score_top1", sm).max()), 4),
                    "score_top3weighted_max": round(float(r.get("score_top3weighted", sm).max()), 4),
                    "score_top5rms_max": round(float(r.get("score_top5rms", sm).max()), 4),
                    "support_frac_max": round(float(r.get("score_support_frac", np.zeros_like(sm)).max()), 4),
                    "sig_pixels": int((mv > CFG["cam_thr"]).sum()),
                    "sig_pct": round(100 * (mv > CFG["cam_thr"]).mean(), 2),
                    "status": "RESPONSE" if r["prob"] > 0.7 else ("WEAK" if r["prob"] > 0.5 else "no_signal"),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  -> {path}")
    return df


def extract_zip_inputs(zip_dir, extract_root):
    zip_dir = Path(zip_dir)
    extract_root = Path(extract_root)
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    for zip_path in sorted(zip_dir.glob("*.zip")):
        folder_name = zip_path.stem
        target = extract_root / folder_name
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.namelist() if m.strip()]
            prefixes = set(m.split("/")[0] for m in members)
            top_dirs = [p for p in prefixes if p and "." not in p]
            if len(top_dirs) == 1 and top_dirs[0].lower() == folder_name.lower():
                zf.extractall(extract_root)
            else:
                zf.extractall(target)
    return extract_root


def resolve_data_root(data_dir, output_dir):
    data_dir = Path(data_dir)
    required = ["susbtrat", "pbs", "analyte"]
    if all((data_dir / name).exists() for name in required):
        return data_dir
    zip_names = {p.stem.lower() for p in data_dir.glob("*.zip")}
    if {"susbtrat", "pbs", "analyte"}.issubset(zip_names):
        extract_root = Path(output_dir) / "_extracted_transfer_data"
        return extract_zip_inputs(data_dir, extract_root)
    raise FileNotFoundError(f"Could not find required folders or zip files under {data_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone transfer-learning workflow extracted from the original biosensor notebook")
    parser.add_argument("--data-dir", default=r"data/gas", help="Folder containing analyte/pbs/susbtrat data or the corresponding zip files.")
    parser.add_argument("--output-dir", default="transfer_learning_results", help="Output directory.")
    parser.add_argument("--model-path", default=r"checkpoints/source_model.pt", help="Source checkpoint to fine-tune.")
    parser.add_argument("--epochs", type=int, default=100, help="Fine-tuning epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Fine-tuning learning rate.")
    parser.add_argument("--batch-size", type=int, default=4, help="Fine-tuning batch size.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze earlier backbone layers during fine-tuning.")
    parser.add_argument("--flip-signal", action="store_true", default=True, help="Flip analyte before/after direction for signal-decrease analytes.")
    parser.add_argument("--no-flip-signal", dest="flip_signal", action="store_false", help="Disable analyte signal flipping.")
    return parser.parse_args()


def main():
    global CFG

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("OK Dependencies ready")
    print(f"Device: {DEVICE}")
    data_root = resolve_data_root(args.data_dir, output_dir)
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Base model checkpoint not found: {model_path}")

    print(f"\nChecking data directory: {data_root}")
    ok = 0
    for folder in ["led_field", "susbtrat", "pbs", "analyte"]:
        p = data_root / folder
        n = len([f for f in p.iterdir() if f.suffix.lower() in [".png", ".tif", ".tiff", ".jpg"]]) if p.exists() else 0
        req = folder != "led_field"
        print(f"  {'OK' if n > 0 else ('ERROR' if req else '-')} {folder}/  {n} images")
        if n > 0 or not req:
            ok += 1
    if ok < 3:
        raise RuntimeError("susbtrat / pbs / analyte must all contain data")

    print(f"\n[1] Loading source model: {model_path}")
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    cfg_ft = dict(ckpt["cfg"])
    cfg_ft["data_dir"] = str(data_root)
    cfg_ft["output_dir"] = str(output_dir)
    cfg_ft["epochs"] = args.epochs
    cfg_ft["lr"] = args.lr
    cfg_ft["batch_size"] = args.batch_size
    cfg_ft["freeze_backbone"] = args.freeze_backbone
    CFG = dict(CFG, **cfg_ft)
    size = (cfg_ft["img_size"], cfg_ft["img_size"])
    print(f"  Source model: epochs={ckpt['cfg']['epochs']}  img_size={cfg_ft['img_size']}")

    print("\n[2] LED flat-field correction...")
    led_map_new = build_led_map(data_root / cfg_ft["led_subdir"], size, cfg_ft["led_bright_thr"], cfg_ft["led_smooth_sigma"])
    print("OK LED correction complete" if led_map_new is not None else "WARNING No LED correction")

    print("\n[3] Loading paired images...")
    pbs_a_new, pbs_b_new, pbs_n_new = load_pairs(data_root / cfg_ft["pbs_subdir"], data_root / cfg_ft["substrate_subdir"], cfg_ft["pbs_prefix"], size, led_map=led_map_new)
    ana_a_new, ana_b_new, ana_n_new = load_pairs(data_root / cfg_ft["analyte_subdir"], data_root / cfg_ft["substrate_subdir"], cfg_ft["analyte_prefix"], size, led_map=led_map_new)
    if args.flip_signal:
        ana_b_new, ana_a_new = ana_a_new, ana_b_new
        print("OK Difference direction reversed for signal-decrease analytes")
    print(f"  PBS: {len(pbs_n_new)} pairs  Analyte: {len(ana_n_new)} pairs")

    mask_new = compute_mask(pbs_a_new + ana_a_new, thr=cfg_ft["mask_thr"], erode=cfg_ft["mask_erode"])
    print(f"  Valid pixels: {mask_new.sum()} / {mask_new.size}  ({mask_new.mean():.1%})")

    print(f"\n[4] Transfer learning (fine-tuning){args.epochs} epochs, lr={args.lr}...")
    model_ft = BiosensorNet(pretrained=False, dropout=cfg_ft["dropout"]).to(DEVICE)
    model_ft.load_state_dict(ckpt["model_state"])
    if args.freeze_backbone:
        for name, param in model_ft.named_parameters():
            if any(x in name for x in ["layer1", "layer2", "layer3", "conv1", "bn1"]):
                param.requires_grad = False
            else:
                param.requires_grad = True
        frozen = sum(1 for p in model_ft.parameters() if not p.requires_grad)
        trainable = sum(1 for p in model_ft.parameters() if p.requires_grad)
        print(f"  Frozen parameter groups:{frozen}  Trainable parameter groups:{trainable}")
    else:
        print("  Fine-tuning all layers")

    full_ds_ft = BioDataset(pbs_a_new, pbs_b_new, ana_a_new, ana_b_new, img_size=cfg_ft["img_size"], augment=False)
    labels_ft = [s[2] for s in full_ds_ft.samples]
    aug_ds_ft = BioDataset(pbs_a_new, pbs_b_new, ana_a_new, ana_b_new, img_size=cfg_ft["img_size"], augment=True)
    if len(set(labels_ft)) >= 2 and len(labels_ft) >= 4:
        tr_idx, val_idx = train_test_split(range(len(full_ds_ft)), test_size=0.25, stratify=labels_ft, random_state=42)
    else:
        tr_idx = val_idx = list(range(len(full_ds_ft)))
    tr_load_ft = DataLoader(Subset(aug_ds_ft, tr_idx), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_load_ft = DataLoader(Subset(full_ds_ft, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    aug_ds_ft.dataset = aug_ds_ft
    print(f"  Train:{len(tr_idx)}  Val:{len(val_idx)}")
    hist_ft = train_model(model_ft, tr_load_ft, val_load_ft, cfg_ft)

    ft_save = output_dir / "finetuned_model.pt"
    torch.save({"model_state": model_ft.state_dict(), "history": hist_ft, "cfg": cfg_ft}, ft_save)
    print(f"\nOK Fine-tuned model saved -> {ft_save}")

    epochs_r = range(1, len(hist_ft["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#111111")

    def _ax(ax):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="w")
        ax.grid(alpha=0.2, color="w")
        [s.set_color("w") for s in ax.spines.values()]

    _ax(axes[0]); axes[0].plot(epochs_r, hist_ft["train_loss"], color="#4a90d9", lw=2, label="Train"); axes[0].plot(epochs_r, hist_ft["val_loss"], color="#e74c3c", lw=2, label="Val"); axes[0].set_title("Fine-tune Loss", color="w", fontweight="bold"); axes[0].set_xlabel("Epoch", color="w"); axes[0].legend()
    _ax(axes[1]); axes[1].plot(epochs_r, hist_ft["train_acc"], color="#4a90d9", lw=2, label="Train Acc"); axes[1].plot(epochs_r, hist_ft["val_acc"], color="#e74c3c", lw=2, label="Val Acc"); axes[1].set_title("Fine-tune Accuracy", color="w", fontweight="bold"); axes[1].set_xlabel("Epoch", color="w"); axes[1].set_ylim(0, 1); axes[1].legend()
    _ax(axes[2]); axes[2].plot(epochs_r, hist_ft["val_auc"], color="#2ed573", lw=2, label="Val AUC"); axes[2].axhline(0.5, color="white", ls="--", alpha=0.4); axes[2].set_title("Fine-tune AUC", color="w", fontweight="bold"); axes[2].set_xlabel("Epoch", color="w"); axes[2].set_ylim(0, 1.05); axes[2].legend()
    plt.suptitle("Fine-tuning Training Curves", color="w", fontsize=13, fontweight="bold")
    plt.tight_layout()
    curve_path = output_dir / "finetune_curves.png"
    plt.savefig(curve_path, dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close()
    print(f"OK Training curves -> {curve_path}")

    print("\n[5] Computing evaluation metrics...")
    model_ft.eval()
    all_probs_ft, all_preds_ft, all_true_ft = [], [], []
    with torch.no_grad():
        for imgs, lbs in val_load_ft:
            out = model_ft(imgs.to(DEVICE))
            prob = torch.softmax(out, 1)[:, 1].cpu().tolist()
            pred = out.argmax(1).cpu().tolist()
            all_probs_ft += prob
            all_preds_ft += pred
            all_true_ft += lbs.tolist()
    all_probs_ft = np.array(all_probs_ft)
    all_preds_ft = np.array(all_preds_ft)
    all_true_ft = np.array(all_true_ft)
    acc = accuracy_score(all_true_ft, all_preds_ft)
    bal_acc = balanced_accuracy_score(all_true_ft, all_preds_ft)
    prec = precision_score(all_true_ft, all_preds_ft, zero_division=0)
    rec = recall_score(all_true_ft, all_preds_ft, zero_division=0)
    f1 = f1_score(all_true_ft, all_preds_ft, zero_division=0)
    mcc = matthews_corrcoef(all_true_ft, all_preds_ft)
    kappa = cohen_kappa_score(all_true_ft, all_preds_ft)
    try:
        auc_roc = roc_auc_score(all_true_ft, all_probs_ft)
    except Exception:
        auc_roc = 0.5
    ap = average_precision_score(all_true_ft, all_probs_ft)

    print("\n=== Fine-tuned validation metrics===")
    metrics_dict = {
        "Accuracy": acc,
        "Balanced Accuracy": bal_acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc,
        "Cohen Kappa": kappa,
        "AUC-ROC": auc_roc,
        "Average Precision": ap,
    }
    for k, v in metrics_dict.items():
        print(f"  {k:22s}: {v:.4f}")
    print("\nClassification report:")
    print(classification_report(all_true_ft, all_preds_ft, target_names=["PBS", "Analyte"]))

    pd.DataFrame([{"Metric": k, "Value": round(v, 4)} for k, v in metrics_dict.items()]).to_csv(output_dir / "finetune_metrics.csv", index=False)
    cm = confusion_matrix(all_true_ft, all_preds_ft)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#111111")
    ax.set_facecolor("#1a1a2e")
    ConfusionMatrixDisplay(cm, display_labels=["PBS", "Analyte"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix (Fine-tuned)", color="w", fontweight="bold")
    ax.tick_params(colors="w")
    ax.xaxis.label.set_color("w")
    ax.yaxis.label.set_color("w")
    [s.set_color("w") for s in ax.spines.values()]
    plt.tight_layout()
    cm_path = output_dir / "finetune_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close()
    print(f"OK Confusion matrix -> {cm_path}")

    print("\n[6] Building pixel-level PBS baseline...")
    pbs_mu_new, pbs_std_new = build_pbs_pixel_baseline(pbs_a_new, pbs_b_new, mask_new, cfg_ft)
    _pbs_raw_new = [_pixel_score(a, b, pbs_mu_new, pbs_std_new, mask_new, cfg_ft) for a, b in zip(pbs_a_new, pbs_b_new)]
    _stk_new = np.stack([m["primary"] for m in _pbs_raw_new])
    pbs_null_mu_new = _stk_new.mean(0)
    pbs_null_std_new = np.clip(_stk_new.std(0, ddof=1), np.percentile(_stk_new.std(0, ddof=1)[mask_new], 10), None)
    print(f"  OK Feature dimension: {pbs_mu_new.shape[0]}")

    pbs_rec_thr_new = infer_all(model_ft, pbs_a_new, pbs_b_new, pbs_n_new, mask_new, cfg_ft, pbs_mu=pbs_mu_new, pbs_std=pbs_std_new, pbs_null_mu=pbs_null_mu_new, pbs_null_std=pbs_null_std_new)
    _all_pbs_new = np.concatenate([r["score"][mask_new] for r in pbs_rec_thr_new])
    adaptive_thr_new = float(np.percentile(_all_pbs_new, 99))
    print(f"  Adaptive threshold: {adaptive_thr_new:.4f}")

    print("\n[7] Running inference...")
    ana_rec_new = infer_all(model_ft, ana_a_new, ana_b_new, ana_n_new, mask_new, cfg_ft, pbs_mu=pbs_mu_new, pbs_std=pbs_std_new, pbs_null_mu=pbs_null_mu_new, pbs_null_std=pbs_null_std_new)
    pbs_rec_new = infer_all(model_ft, pbs_a_new, pbs_b_new, pbs_n_new, mask_new, cfg_ft, pbs_mu=pbs_mu_new, pbs_std=pbs_std_new, pbs_null_mu=pbs_null_mu_new, pbs_null_std=pbs_null_std_new)
    print("\n=== Analyte ===")
    for r in ana_rec_new:
        flag = "RESPONSE" if r["prob"] > 0.7 else ("WEAK" if r["prob"] > 0.5 else "  -")
        print(f"  {r['name']}: P={r['prob']:.3f}  {flag}")
    print("\n=== PBS controls ===")
    for r in pbs_rec_new:
        print(f"  {r['name']}: P={r['prob']:.3f}  {'clean' if r['prob'] < 0.5 else 'WARN'}")

    print("\n[8] Generating heatmaps...")
    cfg_plot_new = dict(cfg_ft, cam_thr=adaptive_thr_new)
    CFG["cam_thr"] = adaptive_thr_new
    plot_grid(ana_rec_new, mask_new, "ANALYTE - Fine-tuned Response Overlay", output_dir / "analyte_overlay.png", cfg_plot_new)
    plot_grid(pbs_rec_new, mask_new, "PBS - Negative Control (Fine-tuned)", output_dir / "pbs_control.png", cfg_plot_new)
    plot_score_dist(ana_rec_new, pbs_rec_new, output_dir / "score_distribution.png")
    df_new = save_csv(ana_rec_new, pbs_rec_new, mask_new, output_dir / "raw_results.csv")
    print("\nSource data:")
    print(df_new.to_string(index=False))

    zip_path = shutil.make_archive(str(output_dir), "zip", str(output_dir))
    print("\nOK All steps complete")
    print(f"OK Output archive -> {zip_path}")
    print("\nOutput file list:")
    for fn in ["finetuned_model.pt", "finetune_curves.png", "finetune_confusion_matrix.png", "finetune_metrics.csv", "analyte_overlay.png", "pbs_control.png", "score_distribution.png", "raw_results.csv", Path(zip_path).name]:
        print(f"  {fn}")


if __name__ == "__main__":
    main()
