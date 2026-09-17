import argparse
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset

from feature_classifier_comparison import ResNet29Classifier, find_best_threshold
from gas_transfer import (
    CFG as BASE_CFG,
    ResNet18_29,
    _ResBlock,
    _compute_channel_stats,
    _make_soft_target,
    _refine_loss,
    build_pbs_baseline_29,
    build_pbs_pixel_baseline,
    build_score_maps,
    extract_feature_bank_29,
    make_balanced_folds,
)
from transfer_learning import DEVICE, build_led_map, compute_mask, load_pairs, resolve_data_root

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CFG = {
    "img_size": 128,
    "led_subdir": "led_field",
    "pbs_subdir": "pbs",
    "analyte_subdir": "analyte",
    "substrate_subdir": "susbtrat",
    "led_bright_thr": 0.35,
    "led_smooth_sigma": 15,
    "feature_bank_mode": "core30",
    "score_primary": "top3_weighted",
    "support_z_thr": 2.5,
    "mask_thr": 20,
    "mask_erode": 8,
    "n_splits": 3,
    "random_state": 42,
    "clf_epochs": 16,
    "clf_batch_size": 8,
    "clf_lr": 2e-4,
    "clf_weight_decay": 1e-5,
    "clf_patience": 5,
    "soft_support_w": 0.6,
    "soft_zconf_w": 0.4,
    "stage2a_epochs": 50,
    "stage2a_batch_size": 8,
    "stage2a_patience": 12,
    "cnn_lr": 3e-4,
    "cnn_wd": 5e-5,
    "resnet_lr": 1.5e-4,
    "resnet_wd": 1e-5,
    "unet_lr": 3e-4,
    "unet_wd": 5e-5,
    "unet_cls_w": 0.2,
    "freeze_warmup_epochs": 3,
    "tv_weight": 5e-4,
    "heatmap_response_thr": 0.5,
    "top_k_ratio": 0.05,
    "dpi": 150,
    "response_percentile_coarse": 99.5,
    "response_percentile_refined": 99.5,
}


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


def stratified_subset_indices(labels, fraction, random_state):
    labels = np.asarray(labels, dtype=np.int32)
    if fraction >= 0.999:
        return np.arange(len(labels), dtype=np.int32)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=random_state)
    keep_idx, _ = next(sss.split(np.zeros(len(labels)), labels))
    return np.sort(keep_idx.astype(np.int32))


class Feature29Dataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class Refine29Dataset(Dataset):
    def __init__(self, after_list, before_list, labels, pbs_mu29, pbs_std29, mask, channel_mean, channel_std):
        self.after_list = after_list
        self.before_list = before_list
        self.labels = labels
        self.pbs_mu29 = pbs_mu29
        self.pbs_std29 = pbs_std29
        self.mask = mask
        self.channel_mean = channel_mean
        self.channel_std = channel_std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        aft = self.after_list[idx]
        bef = self.before_list[idx]
        lbl = int(self.labels[idx])
        feat = extract_feature_bank_29(bef, aft)
        feat = (feat - self.channel_mean[:, None, None]) / (self.channel_std[:, None, None] + 1e-6)
        if lbl == 1:
            soft = make_soft_target_v4(bef, aft, self.pbs_mu29, self.pbs_std29, self.mask)
        else:
            soft = np.zeros(feat.shape[1:], np.float32)
        return torch.from_numpy(feat.astype(np.float32)), torch.from_numpy(soft.astype(np.float32)), torch.tensor(lbl, dtype=torch.long)


def make_soft_target_v4(bef, aft, pbs_mu29, pbs_std29, mask):
    feat = extract_feature_bank_29(bef, aft)
    z = (feat - pbs_mu29) / (pbs_std29 + 1e-6)
    z_abs = np.abs(z)
    support = (z_abs > CFG["support_z_thr"]).sum(0).astype(np.float32)
    z_conf = np.sqrt((z**2).mean(0)).astype(np.float32)

    def _n01(x):
        mn, mx = float(x.min()), float(x.max())
        return np.zeros_like(x, np.float32) if mx - mn < 1e-8 else ((x - mn) / (mx - mn)).astype(np.float32)

    soft = CFG["soft_support_w"] * _n01(support) + CFG["soft_zconf_w"] * _n01(z_conf)
    soft = _n01(soft) * mask.astype(np.float32)
    return soft.astype(np.float32)


class HeatCNN29(nn.Module):
    def __init__(self, in_ch=29):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class UNet29(nn.Module):
    def __init__(self, in_ch=29, base=32):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.enc1 = block(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = block(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = block(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = block(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = block(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)
        self.aux_pool = nn.AdaptiveAvgPool2d(1)
        self.aux_fc = nn.Linear(base * 8, 1)

    def forward(self, x, return_aux=False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        hm = self.head(d1)
        if return_aux:
            aux = self.aux_fc(self.aux_pool(b).flatten(1)).squeeze(1)
            return hm, aux
        return hm


def load_classifier_backbone_into_stage2a(stage2a_model, classifier_ckpt_path):
    ckpt = torch.load(classifier_ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    mapping = {}
    for k, v in state.items():
        if k.startswith("backbone."):
            mapping[k.replace("backbone.", "", 1)] = v
    stage2a_model.load_state_dict(mapping, strict=False)


def load_classifier_init(model, classifier_ckpt_path):
    if not classifier_ckpt_path:
        return False
    ckpt_path = Path(classifier_ckpt_path)
    if not ckpt_path.exists():
        return False
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    return True


def train_classifier_cv(full_feats, labels, names, folds, out_dir, args):
    cv_rows = []
    detail_rows = []
    full_hist = []
    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        x_tr_raw = full_feats[tr_idx]
        y_tr = labels[tr_idx]
        x_va_raw = full_feats[va_idx]
        y_va = labels[va_idx]
        cm = x_tr_raw.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        cs = np.maximum(x_tr_raw.std(axis=(0, 2, 3), keepdims=True).astype(np.float32), 1e-6)
        x_tr = ((x_tr_raw - cm) / cs).astype(np.float32)
        x_va = ((x_va_raw - cm) / cs).astype(np.float32)
        tr_loader = DataLoader(Feature29Dataset(x_tr, y_tr), batch_size=args.clf_batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
        va_loader = DataLoader(Feature29Dataset(x_va, y_va), batch_size=args.clf_batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

        model = ResNet29Classifier(29).to(DEVICE)
        did_init = load_classifier_init(model, args.init_classifier_ckpt)
        if did_init and args.clf_freeze_warmup_epochs > 0:
            for p in model.backbone.stem.parameters():
                p.requires_grad = False
            for p in model.backbone.layer1.parameters():
                p.requires_grad = False
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.clf_lr, weight_decay=args.clf_weight_decay)
        crit = nn.BCEWithLogitsLoss()
        best_state = None
        best_score = -1e9
        wait = 0
        for ep in range(1, args.clf_epochs + 1):
            if did_init and args.clf_freeze_warmup_epochs > 0 and ep == args.clf_freeze_warmup_epochs + 1:
                for p in model.backbone.stem.parameters():
                    p.requires_grad = True
                for p in model.backbone.layer1.parameters():
                    p.requires_grad = True
                opt = torch.optim.Adam(model.parameters(), lr=args.clf_lr * 0.6, weight_decay=args.clf_weight_decay)
            model.train()
            losses = []
            for xb, yb in tr_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            model.eval()
            va_prob = []
            va_true = []
            with torch.no_grad():
                for xb, yb in va_loader:
                    prob = torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
                    va_prob.extend(prob.tolist())
                    va_true.extend(yb.numpy().tolist())
            va_prob = np.asarray(va_prob, dtype=np.float32)
            va_true = np.asarray(va_true, dtype=np.int32)
            metrics = compute_metrics(va_true, (va_prob >= 0.5).astype(int), va_prob)
            full_hist.append(
                {
                    "fold": fold,
                    "epoch": ep,
                    "train_loss": float(np.mean(losses) if losses else np.nan),
                    "val_accuracy": float(metrics["accuracy"]),
                    "val_balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "val_f1": float(metrics["f1"]),
                    "val_roc_auc": float(metrics["roc_auc"]),
                }
            )
            score = float(metrics["balanced_accuracy"])
            if score > best_score + 1e-12:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= args.clf_patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)

        tr_eval_loader = DataLoader(Feature29Dataset(x_tr, y_tr), batch_size=args.clf_batch_size, shuffle=False, num_workers=0)
        tr_prob = []
        with torch.no_grad():
            for xb, _ in tr_eval_loader:
                tr_prob.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy().tolist())
        tr_prob = np.asarray(tr_prob, dtype=np.float32)
        thr, tr_acc, tr_bal = find_best_threshold(y_tr, tr_prob)

        va_prob = []
        with torch.no_grad():
            for xb, _ in va_loader:
                va_prob.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy().tolist())
        va_prob = np.asarray(va_prob, dtype=np.float32)
        va_pred = (va_prob >= thr).astype(int)
        row = {"fold": fold, "decision_threshold": thr, "train_accuracy_at_threshold": tr_acc, "train_balanced_accuracy_at_threshold": tr_bal}
        row.update(compute_metrics(y_va, va_pred, va_prob))
        cv_rows.append(row)
        for idx_local, nm in enumerate([names[i] for i in va_idx]):
            detail_rows.append(
                {
                    "fold": fold,
                    "name": nm,
                    "true_label": "Analyte" if int(y_va[idx_local]) == 1 else "PBS",
                    "predicted_label": "Analyte" if int(va_pred[idx_local]) == 1 else "PBS",
                    "prob_analyte": float(va_prob[idx_local]),
                    "confidence": float(max(va_prob[idx_local], 1.0 - va_prob[idx_local])),
                    "correct": bool(int(va_pred[idx_local]) == int(y_va[idx_local])),
                    "decision_threshold": float(thr),
                }
            )
        torch.save(
            {
                "model_state": model.state_dict(),
                "channel_mean": cm.squeeze().tolist(),
                "channel_std": cs.squeeze().tolist(),
                "threshold": float(thr),
                "fold": fold,
            },
            out_dir / f"classifier_resnet29_fold{fold}.pt",
        )

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(out_dir / "classifier_cv_results.csv", index=False)
    pd.DataFrame(full_hist).to_csv(out_dir / "classifier_training_history.csv", index=False)
    detail_df = pd.DataFrame(detail_rows).sort_values(["fold", "true_label", "name"]).reset_index(drop=True)
    detail_df.to_csv(out_dir / "classifier_fold_details.csv", index=False)
    detail_df[~detail_df["correct"]].to_csv(out_dir / "classifier_fold_errors.csv", index=False)
    detail_df.groupby(["fold", "true_label"]).agg(total=("name", "count"), correct=("correct", "sum"), mean_prob_analyte=("prob_analyte", "mean")).reset_index().assign(
        accuracy_within_label=lambda x: x["correct"] / x["total"]
    ).to_csv(out_dir / "classifier_fold_label_summary.csv", index=False)
    return cv_df


def train_full_classifier_for_init(full_feats, labels, names, train_idx, val_idx, out_dir, args):
    x_tr_raw = full_feats[train_idx]
    y_tr = labels[train_idx]
    x_va_raw = full_feats[val_idx]
    y_va = labels[val_idx]
    cm = x_tr_raw.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    cs = np.maximum(x_tr_raw.std(axis=(0, 2, 3), keepdims=True).astype(np.float32), 1e-6)
    x_tr = ((x_tr_raw - cm) / cs).astype(np.float32)
    x_va = ((x_va_raw - cm) / cs).astype(np.float32)
    tr_loader = DataLoader(Feature29Dataset(x_tr, y_tr), batch_size=args.clf_batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(Feature29Dataset(x_va, y_va), batch_size=args.clf_batch_size, shuffle=False, num_workers=0)
    model = ResNet29Classifier(29).to(DEVICE)
    did_init = load_classifier_init(model, args.init_classifier_ckpt)
    if did_init and args.clf_freeze_warmup_epochs > 0:
        for p in model.backbone.stem.parameters():
            p.requires_grad = False
        for p in model.backbone.layer1.parameters():
            p.requires_grad = False
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.clf_lr, weight_decay=args.clf_weight_decay)
    crit = nn.BCEWithLogitsLoss()
    best_state = None
    best_score = -1e9
    wait = 0
    hist_rows = []
    for ep in range(1, args.clf_epochs + 1):
        if did_init and args.clf_freeze_warmup_epochs > 0 and ep == args.clf_freeze_warmup_epochs + 1:
            for p in model.backbone.stem.parameters():
                p.requires_grad = True
            for p in model.backbone.layer1.parameters():
                p.requires_grad = True
            opt = torch.optim.Adam(model.parameters(), lr=args.clf_lr * 0.6, weight_decay=args.clf_weight_decay)
        model.train()
        losses = []
        for xb, yb in tr_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        va_prob = []
        with torch.no_grad():
            for xb, _ in va_loader:
                va_prob.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy().tolist())
        va_prob = np.asarray(va_prob, dtype=np.float32)
        metrics = compute_metrics(y_va, (va_prob >= 0.5).astype(int), va_prob)
        hist_rows.append(
            {
                "epoch": ep,
                "train_loss": float(np.mean(losses) if losses else np.nan),
                "val_accuracy": float(metrics["accuracy"]),
                "val_balanced_accuracy": float(metrics["balanced_accuracy"]),
                "val_f1": float(metrics["f1"]),
                "val_roc_auc": float(metrics["roc_auc"]),
            }
        )
        score = float(metrics["balanced_accuracy"])
        if score > best_score + 1e-12:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.clf_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = out_dir / "classifier_resnet29_stage2a_init.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "channel_mean": cm.squeeze().tolist(),
            "channel_std": cs.squeeze().tolist(),
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
        },
        ckpt_path,
    )
    pd.DataFrame(hist_rows).to_csv(out_dir / "classifier_stage2a_init_history.csv", index=False)
    return ckpt_path


def train_stage2a_model(model_name, model, train_after, train_before, train_labels, val_after, val_before, val_labels, mask, pbs_mu29, pbs_std29, cm, cs, out_dir, args, init_classifier_ckpt=None, lr=None, weight_decay=None, use_aux=False, aux_cls_w=0.0):
    tr_ds = Refine29Dataset(train_after, train_before, train_labels, pbs_mu29, pbs_std29, mask, cm, cs)
    va_ds = Refine29Dataset(val_after, val_before, val_labels, pbs_mu29, pbs_std29, mask, cm, cs)
    tr_ld = DataLoader(tr_ds, batch_size=args.stage2a_batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    va_ld = DataLoader(va_ds, batch_size=args.stage2a_batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    model = model.to(DEVICE)
    if model_name == "stage2a_resnet" and init_classifier_ckpt is not None:
        load_classifier_backbone_into_stage2a(model, init_classifier_ckpt)
        for p in model.stem.parameters():
            p.requires_grad = False
        for p in model.layer1.parameters():
            p.requires_grad = False

    base_lr = float(lr if lr is not None else CFG["resnet_lr"])
    base_wd = float(weight_decay if weight_decay is not None else CFG["resnet_wd"])
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=base_lr, weight_decay=base_wd)
    aux_crit = nn.BCEWithLogitsLoss()
    best_sep, wait, best_state = -1e9, 0, None
    hist_rows = []
    for ep in range(1, args.stage2a_epochs + 1):
        if model_name == "stage2a_resnet" and ep == args.freeze_warmup_epochs + 1:
            for p in model.stem.parameters():
                p.requires_grad = True
            for p in model.layer1.parameters():
                p.requires_grad = True
            opt = torch.optim.Adam(model.parameters(), lr=base_lr * 0.6, weight_decay=base_wd)
        model.train()
        tr_losses = []
        for feat, soft, _ in tr_ld:
            feat = feat.to(DEVICE)
            soft = soft.to(DEVICE)
            opt.zero_grad()
            if use_aux:
                hm_logit, aux_logit = model(feat, return_aux=True)
                loss = _refine_loss(hm_logit, soft, args.tv_weight)
                y_aux = (soft.flatten(1).mean(1) > 0).float()
                loss = loss + aux_cls_w * aux_crit(aux_logit, y_aux)
                pred_logit = hm_logit
            else:
                pred_logit = model(feat)
                loss = _refine_loss(pred_logit, soft, args.tv_weight)
            loss.backward()
            opt.step()
            tr_losses.append(float(loss.item()))
        model.eval()
        scores_ana, scores_pbs = [], []
        with torch.no_grad():
            for feat, _, lbl in va_ld:
                feat = feat.to(DEVICE)
                if use_aux:
                    hm = torch.sigmoid(model(feat, return_aux=True)[0]).squeeze(1).cpu().numpy()
                else:
                    hm = torch.sigmoid(model(feat)).squeeze(1).cpu().numpy()
                for ib, lb in enumerate(lbl.tolist()):
                    flat = hm[ib].ravel()
                    k = max(1, int(len(flat) * CFG["top_k_ratio"]))
                    sc = np.partition(flat, -k)[-k:].mean()
                    (scores_ana if lb == 1 else scores_pbs).append(sc)
        sep = (np.mean(scores_ana) if scores_ana else 0.0) - 0.5 * (np.mean(scores_pbs) if scores_pbs else 1.0)
        hist_rows.append(
            {
                "model": model_name,
                "epoch": ep,
                "train_loss": float(np.mean(tr_losses) if tr_losses else np.nan),
                "ana_topk": float(np.mean(scores_ana) if scores_ana else 0.0),
                "pbs_topk": float(np.mean(scores_pbs) if scores_pbs else 0.0),
                "sep": float(sep),
            }
        )
        if sep > best_sep + 1e-12:
            best_sep, wait = sep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.stage2a_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model_state": model.state_dict(),
            "channel_mean": cm.tolist(),
            "channel_std": cs.tolist(),
            "pbs_mu29": pbs_mu29,
            "pbs_std29": pbs_std29,
            "best_sep": float(best_sep),
            "model_name": model_name,
        },
        out_dir / f"{model_name}.pt",
    )
    return model, pd.DataFrame(hist_rows)


def infer_heatmaps(model, after_list, before_list, cm, cs):
    model.eval()
    out = []
    with torch.no_grad():
        for aft, bef in zip(after_list, before_list):
            feat = extract_feature_bank_29(bef, aft)
            feat = (feat - cm[:, None, None]) / (cs[:, None, None] + 1e-6)
            x = torch.from_numpy(feat[None].astype(np.float32)).to(DEVICE)
            pred = model(x, return_aux=True)[0] if isinstance(model, UNet29) else model(x)
            hm = torch.sigmoid(pred)[0, 0].cpu().numpy().astype(np.float32)
            hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
            out.append(hm)
    return out


def _img_score(h):
    flat = h.ravel()
    k = max(1, int(len(flat) * CFG["top_k_ratio"]))
    return float(np.partition(flat, -k)[-k:].mean())


def compute_heatmap_metrics(hm_pbs, hm_ana, support_maps, label):
    hmap_thr = CFG["heatmap_response_thr"]
    pbs_fpr = float(np.mean([(h > hmap_thr).mean() for h in hm_pbs])) if hm_pbs else np.nan
    seed_rets = []
    for h, sup in zip(hm_ana, support_maps):
        mn, mx = float(sup.min()), float(sup.max())
        sup_n = np.zeros_like(sup, np.float32) if mx - mn < 1e-8 else ((sup - mn) / (mx - mn)).astype(np.float32)
        smask = sup_n >= np.percentile(sup_n, 90)
        if smask.any():
            seed_rets.append(float(h[smask].mean() / (h.mean() + 1e-8)))
    bg_sups = []
    for h in hm_ana:
        lo = np.percentile(h, 50)
        hi = np.percentile(h, 90)
        bot = h[h <= lo].mean() if (h <= lo).any() else 0.0
        top = h[h >= hi].mean() if (h >= hi).any() else 0.0
        bg_sups.append(float(top / max(bot, 1e-3)))
    scores_ana = np.array([_img_score(h) for h in hm_ana], dtype=np.float32)
    scores_pbs = np.array([_img_score(h) for h in hm_pbs], dtype=np.float32) if hm_pbs else np.array([0.0], dtype=np.float32)
    mu_a, mu_p = float(scores_ana.mean()), float(scores_pbs.mean())
    sig_a = float(scores_ana.std(ddof=min(1, len(scores_ana) - 1))) + 1e-8
    sig_p = float(scores_pbs.std(ddof=min(1, len(scores_pbs) - 1))) + 1e-8
    return {
        "Method": label,
        "pbs_fpr_area": float(pbs_fpr),
        "ana_seed_ret": float(np.mean(seed_rets)) if seed_rets else np.nan,
        "bg_suppress": float(np.mean(bg_sups)) if bg_sups else np.nan,
        "sep_effect": float((mu_a - mu_p) / np.sqrt((sig_a**2 + sig_p**2) / 2)),
        "sep_tail": float(np.percentile(scores_ana, 95) - np.percentile(scores_pbs, 95)),
        "mean_ana_score": mu_a,
        "mean_pbs_score": mu_p,
    }


def plot_heatmap_bars(df_hm, out_png):
    palette = ["#2ed573", "#ffa502", "#a29bfe", "#4a90d9"]
    clip_cfg = {"pbs_fpr_area": None, "ana_seed_ret": 5.0, "bg_suppress": 20.0, "sep_effect": None, "sep_tail": None}
    cols_cfg = [
        ("pbs_fpr_area", "PBS cleanliness\n(1-FPR) (higher)", "inv"),
        ("ana_seed_ret", "Seed retention (higher)", "norm"),
        ("bg_suppress", "Background suppression (higher)", "norm"),
        ("sep_effect", "Effect-size\nSep (higher)", "norm"),
        ("sep_tail", "Tail Sep (higher)", "norm"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(30, 6), facecolor="#111111")
    for ax, (col, title, inv) in zip(axes, cols_cfg):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="w")
        ax.grid(alpha=0.15, color="w", axis="y")
        for sp in ax.spines.values():
            sp.set_color("w")
        raw = df_hm[col].values.astype(float)
        clip_max = clip_cfg[col]
        show_r = np.clip(raw, 0, clip_max) if clip_max else raw.copy()
        show = 1.0 - show_r if inv == "inv" else show_r
        x = np.arange(len(df_hm))
        bars = ax.bar(x, show, color=palette[: len(df_hm)], edgecolor="white", alpha=0.88)
        for b, rv, sv in zip(bars, raw, show):
            ax.text(b.get_x() + b.get_width() / 2, sv + max(abs(show).max() * 0.02, 0.02), f"{rv:.3f}", ha="center", va="bottom", color="white", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(df_hm["Method"].tolist(), rotation=20, ha="right", color="w", fontsize=10)
        ax.set_title(title, color="w", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=CFG["dpi"], bbox_inches="tight", facecolor="#111111")
    plt.close()


def plot_heatmap_radar(df_hm, out_png):
    cats = ["PBS cleanliness", "Seed retention", "Background suppression", "Effect-size Sep", "Tail Sep"]
    ang = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist() + [0]
    palette = ["#2ed573", "#ffa502", "#a29bfe", "#4a90d9"]
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#111111", subplot_kw=dict(projection="polar"))
    ax.set_facecolor("#1a1a2e")
    for i, row in df_hm.iterrows():
        eff_norm = min(max(float(row["sep_effect"]), 0.0), 5.0) / 5.0
        tail_norm = min(max(float(row["sep_tail"]), 0.0), 1.0) / 1.0
        vals = [
            1.0 - float(row["pbs_fpr_area"]),
            min(float(row["ana_seed_ret"]), 5.0) / 5.0,
            min(float(row["bg_suppress"]), 20.0) / 20.0,
            eff_norm,
            tail_norm,
        ]
        ax.plot(ang, vals + [vals[0]], color=palette[i], lw=2, label=row["Method"])
        ax.fill(ang, vals + [vals[0]], color=palette[i], alpha=0.12)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(cats, color="w", fontsize=11)
    ax.set_ylim(0, 1)
    ax.tick_params(colors="w")
    ax.set_title("Heatmap Quality Radar", color="w", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1.12), facecolor="#1a1a2e", labelcolor="w", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=CFG["dpi"], bbox_inches="tight", facecolor="#111111")
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="Gas transfer experiment: 29feat+ResNet classification, then stage2a heatmap comparison.")
    p.add_argument("--data-dir", default=r"data/gas")
    p.add_argument("--output-dir", default=r"outputs/localization_comparison")
    p.add_argument("--base-output-dir", default=r"outputs/hybrid")
    p.add_argument("--subset-fraction", type=float, default=1.0)
    p.add_argument("--clf-epochs", type=int, default=CFG["clf_epochs"])
    p.add_argument("--clf-batch-size", type=int, default=CFG["clf_batch_size"])
    p.add_argument("--clf-lr", type=float, default=CFG["clf_lr"])
    p.add_argument("--clf-weight-decay", type=float, default=CFG["clf_weight_decay"])
    p.add_argument("--clf-patience", type=int, default=CFG["clf_patience"])
    p.add_argument("--init-classifier-ckpt", default=None)
    p.add_argument("--clf-freeze-warmup-epochs", type=int, default=2)
    p.add_argument("--stage2a-epochs", type=int, default=CFG["stage2a_epochs"])
    p.add_argument("--stage2a-batch-size", type=int, default=CFG["stage2a_batch_size"])
    p.add_argument("--stage2a-patience", type=int, default=CFG["stage2a_patience"])
    p.add_argument("--freeze-warmup-epochs", type=int, default=CFG["freeze_warmup_epochs"])
    p.add_argument("--tv-weight", type=float, default=CFG["tv_weight"])
    p.add_argument("--reverse-analyte", action="store_true", default=False)
    p.add_argument("--zip-output", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = resolve_data_root(args.data_dir, out_dir)
    size = (CFG["img_size"], CFG["img_size"])
    led_map = build_led_map(data_root / CFG["led_subdir"], size, CFG["led_bright_thr"], CFG["led_smooth_sigma"])

    pbs_a, pbs_b, pbs_n = load_pairs(data_root / CFG["pbs_subdir"], data_root / CFG["substrate_subdir"], "pbs", size, led_map=led_map)
    ana_a, ana_b, ana_n = load_pairs(data_root / CFG["analyte_subdir"], data_root / CFG["substrate_subdir"], "analyte", size, led_map=led_map)
    if args.reverse_analyte:
        ana_a, ana_b = ana_b, ana_a
        print("OK Signal-decrease adaptation enabled: reverse analyte pairs only; keep PBS unchanged")
    else:
        print("OK Signal-increase mode: keep original before->after order for PBS and analyte")

    all_after = pbs_a + ana_a
    all_before = pbs_b + ana_b
    all_labels = np.array([0] * len(pbs_a) + [1] * len(ana_a), dtype=np.int32)
    all_names = pbs_n + ana_n
    keep_idx = stratified_subset_indices(all_labels, float(args.subset_fraction), CFG["random_state"])
    all_after = [all_after[i] for i in keep_idx]
    all_before = [all_before[i] for i in keep_idx]
    all_labels = all_labels[keep_idx]
    all_names = [all_names[i] for i in keep_idx]
    n_pbs = int((all_labels == 0).sum())
    n_ana = int((all_labels == 1).sum())
    print(f"OK Subset fraction={float(args.subset_fraction):.3f}  sample count={len(all_labels)} (PBS={n_pbs}, Analyte={n_ana})")
    full_feats = np.stack([extract_feature_bank_29(bef, aft) for aft, bef in zip(all_after, all_before)], axis=0).astype(np.float32)
    folds = make_balanced_folds(all_names, all_labels, min(CFG["n_splits"], max(2, len(all_labels))), CFG["random_state"])

    cv_df = train_classifier_cv(full_feats, all_labels, all_names, folds, out_dir, args)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=CFG["random_state"])
    train_idx, val_idx = next(sss.split(np.arange(len(all_labels)), all_labels))
    cls_init_ckpt = train_full_classifier_for_init(full_feats, all_labels, all_names, train_idx, val_idx, out_dir, args)

    train_after = [all_after[i] for i in train_idx]
    train_before = [all_before[i] for i in train_idx]
    train_labels = all_labels[train_idx]
    val_after = [all_after[i] for i in val_idx]
    val_before = [all_before[i] for i in val_idx]
    val_labels = all_labels[val_idx]

    mask = compute_mask(train_after, thr=CFG["mask_thr"], erode=CFG["mask_erode"])
    train_pbs_after = [img for img, lbl in zip(train_after, train_labels) if lbl == 0]
    train_pbs_before = [img for img, lbl in zip(train_before, train_labels) if lbl == 0]
    pbs_mu29, pbs_std29 = build_pbs_baseline_29(train_pbs_after, train_pbs_before, mask)
    cm, cs = _compute_channel_stats(train_after, train_before)
    pbs_mu_px, pbs_std_px = build_pbs_pixel_baseline(train_pbs_after, train_pbs_before, mask, mode=CFG["feature_bank_mode"])

    coarse_maps_all = [
        build_score_maps(bef, aft, pbs_mu_px, pbs_std_px, mask, {"feature_bank_mode": CFG["feature_bank_mode"], "score_primary": CFG["score_primary"], "support_z_thr": CFG["support_z_thr"]})
        for aft, bef in zip(all_after, all_before)
    ]
    support_maps_ana = [maps["support_frac"] for maps, lbl in zip(coarse_maps_all, all_labels) if lbl == 1]
    coarse_hm_pbs = [maps["primary"] for maps, lbl in zip(coarse_maps_all, all_labels) if lbl == 0]
    coarse_hm_ana = [maps["primary"] for maps, lbl in zip(coarse_maps_all, all_labels) if lbl == 1]

    model_specs = [
        ("ResNet18_29_transfer", ResNet18_29(29), cls_init_ckpt, CFG["resnet_lr"], CFG["resnet_wd"], False, 0.0),
    ]
    metrics_rows = [compute_heatmap_metrics(coarse_hm_pbs, coarse_hm_ana, support_maps_ana, "Coarse_top3_weighted")]
    hist_frames = []
    for name, model, init_ckpt, lr, wd, use_aux, aux_w in model_specs:
        trained_model, hist_df = train_stage2a_model(
            "stage2a_resnet" if "ResNet18" in name else ("stage2a_unet" if "UNet" in name else "stage2a_cnn"),
            model,
            train_after,
            train_before,
            train_labels,
            val_after,
            val_before,
            val_labels,
            mask,
            pbs_mu29,
            pbs_std29,
            cm,
            cs,
            out_dir,
            args,
            init_classifier_ckpt=init_ckpt,
            lr=lr,
            weight_decay=wd,
            use_aux=use_aux,
            aux_cls_w=aux_w,
        )
        hist_df["Method"] = name
        hist_frames.append(hist_df)
        hmaps_all = infer_heatmaps(trained_model, all_after, all_before, cm, cs)
        hm_pbs = [h for h, lbl in zip(hmaps_all, all_labels) if lbl == 0]
        hm_ana = [h for h, lbl in zip(hmaps_all, all_labels) if lbl == 1]
        metrics_rows.append(compute_heatmap_metrics(hm_pbs, hm_ana, support_maps_ana, name))

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "heatmap_summary.csv", index=False)
    pd.concat(hist_frames, ignore_index=True).to_csv(out_dir / "stage2a_training_history.csv", index=False)
    pd.DataFrame(
        [
            {
                "classification_model": "29feat+ResNet18 classifier",
                "stage2a_resnet_init": "load classifier backbone, freeze stem+layer1 for warmup, then unfreeze all",
                "stage2a_cnn_init": "from scratch, heat_map_v4 lr/wd",
                "stage2a_unet_init": "from scratch + aux cls head, heat_map_v4 lr/wd",
                "freeze_warmup_epochs": args.freeze_warmup_epochs,
                "clf_freeze_warmup_epochs": args.clf_freeze_warmup_epochs,
                "clf_epochs": args.clf_epochs,
                "stage2a_epochs": args.stage2a_epochs,
                "stage2a_split": "StratifiedShuffleSplit test_size=0.2",
                "subset_fraction": float(args.subset_fraction),
                "subset_total_samples": int(len(all_labels)),
                "subset_pbs": int(n_pbs),
                "subset_analyte": int(n_ana),
                "feature_bank_mode_for_coarse": CFG["feature_bank_mode"],
                "coarse_compare_reference": CFG["score_primary"],
                "reverse_analyte": bool(args.reverse_analyte),
                "init_classifier_ckpt": str(args.init_classifier_ckpt) if args.init_classifier_ckpt else "",
            }
        ]
    ).to_csv(out_dir / "run_strategy_summary.csv", index=False)

    plot_heatmap_bars(metrics_df, out_dir / "heatmap_barplots.png")
    plot_heatmap_radar(metrics_df, out_dir / "heatmap_radar.png")

    with pd.ExcelWriter(out_dir / "results_master.xlsx", engine="openpyxl") as writer:
        pd.DataFrame([vars(args)]).to_excel(writer, sheet_name="run_args", index=False)
        cv_df.to_excel(writer, sheet_name="classifier_cv_results", index=False)
        metrics_df.to_excel(writer, sheet_name="heatmap_summary", index=False)
        pd.read_csv(out_dir / "classifier_fold_label_summary.csv").to_excel(writer, sheet_name="classifier_fold_label", index=False)
        pd.concat(hist_frames, ignore_index=True).to_excel(writer, sheet_name="stage2a_training_history", index=False)

    if args.zip_output:
        zip_path = shutil.make_archive(str(out_dir), "zip", str(out_dir))
        print(f"OK Output archive created: {zip_path}")

    print("OK Classification CV results:", out_dir / "classifier_cv_results.csv")
    print("OK Stage 2a five-metric comparison:", out_dir / "heatmap_summary.csv")
    print("OK Results workbook:", out_dir / "results_master.xlsx")


if __name__ == "__main__":
    main()
