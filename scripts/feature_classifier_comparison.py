import argparse
import json
from pathlib import Path

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
from torch.utils.data import DataLoader, Dataset

from gas_transfer import ResNet18_29, extract_feature_bank_29, make_balanced_folds
from transfer_learning import DEVICE, build_led_map, load_pairs, resolve_data_root


CFG = {
    "img_size": 128,
    "led_subdir": "led_field",
    "pbs_subdir": "pbs",
    "analyte_subdir": "analyte",
    "substrate_subdir": "susbtrat",
    "led_bright_thr": 0.5,
    "led_smooth_sigma": 15,
    "n_splits": 3,
    "random_state": 42,
    "epochs": 16,
    "batch_size": 8,
    "lr": 2e-4,
    "weight_decay": 1e-5,
    "patience": 5,
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


class Feature29Dataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class SimpleCNN29(nn.Module):
    def __init__(self, in_ch=29):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.fc(x).squeeze(1)


class ResNet29Classifier(nn.Module):
    def __init__(self, in_ch=29):
        super().__init__()
        self.backbone = ResNet18_29(in_ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        s = self.backbone.stem(x)
        l1 = self.backbone.layer1(s)
        l2 = self.backbone.layer2(l1)
        l3 = self.backbone.layer3(l2)
        x = self.pool(l3).flatten(1)
        return self.fc(x).squeeze(1)


def load_refine_init(model, ckpt_path):
    if not ckpt_path or not Path(ckpt_path).exists():
        return False
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    return True


def train_one_model(model_name, build_model, feats, labels, names, folds, out_dir, epochs, batch_size, lr, weight_decay, patience, init_ckpt=None):
    model_dir = out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    detail_rows = []
    cv_rows = []
    hist_rows = []

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        x_tr_raw = feats[tr_idx]
        y_tr = labels[tr_idx]
        x_va_raw = feats[va_idx]
        y_va = labels[va_idx]
        names_va = [names[i] for i in va_idx]

        cm = x_tr_raw.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        cs = x_tr_raw.std(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        cs = np.maximum(cs, 1e-6)
        x_tr = ((x_tr_raw - cm) / cs).astype(np.float32)
        x_va = ((x_va_raw - cm) / cs).astype(np.float32)

        tr_loader = DataLoader(Feature29Dataset(x_tr, y_tr), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
        va_loader = DataLoader(Feature29Dataset(x_va, y_va), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

        model = build_model().to(DEVICE)
        init_used = False
        if init_ckpt and model_name == "resnet29":
            init_used = load_refine_init(model, init_ckpt)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        crit = nn.BCEWithLogitsLoss()
        best_state = None
        best_score = -1e9
        wait = 0

        for ep in range(1, epochs + 1):
            model.train()
            tr_losses = []
            for xb, yb in tr_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()
                tr_losses.append(float(loss.item()))

            model.eval()
            va_prob = []
            va_true = []
            with torch.no_grad():
                for xb, yb in va_loader:
                    logits = model(xb.to(DEVICE, non_blocking=True))
                    prob = torch.sigmoid(logits).detach().cpu().numpy()
                    va_prob.extend(prob.tolist())
                    va_true.extend(yb.numpy().tolist())
            va_prob = np.asarray(va_prob, dtype=np.float32)
            va_true = np.asarray(va_true, dtype=np.int32)
            va_pred = (va_prob >= 0.5).astype(int)
            metrics = compute_metrics(va_true, va_pred, va_prob)
            score = float(metrics["balanced_accuracy"])
            hist_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "epoch": ep,
                    "train_loss": float(np.mean(tr_losses) if tr_losses else np.nan),
                    "val_accuracy": float(metrics["accuracy"]),
                    "val_balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "val_f1": float(metrics["f1"]),
                    "val_roc_auc": float(metrics["roc_auc"]),
                }
            )
            if score > best_score + 1e-12:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        tr_loader_eval = DataLoader(Feature29Dataset(x_tr, y_tr), batch_size=batch_size, shuffle=False, num_workers=0)
        tr_prob = []
        with torch.no_grad():
            for xb, _ in tr_loader_eval:
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
        row["used_refine_init"] = bool(init_used)
        cv_rows.append(row)

        for nm, y_true_i, prob_i, pred_i in zip(names_va, y_va, va_prob, va_pred):
            true_label = "Analyte" if int(y_true_i) == 1 else "PBS"
            pred_label = "Analyte" if int(pred_i) == 1 else "PBS"
            detail_rows.append(
                {
                    "model": model_name,
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

        torch.save(
            {
                "model_state": model.state_dict(),
                "channel_mean": cm.squeeze().tolist(),
                "channel_std": cs.squeeze().tolist(),
                "threshold": float(thr),
                "fold": fold,
                "model_name": model_name,
            },
            model_dir / f"{model_name}_fold{fold}.pt",
        )

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(model_dir / f"{model_name}_cv_results.csv", index=False)
    hist_df = pd.DataFrame(hist_rows)
    hist_df.to_csv(model_dir / f"{model_name}_training_history.csv", index=False)
    detail_df = pd.DataFrame(detail_rows).sort_values(["fold", "true_label", "name"]).reset_index(drop=True)
    detail_df.to_csv(model_dir / f"{model_name}_fold_details.csv", index=False)
    detail_df[~detail_df["correct"]].to_csv(model_dir / f"{model_name}_fold_errors.csv", index=False)
    label_summary = (
        detail_df.groupby(["fold", "true_label"])
        .agg(total=("name", "count"), correct=("correct", "sum"), mean_prob_analyte=("prob_analyte", "mean"))
        .reset_index()
    )
    label_summary["accuracy_within_label"] = label_summary["correct"] / label_summary["total"]
    label_summary.to_csv(model_dir / f"{model_name}_fold_label_summary.csv", index=False)
    return cv_df


def parse_args():
    p = argparse.ArgumentParser(description="29feat classification comparison: CNN vs ResNet on gas transfer data.")
    p.add_argument("--data-dir", default=r"data/gas")
    p.add_argument("--output-dir", default=r"outputs/classifier_comparison")
    p.add_argument("--base-output-dir", default=r"outputs/hybrid")
    p.add_argument("--epochs", type=int, default=CFG["epochs"])
    p.add_argument("--batch-size", type=int, default=CFG["batch_size"])
    p.add_argument("--lr", type=float, default=CFG["lr"])
    p.add_argument("--weight-decay", type=float, default=CFG["weight_decay"])
    p.add_argument("--patience", type=int, default=CFG["patience"])
    p.add_argument("--reverse-analyte", action="store_true", default=True)
    p.add_argument("--no-reverse-analyte", dest="reverse_analyte", action="store_false")
    p.add_argument("--models", nargs="+", choices=["cnn29", "resnet29"], default=["cnn29", "resnet29"])
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

    all_after = pbs_a + ana_a
    all_before = pbs_b + ana_b
    all_labels = np.array([0] * len(pbs_a) + [1] * len(ana_a), dtype=np.int32)
    all_names = pbs_n + ana_n
    feats = np.stack([extract_feature_bank_29(bef, aft) for aft, bef in zip(all_after, all_before)], axis=0).astype(np.float32)
    folds = make_balanced_folds(all_names, all_labels, min(CFG["n_splits"], max(2, len(all_labels))), CFG["random_state"])

    resnet_init_ckpt = str(Path(args.base_output_dir) / "stage2a_refine.pt")
    summaries = []
    if "cnn29" in args.models:
        cnn_cv = train_one_model(
            "cnn29",
            lambda: SimpleCNN29(29),
            feats,
            all_labels,
            all_names,
            folds,
            out_dir,
            args.epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            args.patience,
            init_ckpt=None,
        )
        summaries.append(
            {
                "model": "cnn29",
                "mean_accuracy": float(cnn_cv["accuracy"].mean()),
                "mean_balanced_accuracy": float(cnn_cv["balanced_accuracy"].mean()),
                "mean_f1": float(cnn_cv["f1"].mean()),
                "mean_roc_auc": float(cnn_cv["roc_auc"].mean()),
            }
        )
    if "resnet29" in args.models:
        resnet_cv = train_one_model(
            "resnet29",
            lambda: ResNet29Classifier(29),
            feats,
            all_labels,
            all_names,
            folds,
            out_dir,
            args.epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            args.patience,
            init_ckpt=resnet_init_ckpt,
        )
        summaries.append(
            {
                "model": "resnet29",
                "mean_accuracy": float(resnet_cv["accuracy"].mean()),
                "mean_balanced_accuracy": float(resnet_cv["balanced_accuracy"].mean()),
                "mean_f1": float(resnet_cv["f1"].mean()),
                "mean_roc_auc": float(resnet_cv["roc_auc"].mean()),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(out_dir / "model_compare_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
