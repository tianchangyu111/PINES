import argparse
import random
import zipfile
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, uniform_filter
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18


ROOT = Path(r".")
ZIP_ROOT = Path(r"data/concentration_archives")
SPLIT_ROOT = ROOT / "data" / "concentration"
OUT_DIR = ROOT / "outputs" / "concentration_5fold"
IMG_SIZE = 128
RANDOM_STATE = 42
N_SPLITS = 5
BATCH_SIZE = 16
EPOCHS = 18
PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["PBS", "1uM", "10uM", "100uM"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
FEATURE_NAMES = [
    "diff",
    "abs_diff",
    "log1p_diff",
    "ratio_minus1",
    "relative_diff_clip",
    "gauss_diff_s1",
    "gauss_diff_s3",
    "gauss_diff_s6",
    "local_std_delta_w3",
    "local_std_delta_w7",
    "local_std_delta_w15",
    "local_contrast_delta_w5",
    "local_contrast_delta_w9",
    "lowpass_delta",
    "highpass_delta",
    "highpass_energy_delta",
    "bandpass_diff_s1",
    "bandpass_diff_s2",
    "bandpass_diff_s4",
    "before_raw",
    "after_raw",
    "before_gauss_s2",
    "after_gauss_s2",
    "local_std_diff_w3",
    "local_std_diff_w7",
    "gauss_absdiff_delta",
    "square_delta",
    "gauss_absdiff_s3",
    "positive_diff",
]


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_zip_images(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        return sorted(
            n
            for n in zf.namelist()
            if not n.endswith("/")
            and n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        )


def read_zip_gray(zip_path, member):
    with zipfile.ZipFile(zip_path) as zf:
        data = np.frombuffer(zf.read(member), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not decode {member}")
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)


def read_file_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read {path}")
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)


def basename(member):
    return Path(member).name


def local_std(x, win):
    mean = uniform_filter(x, size=win)
    mean2 = uniform_filter(x * x, size=win)
    return np.sqrt(np.maximum(mean2 - mean * mean, 0))


def local_contrast(x, win):
    return local_std(x, win) / (uniform_filter(x, size=win) + 1e-3)


def extract_29(before, after):
    before = before.astype(np.float32)
    after = after.astype(np.float32)
    eps = 1e-3
    diff = after - before
    abs_diff = np.abs(diff)
    low_before = gaussian_filter(before, 8)
    low_after = gaussian_filter(after, 8)
    high_before = before - low_before
    high_after = after - low_after
    feats = [
        diff,
        abs_diff,
        np.log1p(abs_diff) * np.sign(diff),
        after / (before + eps) - 1.0,
        np.clip(diff / (before + eps), -2, 2),
        gaussian_filter(diff, 1),
        gaussian_filter(diff, 3),
        gaussian_filter(diff, 6),
        local_std(after, 3) - local_std(before, 3),
        local_std(after, 7) - local_std(before, 7),
        local_std(after, 15) - local_std(before, 15),
        local_contrast(after, 5) - local_contrast(before, 5),
        local_contrast(after, 9) - local_contrast(before, 9),
        low_after - low_before,
        high_after - high_before,
        high_after * high_after - high_before * high_before,
        gaussian_filter(diff, 1) - gaussian_filter(diff, 2),
        gaussian_filter(diff, 2) - gaussian_filter(diff, 4),
        gaussian_filter(diff, 4) - gaussian_filter(diff, 8),
        before,
        after,
        gaussian_filter(before, 2),
        gaussian_filter(after, 2),
        local_std(diff, 3),
        local_std(diff, 7),
        gaussian_filter(abs_diff, 1) - gaussian_filter(abs_diff, 6),
        diff * diff,
        gaussian_filter(abs_diff, 3),
        np.maximum(diff, 0),
    ]
    return np.stack(feats).astype(np.float32)


def build_records():
    records = []

    pbs_zip = ZIP_ROOT / "pbs.zip"
    sub_zip = ZIP_ROOT / "susbtrat.zip"
    pbs_members = {basename(n): n for n in list_zip_images(pbs_zip)}
    sub_members = {basename(n): n for n in list_zip_images(sub_zip)}
    for name in sorted(set(pbs_members) & set(sub_members)):
        records.append(
            {
                "name": name,
                "label": CLASS_TO_ID["PBS"],
                "label_name": "PBS",
                "after": ("zip", pbs_zip, pbs_members[name]),
                "before": ("zip", sub_zip, sub_members[name]),
            }
        )

    for label_name in ["1uM", "10uM", "100uM"]:
        a_dir = SPLIT_ROOT / label_name / "analyte"
        s_dir = SPLIT_ROOT / label_name / "susbtrat"
        for after_path in sorted(a_dir.glob("*")):
            if not after_path.is_file():
                continue
            before_path = s_dir / after_path.name
            if before_path.exists():
                records.append(
                    {
                        "name": after_path.name,
                        "label": CLASS_TO_ID[label_name],
                        "label_name": label_name,
                        "after": ("file", after_path),
                        "before": ("file", before_path),
                    }
                )
    return records


def read_ref(ref):
    if ref[0] == "zip":
        return read_zip_gray(ref[1], ref[2])
    return read_file_gray(ref[1])


class FeatureDataset(Dataset):
    def __init__(self, records, mean=None, std=None):
        self.records = records
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        x = extract_29(read_ref(r["before"]), read_ref(r["after"]))
        if self.mean is not None:
            x = (x - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-6)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(r["label"], dtype=torch.long), r["name"]


def compute_channel_stats(records):
    sums = np.zeros(29, dtype=np.float64)
    sqs = np.zeros(29, dtype=np.float64)
    n = 0
    for r in records:
        x = extract_29(read_ref(r["before"]), read_ref(r["after"]))
        flat = x.reshape(29, -1)
        sums += flat.sum(axis=1)
        sqs += (flat * flat).sum(axis=1)
        n += flat.shape[1]
    mean = sums / n
    std = np.sqrt(np.maximum(sqs / n - mean * mean, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def compute_feature_mean_table(records):
    rows = []
    for idx, r in enumerate(records):
        x = extract_29(read_ref(r["before"]), read_ref(r["after"]))
        row = {
            "record_index": idx,
            "name": r["name"],
            "label": r["label_name"],
        }
        row.update({name: float(x[i].mean()) for i, name in enumerate(FEATURE_NAMES)})
        rows.append(row)
    return pd.DataFrame(rows)


class ResNet29FourClass(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.net = resnet18(weights=None)
        self.net.conv1 = nn.Conv2d(29, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net.fc = nn.Linear(self.net.fc.in_features, n_classes)

    def forward(self, x):
        return self.net(x)


def train_one_fold(train_records, val_records):
    mean, std = compute_channel_stats(train_records)
    tr_loader = DataLoader(FeatureDataset(train_records, mean, std), batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(FeatureDataset(val_records, mean, std), batch_size=BATCH_SIZE, shuffle=False)
    model = ResNet29FourClass().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    best_state = None
    best_score = -1
    bad = 0
    hist = []
    for ep in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for x, y, _ in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        y_true, y_pred = evaluate(model, va_loader)[:2]
        bal = balanced_accuracy_score(y_true, y_pred)
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_balanced_accuracy": bal})
        if bal > best_score:
            best_score = bal
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, mean, std, pd.DataFrame(hist)


def evaluate(model, loader):
    model.eval()
    ys, preds, probs, names = [], [], [], []
    with torch.no_grad():
        for x, y, nms in loader:
            x = x.to(DEVICE)
            p = torch.softmax(model(x), dim=1).cpu().numpy()
            pred = p.argmax(axis=1)
            ys.extend(y.numpy().tolist())
            preds.extend(pred.tolist())
            probs.extend(p.tolist())
            names.extend(list(nms))
    return np.array(ys), np.array(preds), np.array(probs), names


def metrics(y, pred):
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
    }


def plot_metric_summary(cv):
    metric_cols = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "mcc"]
    rows = [{"metric": c, "mean": cv[c].mean(), "std": cv[c].std()} for c in metric_cols]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "fourclass_metrics_mean_std.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(summary["metric"], summary["mean"], yerr=summary["std"], capsize=5, color="#2f6f8f")
    ax.set_title("Four-class ResNet29 classification metrics")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    for i, row in summary.iterrows():
        ax.text(i, row["mean"] + 0.035, f"{row['mean']:.3f}\n+/-{row['std']:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fourclass_metrics_mean_std.png", dpi=220)
    plt.close(fig)


def plot_fold_metrics(cv):
    metrics_to_plot = ["accuracy", "balanced_accuracy", "f1_macro", "mcc"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(cv))
    width = 0.18
    for j, m in enumerate(metrics_to_plot):
        ax.bar(x + (j - 1.5) * width, cv[m], width, label=m)
    ax.set_title("Fold-wise four-class metrics")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x, [f"Fold {int(f)}" for f in cv["fold"]])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fourclass_metrics_by_fold.png", dpi=220)
    plt.close(fig)


def plot_history():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for fold in range(1, N_SPLITS + 1):
        p = OUT_DIR / f"history_fold{fold}.csv"
        if not p.exists():
            continue
        hist = pd.read_csv(p)
        axes[0].plot(hist["epoch"], hist["train_loss"], marker="o", label=f"fold{fold}")
        axes[1].plot(hist["epoch"], hist["val_balanced_accuracy"], marker="o", label=f"fold{fold}")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Validation balanced accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fourclass_training_curves.png", dpi=220)
    plt.close(fig)


def save_report_tables(details):
    y_true = details["true_label"]
    y_pred = details["predicted_label"]
    report = classification_report(y_true, y_pred, labels=CLASS_NAMES, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(OUT_DIR / "fourclass_classification_report.csv", encoding="utf-8-sig")

    details[details["correct"] == False].to_csv(OUT_DIR / "fourclass_errors.csv", index=False, encoding="utf-8-sig")
    label_summary = (
        details.groupby(["true_label", "predicted_label"]).size().reset_index(name="n").sort_values(["true_label", "predicted_label"])
    )
    label_summary.to_csv(OUT_DIR / "fourclass_label_prediction_counts.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    rep_df = pd.DataFrame(report).T.loc[CLASS_NAMES, ["precision", "recall", "f1-score"]]
    x = np.arange(len(rep_df))
    width = 0.24
    for j, c in enumerate(rep_df.columns):
        ax.bar(x + (j - 1) * width, rep_df[c], width, label=c)
    ax.set_xticks(x, rep_df.index)
    ax.set_ylim(0, 1.08)
    ax.set_title("Per-class precision, recall and F1")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fourclass_per_class_metrics.png", dpi=220)
    plt.close(fig)


def plot_probability_distribution(details):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    for i, cls in enumerate(CLASS_NAMES):
        ax = axes[i]
        sub = details[details["true_label"] == cls]
        prob_col = f"prob_{cls}"
        ax.hist(sub[prob_col], bins=16, color="#2f6f8f", alpha=0.85)
        ax.set_title(f"True {cls}: probability for {cls}")
        ax.set_xlim(0, 1)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fourclass_probability_distributions.png", dpi=220)
    plt.close(fig)


def resnet_gradient_shap_batch(model, x, target, steps=8, noise_std=0.02):
    model.eval()
    total_grad = torch.zeros_like(x)
    baseline = torch.zeros_like(x)
    delta = x - baseline
    for _ in range(steps):
        alpha = torch.rand((x.shape[0], 1, 1, 1), device=x.device)
        noise = torch.randn_like(x) * noise_std
        xi = (baseline + alpha * delta + noise).detach().requires_grad_(True)
        logits = model(xi)
        score = logits[torch.arange(x.shape[0], device=x.device), target].sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        total_grad += xi.grad.detach()
    return (delta * total_grad / steps).detach()


def run_resnet_gradient_shap(fold_payloads, details):
    sample_rows = []
    class_values = {name: [] for name in CLASS_NAMES}
    overall_values = []
    heatmap_examples = []

    for payload in fold_payloads:
        model = payload["model"]
        mean = payload["mean"]
        std = payload["std"]
        val_records = payload["val_records"]
        loader = DataLoader(FeatureDataset(val_records, mean, std), batch_size=8, shuffle=False)

        for x, y, names in loader:
            x = x.to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)
                pred = probs.argmax(dim=1)
                conf = probs.max(dim=1).values
            attr = resnet_gradient_shap_batch(model, x, pred)
            attr_abs = attr.abs().mean(dim=(2, 3)).cpu().numpy()
            pred_np = pred.cpu().numpy()
            true_np = y.numpy()
            conf_np = conf.cpu().numpy()

            for i, name in enumerate(names):
                vals = attr_abs[i]
                vals_norm = vals / (vals.sum() + 1e-12)
                pred_name = CLASS_NAMES[int(pred_np[i])]
                true_name = CLASS_NAMES[int(true_np[i])]
                class_values[pred_name].append(vals_norm)
                overall_values.append(vals_norm)
                top_idx = np.argsort(vals_norm)[::-1][:5]
                row = {
                    "fold": payload["fold"],
                    "name": name,
                    "true_label": true_name,
                    "predicted_label": pred_name,
                    "confidence": float(conf_np[i]),
                    "correct": bool(pred_np[i] == true_np[i]),
                }
                for rank, feat_idx in enumerate(top_idx, start=1):
                    row[f"top{rank}_feature"] = FEATURE_NAMES[feat_idx]
                    row[f"top{rank}_attribution"] = float(vals_norm[feat_idx])
                sample_rows.append(row)

                if len(heatmap_examples) < 16:
                    heat = attr[i].abs().sum(dim=0).detach().cpu().numpy()
                    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-12)
                    heatmap_examples.append((name, true_name, pred_name, heat))

    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(OUT_DIR / "fourclass_resnet_gradient_shap_samples.csv", index=False, encoding="utf-8-sig")

    overall = np.vstack(overall_values).mean(axis=0)
    overall_df = pd.DataFrame({"feature": FEATURE_NAMES, "mean_abs_attribution": overall}).sort_values(
        "mean_abs_attribution", ascending=False
    )
    overall_df.to_csv(OUT_DIR / "fourclass_resnet_gradient_shap_top20_overall.csv", index=False, encoding="utf-8-sig")

    class_rows = []
    for class_name, values in class_values.items():
        if not values:
            continue
        mean_vals = np.vstack(values).mean(axis=0)
        order = np.argsort(mean_vals)[::-1]
        for rank, feat_idx in enumerate(order[:20], start=1):
            class_rows.append(
                {
                    "class": class_name,
                    "rank": rank,
                    "feature": FEATURE_NAMES[feat_idx],
                    "mean_abs_attribution": float(mean_vals[feat_idx]),
                }
            )
    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(OUT_DIR / "fourclass_resnet_gradient_shap_top20_by_class.csv", index=False, encoding="utf-8-sig")

    top = overall_df.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["mean_abs_attribution"], color="#2f6f8f")
    ax.set_title("Direct ResNet18 GradientSHAP-style attribution")
    ax.set_xlabel("normalized mean |attribution|")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fourclass_resnet_gradient_shap_top_features.png", dpi=220)
    plt.close(fig)

    pivot = class_df[class_df["rank"] <= 10].pivot_table(
        index="feature", columns="class", values="mean_abs_attribution", aggfunc="mean", fill_value=0
    )
    pivot = pivot.reindex(overall_df.head(12)["feature"].values).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, cmap="YlGnBu")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Class-wise direct ResNet18 attribution")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fourclass_resnet_gradient_shap_class_heatmap.png", dpi=220)
    plt.close(fig)

    if heatmap_examples:
        cols = 4
        rows = int(np.ceil(len(heatmap_examples) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
        axes = np.asarray(axes).ravel()
        for ax, (name, true_name, pred_name, heat) in zip(axes, heatmap_examples):
            ax.imshow(heat, cmap="inferno")
            ax.set_title(f"{name}\ntrue={true_name}, pred={pred_name}", fontsize=8)
            ax.axis("off")
        for ax in axes[len(heatmap_examples) :]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "fourclass_resnet_gradient_shap_example_maps.png", dpi=220)
        plt.close(fig)

    with pd.ExcelWriter(OUT_DIR / "fourclass_summary_tables.xlsx") as writer:
        pd.read_csv(OUT_DIR / "fourclass_cv_results.csv").to_excel(writer, sheet_name="cv_results", index=False)
        pd.read_csv(OUT_DIR / "fourclass_metrics_mean_std.csv").to_excel(writer, sheet_name="mean_std", index=False)
        pd.read_csv(OUT_DIR / "fourclass_classification_report.csv").to_excel(
            writer, sheet_name="classification_report", index=False
        )
        pd.read_csv(OUT_DIR / "fourclass_confusion_matrix.csv").to_excel(writer, sheet_name="confusion_matrix", index=False)
        class_df.to_excel(writer, sheet_name="resnet_attr_by_class", index=False)
        overall_df.head(20).to_excel(writer, sheet_name="resnet_attr_overall", index=False)
        sample_df.to_excel(writer, sheet_name="sample_attribution", index=False)


def package_outputs():
    zip_path = OUT_DIR.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT_DIR.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(OUT_DIR.parent))
    return zip_path


def main():
    global ZIP_ROOT, SPLIT_ROOT, OUT_DIR, N_SPLITS, EPOCHS, BATCH_SIZE, PATIENCE
    parser = argparse.ArgumentParser(description="29-feature four-class ResNet18 training.")
    parser.add_argument("--zip-root", type=Path, default=ZIP_ROOT, help="Directory containing pbs.zip and susbtrat.zip.")
    parser.add_argument("--split-root", type=Path, default=SPLIT_ROOT, help="Directory with verified 1uM, 10uM, and 100uM paired data.")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--folds", type=int, default=N_SPLITS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args()
    if args.folds < 2 or min(args.epochs, args.batch_size, args.patience) < 1:
        parser.error("folds must be at least 2; epochs, batch-size, and patience must be positive")
    ZIP_ROOT, SPLIT_ROOT, OUT_DIR = args.zip_root, args.split_root, args.output_dir
    N_SPLITS, EPOCHS, BATCH_SIZE, PATIENCE = args.folds, args.epochs, args.batch_size, args.patience
    seed_all(RANDOM_STATE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = build_records()
    for idx, record in enumerate(records):
        record["record_index"] = idx
    pd.DataFrame({"label": [r["label_name"] for r in records]}).value_counts().reset_index(name="n").to_csv(
        OUT_DIR / "dataset_counts.csv", index=False
    )

    y = np.array([r["label"] for r in records])
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows, details, fold_payloads = [], [], []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        train_records = [records[i] for i in tr_idx]
        val_records = [records[i] for i in va_idx]
        model, mean, std, hist = train_one_fold(train_records, val_records)
        torch.save({"model_state": model.state_dict(), "mean": mean, "std": std, "class_names": CLASS_NAMES}, OUT_DIR / f"fold{fold}_resnet29_fourclass.pt")
        fold_payloads.append({"fold": fold, "model": model, "mean": mean, "std": std, "val_records": val_records})
        hist.to_csv(OUT_DIR / f"history_fold{fold}.csv", index=False)
        va_loader = DataLoader(FeatureDataset(val_records, mean, std), batch_size=BATCH_SIZE, shuffle=False)
        yy, pp, prob, names = evaluate(model, va_loader)
        rows.append({"fold": fold, **metrics(yy, pp)})
        for name, t, pred, pr, record in zip(names, yy, pp, prob, val_records):
            details.append(
                {
                    "fold": fold,
                    "record_index": int(record["record_index"]),
                    "name": name,
                    "true_label": CLASS_NAMES[int(t)],
                    "predicted_label": CLASS_NAMES[int(pred)],
                    "correct": bool(t == pred),
                    **{f"prob_{c}": float(pr[i]) for i, c in enumerate(CLASS_NAMES)},
                }
            )
    cv = pd.DataFrame(rows)
    det = pd.DataFrame(details)
    cv.to_csv(OUT_DIR / "fourclass_cv_results.csv", index=False)
    det.to_csv(OUT_DIR / "fourclass_fold_details.csv", index=False)

    cm = confusion_matrix(
        det["true_label"].map(CLASS_TO_ID),
        det["predicted_label"].map(CLASS_TO_ID),
        labels=list(range(len(CLASS_NAMES))),
    )
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(OUT_DIR / "fourclass_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4), CLASS_NAMES, rotation=30)
    ax.set_yticks(range(4), CLASS_NAMES)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fourclass_confusion_matrix.png", dpi=220)
    plt.close(fig)

    save_report_tables(det)
    plot_metric_summary(cv)
    plot_fold_metrics(cv)
    plot_history()
    plot_probability_distribution(det)
    feature_df = compute_feature_mean_table(records)
    feature_df.to_csv(OUT_DIR / "fourclass_29feat_mean_matrix.csv", index=False, encoding="utf-8-sig")
    run_resnet_gradient_shap(fold_payloads, det)

    meta = {
        "model": "29feat + ResNet18 four-class classifier",
        "classes": CLASS_NAMES,
        "n_samples": len(records),
        "n_splits": N_SPLITS,
        "img_size": IMG_SIZE,
        "device": DEVICE,
    }
    pd.Series(meta).to_json(OUT_DIR / "fourclass_run_meta.json", force_ascii=False, indent=2)
    zip_path = package_outputs()

    print(cv)
    print("MEAN")
    print(cv.mean(numeric_only=True))
    print(OUT_DIR)
    print(zip_path)


if __name__ == "__main__":
    main()
