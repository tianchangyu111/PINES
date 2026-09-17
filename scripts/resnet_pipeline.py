import argparse
import json
import shutil
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.model_selection import StratifiedShuffleSplit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hybrid_pipeline import (
    RESPONSE_CMAP,
    _normalize_01,
    build_compact_stage_tile,
    build_response_only_tile,
    export_group_zscore_summary,
    export_led_bundle,
    export_led_effect_examples,
    export_mask_bundle,
    export_baseline_bundle,
    plot_led_correction,
    plot_response_area_comparison,
    save_feature_value_workbook,
    save_group_map_galleries,
    save_image_grid,
)
from feature_classifier_comparison import ResNet29Classifier
from gas_transfer import (
    ResNet18_29,
    _compute_channel_stats,
    build_pbs_baseline_29,
    extract_feature_bank_29,
    make_balanced_folds,
)
from localization_comparison import (
    CFG as STAGE_CFG,
    compute_metrics,
    find_best_threshold,
    load_classifier_backbone_into_stage2a,
    train_classifier_cv,
    train_full_classifier_for_init,
    train_stage2a_model,
)
from transfer_learning import DEVICE, build_led_map, compute_mask, load_pairs, resolve_data_root


CFG = {
    "img_size": 128,
    "led_subdir": "led_field",
    "pbs_subdir": "pbs",
    "analyte_subdir": "analyte",
    "substrate_subdir": "susbtrat",
    "led_bright_thr": 0.35,
    "led_smooth_sigma": 15,
    "mask_thr": 20,
    "mask_erode": 8,
    "n_splits": 3,
    "random_state": 42,
    "clf_epochs": 16,
    "clf_batch_size": 8,
    "clf_lr": 2e-4,
    "clf_weight_decay": 1e-5,
    "clf_patience": 5,
    "clf_freeze_warmup_epochs": 2,
    "support_z_thr": 2.5,
    "stage2a_epochs": 50,
    "stage2a_batch_size": 8,
    "stage2a_patience": 12,
    "freeze_warmup_epochs": 3,
    "resnet_lr": 1.5e-4,
    "resnet_wd": 1e-5,
    "tv_weight": 5e-4,
    "top_k_ratio": 0.05,
    "response_percentile_coarse": 99.5,
    "response_percentile_refined": 99.5,
    "final_gallery_cols": 3,
    "dpi": 150,
}

FEAT29_NAMES = [
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


def build_score_maps_29(before, after, pbs_mu29, pbs_std29, mask, support_z_thr):
    feat = extract_feature_bank_29(before, after)
    z = (feat - pbs_mu29) / (pbs_std29 + 1e-6)
    z = np.where(mask[None], z, 0.0)
    z_pos = np.clip(z, 0.0, None)
    z_sorted = np.sort(z_pos, axis=0)[::-1]
    top1 = z_sorted[0]
    top3 = z_sorted[:3].mean(0)
    top5 = np.sqrt((z_sorted[:5] ** 2).mean(0))
    support_frac = (z > support_z_thr).sum(0).astype(np.float32) / z.shape[0]
    return {
        "top1": _normalize_01(top1),
        "top3_weighted": _normalize_01(0.2 * z_sorted[0] + 0.5 * top3 + 0.3 * z_sorted[:5].mean(0)),
        "top5_rms": _normalize_01(top5),
        "support_frac": _normalize_01(support_frac),
        "feature_names": FEAT29_NAMES,
        "z_stack": z.astype(np.float32),
    }


def summarize_image_features_29(feat, mask):
    rows = {}
    masked = feat[:, mask] if mask is not None and mask.any() else feat.reshape(feat.shape[0], -1)
    for i, nm in enumerate(FEAT29_NAMES):
        vals = masked[i]
        rows[f"{nm}_mean"] = float(vals.mean())
        rows[f"{nm}_std"] = float(vals.std())
        rows[f"{nm}_q95"] = float(np.quantile(vals, 0.95))
    return rows


def export_feature_bank_dictionary_29(out_dir):
    rows = []
    for idx, name in enumerate(FEAT29_NAMES):
        rows.append(
            {
                "feature_idx": idx,
                "feature_name": name,
                "group": "29feat_backbone",
                "formula_or_definition": name,
                "used_in_classifier": True,
                "used_in_stage2a": True,
            }
        )
    df = pd.DataFrame(rows)
    xlsx_path = Path(out_dir) / "feature_bank_dictionary.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame([{"sheet": "feat29", "implemented_count": len(FEAT29_NAMES)}]).to_excel(writer, sheet_name="summary", index=False)
        df.to_excel(writer, sheet_name="feat29", index=False)
    df.to_csv(Path(out_dir) / "feature_bank_core30_dictionary.csv", index=False)
    df.to_csv(Path(out_dir) / "feature_bank_extended48_dictionary.csv", index=False)
    df.to_csv(Path(out_dir) / "feature_bank_selected_dictionary.csv", index=False)
    return df


def plot_curves_generic(df, out_png, title, cols):
    if df.empty:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4), facecolor="#111111")
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, cols):
        ax.set_facecolor("#111111")
        for key, sub in df.groupby(df.columns[0]):
            ax.plot(sub["epoch"], sub[col], label=str(key))
        ax.set_title(col, color="w")
        ax.tick_params(colors="w")
        for spine in ax.spines.values():
            spine.set_color("w")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, labelcolor="w")
    fig.suptitle(title, color="w")
    plt.tight_layout()
    plt.savefig(out_png, dpi=CFG["dpi"], bbox_inches="tight", facecolor="#111111")
    plt.close()


def run_shap_surrogate(final_probs, feat_df, out_dir):
    x = feat_df.values.astype(np.float32)
    reg = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=CFG["random_state"],
    )
    reg.fit(x, final_probs)
    explainer = shap.TreeExplainer(reg)
    sv = explainer.shap_values(x)
    sv = sv[1] if isinstance(sv, list) else sv
    mean_abs = np.abs(sv).mean(0)
    mean_sv = sv.mean(0)
    std_sv = np.abs(sv).std(0)
    top_idx = np.argsort(mean_abs)[::-1][:20]
    shap_df = pd.DataFrame(
        {
            "feature": feat_df.columns[top_idx],
            "mean_abs_shap": mean_abs[top_idx],
            "mean_shap": mean_sv[top_idx],
            "std_shap": std_sv[top_idx],
        }
    )
    shap_df.to_csv(Path(out_dir) / "shap_top20.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#111111")
    for ax in axes:
        ax.set_facecolor("#111111")
        ax.tick_params(colors="w")
        for spine in ax.spines.values():
            spine.set_color("w")
    bar_df = shap_df.head(10).iloc[::-1]
    axes[0].barh(bar_df["feature"], bar_df["mean_abs_shap"], color="#4a90d9", edgecolor="white")
    axes[0].set_title("Top 10 29feat SHAP", color="white")
    axes[0].set_xlabel("Mean |SHAP|", color="white")

    bees_idx = np.argsort(mean_abs)[::-1][:10]
    for i, idx in enumerate(bees_idx):
        xv = sv[:, idx]
        yv = np.random.normal(i, 0.08, size=len(xv))
        axes[1].scatter(xv, yv, c=feat_df.iloc[:, idx], cmap="coolwarm", s=16, alpha=0.7)
    axes[1].set_yticks(range(len(bees_idx)))
    axes[1].set_yticklabels(feat_df.columns[bees_idx], color="white")
    axes[1].set_title("29feat SHAP Beeswarm", color="white")
    axes[1].set_xlabel("SHAP value", color="white")
    fig.suptitle("ResNet29 Surrogate SHAP", color="white", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "shap_summary_dark.png", dpi=CFG["dpi"], bbox_inches="tight", facecolor="#111111")
    plt.close()
    return shap_df


def classifier_prob_from_ckpt(ckpt_path, feats):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = ResNet29Classifier(29).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    cm = np.asarray(ckpt["channel_mean"], dtype=np.float32).reshape(1, -1, 1, 1)
    cs = np.asarray(ckpt["channel_std"], dtype=np.float32).reshape(1, -1, 1, 1)
    x = ((feats - cm) / (cs + 1e-6)).astype(np.float32)
    probs = []
    with torch.no_grad():
        for i in range(0, len(x), CFG["clf_batch_size"]):
            xb = torch.from_numpy(x[i : i + CFG["clf_batch_size"]]).to(DEVICE)
            probs.extend(torch.sigmoid(model(xb)).cpu().numpy().tolist())
    return np.asarray(probs, dtype=np.float32)


def infer_refined_maps(model, after_list, before_list, cm, cs):
    maps = []
    model.eval()
    with torch.no_grad():
        for aft, bef in zip(after_list, before_list):
            feat = extract_feature_bank_29(bef, aft)
            feat = (feat - cm[:, None, None]) / (cs[:, None, None] + 1e-6)
            xb = torch.from_numpy(feat[None].astype(np.float32)).to(DEVICE)
            hm = torch.sigmoid(model(xb))[0, 0].cpu().numpy()
            maps.append(_normalize_01(hm))
    return maps


def compute_heatmap_metrics(hm_pbs, hm_ana, records_ana, label="refined"):
    top_k = CFG["top_k_ratio"]

    def _img_score(h):
        flat = h.ravel()
        k = max(1, int(len(flat) * top_k))
        return float(np.partition(flat, -k)[-k:].mean())

    pbs_fpr = float(np.mean([(h > 0.5).mean() for h in hm_pbs])) if hm_pbs else np.nan
    seed_rets = []
    for h, rec in zip(hm_ana, records_ana):
        sup = rec["maps"]["support_frac"]
        smask = sup >= np.percentile(sup, 90)
        if smask.any():
            seed_rets.append(h[smask].mean() / (h.mean() + 1e-8))
    bg_sups = []
    for h in hm_ana:
        lo = np.percentile(h, 50)
        hi = np.percentile(h, 90)
        bg_sups.append(float(h[h >= hi].mean() / max(h[h <= lo].mean(), 1e-3)))
    scores_ana = np.array([_img_score(h) for h in hm_ana])
    scores_pbs = np.array([_img_score(h) for h in hm_pbs]) if hm_pbs else np.array([0.0])
    mu_a, mu_p = scores_ana.mean(), scores_pbs.mean()
    sig_a = scores_ana.std(ddof=min(1, len(scores_ana) - 1)) + 1e-8
    sig_p = scores_pbs.std(ddof=min(1, len(scores_pbs) - 1)) + 1e-8
    return {
        "Method": label,
        "pbs_fpr_area": pbs_fpr,
        "ana_seed_ret": float(np.mean(seed_rets)) if seed_rets else np.nan,
        "bg_suppress": float(np.mean(bg_sups)) if bg_sups else np.nan,
        "sep_effect": float((mu_a - mu_p) / np.sqrt((sig_a**2 + sig_p**2) / 2)),
        "sep_tail": float(np.percentile(scores_ana, 95) - np.percentile(scores_pbs, 95)),
        "mean_ana_score": float(mu_a),
        "mean_pbs_score": float(mu_p),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Full parity export for 29feat + single ResNet18 backbone.")
    p.add_argument("--data-dir", default=r"data")
    p.add_argument("--output-dir", default=r"outputs/resnet")
    p.add_argument("--base-output-dir", default=r"outputs/hybrid")
    p.add_argument("--zip-output", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = resolve_data_root(args.data_dir, out_dir)
    size = (CFG["img_size"], CFG["img_size"])

    led_map = build_led_map(data_root / CFG["led_subdir"], size, CFG["led_bright_thr"], CFG["led_smooth_sigma"])
    if led_map is not None:
        plot_led_correction(led_map, CFG, out_dir / "led_correction_map.png")
        export_led_bundle(data_root / CFG["led_subdir"], led_map, size, CFG["led_bright_thr"], CFG["led_smooth_sigma"], out_dir, CFG)

    pbs_after, pbs_before, pbs_names = load_pairs(data_root / CFG["pbs_subdir"], data_root / CFG["substrate_subdir"], "pbs", size, led_map=led_map)
    ana_after, ana_before, ana_names = load_pairs(data_root / CFG["analyte_subdir"], data_root / CFG["substrate_subdir"], "analyte", size, led_map=led_map)
    export_led_effect_examples(pbs_before, pbs_after, led_map, out_dir, CFG, prefix="pbs")
    export_led_effect_examples(ana_before, ana_after, led_map, out_dir, CFG, prefix="analyte")

    feature_dict_df = export_feature_bank_dictionary_29(out_dir)
    mask = compute_mask(pbs_before + ana_before, thr=CFG["mask_thr"], erode=CFG["mask_erode"])
    export_mask_bundle(mask, out_dir, CFG)

    pbs_mu29, pbs_std29 = build_pbs_baseline_29(pbs_after, pbs_before, mask)
    export_baseline_bundle(pbs_mu29, pbs_std29, mask, FEAT29_NAMES, out_dir, CFG)

    pbs_records, ana_records = [], []
    for aft, bef, nm in zip(pbs_after, pbs_before, pbs_names):
        maps = build_score_maps_29(bef, aft, pbs_mu29, pbs_std29, mask, CFG["support_z_thr"])
        pbs_records.append({"name": nm, "label": 0, "before_corr": bef, "after_corr": aft, "log_ratio": np.log1p(aft) - np.log1p(bef), "maps": maps})
    for aft, bef, nm in zip(ana_after, ana_before, ana_names):
        maps = build_score_maps_29(bef, aft, pbs_mu29, pbs_std29, mask, CFG["support_z_thr"])
        ana_records.append({"name": nm, "label": 1, "before_corr": bef, "after_corr": aft, "log_ratio": np.log1p(aft) - np.log1p(bef), "maps": maps})
    all_records = pbs_records + ana_records

    save_group_map_galleries(pbs_records, out_dir, "PBS")
    save_group_map_galleries(ana_records, out_dir, "Analyte")
    pbs_summary_df = export_group_zscore_summary(pbs_records, out_dir, mask, "PBS")
    ana_summary_df = export_group_zscore_summary(ana_records, out_dir, mask, "Analyte")

    feat29_rows = []
    full_feats = []
    all_names = []
    all_labels = []
    for rec in all_records:
        feat = extract_feature_bank_29(rec["before_corr"], rec["after_corr"]).astype(np.float32)
        full_feats.append(feat)
        all_names.append(rec["name"])
        all_labels.append(rec["label"])
        feat29_rows.append({"name": rec["name"], "label": "Analyte" if rec["label"] == 1 else "PBS", **summarize_image_features_29(feat, mask)})
    full_feats = np.stack(full_feats, axis=0).astype(np.float32)
    all_labels = np.asarray(all_labels, dtype=np.int32)
    hand_df = pd.DataFrame(feat29_rows)
    save_feature_value_workbook(out_dir, pbs_summary_df, ana_summary_df, hand_df)

    folds = make_balanced_folds(all_names, all_labels, min(CFG["n_splits"], max(2, len(all_labels))), CFG["random_state"])
    clf_args = argparse.Namespace(
        clf_epochs=CFG["clf_epochs"],
        clf_batch_size=CFG["clf_batch_size"],
        clf_lr=CFG["clf_lr"],
        clf_weight_decay=CFG["clf_weight_decay"],
        clf_patience=CFG["clf_patience"],
        clf_freeze_warmup_epochs=CFG["clf_freeze_warmup_epochs"],
        init_classifier_ckpt=str(Path(args.base_output_dir) / "stage2a_refine.pt"),
        stage2a_batch_size=CFG["stage2a_batch_size"],
        stage2a_epochs=CFG["stage2a_epochs"],
        stage2a_patience=CFG["stage2a_patience"],
        freeze_warmup_epochs=CFG["freeze_warmup_epochs"],
        tv_weight=CFG["tv_weight"],
    )
    cv_df = train_classifier_cv(full_feats, all_labels, all_names, folds, out_dir, clf_args)
    plot_curves_generic(pd.read_csv(out_dir / "classifier_training_history.csv"), out_dir / "classifier_training_curves.png", "ResNet29 CV Training", ["val_accuracy", "val_balanced_accuracy", "val_f1", "val_roc_auc"])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=CFG["random_state"])
    train_idx, val_idx = next(sss.split(np.arange(len(all_labels)), all_labels))
    cls_ckpt = train_full_classifier_for_init(full_feats, all_labels, all_names, train_idx, val_idx, out_dir, clf_args)

    final_classifier_dir = out_dir / "final_classifier"
    final_classifier_dir.mkdir(exist_ok=True)
    shutil.copy2(cls_ckpt, final_classifier_dir / "best_classifier.pt")
    cv_df.to_csv(final_classifier_dir / "resnet29_cv_results.csv", index=False)
    meta = {
        "model": "29feat+ResNet18",
        "n_samples": int(len(all_labels)),
        "n_pbs": int((all_labels == 0).sum()),
        "n_analyte": int((all_labels == 1).sum()),
        "mean_accuracy": float(cv_df["accuracy"].mean()),
        "mean_balanced_accuracy": float(cv_df["balanced_accuracy"].mean()),
        "mean_f1": float(cv_df["f1"].mean()),
        "mean_roc_auc": float(cv_df["roc_auc"].mean()),
    }
    (final_classifier_dir / "best_classifier_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    train_after = [pbs_after[i] if i < len(pbs_after) else ana_after[i - len(pbs_after)] for i in train_idx]
    train_before = [pbs_before[i] if i < len(pbs_before) else ana_before[i - len(pbs_before)] for i in train_idx]
    train_labels = all_labels[train_idx]
    val_after = [pbs_after[i] if i < len(pbs_after) else ana_after[i - len(pbs_after)] for i in val_idx]
    val_before = [pbs_before[i] if i < len(pbs_before) else ana_before[i - len(pbs_before)] for i in val_idx]
    val_labels = all_labels[val_idx]

    mask_ref = compute_mask(train_before, thr=CFG["mask_thr"], erode=CFG["mask_erode"])
    train_pbs_after = [img for img, lbl in zip(train_after, train_labels) if lbl == 0]
    train_pbs_before = [img for img, lbl in zip(train_before, train_labels) if lbl == 0]
    pbs_mu29_ref, pbs_std29_ref = build_pbs_baseline_29(train_pbs_after, train_pbs_before, mask_ref)
    cm, cs = _compute_channel_stats(train_after, train_before)

    stage_model = ResNet18_29(29)
    stage_model, hist_df = train_stage2a_model(
        "stage2a_resnet",
        stage_model,
        train_after,
        train_before,
        train_labels,
        val_after,
        val_before,
        val_labels,
        mask_ref,
        pbs_mu29_ref,
        pbs_std29_ref,
        cm,
        cs,
        out_dir,
        clf_args,
        init_classifier_ckpt=cls_ckpt,
        lr=CFG["resnet_lr"],
        weight_decay=CFG["resnet_wd"],
        use_aux=False,
        aux_cls_w=0.0,
    )
    hist_df.to_csv(out_dir / "stage2a_training_history.csv", index=False)
    plot_curves_generic(
        hist_df[["model", "epoch", "train_loss", "sep", "ana_topk", "pbs_topk"]],
        out_dir / "stage2a_training_curves.png",
        "Stage2a ResNet Training",
        ["train_loss", "sep", "ana_topk", "pbs_topk"],
    )
    torch.save(
        {
            "model_state": stage_model.state_dict(),
            "channel_mean": cm.tolist(),
            "channel_std": cs.tolist(),
            "pbs_mu29": pbs_mu29_ref,
            "pbs_std29": pbs_std29_ref,
            "classifier_init_ckpt": str(cls_ckpt),
        },
        out_dir / "stage2a_refine.pt",
    )

    final_probs = classifier_prob_from_ckpt(cls_ckpt, full_feats)
    feat29_mean_df = pd.DataFrame([{"name": n, "label": l, **{FEAT29_NAMES[i]: float((f[i][mask]).mean()) for i in range(f.shape[0])}} for n, l, f in zip(all_names, all_labels, full_feats)])
    shap_df = run_shap_surrogate(final_probs, feat29_mean_df[FEAT29_NAMES], out_dir)

    refined_maps = infer_refined_maps(stage_model, [r["after_corr"] for r in all_records], [r["before_corr"] for r in all_records], cm, cs)
    coarse_maps = [r["maps"]["top3_weighted"] for r in all_records]
    pbs_coarse_vals = np.concatenate([m[mask] for m, r in zip(coarse_maps, all_records) if r["label"] == 0])
    pbs_refined_vals = np.concatenate([m[mask] for m, r in zip(refined_maps, all_records) if r["label"] == 0])
    coarse_thr = float(np.percentile(pbs_coarse_vals, CFG["response_percentile_coarse"]))
    refined_thr = float(np.percentile(pbs_refined_vals, CFG["response_percentile_refined"]))

    final_dir = out_dir / "final_visualizations"
    final_dir.mkdir(exist_ok=True)
    rows = []
    ana_stage_items, pbs_stage_items, ana_resp_items, pbs_resp_items = [], [], [], []
    for rec, prob, coarse, refined in zip(all_records, final_probs, coarse_maps, refined_maps):
        coarse_mask = (coarse > coarse_thr) & mask
        refined_mask = (refined > refined_thr) & mask
        stage_tile = build_compact_stage_tile(
            rec["before_corr"],
            rec["after_corr"],
            _normalize_01(rec["after_corr"] - rec["before_corr"]),
            _normalize_01(rec["log_ratio"]),
            coarse,
            refined,
        )
        resp_tile = build_response_only_tile(rec["after_corr"], coarse, refined, coarse_mask, refined_mask)
        row = {
            "name": rec["name"],
            "label": "Analyte" if rec["label"] == 1 else "PBS",
            "classifier": "29feat+ResNet18",
            "prob_analyte": float(prob),
            "predicted_label": "Analyte" if prob >= 0.5 else "PBS",
            "coarse_max": float(coarse.max()),
            "refined_max": float(refined.max()),
            "coarse_area_pct": float(100.0 * coarse_mask.sum() / max(mask.sum(), 1)),
            "refined_area_pct": float(100.0 * refined_mask.sum() / max(mask.sum(), 1)),
            "area_delta_pct": float(100.0 * (refined_mask.sum() - coarse_mask.sum()) / max(mask.sum(), 1)),
        }
        rows.append(row)
        item_s = {"img": stage_tile, "title": f"{rec['name']} | p={prob:.3f}"}
        item_r = {"img": resp_tile, "title": f"{rec['name']} | coarse->refined"}
        if rec["label"] == 1:
            ana_stage_items.append(item_s)
            ana_resp_items.append(item_r)
        else:
            pbs_stage_items.append(item_s)
            pbs_resp_items.append(item_r)

    result_df = pd.DataFrame(rows).sort_values(["label", "name"]).reset_index(drop=True)
    result_df.to_csv(final_dir / "final_prediction_summary.csv", index=False)
    if ana_stage_items:
        save_image_grid(ana_stage_items, final_dir / "analyte_final_stage_gallery.png", ncols=CFG["final_gallery_cols"], title="Analyte - Final stage")
        save_image_grid(ana_resp_items, final_dir / "analyte_response_compare_gallery.png", ncols=CFG["final_gallery_cols"], title="Analyte - coarse vs refined")
    if pbs_stage_items:
        save_image_grid(pbs_stage_items, final_dir / "pbs_final_stage_gallery.png", ncols=CFG["final_gallery_cols"], title="PBS - Final stage")
        save_image_grid(pbs_resp_items, final_dir / "pbs_response_compare_gallery.png", ncols=CFG["final_gallery_cols"], title="PBS - coarse vs refined")
    plot_response_area_comparison(result_df, final_dir / "response_area_comparison.png")

    hm_pbs_refined = [m for m, r in zip(refined_maps, all_records) if r["label"] == 0]
    hm_ana_refined = [m for m, r in zip(refined_maps, all_records) if r["label"] == 1]
    hm_pbs_coarse = [m for m, r in zip(coarse_maps, all_records) if r["label"] == 0]
    hm_ana_coarse = [m for m, r in zip(coarse_maps, all_records) if r["label"] == 1]
    metrics_df = pd.DataFrame(
        [
            compute_heatmap_metrics(hm_pbs_coarse, hm_ana_coarse, ana_records, "coarse"),
            compute_heatmap_metrics(hm_pbs_refined, hm_ana_refined, ana_records, "refined"),
        ]
    )
    metrics_df.to_csv(final_dir / "heatmap_metrics.csv", index=False)

    with pd.ExcelWriter(final_dir / "results_master.xlsx", engine="openpyxl") as writer:
        pd.DataFrame([CFG]).to_excel(writer, sheet_name="run_config", index=False)
        cv_df.to_excel(writer, sheet_name="clf_cv_results", index=False)
        result_df.to_excel(writer, sheet_name="final_prediction_summary", index=False)
        metrics_df.to_excel(writer, sheet_name="heatmap_metrics", index=False)
        shap_df.to_excel(writer, sheet_name="shap_top20", index=False)

    if args.zip_output:
        zip_path = shutil.make_archive(str(out_dir), "zip", str(out_dir))
        print(f"OK Output archive created: {zip_path}")


if __name__ == "__main__":
    main()
