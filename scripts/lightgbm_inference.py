import argparse
import json
import os
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from hybrid_pipeline import (
    CFG,
    _normalize_01,
    build_compact_stage_tile,
    build_led_map,
    build_response_only_tile,
    build_score_maps,
    compute_mask,
    load_pairs,
    plot_response_area_comparison,
    save_image_grid,
    summarize_image_features,
)
from gas_transfer import ResNet18_29, extract_feature_bank_29
from resnet_inference import extract_unknown_zipset
from transfer_learning import DEVICE


HMAP_THR = 0.5
TOP_K = 0.05


def _img_score(h):
    flat = h.flatten()
    k = max(1, int(len(flat) * TOP_K))
    return float(np.partition(flat, -k)[-k:].mean())


def compute_heatmap_metrics(hm_pbs, hm_ana, records_ana, label="refined"):
    pbs_fpr = float(np.mean([(h > HMAP_THR).mean() for h in hm_pbs])) if hm_pbs else np.nan

    seed_rets = []
    for h, rec in zip(hm_ana, records_ana):
        sup = rec["maps"].get("support_frac", rec["maps"].get("multi_feature_support_count_map"))
        if sup is None:
            continue
        mn, mx = sup.min(), sup.max()
        sup_n = (sup - mn) / (mx - mn + 1e-8)
        smask = sup_n >= np.percentile(sup_n, 90)
        if not smask.any():
            continue
        seed_rets.append(h[smask].mean() / (h.mean() + 1e-8))
    ana_seed = float(np.mean(seed_rets)) if seed_rets else np.nan

    bg_sups = []
    for h in hm_ana:
        lo = np.percentile(h, 50)
        hi = np.percentile(h, 90)
        bot = h[h <= lo].mean() if (h <= lo).any() else 0.0
        top = h[h >= hi].mean() if (h >= hi).any() else 0.0
        bg_sups.append(float(top / max(bot, 1e-3)))
    bg_sup = float(np.mean(bg_sups)) if bg_sups else np.nan

    scores_ana = np.array([_img_score(h) for h in hm_ana])
    scores_pbs = np.array([_img_score(h) for h in hm_pbs]) if hm_pbs else np.array([0.0])
    mu_a, mu_p = scores_ana.mean(), scores_pbs.mean()
    sig_a = scores_ana.std(ddof=min(1, len(scores_ana) - 1)) + 1e-8
    sig_p = scores_pbs.std(ddof=min(1, len(scores_pbs) - 1)) + 1e-8
    sep_effect = float((mu_a - mu_p) / np.sqrt((sig_a**2 + sig_p**2) / 2))

    q95_a = float(np.percentile(scores_ana, 95))
    q95_p = float(np.percentile(scores_pbs, 95))
    sep_tail = float(q95_a - q95_p)

    return {
        "Method": label,
        "pbs_fpr_area": pbs_fpr,
        "ana_seed_ret": ana_seed,
        "bg_suppress": bg_sup,
        "sep_effect": sep_effect,
        "sep_tail": sep_tail,
        "mean_ana_score": float(mu_a),
        "mean_pbs_score": float(mu_p),
    }


def refine_one(rec, refine_model, cm, cs):
    feat = extract_feature_bank_29(rec["before_corr"], rec["after_corr"])
    feat = (feat - cm[:, None, None]) / (cs[:, None, None] + 1e-6)
    x = torch.from_numpy(feat[None].astype(np.float32)).to(DEVICE)
    with torch.no_grad():
        hm = torch.sigmoid(refine_model(x))[0, 0].cpu().numpy()
    mn, mx = hm.min(), hm.max()
    if mx - mn < 1e-8:
        return np.zeros_like(hm, np.float32)
    return ((hm - mn) / (mx - mn)).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(description="Inference on unknown zipped data using handcrafted+LightGBM + stage2a refinement.")
    p.add_argument("--zip-dir", required=True)
    p.add_argument("--work-data-dir", required=True)
    p.add_argument("--model-dir", default=r"outputs/hybrid")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--zip-output", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = extract_unknown_zipset(args.zip_dir, args.work_data_dir)

    led_map = build_led_map(
        Path(data_dir) / "led_field",
        (CFG["img_size"], CFG["img_size"]),
        CFG["led_bright_thr"],
        CFG["led_smooth_sigma"],
    )
    pbs_after, pbs_before, pbs_names = load_pairs(Path(data_dir) / "pbs", Path(data_dir) / "susbtrat", "pbs", (CFG["img_size"], CFG["img_size"]), led_map=led_map)
    ana_after, ana_before, ana_names = load_pairs(Path(data_dir) / "analyte", Path(data_dir) / "susbtrat", "analyte", (CFG["img_size"], CFG["img_size"]), led_map=led_map)

    all_after = pbs_after + ana_after
    all_before = pbs_before + ana_before
    all_names = pbs_names + ana_names
    all_labels = [0] * len(pbs_after) + [1] * len(ana_after)
    mask = compute_mask(all_before, thr=CFG["mask_thr"], erode=CFG["mask_erode"])

    model_dir = Path(args.model_dir)
    clf = joblib.load(model_dir / "final_classifier" / "best_classifier.joblib")
    meta = json.loads((model_dir / "final_classifier" / "best_classifier_meta.json").read_text(encoding="utf-8"))
    feature_cols = meta["feature_cols"]

    ckpt = torch.load(model_dir / "stage2a_refine.pt", map_location=DEVICE, weights_only=False)
    refine_model = ResNet18_29(29).to(DEVICE)
    refine_model.load_state_dict(ckpt["model_state"])
    refine_model.eval()
    cm = np.asarray(ckpt["channel_mean"], dtype=np.float32)
    cs = np.asarray(ckpt["channel_std"], dtype=np.float32)
    pbs_mu_px = ckpt["pbs_mu_px"]
    pbs_std_px = ckpt["pbs_std_px"]

    records = []
    for name, label, before, after in zip(all_names, all_labels, all_before, all_after):
        maps = build_score_maps(before, after, pbs_mu_px, pbs_std_px, mask, CFG)
        rec = {
            "name": name,
            "label": label,
            "before_corr": before,
            "after_corr": after,
            "maps": maps,
        }
        row = summarize_image_features(maps, mask)
        x = pd.DataFrame([row])[feature_cols].values.astype(np.float32)
        prob = float(clf.predict_proba(x)[:, 1][0])
        refined = refine_one(rec, refine_model, cm, cs)
        rec["prob"] = prob
        rec["coarse"] = maps[CFG["score_primary"]]
        rec["refined"] = refined
        records.append(rec)

    pbs_coarse_vals = np.concatenate([r["coarse"][mask] for r in records if r["label"] == 0])
    pbs_refined_vals = np.concatenate([r["refined"][mask] for r in records if r["label"] == 0])
    coarse_thr = float(np.percentile(pbs_coarse_vals, CFG["response_percentile_coarse"]))
    refined_thr = float(np.percentile(pbs_refined_vals, CFG["response_percentile_refined"]))

    final_dir = out_dir / "final_visualizations"
    final_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    ana_stage_items, pbs_stage_items, ana_resp_items, pbs_resp_items = [], [], [], []
    for rec in records:
        coarse_mask = (rec["coarse"] > coarse_thr) & mask
        refined_mask = (rec["refined"] > refined_thr) & mask
        stage_tile = build_compact_stage_tile(
            rec["before_corr"],
            rec["after_corr"],
            _normalize_01(rec["after_corr"] - rec["before_corr"]),
            _normalize_01(np.log1p(rec["after_corr"]) - np.log1p(rec["before_corr"])),
            rec["coarse"],
            rec["refined"],
        )
        resp_tile = build_response_only_tile(
            rec["after_corr"], rec["coarse"], rec["refined"], coarse_mask, refined_mask
        )
        row = {
            "name": rec["name"],
            "true_label": "Analyte" if rec["label"] == 1 else "PBS",
            "predicted_label": "Analyte" if rec["prob"] >= 0.5 else "PBS",
            "prob_analyte": rec["prob"],
            "confidence": max(rec["prob"], 1.0 - rec["prob"]),
            "correct_vs_folder_label": bool((rec["prob"] >= 0.5) == bool(rec["label"])),
            "coarse_area_pct": float(100.0 * coarse_mask.sum() / max(mask.sum(), 1)),
            "refined_area_pct": float(100.0 * refined_mask.sum() / max(mask.sum(), 1)),
        }
        rows.append(row)
        item_s = {"img": stage_tile, "title": f"{rec['name']} | p={rec['prob']:.3f}"}
        item_r = {"img": resp_tile, "title": f"{rec['name']} | coarse->refined"}
        if rec["label"] == 1:
            ana_stage_items.append(item_s)
            ana_resp_items.append(item_r)
        else:
            pbs_stage_items.append(item_s)
            pbs_resp_items.append(item_r)

    result_df = pd.DataFrame(rows).sort_values(["true_label", "name"]).reset_index(drop=True)
    result_df["area_delta_pct"] = result_df["refined_area_pct"] - result_df["coarse_area_pct"]
    result_df.to_csv(final_dir / "final_prediction_summary.csv", index=False)

    if ana_stage_items:
        save_image_grid(ana_stage_items, final_dir / "analyte_final_stage_gallery.png", ncols=CFG["final_gallery_cols"], title="Unknown analyte - final stage (LightGBM)")
        save_image_grid(ana_resp_items, final_dir / "analyte_response_compare_gallery.png", ncols=CFG["final_gallery_cols"], title="Unknown analyte - coarse vs refined (LightGBM)")
    if pbs_stage_items:
        save_image_grid(pbs_stage_items, final_dir / "pbs_final_stage_gallery.png", ncols=CFG["final_gallery_cols"], title="Unknown pbs - final stage (LightGBM)")
        save_image_grid(pbs_resp_items, final_dir / "pbs_response_compare_gallery.png", ncols=CFG["final_gallery_cols"], title="Unknown pbs - coarse vs refined (LightGBM)")

    plot_response_area_comparison(
        result_df.rename(columns={"true_label": "label"}),
        final_dir / "response_area_comparison.png",
    )

    hm_df = pd.DataFrame(
        [
            compute_heatmap_metrics(
                [r["coarse"] for r in records if r["label"] == 0],
                [r["coarse"] for r in records if r["label"] == 1],
                [r for r in records if r["label"] == 1],
                "coarse",
            ),
            compute_heatmap_metrics(
                [r["refined"] for r in records if r["label"] == 0],
                [r["refined"] for r in records if r["label"] == 1],
                [r for r in records if r["label"] == 1],
                "refined",
            ),
        ]
    )
    hm_df.to_csv(final_dir / "heatmap_metrics.csv", index=False)

    summary = {
        "n_total": int(len(result_df)),
        "n_pbs": int((result_df["true_label"] == "PBS").sum()),
        "n_analyte": int((result_df["true_label"] == "Analyte").sum()),
        "acc_vs_folder_label": float(result_df["correct_vs_folder_label"].mean()) if len(result_df) else float("nan"),
        "mean_prob_pbs": float(result_df.loc[result_df["true_label"] == "PBS", "prob_analyte"].mean()),
        "mean_prob_analyte": float(result_df.loc[result_df["true_label"] == "Analyte", "prob_analyte"].mean()),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "inference_summary.csv", index=False)

    with pd.ExcelWriter(final_dir / "results_master.xlsx", engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="final_prediction_summary", index=False)
        hm_df.to_excel(writer, sheet_name="heatmap_metrics", index=False)
        pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)

    if args.zip_output:
        zip_path = shutil.make_archive(str(out_dir), "zip", str(out_dir))
        print(f"OK Output archive created: {zip_path}")


if __name__ == "__main__":
    main()
