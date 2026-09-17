import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hybrid_pipeline import (
    _normalize_01,
    build_compact_stage_tile,
    build_response_only_tile,
    plot_response_area_comparison,
    save_image_grid,
)
from gas_transfer import extract_feature_bank_29
from resnet_pipeline import (
    CFG,
    FEAT29_NAMES,
    ResNet18_29,
    ResNet29Classifier,
    build_score_maps_29,
    compute_heatmap_metrics,
)
from transfer_learning import DEVICE, build_led_map, compute_mask, load_pairs


def extract_unknown_zipset(zip_dir, target_dir):
    zip_dir = Path(zip_dir)
    target_dir = Path(target_dir)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    mapping = [
        (["led.zip"], "led_field"),
        (["pbs.zip"], "pbs"),
        (["analyte.zip"], "analyte"),
        (["susbtrat.zip", "susbrat.zip"], "susbtrat"),
    ]
    for zip_names, folder_name in mapping:
        src = None
        for zip_name in zip_names:
            cand = zip_dir / zip_name
            if cand.exists():
                src = cand
                break
        if src is None:
            raise FileNotFoundError(f"Missing zip for {folder_name}: tried {zip_names}")
        out = target_dir / folder_name
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(out)
    return target_dir


def load_classifier_probs(ckpt_path, feats):
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


def load_refine_outputs(ckpt_path, after_list, before_list):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = ResNet18_29(29).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    cm = np.asarray(ckpt["channel_mean"], dtype=np.float32)
    cs = np.asarray(ckpt["channel_std"], dtype=np.float32)
    pbs_mu29 = ckpt["pbs_mu29"]
    pbs_std29 = ckpt["pbs_std29"]
    refined = []
    coarse = []
    feat_stack = []
    with torch.no_grad():
        for aft, bef in zip(after_list, before_list):
            feat = extract_feature_bank_29(bef, aft).astype(np.float32)
            feat_stack.append(feat)
            coarse.append(build_score_maps_29(bef, aft, pbs_mu29, pbs_std29, np.ones_like(bef, dtype=bool), CFG["support_z_thr"])["top3_weighted"])
            x = (feat - cm[:, None, None]) / (cs[:, None, None] + 1e-6)
            hm = torch.sigmoid(model(torch.from_numpy(x[None]).to(DEVICE)))[0, 0].cpu().numpy()
            refined.append(_normalize_01(hm))
    return np.stack(feat_stack, axis=0), coarse, refined


def parse_args():
    p = argparse.ArgumentParser(description="Inference on unknown zipped data using the full parity ResNet29 model.")
    p.add_argument("--zip-dir", default=r"data/unknown_archives")
    p.add_argument("--work-data-dir", default=r"data/unknown_extracted")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--output-dir", default=r"outputs/unknown_resnet")
    p.add_argument("--zip-output", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = extract_unknown_zipset(args.zip_dir, args.work_data_dir)

    led_map = build_led_map(data_dir / "led_field", (CFG["img_size"], CFG["img_size"]), CFG["led_bright_thr"], CFG["led_smooth_sigma"])
    pbs_after, pbs_before, pbs_names = load_pairs(data_dir / "pbs", data_dir / "susbtrat", "pbs", (CFG["img_size"], CFG["img_size"]), led_map=led_map)
    ana_after, ana_before, ana_names = load_pairs(data_dir / "analyte", data_dir / "susbtrat", "analyte", (CFG["img_size"], CFG["img_size"]), led_map=led_map)

    all_after = pbs_after + ana_after
    all_before = pbs_before + ana_before
    all_names = pbs_names + ana_names
    all_labels = np.array([0] * len(pbs_after) + [1] * len(ana_after), dtype=np.int32)
    mask = compute_mask(all_before, thr=CFG["mask_thr"], erode=CFG["mask_erode"])

    model_dir = Path(args.model_dir)
    clf_ckpt = model_dir / "binary_classifier.pt"
    refine_ckpt = model_dir / "response_localizer.pt"
    if not clf_ckpt.is_file():
        clf_ckpt = model_dir / "final_classifier" / "best_classifier.pt"
    if not refine_ckpt.is_file():
        refine_ckpt = model_dir / "stage2a_refine.pt"
    feats, coarse_maps, refined_maps = load_refine_outputs(refine_ckpt, all_after, all_before)
    probs = load_classifier_probs(clf_ckpt, feats)

    pbs_coarse_vals = np.concatenate([m[mask] for m, lbl in zip(coarse_maps, all_labels) if lbl == 0])
    pbs_refined_vals = np.concatenate([m[mask] for m, lbl in zip(refined_maps, all_labels) if lbl == 0])
    coarse_thr = float(np.percentile(pbs_coarse_vals, CFG["response_percentile_coarse"]))
    refined_thr = float(np.percentile(pbs_refined_vals, CFG["response_percentile_refined"]))

    final_dir = out_dir / "final_visualizations"
    final_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    ana_stage_items, pbs_stage_items, ana_resp_items, pbs_resp_items = [], [], [], []
    records = []
    for nm, lbl, bef, aft, prob, coarse, refined in zip(all_names, all_labels, all_before, all_after, probs, coarse_maps, refined_maps):
        records.append({"name": nm, "label": lbl, "before_corr": bef, "after_corr": aft, "maps": {"support_frac": np.zeros_like(bef), "top3_weighted": coarse}})
        coarse_mask = (coarse > coarse_thr) & mask
        refined_mask = (refined > refined_thr) & mask
        stage_tile = build_compact_stage_tile(bef, aft, _normalize_01(aft - bef), _normalize_01(np.log1p(aft) - np.log1p(bef)), coarse, refined)
        resp_tile = build_response_only_tile(aft, coarse, refined, coarse_mask, refined_mask)
        rows.append(
            {
                "name": nm,
                "true_label": "Analyte" if lbl == 1 else "PBS",
                "predicted_label": "Analyte" if prob >= 0.5 else "PBS",
                "prob_analyte": float(prob),
                "confidence": float(max(prob, 1.0 - prob)),
                "correct_vs_folder_label": bool((prob >= 0.5) == bool(lbl)),
                "coarse_area_pct": float(100.0 * coarse_mask.sum() / max(mask.sum(), 1)),
                "refined_area_pct": float(100.0 * refined_mask.sum() / max(mask.sum(), 1)),
            }
        )
        item_s = {"img": stage_tile, "title": f"{nm} | p={prob:.3f}"}
        item_r = {"img": resp_tile, "title": f"{nm} | coarse->refined"}
        if lbl == 1:
            ana_stage_items.append(item_s)
            ana_resp_items.append(item_r)
        else:
            pbs_stage_items.append(item_s)
            pbs_resp_items.append(item_r)

    result_df = pd.DataFrame(rows).sort_values(["true_label", "name"]).reset_index(drop=True)
    result_df.to_csv(final_dir / "final_prediction_summary.csv", index=False)
    if ana_stage_items:
        save_image_grid(ana_stage_items, final_dir / "analyte_final_stage_gallery.png", ncols=CFG["final_gallery_cols"], title="Unknown analyte - final stage")
        save_image_grid(ana_resp_items, final_dir / "analyte_response_compare_gallery.png", ncols=CFG["final_gallery_cols"], title="Unknown analyte - coarse vs refined")
    if pbs_stage_items:
        save_image_grid(pbs_stage_items, final_dir / "pbs_final_stage_gallery.png", ncols=CFG["final_gallery_cols"], title="Unknown pbs - final stage")
        save_image_grid(pbs_resp_items, final_dir / "pbs_response_compare_gallery.png", ncols=CFG["final_gallery_cols"], title="Unknown pbs - coarse vs refined")
    plot_response_area_comparison(
        result_df.rename(columns={"true_label": "label", "prob_analyte": "prob_analyte", "coarse_area_pct": "coarse_area_pct", "refined_area_pct": "refined_area_pct"}).assign(area_delta_pct=lambda x: x["refined_area_pct"] - x["coarse_area_pct"]),
        final_dir / "response_area_comparison.png",
    )

    hm_df = pd.DataFrame(
        [
            compute_heatmap_metrics([m for m, y in zip(coarse_maps, all_labels) if y == 0], [m for m, y in zip(coarse_maps, all_labels) if y == 1], [r for r in records if r["label"] == 1], "coarse"),
            compute_heatmap_metrics([m for m, y in zip(refined_maps, all_labels) if y == 0], [m for m, y in zip(refined_maps, all_labels) if y == 1], [r for r in records if r["label"] == 1], "refined"),
        ]
    )
    hm_df.to_csv(final_dir / "heatmap_metrics.csv", index=False)

    summary = {
        "n_total": int(len(result_df)),
        "n_pbs": int((all_labels == 0).sum()),
        "n_analyte": int((all_labels == 1).sum()),
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
