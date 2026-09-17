"""
PyCharm/local runnable version converted from biosensor_hybrid_v6.ipynb.

Recommended folder structure:
    data/
      led_field/
      pbs/
      analyte/
      susbtrat/

You can either:
1) prepare the folders directly under data_dir, or
2) pass one big zip / multiple zips with --zip.

Example:
    python hybrid_pipeline.py --data-dir data --output-dir results_hybrid_v3
    python hybrid_pipeline.py --data-dir data --output-dir results_hybrid_v3 --zip dataset.zip

Optional flags:
    --run-pca-tsne
    --run-benchmark
    --run-shap
    --zip-output
"""

import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

CFG = {
    
    "data_dir"          : "data",
    "output_dir"        : "results_hybrid_v3",
    "led_subdir"        : "led_field",
    "pbs_subdir"        : "pbs",
    "analyte_subdir"    : "analyte",
    "substrate_subdir"  : "susbtrat",
    "pbs_prefix"        : "ap",
    "analyte_prefix"    : "ab",

    
    "img_size"          : 128,
    "mask_thr"          : 20,
    "mask_erode"        : 8,
    "led_bright_thr"    : 0.35,
    "led_smooth_sigma"  : 15,

    
    "feature_bank_mode" : "extended48",
    "exclude_feature_keys": ["diff"],
    "score_primary"     : "top3_weighted",
    "support_z_thr"     : 2.5,
    "proposal_percentile": 99.0,

    
    "n_splits"          : 3,
    "random_state"      : 42,

    
    "refine_in_channels": 29,      
    "refine_epochs"     : 50,
    "refine_lr"         : 1.5e-4,
    "refine_wd"         : 1e-5,
    "refine_batch_size" : 8,
    "refine_patience"   : 12,
    "tv_weight"         : 0.0002,  
    "soft_support_w"    : 0.6,
    "soft_zconf_w"      : 0.4,

    
    "run_stage2b"       : False,

    
    "cam_sigma"         : 3.0,
    "cam_thr_scale"     : 1.0,
    "dpi"               : 160,
    "save_individual_maps": True,
    "response_percentile_coarse": 99.5,
    "response_percentile_refined": 98.0,  
    "final_gallery_cols" : 3,
    "overlay_thr_pct"   : 85,      
    "save_raw_npz"       : False,   
}
if __name__ == "__main__":
    print(CFG)

import os, re, math, json, shutil, warnings, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, uniform_filter, sobel, laplace
from scipy.stats import pearsonr
from scipy.fft import fft2, fftshift
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef, cohen_kappa_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

THERMAL = matplotlib.colors.LinearSegmentedColormap.from_list(
    "thermal_custom",
    ["#0b0f1a", "#173b7a", "#19b7d8", "#5ee05b", "#f5ef63", "#ff9f1a", "#ff2a1a"]
)




def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(CFG["random_state"])

def _img_read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32)

def _resize(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

def _normalize_01(img):
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)

def _safe_std(x, eps=1e-6):
    s = float(np.std(x))
    return s if s > eps else eps

def _list_images(folder):
    if not os.path.exists(folder):
        return []
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if os.path.splitext(f.lower())[1] in exts]
    return sorted(files)

def _extract_numeric_id(name):
    nums = re.findall(r'\d+', os.path.basename(name))
    return int(nums[0]) if nums else None

def _flat_mean(imgs):
    if len(imgs) == 0:
        return None
    return np.mean(np.stack(imgs, 0), axis=0)


# LED flat-field

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def _iter_image_files(folder):
    if not os.path.exists(folder):
        return []
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMG_EXTS)
    ])

def _read_gray(fp):
    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img.astype(np.float32)

def _resize_with_pad(img, target_size):
    'Resize with pad.'
    th, tw = target_size
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((th, tw), dtype=np.float32)

    scale = min(tw / w, th / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((th, tw), dtype=np.float32)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def _find_led_bbox(img, thr_ratio=0.25, pad=6):
    'Find led bbox.'
    if img is None or img.size == 0:
        return 0, 0, img.shape[1], img.shape[0]

    thr = float(img.max()) * thr_ratio
    mask = (img > thr).astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, img.shape[1], img.shape[0]

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img.shape[1], x2 + pad)
    y2 = min(img.shape[0], y2 + pad)

    return x1, y1, x2, y2

def build_led_map(led_dir, target_size=(128, 128), bright_thr=0.25, sigma=15):
    'Build the flat field from filtered bright frames. Crop each illuminated region, resize without zero padding, average, smooth, and normalize to unit mean.'
    files = _iter_image_files(led_dir)
    if len(files) == 0:
        print("  [LED] led_field directory not found; skipping flat-field correction")
        return None

    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    th, tw = target_size

    bright_frames = []
    for fp in files:
        img = _read_gray(fp)
        if img is None:
            continue
        if img.max() < bright_thr * 255:
            continue
        
        x1, y1, x2, y2 = _find_led_bbox(img, thr_ratio=0.25, pad=6)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            crop = img
        resized = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
        bright_frames.append(resized.astype(np.float32))

    if len(bright_frames) == 0:
        print("  [LED] No bright frames passed filtering; skipping flat-field correction")
        return None

    print(f"  [LED] Total {len(bright_frames)} bright frames; computing illumination map...")

    led_mean = np.mean(np.stack(bright_frames, axis=0), axis=0)
    led_map  = gaussian_filter(led_mean, sigma=sigma)
    led_map  = np.clip(led_map, 1e-6, None)
    led_map  = led_map / (led_map.mean() + 1e-8)

    print(f"  [LED] LED map range: [{led_map.min():.3f}, {led_map.max():.3f}]")
    print(f"  [LED] Output size: {led_map.shape[1]}x{led_map.shape[0]}")
    return led_map


def plot_led_correction(led_map, cfg, save_path):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white")

    axes[0].imshow(led_map, cmap="viridis")
    axes[0].set_title("LED Correction Map")
    axes[0].axis("off")

    axes[1].plot(led_map[led_map.shape[0] // 2])
    axes[1].set_title("Horizontal Profile")
    axes[1].grid(alpha=0.2)

    axes[2].plot(led_map[:, led_map.shape[1] // 2])
    axes[2].set_title("Vertical Profile")
    axes[2].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg["dpi"], bbox_inches="tight")
    plt.close()




def load_pairs(after_dir, before_dir, prefix=None, size=(128,128), led_map=None):
    'Pair before and after images by matching filenames. The prefix argument is retained for compatibility.'
    after_dir = Path(after_dir)
    before_dir = Path(before_dir)

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    after_files = sorted([f for f in after_dir.iterdir() if f.suffix.lower() in valid_exts])
    before_files = sorted([f for f in before_dir.iterdir() if f.suffix.lower() in valid_exts])

    before_map = {f.name: f for f in before_files}

    A, B, N = [], [], []
    missed = []

    for aft in after_files:
        nm = aft.name

        
        if nm not in before_map:
            missed.append(nm)
            continue

        after = _img_read_gray(str(aft))
        before = _img_read_gray(str(before_map[nm]))

        after = _resize(after, size[0])
        before = _resize(before, size[0])

        if led_map is not None:
            after = after / (led_map + 1e-6)
            before = before / (led_map + 1e-6)

        A.append(after.astype(np.float32))
        B.append(before.astype(np.float32))
        N.append(nm)

    print(f"  [PAIR] {after_dir.name}: Matched {len(N)} pairs / total {len(after_files)} after images")
    if len(missed) > 0:
        print(f"  [PAIR] Unmatched {len(missed)} images; first 10 examples: {missed[:10]}")

    return A, B, N

def compute_mask(before_imgs, thr=20, erode=8):
    if len(before_imgs) == 0:
        raise ValueError("before_imgs empty")
    mean_before = np.mean(np.stack(before_imgs, 0), 0)
    mask = mean_before > thr
    if erode > 0:
        kernel = np.ones((erode, erode), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask


# handcrafted feature bank

def _local_stats(x, w):
    mu = uniform_filter(x, size=w)
    mu2 = uniform_filter(x*x, size=w)
    std = np.sqrt(np.maximum(mu2 - mu*mu, 0))
    return mu, std

def _dog(x, s1, s2):
    return gaussian_filter(x, s1) - gaussian_filter(x, s2)

def _speckle_contrast(x, w):
    mu, std = _local_stats(x, w)
    return std / (mu + 1e-6)

def _high_low_freq(x):
    f = np.abs(fftshift(fft2(x)))
    h, w = x.shape
    cy, cx = h//2, w//2
    rr = min(h,w)//8
    yy, xx = np.ogrid[:h, :w]
    dist2 = (yy-cy)**2 + (xx-cx)**2
    low_mask = dist2 <= rr*rr
    high_mask = dist2 >= (2*rr)*(2*rr)
    lf = f[low_mask].mean() if low_mask.sum() else 0.
    hf = f[high_mask].mean() if high_mask.sum() else 0.
    return np.full_like(x, lf, dtype=np.float32), np.full_like(x, hf, dtype=np.float32)

def _grad_mag(x):
    gx = sobel(x, axis=1)
    gy = sobel(x, axis=0)
    return np.sqrt(gx*gx + gy*gy)

def _local_corr_drop(a, b, w):
    ma = uniform_filter(a, size=w); mb = uniform_filter(b, size=w)
    va = uniform_filter(a*a, size=w) - ma*ma
    vb = uniform_filter(b*b, size=w) - mb*mb
    cab = uniform_filter(a*b, size=w) - ma*mb
    corr = cab / (np.sqrt(np.maximum(va,0))*np.sqrt(np.maximum(vb,0)) + 1e-6)
    return 1.0 - np.clip(corr, -1, 1)

def _ssim_like_map(a, b, w):
    C1, C2 = 1e-4, 9e-4
    mu_a = uniform_filter(a, size=w); mu_b = uniform_filter(b, size=w)
    var_a = uniform_filter(a*a, size=w) - mu_a*mu_a
    var_b = uniform_filter(b*b, size=w) - mu_b*mu_b
    cov   = uniform_filter(a*b, size=w) - mu_a*mu_b
    num = (2*mu_a*mu_b + C1) * (2*cov + C2)
    den = (mu_a*mu_a + mu_b*mu_b + C1) * (var_a + var_b + C2)
    ssim = num / (den + 1e-6)
    return 1.0 - ssim

def _local_percentile_shift(a, b, w, q=0.8):
    
    ga = gaussian_filter(a, sigma=max(1, w/6))
    gb = gaussian_filter(b, sigma=max(1, w/6))
    return gb - ga

def _rank_diff(a, b, w):
    
    ma, sa = _local_stats(a, w)
    mb, sb = _local_stats(b, w)
    za = (a-ma)/(sa+1e-6)
    zb = (b-mb)/(sb+1e-6)
    return zb - za

def _anscombe(x):
    return 2.0*np.sqrt(np.maximum(x, 0) + 3/8)

def _pixel_features_core30(before, after):
    before = before.astype(np.float32)
    after  = after.astype(np.float32)
    diff   = after - before
    log_r  = np.log(after + 1.) - np.log(before + 1.)
    feats = [diff, log_r]
    for s in [1,2,4,8,12]:
        feats.append(gaussian_filter(diff, sigma=s))
        feats.append(gaussian_filter(log_r, sigma=s))
    for s1,s2 in [(1,2),(2,4),(4,8),(1,4)]:
        feats.append(_dog(diff, s1, s2))
    for w in [7,13,21]:
        mu, std = _local_stats(diff, w)
        snr = diff / (std + 1e-6)
        feats += [mu, std, snr]
    for w in [7,13]:
        k_after = _speckle_contrast(after, w)
        k_before = _speckle_contrast(before, w)
        feats.append(k_after - k_before)
    lf, hf = _high_low_freq(diff)
    feats += [lf, hf]
    return np.stack(feats, 0).astype(np.float32)  # 29, H, W

def _pixel_features_extended48(before, after):
    core = _pixel_features_core30(before, after)
    before = before.astype(np.float32)
    after  = after.astype(np.float32)
    diff   = after - before
    log_r  = np.log(after + 1.) - np.log(before + 1.)
    rel_diff = diff / (before + 1e-3)
    sym_ratio = diff / (after + before + 1e-3)
    _, stdb7 = _local_stats(before, 7)
    z_before_norm = diff / (stdb7 + 1e-6)
    anscombe_diff = _anscombe(after) - _anscombe(before)

    ext = [rel_diff, sym_ratio, z_before_norm, anscombe_diff]

    for w in [7,13]:
        ext.append(_local_corr_drop(before, after, w))
        ext.append(_ssim_like_map(before, after, w))

    for s in [1,2]:
        gdiff = gaussian_filter(_grad_mag(after), sigma=s) - gaussian_filter(_grad_mag(before), sigma=s)
        ext.append(gdiff)
    for s in [1,2]:
        log_like = gaussian_filter(laplace(after), sigma=s) - gaussian_filter(laplace(before), sigma=s)
        ext.append(log_like)

    ext.append(laplace(after) - laplace(before))
    
    ga = _grad_mag(after); gb = _grad_mag(before)
    ext.append(ga - gb)

    for w in [7,13]:
        ext.append(_local_percentile_shift(before, after, w))
        ext.append(_rank_diff(before, after, w))

    ext = np.stack(ext, 0).astype(np.float32)   # 18, H, W
    return np.concatenate([core, ext], 0)       # 47, H, W

def _pixel_features(before, after, mode="extended48"):
    if mode == "core30":
        return _pixel_features_core30(before, after)
    elif mode == "extended48":
        return _pixel_features_extended48(before, after)
    else:
        raise ValueError(f"unknown mode: {mode}")

def _feature_names_core30():
    names = ["diff", "log_ratio"]
    for s in [1,2,4,8,12]:
        names += [f"gauss_diff_s{s}", f"gauss_logr_s{s}"]
    for s1,s2 in [(1,2),(2,4),(4,8),(1,4)]:
        names += [f"dog_diff_{s1}_{s2}"]
    for w in [7,13,21]:
        names += [f"local_mu_w{w}", f"local_std_w{w}", f"local_snr_w{w}"]
    for w in [7,13]:
        names += [f"delta_speckle_w{w}"]
    names += ["fft_low", "fft_high"]
    return names

def _feature_names_extended48():
    names = _feature_names_core30()
    names += [
        "relative_diff","sym_ratio","z_before_norm","anscombe_diff",
        "corr_drop_w7","ssim_drop_w7",
        "corr_drop_w13","ssim_drop_w13",
        "grad_diff_s1","grad_diff_s2",
        "log_like_s1","log_like_s2",
        "laplace_diff","tensor_aniso_change",
        "pct_shift_w7","rank_diff_w7",
        "pct_shift_w13","rank_diff_w13",
    ]
    return names

def get_feature_names(mode="extended48"):
    return _feature_names_extended48() if mode=="extended48" else _feature_names_core30()


# PBS baseline + top-k aggregation

def build_pbs_pixel_baseline(after_list, before_list, mask, mode="extended48"):
    Fs = [_pixel_features(bef, aft, mode=mode) for aft, bef in zip(after_list, before_list)]
    stk = np.stack(Fs, 0)  # N,F,H,W
    mu = stk.mean(0)
    std_raw = stk.std(0, ddof=1)  # [F,H,W]
    
    floor = np.percentile(std_raw[:, mask], 5, axis=1)[:, None, None]  # [F,1,1]
    std = np.maximum(std_raw, floor)
    return mu.astype(np.float32), std.astype(np.float32)

def build_score_maps(before, after, pbs_mu, pbs_std, mask, cfg):
    feats = _pixel_features(before, after, mode=cfg["feature_bank_mode"])
    z = (feats - pbs_mu) / (pbs_std + 1e-6)
    z = np.where(mask[None], z, 0.0)

    z_sorted = np.sort(z, axis=0)[::-1]
    top1 = z_sorted[0]
    top3_mean = z_sorted[:3].mean(0)
    top3_weighted = 0.6*z_sorted[0] + 0.3*z_sorted[1] + 0.1*z_sorted[2]
    top5_rms = np.sqrt((z_sorted[:5]**2).mean(0))
    support_count = (z > cfg["support_z_thr"]).sum(0).astype(np.float32)
    support_frac  = support_count / z.shape[0]

    maps = {
        "top1": top1.astype(np.float32),
        "top3_mean": top3_mean.astype(np.float32),
        "top3_weighted": top3_weighted.astype(np.float32),
        "top5_rms": top5_rms.astype(np.float32),
        "support_count": support_count.astype(np.float32),
        "support_frac": support_frac.astype(np.float32),
        "z_stack": z.astype(np.float32),
        "feature_stack": feats.astype(np.float32),
    }
    maps["primary"] = maps[cfg["score_primary"]]
    return maps

def summarize_image_features(score_maps, mask):
    z = score_maps["z_stack"]
    feat = score_maps["feature_stack"]
    out = {}
    for k in ["top1","top3_mean","top3_weighted","top5_rms","support_count","support_frac"]:
        m = score_maps[k][mask]
        out[f"{k}_mean"] = float(m.mean())
        out[f"{k}_std"]  = float(m.std())
        out[f"{k}_max"]  = float(m.max())
        out[f"{k}_q95"]  = float(np.quantile(m, .95))
        out[f"{k}_sigpct"] = float((m > np.quantile(m, .95)).mean())
    
    for i in range(z.shape[0]):
        zi = z[i][mask]
        out[f"z{i:02d}_mean"] = float(zi.mean())
        out[f"z{i:02d}_std"]  = float(zi.std())
        out[f"z{i:02d}_q95"]  = float(np.quantile(zi, .95))
    return out




class PairImageDataset(Dataset):
    def __init__(self, pbs_A, pbs_B, ana_A, ana_B, img_size=128, augment=False):
        self.samples = []
        for a,b in zip(pbs_A,pbs_B):
            self.samples.append((a,b,0))
        for a,b in zip(ana_A,ana_B):
            self.samples.append((a,b,1))
        self.img_size = img_size
        self.augment = augment

    def __len__(self): return len(self.samples)

    def _aug(self, x):
        if np.random.rand() < 0.5:
            x = np.flip(x, 1).copy()
        if np.random.rand() < 0.5:
            x = np.flip(x, 2).copy()
        k = np.random.randint(0,4)
        x = np.rot90(x, k, axes=(1,2)).copy()
        x += np.random.normal(0, 0.01, x.shape).astype(np.float32)
        return x

    def __getitem__(self, idx):
        after, before, y = self.samples[idx]
        after_n = _normalize_01(after)
        before_n = _normalize_01(before)
        diff_n = _normalize_01(after - before)
        x = np.stack([before_n, after_n, diff_n], 0).astype(np.float32)
        if self.augment:
            x = self._aug(x)
        return torch.from_numpy(x), torch.tensor(y).long()

def make_overlay(gray, score, mask, thr, sigma=3.0):
    base = _normalize_01(gray)
    base_rgb = np.dstack([base, base, base])
    scr = gaussian_filter(score.astype(np.float32), sigma=sigma)
    scr = np.where(mask, scr, 0.)
    scr = np.clip((scr - thr) / (scr.max() - thr + 1e-6), 0, 1)
    rgba = THERMAL(scr)[...,:3]
    alpha = np.clip(scr**0.7, 0, 1) * 0.9
    out = base_rgb*(1-alpha[...,None]) + rgba*alpha[...,None]
    return np.clip(out,0,1)

def plot_grid(records, mask, title, save_path, cfg, use_key="refined_gated"):
    n = len(records); cols = min(5, n); rows = (n-1)//cols + 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4.5), facecolor="#111111")
    axes = np.array(axes).reshape(rows, cols)
    thr = cfg.get("cam_thr", 0.4)
    sigma = cfg.get("cam_sigma", 3.0)
    for i, r in enumerate(records):
        ax = axes[i//cols, i%cols]
        ax.set_facecolor("#111111")
        _val = r.get(use_key) if use_key in r else r.get("refined", np.zeros_like(r["after"]))
        ov = make_overlay(r["after"], _val, mask, thr, sigma)
        ax.imshow(ov, interpolation="bilinear")
        ax.axis("off")
        p = r["prob"]
        ax.set_title(f"{r['name']}\nP={p:.3f}", color="w", fontsize=9)
    for j in range(n, rows*cols):
        axes[j//cols, j%cols].axis("off")
    plt.suptitle(title, color="w", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg["dpi"], bbox_inches="tight", facecolor="#111111")
    plt.close()

def compute_metrics(y_true, y_pred, y_prob):
    out = {}
    out["accuracy"] = accuracy_score(y_true, y_pred)
    out["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    out["precision"] = precision_score(y_true, y_pred, zero_division=0)
    out["recall"] = recall_score(y_true, y_pred, zero_division=0)
    out["f1"] = f1_score(y_true, y_pred, zero_division=0)
    out["mcc"] = matthews_corrcoef(y_true, y_pred)
    out["kappa"] = cohen_kappa_score(y_true, y_pred)
    try: out["roc_auc"] = roc_auc_score(y_true, y_prob)
    except: out["roc_auc"] = np.nan
    try: out["pr_auc"] = average_precision_score(y_true, y_prob)
    except: out["pr_auc"] = np.nan
    return out

def save_metrics_panel(result_dict, save_path):
    df = pd.DataFrame(result_dict).T
    fig, axes = plt.subplots(1,2, figsize=(15,5))
    df[["accuracy","balanced_accuracy","precision","recall","f1","mcc"]].plot(kind="bar", ax=axes[0])
    axes[0].set_ylim(0,1.05); axes[0].set_title("Classification Metrics")
    df[["roc_auc","pr_auc"]].plot(kind="bar", ax=axes[1], color=["#4C72B0","#55A868"])
    axes[1].set_ylim(0,1.05); axes[1].set_title("AUC Metrics")
    for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(save_path, dpi=CFG["dpi"], bbox_inches="tight"); plt.close()

def plot_curves(hist, save_path, title="Training Curves"):
    n_panels = 2 if "val_acc" in hist else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6*n_panels, 4), facecolor="white")
    if n_panels == 1:
        axes = [axes]
    # Loss panel
    ep = range(1, len(hist["tr_loss"])+1)
    axes[0].plot(ep, hist["tr_loss"], "#4C72B0", linewidth=1.8, label="train")
    axes[0].plot(ep, hist["val_loss"], "#DD8452", linewidth=1.8, label="val")
    best_ep = int(np.argmin(hist["val_loss"])) + 1
    axes[0].axvline(best_ep, color="gray", linestyle=":", linewidth=1.2, label=f"best ep={best_ep}")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss", fontweight="bold"); axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.25); axes[0].spines[["top","right"]].set_visible(False)
    # Metrics panel
    if "val_acc" in hist:
        axes[1].plot(ep, hist["val_acc"], "#55A868", linewidth=1.8, label="val acc")
        if "val_auc" in hist:
            ax2 = axes[1].twinx()
            ax2.plot(ep, hist["val_auc"], "#C44E52", linewidth=1.8, linestyle="--", label="val AUC")
            ax2.set_ylabel("AUC", color="#C44E52"); ax2.tick_params(axis="y", labelcolor="#C44E52")
            ax2.set_ylim(0, 1.05); ax2.spines[["top"]].set_visible(False)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy", color="#55A868")
        axes[1].tick_params(axis="y", labelcolor="#55A868")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("Val Metrics", fontweight="bold")
        axes[1].grid(alpha=0.2); axes[1].spines[["top","right"]].set_visible(False)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=CFG["dpi"], bbox_inches="tight")
    plt.close()

def export_individual_maps(records, mask, out_dir, cfg):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for r in records:
        nm = os.path.splitext(r["name"])[0]
        sub = os.path.join(out_dir, nm)
        os.makedirs(sub, exist_ok=True)
        imageio.imwrite(os.path.join(sub, "before.png"), (_normalize_01(r["before"])*255).astype(np.uint8))
        imageio.imwrite(os.path.join(sub, "after.png"), (_normalize_01(r["after"])*255).astype(np.uint8))
        
        _maps = r.get("maps", r.get("score_maps", {}))
        for key in ["top1","top3_mean","top3_weighted","top5_rms","support_count","support_frac"]:
            if key in _maps:
                np.save(os.path.join(sub, f"{key}.npy"), _maps[key])
        np.save(os.path.join(sub, "refined.npy"), r["refined"])
        _refined_show = r.get("refined_gated", r["refined"])
        np.save(os.path.join(sub, "refined_gated.npy"), _refined_show)
        ov = make_overlay(r["after"], _refined_show, mask, cfg.get("cam_thr", 0.4), cfg.get("cam_sigma", 3.0))
        imageio.imwrite(os.path.join(sub, "overlay.png"), (ov*255).astype(np.uint8))
        rows.append({
            "name": r["name"],
            "prob": r["prob"],
            "refined_max": float(r["refined"].max()),
            "refined_mean_masked": float(r["refined"][mask].mean()),
            "support_frac_mean": float(r.get("maps", r.get("score_maps", {})).get("support_frac", np.zeros_like(r["after"]))[mask].mean()),
        })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "manifest.csv"), index=False)

import gc
import time
import joblib
# =========================================================

# =========================================================

from sklearn.neural_network import MLPClassifier

# ---------- visualization helpers ----------
_HEATMAP_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "biosensor_heat",
    ["#0d0221", "#1a0a6b", "#0055d4", "#00b4d8", "#48cae4",
     "#80ed99", "#f9c74f", "#f3722c", "#f94144"]
)

def normalize01(arr):
    arr = np.asarray(arr, dtype=np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)

def save_gray(path, arr, cmap="gray", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="black")
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=CFG["dpi"], bbox_inches="tight", pad_inches=0, facecolor="black")
    plt.close()

def save_overlay(path, base_img, heatmap, alpha=0.50, thr_pct=70):
    """Overlay heatmap on base image. Only shows pixels above thr_pct percentile."""
    base = normalize01(base_img)
    heat = normalize01(heatmap)
    thr  = float(np.percentile(heat, thr_pct))
    heat_masked = np.ma.masked_less(heat, thr)
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="black")
    ax.imshow(base, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    im = ax.imshow(heat_masked, cmap=_HEATMAP_CMAP, alpha=alpha,
                   vmin=thr, vmax=1.0, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=CFG["dpi"], bbox_inches="tight", pad_inches=0, facecolor="black")
    plt.close()

def save_image_grid(items, out_png, ncols=4, title=None):
    if len(items) == 0:
        return
    n = len(items)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.5*ncols, 3.5*nrows),
                             facecolor="#111111")
    axes = np.array(axes).reshape(nrows, ncols)
    for ax in axes.ravel():
        ax.set_facecolor("#111111")
        ax.axis("off")
    for ax, item in zip(axes.ravel(), items):
        ax.imshow(item["img"], cmap=item.get("cmap", None),
                  interpolation="nearest")
        lbl = item.get("title", "")
        ax.set_title(lbl, fontsize=8, color="white", pad=3)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=13, color="white", y=1.01, fontweight="bold")
    plt.tight_layout(pad=0.4)
    plt.savefig(out_png, dpi=CFG["dpi"], bbox_inches="tight",
                facecolor="#111111")
    plt.close()

# ---------- memory helpers ----------
def cleanup_memory(*objs):
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------- feature filtering ----------
def _selected_feature_indices(mode="extended48", cfg=None):
    cfg = CFG if cfg is None else cfg
    names = get_feature_names(mode)
    exclude = set(cfg.get("exclude_feature_keys", []))
    keep = [i for i, nm in enumerate(names) if nm not in exclude]
    keep_names = [names[i] for i in keep]
    return keep, keep_names

def build_pbs_pixel_baseline(after_list, before_list, mask, mode="extended48", cfg=None):
    cfg = CFG if cfg is None else cfg
    keep, _ = _selected_feature_indices(mode, cfg)
    Fs = []
    for aft, bef in zip(after_list, before_list):
        feats = _pixel_features(bef, aft, mode=mode)[keep]
        Fs.append(feats.astype(np.float32))
    stk = np.stack(Fs, axis=0)   # [N,F,H,W]
    mu = stk.mean(axis=0)        # [F,H,W]
    std = stk.std(axis=0, ddof=1) if stk.shape[0] > 1 else np.zeros_like(mu)

    if mask is not None and mask.any():
        
        floor = np.percentile(std[:, mask], 5, axis=1)[:, None, None]   # [F,1,1]
        std = np.maximum(std, floor)
    else:
        std = np.maximum(std, 1e-3)

    return mu.astype(np.float32), std.astype(np.float32)

def build_score_maps(before, after, pbs_mu, pbs_std, mask, cfg):
    keep, feat_names = _selected_feature_indices(cfg["feature_bank_mode"], cfg)
    feats_all = _pixel_features(before, after, mode=cfg["feature_bank_mode"])
    feats = feats_all[keep]
    z = (feats - pbs_mu) / (pbs_std + 1e-6)
    z = np.where(mask[None], z, 0.0)
    idx_sorted = np.argsort(z, axis=0)[::-1]
    z_sorted = np.take_along_axis(z, idx_sorted, axis=0)

    top1 = z_sorted[0]
    top3_mean = z_sorted[:3].mean(0)
    top3_weighted = 0.6*z_sorted[0] + 0.3*z_sorted[1] + 0.1*z_sorted[2]
    top5_rms = np.sqrt((z_sorted[:5]**2).mean(0))
    support_count = (z > cfg["support_z_thr"]).sum(0).astype(np.float32)
    support_frac  = support_count / z.shape[0]

    maps = {
        "top1": top1.astype(np.float32),
        "top3_mean": top3_mean.astype(np.float32),
        "top3_weighted": top3_weighted.astype(np.float32),
        "top5_rms": top5_rms.astype(np.float32),
        "support_count": support_count.astype(np.float32),
        "support_frac": support_frac.astype(np.float32),
        "z_stack": z.astype(np.float32),
        "feature_stack": feats.astype(np.float32),
        "feature_names": feat_names,
        "top_idx_stack": idx_sorted[:5].astype(np.int16),
    }
    maps["primary"] = maps[cfg["score_primary"]]

    # formal aliases
    maps["single_strongest_feature_response_map"] = maps["top1"]
    maps["three_strongest_feature_mean_response_map"] = maps["top3_mean"]
    maps["three_strongest_feature_weighted_response_map"] = maps["top3_weighted"]
    maps["five_strongest_feature_rms_response_map"] = maps["top5_rms"]
    maps["multi_feature_support_count_map"] = maps["support_count"]
    return maps

def summarize_image_features(score_maps, mask):
    z = score_maps["z_stack"]
    out = {}
    for k in ["top1","top3_mean","top3_weighted","top5_rms","support_count","support_frac"]:
        m = score_maps[k][mask]
        out[f"{k}_mean"] = float(m.mean())
        out[f"{k}_std"]  = float(m.std())
        out[f"{k}_max"]  = float(m.max())
        out[f"{k}_q95"]  = float(np.quantile(m, .95))
        out[f"{k}_sigpct"] = float((m > np.quantile(m, .95)).mean())
    for i in range(z.shape[0]):
        zi = z[i][mask]
        out[f"z{i:02d}_mean"] = float(zi.mean())
        out[f"z{i:02d}_std"]  = float(zi.std())
        out[f"z{i:02d}_q95"]  = float(np.quantile(zi, .95))
    return out

def feature_rank_tables(score_maps):
    feat_names = score_maps["feature_names"]
    idx_stack = score_maps["top_idx_stack"]   # [5,H,W]
    tables = {}
    for rank_i in range(idx_stack.shape[0]):
        idx_map = idx_stack[rank_i]
        uniq, cnt = np.unique(idx_map, return_counts=True)
        rows = []
        total = cnt.sum()
        for u, c in zip(uniq, cnt):
            rows.append({
                "feature_idx": int(u),
                "feature_name": feat_names[int(u)],
                "count": int(c),
                "fraction": float(c / total)
            })
        tables[f"rank{rank_i+1}"] = pd.DataFrame(rows).sort_values("count", ascending=False)
    return tables


def _to_rgb_uint8(img, cmap=None):
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        rgb = arr.astype(np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        return np.clip(rgb * 255, 0, 255).astype(np.uint8)
    arr = normalize01(arr.astype(np.float32))
    if cmap is None or cmap == "gray":
        rgb = np.stack([arr, arr, arr], axis=-1)
    else:
        rgb = plt.get_cmap(cmap)(arr)[..., :3]
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)

def _label_tile(img_u8, text, bar_h=18):
    h, w = img_u8.shape[:2]
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    out = np.vstack([bar, img_u8])
    cv2.putText(out, text, (6, bar_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out

def _concat_grid_rgb(imgs, nrows, ncols, pad=4, bg=(8, 8, 8)):
    hs = [im.shape[0] for im in imgs]
    ws = [im.shape[1] for im in imgs]
    cell_h, cell_w = max(hs), max(ws)
    canvas = np.zeros((nrows*cell_h + pad*(nrows-1), ncols*cell_w + pad*(ncols-1), 3), dtype=np.uint8)
    canvas[:] = np.array(bg, dtype=np.uint8)
    for idx, im in enumerate(imgs):
        r, c = divmod(idx, ncols)
        y = r*(cell_h + pad)
        x = c*(cell_w + pad)
        h, w = im.shape[:2]
        canvas[y:y+h, x:x+w] = im
    return canvas

def build_compact_stage_tile(before_corr, after_corr, diff_img, log_ratio, coarse_map, refined_map):
    imgs = [
        _label_tile(_to_rgb_uint8(before_corr, "gray"), "before"),
        _label_tile(_to_rgb_uint8(after_corr, "gray"), "after"),
        _label_tile(_to_rgb_uint8(diff_img, RESPONSE_CMAP), "diff"),
        _label_tile(_to_rgb_uint8(log_ratio, RESPONSE_CMAP), "log_ratio"),
        _label_tile(_to_rgb_uint8(coarse_map, RESPONSE_CMAP), "coarse"),
        _label_tile(_to_rgb_uint8(refined_map, RESPONSE_CMAP), "refined"),
    ]
    return _concat_grid_rgb(imgs, 2, 3, pad=4)

def build_response_only_tile(after_corr, coarse_map, refined_map, coarse_mask, refined_mask):
    def _overlay(base, heat, mask):
        base_rgb = _to_rgb_uint8(base, "gray").astype(np.float32)
        heat_rgb = _to_rgb_uint8(heat, RESPONSE_CMAP).astype(np.float32)
        out = base_rgb.copy()
        m = mask.astype(bool)
        out[m] = 0.35 * base_rgb[m] + 0.65 * heat_rgb[m]
        return out.clip(0, 255).astype(np.uint8)

    imgs = [
        _label_tile(_to_rgb_uint8(after_corr, "gray"), "after"),
        _label_tile(_overlay(after_corr, coarse_map, coarse_mask), "coarse-only"),
        _label_tile(_overlay(after_corr, refined_map, refined_mask), "refined-only"),
    ]
    return _concat_grid_rgb(imgs, 1, 3, pad=4)

def plot_response_area_comparison(df, out_png):
    if len(df) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="#0b0b0b")
    for ax in axes:
        ax.set_facecolor("#0b0b0b")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        ax.grid(alpha=0.2, color="white")

    order = ["PBS", "Analyte"]
    palette = {"PBS":"#4C9AFF", "Analyte":"#FF7A00"}

    # boxplot
    for i, stage in enumerate(["coarse_area_pct", "refined_area_pct"]):
        pos = np.array([0, 1]) + (i*0.28 - 0.14)
        data = [df[df["label"] == g][stage].values for g in order]
        bp = axes[0].boxplot(data, positions=pos, widths=0.22, patch_artist=True, showfliers=False)
        colors = ["#3B82F6" if stage=="coarse_area_pct" else "#F97316"]*2
        for box, c in zip(bp["boxes"], colors):
            box.set_facecolor(c); box.set_alpha(0.65); box.set_edgecolor("white")
        for key in ["whiskers","caps","medians"]:
            for obj in bp[key]:
                obj.set_color("white")
    axes[0].set_xticks([0,1]); axes[0].set_xticklabels(order, color="white")
    axes[0].set_ylabel("Response area (%)", color="white")
    axes[0].set_title("Response area before / after weak supervision", color="white", fontweight="bold")
    axes[0].legend(handles=[
        plt.Line2D([0],[0], color="#3B82F6", lw=8, label="coarse"),
        plt.Line2D([0],[0], color="#F97316", lw=8, label="refined")
    ], frameon=False, labelcolor="white", loc="upper right")

    grp = df.groupby("label")[["coarse_area_pct","refined_area_pct","area_delta_pct"]].mean().reindex(order)
    x = np.arange(len(order))
    axes[1].bar(x - 0.22, grp["coarse_area_pct"], width=0.22, color="#3B82F6", alpha=0.8, label="coarse")
    axes[1].bar(x, grp["refined_area_pct"], width=0.22, color="#F97316", alpha=0.8, label="refined")
    axes[1].bar(x + 0.22, grp["area_delta_pct"], width=0.22, color="#22C55E", alpha=0.8, label="delta")
    axes[1].set_xticks(x); axes[1].set_xticklabels(order, color="white")
    axes[1].set_ylabel("Mean area (%)", color="white")
    axes[1].set_title("Group mean response-area summary", color="white", fontweight="bold")
    axes[1].legend(frameon=False, labelcolor="white", loc="upper right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=CFG["dpi"], bbox_inches="tight", facecolor="#0b0b0b")
    plt.close()


def save_simple_grid(records, out_png, title, use_key="overlay_path", ncols=4):
    if len(records) == 0:
        return
    items = []
    for r in records:
        img = plt.imread(r[use_key])
        items.append({"img": img, "title": f"{r['name']} | p={r.get('prob', np.nan):.3f}", "cmap": None})
    save_image_grid(items, out_png, ncols=ncols, title=title)

RESPONSE_CMAP = "jet"


# ---------- datasets ----------
class PairImageDataset(Dataset):
    def __init__(self, pbs_A, pbs_B, ana_A, ana_B, img_size=128, augment=False):
        self.samples = []
        for a, b in zip(pbs_A, pbs_B):
            self.samples.append((a, b, 0))
        for a, b in zip(ana_A, ana_B):
            self.samples.append((a, b, 1))
        self.img_size = img_size
        self.augment = augment

    def __len__(self): return len(self.samples)

    def _aug(self, x):
        if np.random.rand() < 0.5: x = np.flip(x, 1).copy()
        if np.random.rand() < 0.5: x = np.flip(x, 2).copy()
        k = np.random.randint(0, 4)
        x = np.rot90(x, k, axes=(1,2)).copy()
        x += np.random.normal(0, 0.01, x.shape).astype(np.float32)
        return x

    def __getitem__(self, idx):
        after, before, y = self.samples[idx]
        after_n = _normalize_01(after)
        before_n = _normalize_01(before)
        log_r = _normalize_01(np.log(after+1.0) - np.log(before+1.0))
        x = np.stack([before_n, after_n, log_r], 0).astype(np.float32)
        if self.augment:
            x = self._aug(x)
        return torch.from_numpy(x), torch.tensor(y).long()

class PairMonoDataset(Dataset):
    def __init__(self, after_list, before_list, labels):
        self.after_list = after_list
        self.before_list = before_list
        self.labels = labels

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        after = _normalize_01(self.after_list[idx])[None, ...].astype(np.float32)
        before = _normalize_01(self.before_list[idx])[None, ...].astype(np.float32)
        y = int(self.labels[idx])
        return torch.from_numpy(before), torch.from_numpy(after), torch.tensor(y).long()

# ---------- DL baselines ----------
class SimpleCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

class TinyPairTransformer(nn.Module):
    def __init__(self, in_ch=1, emb_dim=64, nhead=4, depth=2, patch=16, dropout=0.1):
        super().__init__()
        self.embed_before = nn.Conv2d(in_ch, emb_dim, kernel_size=patch, stride=patch)
        self.embed_after  = nn.Conv2d(in_ch, emb_dim, kernel_size=patch, stride=patch)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=emb_dim*2,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.cls = nn.Sequential(
            nn.Linear(emb_dim*3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
    def forward(self, before_img, after_img):
        xb = self.embed_before(before_img).flatten(2).transpose(1, 2)
        xa = self.embed_after(after_img).flatten(2).transpose(1, 2)
        fb = self.encoder(xb).mean(dim=1)
        fa = self.encoder(xa).mean(dim=1)
        fd = fa - fb
        return self.cls(torch.cat([fb, fa, fd], dim=1))

# ---------- generic dl train / pred ----------
def train_dl_classifier(model, tr_loader, val_loader, epochs=8, lr=1e-4, weight_decay=1e-4, patience=3, mode="single"):
    model = model.to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    best = {"val_loss": np.inf, "state": None, "epoch": -1}
    hist = {"tr_loss": [], "val_loss": [], "val_acc": []}
    bad = 0

    for ep in range(epochs):
        model.train()
        tr_losses = []
        for batch in tr_loader:
            opt.zero_grad()
            if mode == "pair":
                bef, aft, y = batch
                bef, aft, y = bef.to(DEVICE), aft.to(DEVICE), y.to(DEVICE)
                out = model(bef, aft)
            else:
                x, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        val_losses, ys, probs, preds = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                if mode == "pair":
                    bef, aft, y = batch
                    bef, aft, y = bef.to(DEVICE), aft.to(DEVICE), y.to(DEVICE)
                    out = model(bef, aft)
                else:
                    x, y = batch
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    out = model(x)
                loss = crit(out, y)
                val_losses.append(loss.item())
                p = torch.softmax(out, 1)[:, 1].cpu().numpy()
                pr = out.argmax(1).cpu().numpy()
                probs.extend(p.tolist()); preds.extend(pr.tolist()); ys.extend(y.cpu().numpy().tolist())

        trm = float(np.mean(tr_losses))
        vam = float(np.mean(val_losses))
        vacc = accuracy_score(ys, preds)
        hist["tr_loss"].append(trm); hist["val_loss"].append(vam); hist["val_acc"].append(vacc)

        if vam < best["val_loss"]:
            best = {"val_loss": vam, "state": {k:v.cpu().clone() for k,v in model.state_dict().items()}, "epoch": ep}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, hist

@torch.no_grad()
def predict_dl_classifier(model, loader, mode="single"):
    model.eval()
    ys, preds, probs = [], [], []
    for batch in loader:
        if mode == "pair":
            bef, aft, y = batch
            bef, aft = bef.to(DEVICE), aft.to(DEVICE)
            out = model(bef, aft)
        else:
            x, y = batch
            x = x.to(DEVICE)
            out = model(x)
        p = torch.softmax(out,1)[:,1].cpu().numpy()
        pr = out.argmax(1).cpu().numpy()
        probs.extend(p.tolist()); preds.extend(pr.tolist()); ys.extend(y.numpy().tolist())
    return np.array(ys), np.array(preds), np.array(probs)

def compute_model_composite(summary_df, speed_df=None):
    rows = []
    for model_name, row in summary_df.iterrows():
        f1 = row.get("f1_mean", np.nan)
        mcc = row.get("mcc_mean", np.nan)
        pr_auc = row.get("pr_auc_mean", np.nan)
        roc_auc = row.get("roc_auc_mean", np.nan)
        bal = row.get("balanced_accuracy_mean", np.nan)
        f1_std = row.get("f1_std", 0.0)
        sp = float(speed_df.loc[model_name, "mean_seconds"]) if speed_df is not None and model_name in speed_df.index else 0.0
        sp_norm = sp / max(float(speed_df["mean_seconds"].max()), 1.0) if speed_df is not None else 0.0
        score = (
            0.25*f1 + 0.20*mcc + 0.20*pr_auc + 0.15*roc_auc + 0.10*bal
            - 0.05*f1_std - 0.05*sp_norm
        )
        rows.append({
            "model": model_name,
            "composite_score": score,
            "f1_mean": f1,
            "f1_std": f1_std,
            "mcc_mean": mcc,
            "pr_auc_mean": pr_auc,
            "roc_auc_mean": roc_auc,
            "balanced_accuracy_mean": bal
        })
    return pd.DataFrame(rows).sort_values("composite_score", ascending=False)

def cleanup_memory(*objs):
    for obj in objs:
        try:
            del obj
        except:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------- detailed export helpers ----------
def _sheet_safe(name, maxlen=31):
    name = str(name).replace("/", "_").replace("\\", "_").replace(":", "_").replace("*","_").replace("?","_").replace("[","(").replace("]",")")
    return name[:maxlen]

def _summarize_array(arr, prefix=""):
    arr = np.asarray(arr, dtype=np.float32)
    flat = arr.reshape(-1)
    return {
        f"{prefix}shape": "x".join(map(str, arr.shape)),
        f"{prefix}min": float(np.min(flat)),
        f"{prefix}max": float(np.max(flat)),
        f"{prefix}mean": float(np.mean(flat)),
        f"{prefix}std": float(np.std(flat)),
        f"{prefix}q01": float(np.quantile(flat, 0.01)),
        f"{prefix}q05": float(np.quantile(flat, 0.05)),
        f"{prefix}q50": float(np.quantile(flat, 0.50)),
        f"{prefix}q95": float(np.quantile(flat, 0.95)),
        f"{prefix}q99": float(np.quantile(flat, 0.99)),
    }

def export_mask_bundle(mask, out_dir, cfg):
    mask_dir = os.path.join(out_dir, "mask_bundle")
    os.makedirs(mask_dir, exist_ok=True)
    save_gray(os.path.join(mask_dir, "mask.png"), mask.astype(np.float32), cmap="gray", vmin=0, vmax=1)
    pd.DataFrame([{
        "mask_shape": "x".join(map(str, mask.shape)),
        "valid_pixels": int(mask.sum()),
        "total_pixels": int(mask.size),
        "valid_ratio": float(mask.mean()),
        "mask_thr": cfg["mask_thr"],
        "mask_erode": cfg["mask_erode"],
    }]).to_csv(os.path.join(mask_dir, "mask_summary.csv"), index=False)
    return mask_dir

def _feature_meta_row(idx, name):
    row = {"feature_idx": idx, "feature_name": name}
    if name == "diff":
        row.update({"group":"intensity","formula_or_definition":"after - before","input_type":"pixelwise difference",
                    "spatial_scale":"pixel","physical_meaning":"Absolute brightness change","expected_response_type":"Local brightening or dimming"})
    elif name == "log_ratio":
        row.update({"group":"intensity","formula_or_definition":"log(after+1)-log(before+1)","input_type":"log ratio",
                    "spatial_scale":"pixel","physical_meaning":"Relative brightness change","expected_response_type":"Exposure-normalized change"})
    elif name.startswith("gauss_diff_s"):
        s=name.split("s")[-1]
        row.update({"group":"multi_scale_smoothing","formula_or_definition":f"Gaussian(diff, sigma={s})","input_type":"smoothed diff",
                    "spatial_scale":f"sigma={s}","physical_meaning":"Slowly varying response at multiple scales","expected_response_type":"Regional or diffuse change"})
    elif name.startswith("gauss_logr_s"):
        s=name.split("s")[-1]
        row.update({"group":"multi_scale_smoothing","formula_or_definition":f"Gaussian(log_ratio, sigma={s})","input_type":"smoothed log-ratio",
                    "spatial_scale":f"sigma={s}","physical_meaning":"Relative change at multiple scales","expected_response_type":"Slow spatial variation"})
    elif name.startswith("dog_diff_"):
        row.update({"group":"DoG_blob","formula_or_definition":name.replace("dog_diff_","DoG(diff, ") + ")",
                    "input_type":"difference of Gaussians","spatial_scale":"band-pass","physical_meaning":"Spot or local hotspot enhancement","expected_response_type":"Point-like hotspots"})
    elif name.startswith("local_mu_w"):
        w=name.split("w")[-1]
        row.update({"group":"local_statistics","formula_or_definition":f"local mean of diff, window={w}","input_type":"local mean",
                    "spatial_scale":f"window={w}","physical_meaning":"Local mean shift","expected_response_type":"Regional intensity change"})
    elif name.startswith("local_std_w"):
        w=name.split("w")[-1]
        row.update({"group":"local_statistics","formula_or_definition":f"local std of diff, window={w}","input_type":"local std",
                    "spatial_scale":f"window={w}","physical_meaning":"Local fluctuation or texture change","expected_response_type":"Increased texture or roughness"})
    elif name.startswith("local_snr_w"):
        w=name.split("w")[-1]
        row.update({"group":"local_statistics","formula_or_definition":f"diff / local_std, window={w}","input_type":"local snr",
                    "spatial_scale":f"window={w}","physical_meaning":"Local signal-to-noise change","expected_response_type":"Weak-signal prominence"})
    elif name.startswith("delta_speckle_w"):
        w=name.split("w")[-1]
        row.update({"group":"texture_speckle","formula_or_definition":f"K(after)-K(before), window={w}","input_type":"speckle contrast delta",
                    "spatial_scale":f"window={w}","physical_meaning":"Local speckle or texture-complexity change","expected_response_type":"Texture response"})
    elif name in ("fft_low","fft_high"):
        row.update({"group":"frequency","formula_or_definition":name,"input_type":"frequency energy",
                    "spatial_scale":"global-local mixed","physical_meaning":"Low/high-frequency energy change","expected_response_type":"Smooth or fine-structure change"})
    elif name == "relative_diff":
        row.update({"group":"normalized_difference","formula_or_definition":"(after-before)/(before+eps)","input_type":"relative diff",
                    "spatial_scale":"pixel","physical_meaning":"Normalized brightness change","expected_response_type":"Batch-shift normalization"})
    elif name == "sym_ratio":
        row.update({"group":"normalized_difference","formula_or_definition":"(after-before)/(after+before+eps)","input_type":"symmetric ratio",
                    "spatial_scale":"pixel","physical_meaning":"Symmetric normalized difference","expected_response_type":"Exposure normalization"})
    elif name == "z_before_norm":
        row.update({"group":"normalized_difference","formula_or_definition":"diff/(local_std_before+eps)","input_type":"before-normalized z",
                    "spatial_scale":"local","physical_meaning":"Normalization by before-image noise","expected_response_type":"Weak-signal enhancement"})
    elif name == "anscombe_diff":
        row.update({"group":"poisson_stabilized","formula_or_definition":"Anscombe(after)-Anscombe(before)","input_type":"variance-stabilized diff",
                    "spatial_scale":"pixel","physical_meaning":"Change after Poisson variance stabilization","expected_response_type":"Low-count fluorescence change"})
    elif name.startswith("corr_drop_w"):
        w=name.split("w")[-1]
        row.update({"group":"local_similarity","formula_or_definition":f"local correlation drop, window={w}","input_type":"correlation drop",
                    "spatial_scale":f"window={w}","physical_meaning":"Reduced local before/after similarity","expected_response_type":"Structural change"})
    elif name.startswith("ssim_drop_w"):
        w=name.split("w")[-1]
        row.update({"group":"local_similarity","formula_or_definition":f"SSIM-like drop, window={w}","input_type":"ssim-like drop",
                    "spatial_scale":f"window={w}","physical_meaning":"Local structural-similarity change","expected_response_type":"Morphology or texture change"})
    elif name.startswith("grad_diff_s"):
        s=name.split("s")[-1]
        row.update({"group":"gradient_edge","formula_or_definition":f"Gaussian(grad(after)-grad(before), sigma={s})","input_type":"gradient magnitude diff",
                    "spatial_scale":f"sigma={s}","physical_meaning":"Edge-intensity change","expected_response_type":"Boundary or contour change"})
    elif name.startswith("log_like_s"):
        s=name.split("s")[-1]
        row.update({"group":"laplacian","formula_or_definition":f"Gaussian(Laplace(after)-Laplace(before), sigma={s})","input_type":"LoG-like diff",
                    "spatial_scale":f"sigma={s}","physical_meaning":"Local curvature or spot change","expected_response_type":"Bright-spot or void change"})
    elif name == "laplace_diff":
        row.update({"group":"laplacian","formula_or_definition":"Laplace(after)-Laplace(before)","input_type":"laplacian diff",
                    "spatial_scale":"pixel-local","physical_meaning":"Curvature change","expected_response_type":"Spot or edge response"})
    elif name == "tensor_aniso_change":
        row.update({"group":"gradient_edge","formula_or_definition":"grad_mag(after)-grad_mag(before)","input_type":"anisotropy proxy",
                    "spatial_scale":"local","physical_meaning":"Local orientation change","expected_response_type":"Stripe or directional-texture change"})
    elif name.startswith("pct_shift_w"):
        w=name.split("w")[-1]
        row.update({"group":"rank_percentile","formula_or_definition":f"local percentile shift, window={w}","input_type":"percentile shift",
                    "spatial_scale":f"window={w}","physical_meaning":"Local quantile change","expected_response_type":"Robust local hotspot"})
    elif name.startswith("rank_diff_w"):
        w=name.split("w")[-1]
        row.update({"group":"rank_percentile","formula_or_definition":f"local rank transform diff, window={w}","input_type":"rank diff",
                    "spatial_scale":f"window={w}","physical_meaning":"Rank change","expected_response_type":"Brightness-drift-normalized structural change"})
    else:
        row.update({"group":"misc","formula_or_definition":name,"input_type":"custom","spatial_scale":"unknown",
                    "physical_meaning":"Not specified","expected_response_type":"Not specified"})
    row["used_in_topk"] = True
    row["used_in_refinement"] = True
    return row

def export_feature_bank_dictionary(out_dir, cfg):
    core_names = _feature_names_core30()
    ext_names  = _feature_names_extended48()
    keep_idx, keep_names = _selected_feature_indices(cfg["feature_bank_mode"], cfg)
    core_df = pd.DataFrame([_feature_meta_row(i, n) for i, n in enumerate(core_names)])
    ext_df  = pd.DataFrame([_feature_meta_row(i, n) for i, n in enumerate(ext_names)])
    selected_df = pd.DataFrame([_feature_meta_row(i, n) for i, n in enumerate(keep_names)])
    summary_df = pd.DataFrame([
        {"sheet":"core30", "declared_name":"core30", "implemented_count":len(core_names)},
        {"sheet":"extended48", "declared_name":"extended48", "implemented_count":len(ext_names)},
        {"sheet":"selected_current_run", "feature_bank_mode":cfg["feature_bank_mode"], "selected_count":len(keep_names),
         "excluded": ",".join(cfg.get("exclude_feature_keys", []))}
    ])
    xlsx_path = os.path.join(out_dir, "feature_bank_dictionary.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        core_df.to_excel(writer, sheet_name="core30", index=False)
        ext_df.to_excel(writer, sheet_name="extended48", index=False)
        selected_df.to_excel(writer, sheet_name="selected_current_run", index=False)
        core_df.groupby("group").size().reset_index(name="n_features").to_excel(writer, sheet_name="core30_groups", index=False)
        ext_df.groupby("group").size().reset_index(name="n_features").to_excel(writer, sheet_name="extended48_groups", index=False)
    core_df.to_csv(os.path.join(out_dir, "feature_bank_core30_dictionary.csv"), index=False)
    ext_df.to_csv(os.path.join(out_dir, "feature_bank_extended48_dictionary.csv"), index=False)
    selected_df.to_csv(os.path.join(out_dir, "feature_bank_selected_dictionary.csv"), index=False)
    return {"core": core_df, "extended": ext_df, "selected": selected_df, "xlsx": xlsx_path}

def export_led_bundle(led_dir, led_map, target_size, bright_thr, sigma, out_dir, cfg):
    led_out = os.path.join(out_dir, "led_diagnostics")
    os.makedirs(led_out, exist_ok=True)
    if led_map is None:
        pd.DataFrame([{"status":"no_led_map"}]).to_csv(os.path.join(led_out, "led_summary.csv"), index=False)
        return led_out

    files = _iter_image_files(led_dir)
    rows = []
    bright_frames = []
    for fp in files:
        img = _read_gray(fp)
        if img is None:
            continue
        kept = bool(img.max() >= bright_thr * 255)
        row = {
            "filename": os.path.basename(fp),
            "shape": f"{img.shape[1]}x{img.shape[0]}",
            "mean_intensity": float(img.mean()),
            "std_intensity": float(img.std()),
            "max_intensity": float(img.max()),
            "min_intensity": float(img.min()),
            "kept": kept,
            "reason": "bright" if kept else "dark_filtered"
        }
        if kept:
            x1, y1, x2, y2 = _find_led_bbox(img, thr_ratio=0.25, pad=6)
            row.update({"bbox_x1":x1, "bbox_y1":y1, "bbox_x2":x2, "bbox_y2":y2,
                        "bbox_w": x2-x1, "bbox_h": y2-y1})
            crop = img[y1:y2, x1:x2]
            bright_frames.append(_resize_with_pad(crop, target_size))
        rows.append(row)

    manifest_df = pd.DataFrame(rows).sort_values(["kept","filename"], ascending=[False, True])
    manifest_df.to_csv(os.path.join(led_out, "led_frame_manifest.csv"), index=False)
    pd.DataFrame(led_map).to_csv(os.path.join(led_out, "led_map.csv"), index=False, header=False)
    horizontal = pd.DataFrame({"x": np.arange(led_map.shape[1]), "value": led_map[led_map.shape[0]//2]})
    vertical = pd.DataFrame({"y": np.arange(led_map.shape[0]), "value": led_map[:, led_map.shape[1]//2]})
    horizontal.to_csv(os.path.join(led_out, "led_horizontal_profile.csv"), index=False)
    vertical.to_csv(os.path.join(led_out, "led_vertical_profile.csv"), index=False)
    sum_row = _summarize_array(led_map)
    sum_row.update({
        "bright_thr": bright_thr,
        "smooth_sigma": sigma,
        "target_size": str(target_size),
        "n_total_frames": int(len(files)),
        "n_kept_frames": int(manifest_df["kept"].sum()) if len(manifest_df) else 0,
        "n_dropped_frames": int((~manifest_df["kept"]).sum()) if len(manifest_df) else 0,
    })
    pd.DataFrame([sum_row]).to_csv(os.path.join(led_out, "led_summary.csv"), index=False)

    
    if len(bright_frames):
        items = [{"img": normalize01(fr), "title": f"LED bright #{i+1}", "cmap":"gray"} for i, fr in enumerate(bright_frames[:16])]
        save_image_grid(items, os.path.join(led_out, "led_kept_frames_preview.png"), ncols=4, title="LED Kept Frames Preview")
    return led_out

def export_led_effect_examples(before_list, after_list, led_map, out_dir, cfg, prefix="pair"):
    if led_map is None or len(before_list) == 0 or len(after_list) == 0:
        return None
    ex_dir = os.path.join(out_dir, "led_diagnostics")
    os.makedirs(ex_dir, exist_ok=True)
    
    sample_rows = []
    for i, (bef, aft) in enumerate(zip(before_list[:min(6, len(before_list))], after_list[:min(6, len(after_list))]), 1):
        sample_rows.append({
            "sample_id": f"{prefix}_{i}",
            "before_mean": float(np.mean(bef)),
            "before_std": float(np.std(bef)),
            "after_mean": float(np.mean(aft)),
            "after_std": float(np.std(aft)),
            "diff_mean": float(np.mean(aft-bef)),
            "diff_std": float(np.std(aft-bef)),
        })
        items = [
            {"img": bef, "title": f"{prefix}_{i} before_corr", "cmap":"gray"},
            {"img": aft, "title": f"{prefix}_{i} after_corr", "cmap":"gray"},
            {"img": aft-bef, "title": f"{prefix}_{i} diff", "cmap":"coolwarm"},
        ]
        save_image_grid(items, os.path.join(ex_dir, f"{prefix}_{i}_corrected_panel.png"), ncols=3, title=f"{prefix}_{i} corrected example")
    pd.DataFrame(sample_rows).to_csv(os.path.join(ex_dir, f"{prefix}_corrected_examples_summary.csv"), index=False)
    return ex_dir

def export_baseline_bundle(pbs_mu_px, pbs_std_px, mask, feature_names, out_dir, cfg):
    base_dir = os.path.join(out_dir, "pbs_baseline_bundle")
    os.makedirs(base_dir, exist_ok=True)
    rows = []
    mu_items, std_items = [], []
    for i, nm in enumerate(feature_names):
        mu = pbs_mu_px[i]; sd = pbs_std_px[i]
        if mask is not None and mask.any():
            mu_vals = mu[mask]; sd_vals = sd[mask]
        else:
            mu_vals = mu.reshape(-1); sd_vals = sd.reshape(-1)
        rows.append({
            "feature_idx": i,
            "feature_name": nm,
            "mu_mean_masked": float(np.mean(mu_vals)),
            "mu_std_masked": float(np.std(mu_vals)),
            "mu_q95_masked": float(np.quantile(mu_vals, 0.95)),
            "std_mean_masked": float(np.mean(sd_vals)),
            "std_std_masked": float(np.std(sd_vals)),
            "std_q95_masked": float(np.quantile(sd_vals, 0.95)),
            "mu_min": float(mu.min()),
            "mu_max": float(mu.max()),
            "std_min": float(sd.min()),
            "std_max": float(sd.max()),
        })
        mu_items.append({"img": normalize01(mu), "title": f"{i:02d} {nm}", "cmap":"viridis"})
        std_items.append({"img": normalize01(sd), "title": f"{i:02d} {nm}", "cmap":"magma"})
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(base_dir, "pbs_baseline_feature_summary.csv"), index=False)
    save_image_grid(mu_items, os.path.join(base_dir, "pbs_mu_feature_gallery.png"), ncols=4, title="PBS baseline u maps")
    save_image_grid(std_items, os.path.join(base_dir, "pbs_std_feature_gallery.png"), ncols=4, title="PBS baseline sigma maps")
    return base_dir, summary_df

def export_sample_zscore_bundle(sdir, maps, mask):
    feat_names = maps.get("feature_names", [])
    z = maps.get("z_stack")
    if z is None:
        return pd.DataFrame()
    rows = []
    for i, nm in enumerate(feat_names):
        zi = z[i]
        vals = zi[mask] if (mask is not None and mask.any()) else zi.reshape(-1)
        rows.append({
            "feature_idx": i,
            "feature_name": nm,
            "z_mean_masked": float(np.mean(vals)),
            "z_std_masked": float(np.std(vals)),
            "z_q95_masked": float(np.quantile(vals, 0.95)),
            "z_q99_masked": float(np.quantile(vals, 0.99)),
            "z_max_masked": float(np.max(vals)),
        })
    return pd.DataFrame(rows).sort_values("z_q99_masked", ascending=False)

def save_proposal_stage_panel(sdir, before_corr, after_corr, log_ratio, maps, title_prefix="proposal"):
    diff_vis = _normalize_01(after_corr - before_corr)
    log_vis  = _normalize_01(log_ratio)
    items = [
        {"img": diff_vis, "title":"diff", "cmap":RESPONSE_CMAP},
        {"img": log_vis, "title":"log_ratio", "cmap":RESPONSE_CMAP},
        {"img": maps["top1"], "title":"top1", "cmap":RESPONSE_CMAP},
        {"img": maps["top3_weighted"], "title":"top3_weighted", "cmap":RESPONSE_CMAP},
        {"img": maps["top5_rms"], "title":"top5_rms", "cmap":RESPONSE_CMAP},
        {"img": maps["support_frac"], "title":"support_frac", "cmap":RESPONSE_CMAP},
    ]
    save_image_grid(items, os.path.join(sdir, f"{title_prefix}_stage_panel.png"), ncols=3, title=f"{title_prefix} stage panel")


def save_group_map_galleries(records, out_dir, group_name="Analyte"):
    if len(records) == 0:
        return {}
    group_dir = os.path.join(out_dir, f"{group_name.lower()}_feature_galleries")
    os.makedirs(group_dir, exist_ok=True)
    spec = [
        ("diff", lambda r: _normalize_01(r["after_corr"] - r["before_corr"]), RESPONSE_CMAP),
        ("log_ratio", lambda r: _normalize_01(r["log_ratio"]), RESPONSE_CMAP),
        ("top1", lambda r: r["maps"]["top1"], RESPONSE_CMAP),
        ("top3_weighted", lambda r: r["maps"]["top3_weighted"], RESPONSE_CMAP),
        ("top5_rms", lambda r: r["maps"]["top5_rms"], RESPONSE_CMAP),
        ("support_frac", lambda r: r["maps"]["support_frac"], RESPONSE_CMAP),
    ]
    out = {}
    for key, getter, cmap in spec:
        items = []
        for rec in records:
            items.append({"img": getter(rec), "title": str(rec["name"]), "cmap": cmap})
        png = os.path.join(group_dir, f"{group_name.lower()}_{key}_gallery.png")
        save_image_grid(items, png, ncols=4, title=f"{group_name} | {key}")
        out[key] = png
    return out

def export_group_zscore_summary(records, out_dir, mask, group_name="Analyte"):
    if len(records) == 0:
        return None
    feat_names = records[0]["maps"].get("feature_names", [])
    rows = []
    for i, nm in enumerate(feat_names):
        vals = []
        for rec in records:
            z = rec["maps"]["z_stack"][i]
            vals.append(z[mask] if (mask is not None and mask.any()) else z.reshape(-1))
        vals = np.concatenate(vals)
        rows.append({
            "feature_idx": i,
            "feature_name": nm,
            "group_name": group_name,
            "z_mean": float(np.mean(vals)),
            "z_std": float(np.std(vals)),
            "z_q95": float(np.quantile(vals, 0.95)),
            "z_q99": float(np.quantile(vals, 0.99)),
            "z_max": float(np.max(vals)),
            "sig_ratio_gt2": float((vals > 2.0).mean()),
            "sig_ratio_gt3": float((vals > 3.0).mean()),
        })
    df = pd.DataFrame(rows).sort_values("z_q99", ascending=False)
    df.to_csv(os.path.join(out_dir, f"{group_name.lower()}_zscore_group_summary.csv"), index=False)
    return df

def save_feature_value_workbook(out_dir, pbs_summary_df=None, ana_summary_df=None, hand_df=None):
    xlsx = os.path.join(out_dir, "feature_bank_values_summary.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        if pbs_summary_df is not None: pbs_summary_df.to_excel(writer, sheet_name="pbs_feature_summary", index=False)
        if ana_summary_df is not None: ana_summary_df.to_excel(writer, sheet_name="analyte_feature_summary", index=False)
        if hand_df is not None: hand_df.to_excel(writer, sheet_name="handcrafted_imagelevel", index=False)
    return xlsx

def write_results_master_xlsx(out_dir, extra_tables=None):
    extra_tables = extra_tables or {}
    xlsx = os.path.join(out_dir, "results_master.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        for name, obj in extra_tables.items():
            if obj is None:
                continue
            if isinstance(obj, pd.DataFrame):
                obj.to_excel(writer, sheet_name=_sheet_safe(name), index=False)
            elif isinstance(obj, str) and os.path.exists(obj) and obj.lower().endswith(".csv"):
                try:
                    pd.read_csv(obj).to_excel(writer, sheet_name=_sheet_safe(name), index=False)
                except Exception:
                    pass
    return xlsx


def extract_zip_inputs(zip_paths, data_dir):
    data_dir = Path(data_dir)
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for zip_path in zip_paths:
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP not found: {zip_path}")
        if zip_path.suffix.lower() != ".zip":
            raise ValueError(f"Only .zip files are supported here: {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.namelist() if m.strip()]
            prefixes = set(m.split("/")[0] for m in members)
            top_dirs = [p for p in prefixes if p and "." not in p]
            if len(top_dirs) == 1:
                zf.extractall(data_dir)
            else:
                folder_name = zip_path.stem
                target = data_dir / folder_name
                target.mkdir(parents=True, exist_ok=True)
                zf.extractall(target)


def validate_data_tree(cfg):
    required = [cfg["led_subdir"], cfg["pbs_subdir"], cfg["analyte_subdir"], cfg["substrate_subdir"]]
    print("\n[Data check]")
    missing = []
    for folder in required:
        p = Path(cfg["data_dir"]) / folder
        n = len(list(p.iterdir())) if p.exists() else 0
        ok = p.exists() and n > 0
        print(f"  {'OK' if ok else 'FAIL'} {p}  ({n} files)")
        if not ok:
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            "Missing required input folders/files:\n" + "\n".join(missing)
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run biosensor hybrid pipeline locally in PyCharm or terminal.")
    parser.add_argument("--data-dir", default="data", help="Input data root directory.")
    parser.add_argument("--output-dir", default="results_hybrid_v3", help="Output directory.")
    parser.add_argument("--zip", nargs="*", default=[], help="Optional zip file(s) to extract into data-dir before running.")
    parser.add_argument("--run-pca-tsne", action="store_true", help="Run optional PCA/t-SNE visualization stage.")
    parser.add_argument("--run-benchmark", action="store_true", help="Run optional handcrafted classifier benchmark.")
    parser.add_argument("--run-shap", action="store_true", help="Run optional SHAP analysis.")
    parser.add_argument("--zip-output", action="store_true", help="Zip the output directory at the end.")
    return parser.parse_args()

def main():
    args = parse_args()
    CFG["data_dir"] = args.data_dir
    CFG["output_dir"] = args.output_dir
    os.makedirs(CFG["output_dir"], exist_ok=True)

    if args.zip:
        print("[Input] Extracting zip package(s)...")
        extract_zip_inputs(args.zip, CFG["data_dir"])

    validate_data_tree(CFG)


    SIZE = (CFG["img_size"], CFG["img_size"])
    os.makedirs(CFG["output_dir"], exist_ok=True)

    print("[1] Building LED flat-field correction map...")
    led_map = build_led_map(
        os.path.join(CFG["data_dir"], CFG["led_subdir"]),
        target_size=SIZE,
        bright_thr=CFG["led_bright_thr"],
        sigma=CFG["led_smooth_sigma"],
    )
    if led_map is not None:
        plot_led_correction(led_map, CFG, os.path.join(CFG["output_dir"], "led_correction_map.png"))
        export_led_bundle(
            os.path.join(CFG["data_dir"], CFG["led_subdir"]),
            led_map=led_map,
            target_size=SIZE,
            bright_thr=CFG["led_bright_thr"],
            sigma=CFG["led_smooth_sigma"],
            out_dir=CFG["output_dir"],
            cfg=CFG
        )
        print("OK LED correction map and diagnostics generated")
    else:
        print("! LED map unavailable; skipping diagnostics")

    print("\n[1-B] Exporting feature dictionary...")
    feature_dict_info = export_feature_bank_dictionary(CFG["output_dir"], CFG)
    print("OK feature_bank_dictionary.xlsx saved")

    print("\n[2] Loading paired images...")
    pbs_A, pbs_B, pbs_N = load_pairs(
        os.path.join(CFG["data_dir"], CFG["pbs_subdir"]),
        os.path.join(CFG["data_dir"], CFG["substrate_subdir"]),
        CFG["pbs_prefix"], SIZE, led_map=led_map)
    ana_A, ana_B, ana_N = load_pairs(
        os.path.join(CFG["data_dir"], CFG["analyte_subdir"]),
        os.path.join(CFG["data_dir"], CFG["substrate_subdir"]),
        CFG["analyte_prefix"], SIZE, led_map=led_map)

    print(f"  PBS: {len(pbs_N)}  pairs  |  Analyte: {len(ana_N)}  pairs")
    export_led_effect_examples(pbs_B, pbs_A, led_map, CFG["output_dir"], CFG, prefix="pbs")
    export_led_effect_examples(ana_B, ana_A, led_map, CFG["output_dir"], CFG, prefix="analyte")

    pair_manifest = []
    for nm, aft, bef in zip(pbs_N, pbs_A, pbs_B):
        pair_manifest.append({"name": nm, "label": "PBS", "after_shape": str(aft.shape), "before_shape": str(bef.shape)})
    for nm, aft, bef in zip(ana_N, ana_A, ana_B):
        pair_manifest.append({"name": nm, "label": "Analyte", "after_shape": str(aft.shape), "before_shape": str(bef.shape)})
    pair_manifest_df = pd.DataFrame(pair_manifest)

    mask = compute_mask(pbs_B + ana_B, thr=CFG["mask_thr"], erode=CFG["mask_erode"])
    print(f"  Valid pixels: {mask.sum()} / {mask.size}  ({mask.mean():.1%})")
    export_mask_bundle(mask, CFG["output_dir"], CFG)
    print("OK mask_bundle saved")

    print("\n[3] Building PBS pixel-level baseline...")
    pbs_mu_px, pbs_std_px = build_pbs_pixel_baseline(pbs_A, pbs_B, mask, mode=CFG["feature_bank_mode"], cfg=CFG)
    keep_idx, keep_names = _selected_feature_indices(CFG["feature_bank_mode"], CFG)
    print(f"  Feature dimension: {pbs_mu_px.shape[0]}  ({CFG['feature_bank_mode']})")
    baseline_dir, baseline_summary_df = export_baseline_bundle(
        pbs_mu_px, pbs_std_px, mask, keep_names, CFG["output_dir"], CFG
    )
    print("OK pbs_baseline_bundle saved")

    print("\n[4] Generating proposal maps...")
    pbs_prop = [build_score_maps(bef, aft, pbs_mu_px, pbs_std_px, mask, CFG) for aft, bef in zip(pbs_A, pbs_B)]
    ana_prop = [build_score_maps(bef, aft, pbs_mu_px, pbs_std_px, mask, CFG) for aft, bef in zip(ana_A, ana_B)]

    _all_clean = np.concatenate([m["primary"][mask] for m in pbs_prop]) if len(pbs_prop) else np.array([0.0], dtype=np.float32)
    ADAPTIVE_THR = float(np.percentile(_all_clean, CFG["proposal_percentile"]))
    CFG["cam_thr"] = ADAPTIVE_THR * CFG["cam_thr_scale"]
    print(f"  ADAPTIVE_THR={ADAPTIVE_THR:.4f}  -> cam_thr={CFG['cam_thr']:.4f}")

    # handcrafted image-level matrix
    hand_rows = []
    for nm, maps in zip(pbs_N, pbs_prop):
        row = summarize_image_features(maps, mask)
        row["name"] = nm; row["label"] = 0
        hand_rows.append(row)
    for nm, maps in zip(ana_N, ana_prop):
        row = summarize_image_features(maps, mask)
        row["name"] = nm; row["label"] = 1
        hand_rows.append(row)

    hand_df = pd.DataFrame(hand_rows)
    print(f"OK Handcrafted image-level features prepared in memory: {hand_df.shape[0]} samples x {hand_df.shape[1]-2} features")

    # raw pixel debug
    if CFG.get("save_raw_npz", False):
        np.savez_compressed(
            os.path.join(CFG["output_dir"], "proposal_debug_arrays.npz"),
            pbs_mu_px=pbs_mu_px, pbs_std_px=pbs_std_px, mask=mask.astype(np.uint8)
        )

    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    pbs_records = []
    ana_records = []
    all_records = []

    for nm, aft, bef, maps in zip(pbs_N, pbs_A, pbs_B, pbs_prop):
        rec = {
            "name": nm,
            "label": 0,
            "before_corr": bef.astype(np.float32),
            "after_corr": aft.astype(np.float32),
            "log_ratio": (np.log(aft.astype(np.float32) + 1.0) - np.log(bef.astype(np.float32) + 1.0)).astype(np.float32),
            "maps": maps,
        }
        pbs_records.append(rec); all_records.append(rec)

    for nm, aft, bef, maps in zip(ana_N, ana_A, ana_B, ana_prop):
        rec = {
            "name": nm,
            "label": 1,
            "before_corr": bef.astype(np.float32),
            "after_corr": aft.astype(np.float32),
            "log_ratio": (np.log(aft.astype(np.float32) + 1.0) - np.log(bef.astype(np.float32) + 1.0)).astype(np.float32),
            "maps": maps,
        }
        ana_records.append(rec); all_records.append(rec)

    
    pbs_z_df = export_group_zscore_summary(pbs_records, CFG["output_dir"], mask, "PBS")
    ana_z_df = export_group_zscore_summary(ana_records, CFG["output_dir"], mask, "Analyte")
    save_feature_value_workbook(CFG["output_dir"], pbs_summary_df=pbs_z_df, ana_summary_df=ana_z_df, hand_df=hand_df)

    
    pbs_gallery_paths = save_group_map_galleries(pbs_records, CFG["output_dir"], "PBS")
    ana_gallery_paths = save_group_map_galleries(ana_records, CFG["output_dir"], "Analyte")

    
    _pbs_top3  = np.concatenate([m["primary"][mask] for m in pbs_prop]) if len(pbs_prop) else np.array([0.0])
    _ana_top3  = np.concatenate([m["primary"][mask] for m in ana_prop]) if len(ana_prop) else np.array([0.0])
    _clip_max  = float(np.percentile(np.concatenate([_pbs_top3, _ana_top3]), 99.5))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    axes[0].hist(_pbs_top3.clip(-2, _clip_max), bins=80, alpha=0.6,
                 color="#4C72B0", label=f"PBS  (n={len(pbs_N)} imgs)", density=True)
    axes[0].hist(_ana_top3.clip(-2, _clip_max), bins=80, alpha=0.6,
                 color="#DD8452", label=f"Analyte  (n={len(ana_N)} imgs)", density=True)
    axes[0].axvline(ADAPTIVE_THR, color="red", linestyle="--", linewidth=1.5,
                    label=f"cam_thr = {ADAPTIVE_THR:.3f}")
    axes[0].set_xlabel(f"{CFG['score_primary']} Z-score (pixel)", fontsize=11)
    axes[0].set_ylabel("Density", fontsize=11)
    axes[0].set_title("Pixel-level Response Distribution", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.25); axes[0].spines[["top","right"]].set_visible(False)

    for arr, label, col in [(_pbs_top3, "PBS", "#4C72B0"), (_ana_top3, "Analyte", "#DD8452")]:
        s = np.sort(arr.clip(-2, _clip_max))
        cdf = np.arange(1, len(s)+1) / len(s)
        axes[1].plot(s, cdf, label=label, color=col, linewidth=1.5)
    axes[1].axvline(ADAPTIVE_THR, color="red", linestyle="--", linewidth=1.5,
                    label=f"cam_thr")
    axes[1].set_xlabel(f"{CFG['score_primary']} Z-score", fontsize=11)
    axes[1].set_ylabel("CDF", fontsize=11)
    axes[1].set_title("Cumulative Distribution", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.25); axes[1].spines[["top","right"]].set_visible(False)

    plt.suptitle(f"Physics-guided Proposal - Stage 1 Statistics  [{CFG['feature_bank_mode']}]",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG["output_dir"], "proposal_zscore_distribution.png"),
                dpi=CFG["dpi"], bbox_inches="tight")
    plt.close()

    print("OK Proposal galleries, group z-score summaries, and baseline saved")
    print("OK proposal_zscore_distribution.png saved")

    
    
    for rec in all_records:
        rec["maps"].pop("feature_stack", None)
    for rec in pbs_records:
        rec["maps"].pop("feature_stack", None)
    for rec in ana_records:
        rec["maps"].pop("feature_stack", None)

    del pbs_prop, ana_prop
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("OK Released proposal arrays (z_stack / feature_stack)")

    if args.run_pca_tsne:

        
        rows = []
        X = []
        for nm, aft, bef in zip(pbs_N, pbs_A, pbs_B):
            vec = np.concatenate([_normalize_01(bef).ravel(), _normalize_01(aft).ravel(), _normalize_01(aft-bef).ravel()])
            X.append(vec); rows.append({"name": nm, "group": "PBS"})
        for nm, aft, bef in zip(ana_N, ana_A, ana_B):
            vec = np.concatenate([_normalize_01(bef).ravel(), _normalize_01(aft).ravel(), _normalize_01(aft-bef).ravel()])
            X.append(vec); rows.append({"name": nm, "group": "Analyte"})

        # substrate only
        sub_files = _list_images(os.path.join(CFG["data_dir"], CFG["substrate_subdir"]))
        sub_files = [f for f in sub_files if not (os.path.basename(f).startswith(CFG["pbs_prefix"]) or os.path.basename(f).startswith(CFG["analyte_prefix"]))]
        for f in sub_files[:min(100, len(sub_files))]:
            img = _resize(_img_read_gray(f), CFG["img_size"])
            if led_map is not None:
                img = img / (led_map + 1e-6)
            vec = np.concatenate([_normalize_01(img).ravel(), _normalize_01(img).ravel(), np.zeros_like(img).ravel()])
            X.append(vec); rows.append({"name": os.path.basename(f), "group": "Substrate"})

        X = np.stack(X, 0)
        meta = pd.DataFrame(rows)
        Xs = StandardScaler().fit_transform(X)
        pca = PCA(n_components=5, random_state=CFG["random_state"])
        Xp = pca.fit_transform(Xs)
        tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(X)//5)), random_state=CFG["random_state"], init="pca")
        Xt = tsne.fit_transform(Xp[:,:min(20, Xp.shape[1])])

        pca_df = meta.copy()
        for i in range(Xp.shape[1]):
            pca_df[f"PC{i+1}"] = Xp[:,i]
        pca_df.to_csv(os.path.join(CFG["output_dir"], "pca_scores.csv"), index=False)

        tsne_df = meta.copy()
        tsne_df["tSNE1"] = Xt[:,0]; tsne_df["tSNE2"] = Xt[:,1]
        tsne_df.to_csv(os.path.join(CFG["output_dir"], "tsne_scores.csv"), index=False)

        fig, axes = plt.subplots(1,3, figsize=(16,4))
        for g, c in zip(["Substrate","PBS","Analyte"], ["gray","#4C72B0","#DD8452"]):
            idx = meta["group"] == g
            axes[0].scatter(Xp[idx,0], Xp[idx,1], s=18, alpha=.75, label=g, c=c)
            axes[1].scatter(Xp[idx,0], Xp[idx,2], s=18, alpha=.75, label=g, c=c)
            axes[2].scatter(Xt[idx,0], Xt[idx,1], s=18, alpha=.75, label=g, c=c)
        axes[0].set_title("PCA: PC1 vs PC2")
        axes[1].set_title("PCA: PC1 vs PC3")
        axes[2].set_title("t-SNE")
        for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(CFG["output_dir"], "pca_tsne.png"), dpi=CFG["dpi"], bbox_inches="tight")
        plt.close()

        pd.DataFrame({
            "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_
        }).to_csv(os.path.join(CFG["output_dir"], "pca_explained_variance.csv"), index=False)
        print("OK PCA / t-SNE saved")

    if args.run_benchmark:
        # =========================================================
        
        
        # =========================================================
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        import time

        all_names  = pbs_N + ana_N
        all_groups = np.array([
            _extract_numeric_id(nm) if _extract_numeric_id(nm) is not None else i
            for i, nm in enumerate(all_names)
        ])
        feature_cols = [c for c in hand_df.columns if c not in ["name","label"]]
        X_hand = hand_df[feature_cols].values.astype(np.float32)
        y_all  = hand_df["label"].values.astype(int)

        gkf = GroupKFold(n_splits=CFG["n_splits"])
        bench_rows = []

        shallow_registry = {
            "handcrafted+LightGBM": lambda: lgb.LGBMClassifier(
                n_estimators=120, learning_rate=0.08, num_leaves=15, max_depth=4,
                subsample=0.9, colsample_bytree=0.8,
                random_state=CFG["random_state"], verbose=-1
            ),
            "handcrafted+SVM": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", probability=True,
                            class_weight="balanced", random_state=CFG["random_state"]))
            ]),
            "handcrafted+MLP": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(
                    hidden_layer_sizes=(128,), activation="relu",
                    alpha=1e-4, max_iter=300,
                    early_stopping=True, random_state=CFG["random_state"]
                ))
            ]),
        }

        for model_name, model_fn in shallow_registry.items():
            print(f"\n[Benchmark] {model_name}")
            fold_aucs = []
            for fold, (tr_idx, val_idx) in enumerate(
                    gkf.split(np.arange(len(y_all)), y_all, all_groups), 1):
                t1 = time.time()
                mdl = model_fn()
                mdl.fit(X_hand[tr_idx], y_all[tr_idx])
                prob = mdl.predict_proba(X_hand[val_idx])[:, 1]
                pred = (prob >= 0.5).astype(int)
                m = compute_metrics(y_all[val_idx], pred, prob)
                bench_rows.append({"model": model_name, "fold": fold,
                                   "seconds": time.time()-t1, **m})
                fold_aucs.append(m["roc_auc"])
                print(f"  fold{fold}: AUC={m['roc_auc']:.4f}  Acc={m['accuracy']:.4f}")
            print(f"  => Mean AUC={np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")

        bench_df = pd.DataFrame(bench_rows)
        bench_df.to_csv(os.path.join(CFG["output_dir"], "benchmark_handcrafted.csv"), index=False)
        print("\nOK Benchmark complete; resultssaved")
        print(bench_df.groupby("model")["roc_auc"].agg(["mean","std"]).round(4).to_string())

    # =========================================================
    
    # =========================================================
    import joblib

    final_clf_dir = os.path.join(CFG["output_dir"], "final_classifier")
    os.makedirs(final_clf_dir, exist_ok=True)

    feature_cols = [c for c in hand_df.columns if c not in ["name", "label"]]
    X_hand = hand_df[feature_cols].values.astype(np.float32)
    y_all  = hand_df["label"].values.astype(int)

    
    all_names  = pbs_N + ana_N
    all_groups = np.array([
        _extract_numeric_id(nm) if _extract_numeric_id(nm) is not None else i
        for i, nm in enumerate(all_names)
    ])
    gkf = GroupKFold(n_splits=CFG["n_splits"])
    cv_rows = []
    for fold, (tr_idx, val_idx) in enumerate(
            gkf.split(np.arange(len(y_all)), y_all, all_groups), 1):
        clf_cv = lgb.LGBMClassifier(
            n_estimators=120, learning_rate=0.08, num_leaves=15, max_depth=4,
            subsample=0.9, colsample_bytree=0.8,
            random_state=CFG["random_state"], verbose=-1
        )
        clf_cv.fit(X_hand[tr_idx], y_all[tr_idx])
        prob = clf_cv.predict_proba(X_hand[val_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        m = compute_metrics(y_all[val_idx], pred, prob)
        cv_rows.append({"fold": fold, **m})
        print(f"  Fold {fold}: AUC={m['roc_auc']:.4f}  Acc={m['accuracy']:.4f}")

    cv_df = pd.DataFrame(cv_rows)
    print(f"\n  Mean AUC = {cv_df['roc_auc'].mean():.4f} +/- {cv_df['roc_auc'].std():.4f}")
    cv_df.to_csv(os.path.join(final_clf_dir, "lgbm_cv_results.csv"), index=False)

    
    final_clf = lgb.LGBMClassifier(
        n_estimators=120, learning_rate=0.08, num_leaves=15, max_depth=4,
        subsample=0.9, colsample_bytree=0.8,
        random_state=CFG["random_state"], verbose=-1
    )
    final_clf.fit(X_hand, y_all)
    joblib.dump(final_clf, os.path.join(final_clf_dir, "best_classifier.joblib"))

    meta = {
        "best_model_name": "handcrafted+LightGBM",
        "kind": "shallow_handcrafted",
        "feature_cols": feature_cols,
    }
    with open(os.path.join(final_clf_dir, "best_classifier_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("OK Full-data handcrafted+LightGBM training complete; saved")

    if args.run_shap:
        import shap

        
        feature_cols = [c for c in hand_df.columns if c not in ["name","label"]]
        X_hand = hand_df[feature_cols].values.astype(np.float32)
        y_hand = hand_df["label"].values.astype(int)

        gbm = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=CFG["random_state"],
            verbosity=-1
        )
        gbm.fit(X_hand, y_hand)

        explainer = shap.TreeExplainer(gbm)
        shap_vals = explainer.shap_values(X_hand)
        sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

        mean_abs = np.abs(sv).mean(0)
        mean_shap = sv.mean(0)
        std_shap = np.abs(sv).std(0)

        top_n = 20
        top_idx = np.argsort(mean_abs)[::-1][:top_n]
        shap_df = pd.DataFrame({
            "feature": np.array(feature_cols)[top_idx],
            "mean_abs_shap": mean_abs[top_idx],
            "mean_shap": mean_shap[top_idx],
            "std_shap": std_shap[top_idx],
        })
        shap_df.to_csv(os.path.join(CFG["output_dir"], "shap_top20.csv"), index=False)

        top10_names = list(shap_df.head(10)["feature"])
        top10_cols_idx = [list(feature_cols).index(f) for f in top10_names]
        sv_top10 = sv[:, top10_cols_idx]
        X_top10 = X_hand[:, top10_cols_idx]

        plt.style.use("default")
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="#0a0d18")
        for ax in axes:
            ax.set_facecolor("#171a2e")
            ax.tick_params(colors="white", labelsize=10)
            for spine in ax.spines.values():
                spine.set_color("white")
            ax.grid(axis="x", color="white", alpha=0.12)

        # Left: bar
        bar_df = shap_df.head(10).iloc[::-1].reset_index(drop=True)
        colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(bar_df)))
        bars = axes[0].barh(bar_df["feature"], bar_df["mean_abs_shap"], color=colors, edgecolor="white", alpha=0.9)
        for bar, val in zip(bars, bar_df["mean_abs_shap"]):
            axes[0].text(val + 0.005*np.max(bar_df["mean_abs_shap"]), bar.get_y()+bar.get_height()/2,
                         f"{val:.4f}", va="center", color="white", fontsize=8)
        axes[0].set_xlabel("Mean |SHAP value|", color="white", fontsize=11)
        axes[0].set_title("Top 10 Feature Importance (SHAP)\n(Higher = more important for PBS/Analyte discrimination)",
                          color="white", fontsize=13, fontweight="bold", pad=10)

        # Right: beeswarm-like scatter
        feature_labels_top10 = top10_names[::-1]
        y_positions = np.arange(len(feature_labels_top10))
        for fi, fname in enumerate(feature_labels_top10):
            idx = top10_names.index(fname)
            sv_col = sv_top10[:, idx]
            x_col = X_top10[:, idx]
            xn = (x_col - x_col.min()) / (np.ptp(x_col) + 1e-8)
            jitter = np.random.normal(0, 0.085, len(sv_col))
            sc = axes[1].scatter(
                sv_col,
                y_positions[fi] + jitter,
                c=xn,
                cmap="coolwarm",
                s=14,
                alpha=0.85,
                vmin=0,
                vmax=1,
                edgecolors="none"
            )
        axes[1].axvline(0, color="white", linewidth=1.0, alpha=0.45)
        axes[1].set_yticks(y_positions)
        axes[1].set_yticklabels(feature_labels_top10, color="white", fontsize=10)
        axes[1].set_xlabel("SHAP value (impact on Analyte prediction)", color="white", fontsize=11)
        axes[1].set_title("SHAP Beeswarm\n(Red=high feature value  Blue=low feature value)",
                          color="white", fontsize=13, fontweight="bold", pad=10)

        fig.suptitle("SHAP Feature Importance Analysis", color="white", fontsize=17, fontweight="bold", y=0.99)
        plt.tight_layout()
        plt.savefig(os.path.join(CFG["output_dir"], "shap_summary_dark.png"),
                    dpi=CFG["dpi"], bbox_inches="tight", facecolor="#0a0d18")
        plt.close()

        print("OK SHAP complete")
        print("  OK shap_summary_dark.png")
        print(shap_df.head(10).to_string(index=False))

    import copy
    # =========================================================
    
    
    # =========================================================
    import torch.nn.functional as F_nn

    
    class _ResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
            )
            self.relu = nn.ReLU(inplace=True)
        def forward(self, x):
            return self.relu(self.block(x) + x)

    class ResNet18_29(nn.Module):
        'Map 29-channel feature images to a one-channel logit map at the input resolution.'
        def __init__(self, in_ch=29):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            )
            self.layer1 = nn.Sequential(_ResBlock(64), _ResBlock(64))
            self.layer2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                _ResBlock(128), _ResBlock(128),
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                _ResBlock(256),
            )
            # decoder
            self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dec3 = _ResBlock(128)
            self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec2 = _ResBlock(64)
            self.head = nn.Conv2d(64, 1, 1)

        def forward(self, x):
            s  = self.stem(x)       # [B,64,H,W]
            l1 = self.layer1(s)     # [B,64,H,W]
            l2 = self.layer2(l1)    # [B,128,H/2,W/2]
            l3 = self.layer3(l2)    # [B,256,H/4,W/4]
            d3 = self.dec3(self.up3(l3) + l2)
            d2 = self.dec2(self.up2(d3) + l1)
            return self.head(d2)    # [B,1,H,W]

    
    def extract_feature_bank_29(bef, aft):
        from scipy.ndimage import gaussian_filter as _gf
        
        def _to01(x):
            x = x.astype(np.float32)
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-8)
        b = _to01(bef)
        a = _to01(aft)
        d = a - b; eps = 1e-6
        def _lstd(x, w):
            mu = _gf(x,w/2); mu2 = _gf(x**2,w/2)
            return np.sqrt(np.maximum(mu2-mu**2, 0))
        def _sc(x,w): return _lstd(x,w)/(_gf(x,w/2)+1e-6)
        lb=_gf(b,4); hb=np.abs(b-lb)
        la=_gf(a,4); ha=np.abs(a-la)
        feats = [
            d, np.abs(d),
            np.log1p(a+eps)-np.log1p(b+eps),
            a/(b+eps)-1, np.clip(d/(b+eps),-5,5),
            _gf(d,1), _gf(d,3), _gf(d,6),
            _lstd(a,3)-_lstd(b,3), _lstd(a,7)-_lstd(b,7), _lstd(a,15)-_lstd(b,15),
            _sc(a,5)-_sc(b,5), _sc(a,9)-_sc(b,9),
            la-lb, ha-hb, ha**2-hb**2,
        ]
        for s in [1,2,4]:
            feats.append((a-_gf(a,s*2))-(b-_gf(b,s*2)))
        feats += [
            b, a, _gf(b,2), _gf(a,2),
            _lstd(d,3), _lstd(d,7),
            np.abs(_gf(d,2)-_gf(d,6)),
            a**2-b**2, _gf(np.abs(d),3), np.maximum(d,0),
        ]
        assert len(feats) == 29
        return np.stack(feats, 0).astype(np.float32)

    
    def build_pbs_baseline_29(pbs_after_list, pbs_before_list, mask):
        'Compute per-pixel means and standard deviations of the 29 PBS feature maps.'
        Fs = [extract_feature_bank_29(b, a)
              for a, b in zip(pbs_after_list, pbs_before_list)]
        stk = np.stack(Fs, 0)           # [N,29,H,W]
        mu  = stk.mean(axis=0)          # [29,H,W]
        std_raw = stk.std(axis=0, ddof=1) if stk.shape[0] > 1 else np.zeros_like(mu)
        
        floor = np.percentile(std_raw[:, mask], 5, axis=1)[:, None, None]  # [29,1,1]
        std   = np.maximum(std_raw, floor)
        return mu.astype(np.float32), std.astype(np.float32)


    def _make_soft_target(bef, aft, pbs_mu29, pbs_std29, mask, support_z_thr=2.0):
        'Construct a soft target from PBS-referenced feature z-scores using weighted top responses and feature-support gating.'
        feat = extract_feature_bank_29(bef, aft)           # [29,H,W]
        z    = (feat - pbs_mu29) / (pbs_std29 + 1e-6)     # [29,H,W]
        z    = np.where(mask[None], z, 0.0)

        z_sorted = np.sort(z, axis=0)[::-1]               
        top1     = z_sorted[0]
        top3     = z_sorted[:3].mean(0)
        top5     = z_sorted[:5].mean(0)
        support  = (z > support_z_thr).sum(0).astype(np.float32)  # [H,W]

        def _n01(x):
            mn, mx = float(x.min()), float(x.max())
            if mx - mn < 1e-8: return np.zeros_like(x, dtype=np.float32)
            return ((x - mn) / (mx - mn)).astype(np.float32)

        top1_n    = _n01(np.clip(top1,  0, None))
        top3_n    = _n01(np.clip(top3,  0, None))
        top5_n    = _n01(np.clip(top5,  0, None))
        support_n = _n01(support)

        fused = _n01(
            (0.20*top1_n + 0.50*top3_n + 0.30*top5_n) * (0.65 + 0.35*support_n)
        )
        return fused.astype(np.float32)

    
    def _tv_loss(x):
        return (torch.abs(x[:,:,1:,:]-x[:,:,:-1,:]).mean() +
                torch.abs(x[:,:,:,1:]-x[:,:,:,:-1]).mean())

    def _refine_loss(logit, target, tv_w):
        pred = torch.sigmoid(logit).squeeze(1)
        return F.smooth_l1_loss(pred, target) + tv_w * _tv_loss(logit)

    
    class Refine29Dataset(Dataset):
        def __init__(self, after_list, before_list, labels,
                     pbs_mu, pbs_std, mask, cfg, cm=None, cs=None,
                     records=None):
            self.after   = after_list
            self.before  = before_list
            self.labels  = labels
            self.pbs_mu  = pbs_mu
            self.pbs_std = pbs_std
            self.mask    = mask
            self.cfg     = cfg
            self.cm      = cm        # channel mean [29]
            self.cs      = cs        # channel std  [29]
            self.records = records   

        def __len__(self): return len(self.labels)

        def __getitem__(self, idx):
            aft  = self.after[idx]
            bef  = self.before[idx]
            lbl  = self.labels[idx]
            feat = extract_feature_bank_29(bef, aft)   # [29,H,W]
            
            if self.cm is not None:
                feat = (feat - self.cm[:,None,None]) / (self.cs[:,None,None] + 1e-6)
            
            if lbl == 1 and self.pbs_mu is not None:
                soft = _make_soft_target(bef, aft, self.pbs_mu, self.pbs_std,
                                         self.mask, support_z_thr=2.0)
            else:
                soft = np.zeros(feat.shape[1:], np.float32)
            return (torch.from_numpy(feat),
                    torch.from_numpy(soft),
                    torch.tensor(lbl, dtype=torch.long))

    
    def _compute_channel_stats(after_list, before_list):
        
        feats = [extract_feature_bank_29(b, a)
                 for a, b in zip(after_list, before_list)]
        stk = np.stack(feats, 0)   # [N,29,H,W]
        cm  = stk.mean(axis=(0,2,3)).astype(np.float32)
        cs  = stk.std (axis=(0,2,3)).astype(np.float32)
        return cm, cs

    
    def run_stage2a(pbs_mu_px, pbs_std_px):
        'Run stage2a.'
        all_after  = pbs_A + ana_A
        all_before = pbs_B + ana_B
        all_labels = [0]*len(pbs_A) + [1]*len(ana_A)
        all_names_loc = pbs_N + ana_N

        all_groups = np.array([
            _extract_numeric_id(nm) if _extract_numeric_id(nm) is not None else i
            for i, nm in enumerate(all_names_loc)
        ])
        gkf = GroupKFold(n_splits=CFG["n_splits"])
        tr_idx, val_idx = next(gkf.split(
            np.arange(len(all_labels)), all_labels, all_groups))

        print(f"  Stage2a: train={len(tr_idx)}  val={len(val_idx)}")

        
        pbs_after_list  = [all_after[i]  for i, l in enumerate(all_labels) if l == 0]
        pbs_before_list = [all_before[i] for i, l in enumerate(all_labels) if l == 0]
        pbs_mu29, pbs_std29 = build_pbs_baseline_29(pbs_after_list, pbs_before_list, mask)
        print(f"  29-channel PBS baseline: shape={pbs_mu29.shape}")

        
        cm, cs = _compute_channel_stats(all_after, all_before)

        
        all_recs = [r for r in pbs_records] + [r for r in ana_records]

        def _make_ds(idx_list):
            return Refine29Dataset(
                [all_after[i]  for i in idx_list],
                [all_before[i] for i in idx_list],
                [all_labels[i] for i in idx_list],
                pbs_mu29, pbs_std29, mask, CFG, cm, cs,  
                records=[all_recs[i] for i in idx_list],
            )

        tr_ld = DataLoader(_make_ds(tr_idx), CFG["refine_batch_size"],
                           shuffle=True,  num_workers=0)
        va_ld = DataLoader(_make_ds(val_idx), CFG["refine_batch_size"],
                           shuffle=False, num_workers=0)

        model = ResNet18_29(29).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(),
                                 lr=CFG["refine_lr"], weight_decay=CFG["refine_wd"])
        sch   = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=CFG["refine_epochs"], eta_min=CFG["refine_lr"]/20)

        best_sc, wait, best_st = -1e9, 0, None
        for ep in range(1, CFG["refine_epochs"]+1):
            model.train()
            for feat, soft, _ in tr_ld:
                feat, soft = feat.to(DEVICE), soft.to(DEVICE)
                opt.zero_grad()
                _refine_loss(model(feat), soft, CFG["tv_weight"]).backward()
                opt.step()
            sch.step()

            
            model.eval()
            scores_ana, scores_pbs = [], []
            with torch.no_grad():
                for feat, soft, lbl in va_ld:
                    feat = feat.to(DEVICE)
                    hm = torch.sigmoid(model(feat)).squeeze(1).cpu().numpy()
                    for ib, lb in enumerate(lbl.tolist()):
                        flat = hm[ib].flatten()
                        
                        k = max(1, int(len(flat)*0.20))
                        sc = np.partition(flat, -k)[-k:].mean()
                        (scores_ana if lb==1 else scores_pbs).append(sc)
            sep = (np.mean(scores_ana) if scores_ana else 0.0) -               0.5*(np.mean(scores_pbs) if scores_pbs else 1.0)
            print(f"    ep{ep:3d}: ana={np.mean(scores_ana) if scores_ana else 0:.3f}"
                  f"  pbs={np.mean(scores_pbs) if scores_pbs else 1:.3f}  sep={sep:.4f}")
            if sep > best_sc:
                best_sc = sep; wait = 0
                best_st = {k:v.clone() for k,v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= CFG["refine_patience"]:
                    print(f"    Early stopping at epoch {ep}"); break

        if best_st:
            model.load_state_dict(best_st)

        ckpt_path = os.path.join(CFG["output_dir"], "stage2a_refine.pt")
        torch.save({
            "model_state" : model.state_dict(),
            "cfg"         : CFG,
            "channel_mean": cm.tolist(),
            "channel_std" : cs.tolist(),
            "pbs_mu_px"   : pbs_mu_px,    
            "pbs_std_px"  : pbs_std_px,   
            "pbs_mu29"    : pbs_mu29,     
            "pbs_std29"   : pbs_std29,    
        }, ckpt_path)
        print(f"OK stage2a_refine.pt saved  (best_sep={best_sc:.4f})")
        return model, cm, cs, pbs_mu29, pbs_std29  

    refine_model_2a, ch_mean, ch_std, pbs_mu29, pbs_std29 = run_stage2a(pbs_mu_px, pbs_std_px)

    
    
    
    if CFG.get("run_stage2b", False):
        print("[Stage 2b] Full-dataset fine-tuning of ResNet18_29...")
        all_after  = pbs_A + ana_A
        all_before = pbs_B + ana_B
        all_labels = [0]*len(pbs_A) + [1]*len(ana_A)
        all_recs_full = [r for r in pbs_records] + [r for r in ana_records]
        full_ds = Refine29Dataset(
            all_after, all_before, all_labels,
            pbs_mu29, pbs_std29, mask, CFG, ch_mean, ch_std,  
            records=all_recs_full,
        )
        full_ld = DataLoader(full_ds, CFG["refine_batch_size"], shuffle=True, num_workers=0)
        model_2b = copy.deepcopy(refine_model_2a)
        opt2 = torch.optim.Adam(model_2b.parameters(),
                                lr=CFG["refine_lr"]/5, weight_decay=CFG["refine_wd"])
        for ep in range(1, 11):
            model_2b.train()
            for feat, soft, _ in full_ld:
                feat, soft = feat.to(DEVICE), soft.to(DEVICE)
                opt2.zero_grad()
                _refine_loss(model_2b(feat), soft, CFG["tv_weight"]).backward()
                opt2.step()
            print(f"    2b ep{ep:02d} done")
        ckpt_2b = os.path.join(CFG["output_dir"], "stage2b_refine.pt")
        torch.save({
            "model_state" : model_2b.state_dict(),
            "cfg"         : CFG,
            "channel_mean": ch_mean.tolist(),
            "channel_std" : ch_std.tolist(),
            "pbs_mu_px"   : pbs_mu_px,
            "pbs_std_px"  : pbs_std_px,
            "pbs_mu29"    : pbs_mu29,
            "pbs_std29"   : pbs_std29,
        }, ckpt_2b)
        refine_model_2a = model_2b
        print(f"OK stage2b_refine.pt saved")
    else:
        print("[Stage 2b] Skipped (CFG[\'run_stage2b\']=False)")

    # =========================================================
    
    # =========================================================
    import joblib
    import copy as _copy

    print("[8] Final inference...")

    final_clf_dir = os.path.join(CFG["output_dir"], "final_classifier")
    final_dir     = os.path.join(CFG["output_dir"], "final_visualizations")
    os.makedirs(final_dir, exist_ok=True)

    
    final_clf = joblib.load(os.path.join(final_clf_dir, "best_classifier.joblib"))
    feature_cols = [c for c in hand_df.columns if c not in ["name","label"]]

    def predict_prob(rec):
        row = summarize_image_features(rec["maps"], mask)
        x   = pd.DataFrame([row])[feature_cols].values.astype(np.float32)
        return float(final_clf.predict_proba(x)[:, 1][0])

    
    ckpt = torch.load(
        os.path.join(CFG["output_dir"], "stage2a_refine.pt"),
        map_location=DEVICE,
        weights_only=False,
    )
    refine_model_inf = ResNet18_29(29).to(DEVICE)
    refine_model_inf.load_state_dict(ckpt["model_state"])
    refine_model_inf.eval()

    _cm_inf = np.array(ckpt["channel_mean"], dtype=np.float32)
    _cs_inf = np.array(ckpt["channel_std"],  dtype=np.float32)
    _pbs_mu_inf  = ckpt["pbs_mu_px"]
    _pbs_std_inf = ckpt["pbs_std_px"]

    def refine_one(rec):
        feat = extract_feature_bank_29(rec["before_corr"], rec["after_corr"])
        feat = (feat - _cm_inf[:,None,None]) / (_cs_inf[:,None,None] + 1e-6)
        x = torch.from_numpy(feat[None]).to(DEVICE)
        with torch.no_grad():
            hm = torch.sigmoid(refine_model_inf(x))[0,0].cpu().numpy()
        mn, mx = hm.min(), hm.max()
        if mx - mn < 1e-8:
            return np.zeros_like(hm, np.float32)
        return ((hm - mn) / (mx - mn)).astype(np.float32)

    
    temp = []
    for rec in all_records:
        prob    = predict_prob(rec)
        refined = refine_one(rec)
        temp.append({
            "name"      : rec["name"],
            "label"     : "Analyte" if rec["label"]==1 else "PBS",
            "prob"      : prob,
            "before"    : rec["before_corr"],
            "after"     : rec["after_corr"],
            "coarse"    : rec["maps"][CFG["score_primary"]],
            "refined"   : refined,
        })

    pbs_coarse_vals  = np.concatenate([r["coarse"][mask]  for r in temp if r["label"]=="PBS"])                    if any(r["label"]=="PBS" for r in temp)                    else np.array([0.], np.float32)
    pbs_refined_vals = np.concatenate([r["refined"][mask] for r in temp if r["label"]=="PBS"])                    if any(r["label"]=="PBS" for r in temp)                    else np.array([0.], np.float32)

    coarse_thr  = float(np.percentile(pbs_coarse_vals,  CFG["response_percentile_coarse"]))
    refined_thr = float(np.percentile(pbs_refined_vals, CFG["response_percentile_refined"]))
    print(f"  coarse thr={coarse_thr:.4f}  refined thr={refined_thr:.4f}")

    
    rows = []
    ana_stage_items, pbs_stage_items = [], []
    ana_resp_items,  pbs_resp_items  = [], []

    OVR_THR = CFG.get("overlay_thr_pct", 85)  

    for r in temp:
        nm = r["name"]
        coarse_mask  = (r["coarse"]  > coarse_thr)  & mask
        refined_mask = (r["refined"] > refined_thr) & mask

        
        
        if r["label"] == "Analyte":
            heat = r["refined"]
            thr_fallback = float(np.percentile(heat[mask], OVR_THR))
            refined_mask_show = (heat > thr_fallback) & mask
            if not refined_mask_show.any():
                
                flat = heat[mask].flatten()
                k = max(1, int(len(flat)*0.01))
                min_val = np.partition(flat, -k)[-k:].min()
                refined_mask_show = (heat >= min_val) & mask
        else:
            refined_mask_show = refined_mask

        diff_vis = _normalize_01(r["after"]  - r["before"])
        log_vis  = _normalize_01(np.log(r["after"]+1) - np.log(r["before"]+1))

        stage_tile = build_compact_stage_tile(
            r["before"], r["after"], diff_vis, log_vis,
            r["coarse"], r["refined"]
        )
        resp_tile = build_response_only_tile(
            r["after"], r["coarse"], r["refined"],
            coarse_mask, refined_mask_show
        )

        row = {
            "name"              : nm,
            "label"             : r["label"],
            "classifier"        : "handcrafted+LightGBM",
            "prob_analyte"      : r["prob"],
            "coarse_max"        : float(np.max(r["coarse"])),
            "coarse_q95"        : float(np.quantile(r["coarse"][mask], 0.95)),
            "refined_max"       : float(np.max(r["refined"])),
            "refined_q95"       : float(np.quantile(r["refined"][mask], 0.95)),
            "coarse_thr"        : coarse_thr,
            "refined_thr"       : refined_thr,
            "coarse_area_pct"   : float(100.0 * coarse_mask.sum()  / max(mask.sum(),1)),
            "refined_area_pct"  : float(100.0 * refined_mask_show.sum() / max(mask.sum(),1)),
            "area_delta_pct"    : float(100.0 * (refined_mask_show.sum()-coarse_mask.sum()) / max(mask.sum(),1)),
        }
        rows.append(row)

        item_s = {"img": stage_tile, "title": f"{nm} | p={r['prob']:.2f}"}
        item_r = {"img": resp_tile,  "title": f"{nm} | coarse->refined"}

        if r["label"] == "Analyte":
            ana_stage_items.append(item_s); ana_resp_items.append(item_r)
        else:
            pbs_stage_items.append(item_s); pbs_resp_items.append(item_r)

    result_df = pd.DataFrame(rows).sort_values(["label","name"]).reset_index(drop=True)
    result_df.to_csv(os.path.join(final_dir, "final_prediction_summary.csv"), index=False)

    if ana_stage_items:
        save_image_grid(ana_stage_items,
            os.path.join(final_dir, "analyte_final_stage_gallery.png"),
            ncols=CFG.get("final_gallery_cols",3), title="Analyte - Final stage")
    if pbs_stage_items:
        save_image_grid(pbs_stage_items,
            os.path.join(final_dir, "pbs_final_stage_gallery.png"),
            ncols=CFG.get("final_gallery_cols",3), title="PBS - Final stage")
    if ana_resp_items:
        save_image_grid(ana_resp_items,
            os.path.join(final_dir, "analyte_response_compare_gallery.png"),
            ncols=CFG.get("final_gallery_cols",3), title="Analyte - coarse vs refined")
    if pbs_resp_items:
        save_image_grid(pbs_resp_items,
            os.path.join(final_dir, "pbs_response_compare_gallery.png"),
            ncols=CFG.get("final_gallery_cols",3), title="PBS - coarse vs refined")

    plot_response_area_comparison(result_df, os.path.join(final_dir, "response_area_comparison.png"))

    
    
    
    HMAP_THR  = 0.5
    TOP_K     = 0.05   

    def _img_score(h):
        flat = h.flatten()
        k = max(1, int(len(flat) * TOP_K))
        return float(np.partition(flat, -k)[-k:].mean())

    hm_pbs_refined = [r["refined"] for r in temp if r["label"] == "PBS"]
    hm_ana_refined = [r["refined"] for r in temp if r["label"] == "Analyte"]
    hm_pbs_coarse  = [r["coarse"]  for r in temp if r["label"] == "PBS"]
    hm_ana_coarse  = [r["coarse"]  for r in temp if r["label"] == "Analyte"]
    ana_recs_order = [rec for rec in all_records if rec["label"] == 1]

    def compute_heatmap_metrics(hm_pbs, hm_ana, records_ana, label="refined"):
        
        pbs_fpr = float(np.mean([(h > HMAP_THR).mean() for h in hm_pbs])) if hm_pbs else np.nan

        
        seed_rets = []
        for h, rec in zip(hm_ana, records_ana):
            sup = rec["maps"].get("support_frac",
                  rec["maps"].get("multi_feature_support_count_map", None))
            if sup is None: continue
            mn, mx = sup.min(), sup.max()
            sup_n = (sup - mn) / (mx - mn + 1e-8)
            smask = sup_n >= np.percentile(sup_n, 90)
            if not smask.any(): continue
            seed_rets.append(h[smask].mean() / (h.mean() + 1e-8))
        ana_seed = float(np.mean(seed_rets)) if seed_rets else np.nan

        
        bg_sups = []
        for h in hm_ana:
            lo = np.percentile(h, 50); hi = np.percentile(h, 90)
            bot = h[h <= lo].mean() if (h <= lo).any() else 0.0
            top = h[h >= hi].mean() if (h >= hi).any() else 0.0
            bg_sups.append(float(top / max(bot, 1e-3)))
        bg_sup = float(np.mean(bg_sups)) if bg_sups else np.nan

        
        scores_ana = np.array([_img_score(h) for h in hm_ana])
        scores_pbs = np.array([_img_score(h) for h in hm_pbs]) if hm_pbs else np.array([0.0])
        mu_a, mu_p = scores_ana.mean(), scores_pbs.mean()
        sig_a = scores_ana.std(ddof=min(1, len(scores_ana)-1)) + 1e-8
        sig_p = scores_pbs.std(ddof=min(1, len(scores_pbs)-1)) + 1e-8
        sep_effect = float((mu_a - mu_p) / np.sqrt((sig_a**2 + sig_p**2) / 2))

        
        q95_a = float(np.percentile(scores_ana, 95))
        q95_p = float(np.percentile(scores_pbs, 95))
        sep_tail = float(q95_a - q95_p)

        return {
            f"{label}_pbs_fpr"    : pbs_fpr,
            f"{label}_seed_ret"   : ana_seed,
            f"{label}_bg_suppress": bg_sup,
            f"{label}_sep_effect" : sep_effect,
            f"{label}_sep_tail"   : sep_tail,
        }

    hm_metrics_refined = compute_heatmap_metrics(hm_pbs_refined, hm_ana_refined, ana_recs_order, "refined")
    hm_metrics_coarse  = compute_heatmap_metrics(hm_pbs_coarse,  hm_ana_coarse,  ana_recs_order, "coarse")

    hm_metrics_df = pd.DataFrame([{**hm_metrics_coarse, **hm_metrics_refined}])
    hm_metrics_df.to_csv(os.path.join(final_dir, "heatmap_metrics.csv"), index=False)

    print("\n=== Heatmap localization metrics ===")
    print(f"{'Metric':<30} {'Coarse':>10} {'Refined':>10}")
    print("-" * 52)
    metrics_pairs = [
        ("PBS cleanliness (1-FPR)  (higher)",  "coarse_pbs_fpr",     "refined_pbs_fpr",     True),
        ("Seed retention  (higher)",          "coarse_seed_ret",    "refined_seed_ret",    False),
        ("Background suppression  (higher)",             "coarse_bg_suppress", "refined_bg_suppress", False),
        ("Effect-size Sep  (higher)",      "coarse_sep_effect",  "refined_sep_effect",  False),
        ("Tail Sep  (higher)",             "coarse_sep_tail",    "refined_sep_tail",    False),
    ]
    for name, ck, rk, invert in metrics_pairs:
        cv = hm_metrics_coarse.get(ck, np.nan)
        rv = hm_metrics_refined.get(rk, np.nan)
        if invert:
            cv_show = 1.0 - cv if not np.isnan(cv) else np.nan
            rv_show = 1.0 - rv if not np.isnan(rv) else np.nan
        else:
            cv_show, rv_show = cv, rv
        print(f"  {name:<28} {cv_show:>10.4f} {rv_show:>10.4f}")

    
    
    
    clf_csv = os.path.join(CFG["output_dir"], "final_classifier", "lgbm_cv_results.csv")
    if os.path.exists(clf_csv):
        cv_df = pd.read_csv(clf_csv)
        print("\n=== Classification metrics (handcrafted+LightGBM GroupKFold)===")
        cols_show = [c for c in ["accuracy","balanced_accuracy","f1","mcc","roc_auc","pr_auc"] if c in cv_df.columns]
        for col in cols_show:
            print(f"  {col:<25} mean={cv_df[col].mean():.4f}  std={cv_df[col].std():.4f}")

    with pd.ExcelWriter(os.path.join(final_dir, "results_master.xlsx"), engine="openpyxl") as writer:
        pd.DataFrame([CFG]).to_excel(writer, sheet_name="run_config", index=False)
        result_df.to_excel(writer, sheet_name="final_prediction_summary", index=False)
        hm_metrics_df.to_excel(writer, sheet_name="heatmap_metrics", index=False)
        if os.path.exists(clf_csv):
            pd.read_csv(clf_csv).to_excel(writer, sheet_name="clf_cv_results", index=False)

    print("\nOK Inference complete")
    print("  analyte samples:", len(ana_stage_items))
    print("  pbs samples:",     len(pbs_stage_items))
    print("  OK heatmap_metrics.csv")
    print("  OK results_master.xlsx (includes heatmap_metrics + clf_cv_results)")

    if args.zip_output:
        zip_path = shutil.make_archive(CFG["output_dir"], "zip", CFG["output_dir"])
        print(f"OK Output archive created: {zip_path}")


if __name__ == '__main__':
    main()
