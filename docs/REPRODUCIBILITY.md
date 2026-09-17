# Reproducibility and Interpretation

## Scope

Nine source scripts were copied and adapted for an English-only repository.
Local import names and path defaults were updated. Chinese comments were
removed; Chinese docstrings, labels, and log messages were translated or
replaced with short English descriptions. No numerical training settings were
changed as part of packaging. A command-line interface was added to the
four-class script while retaining its current defaults.

`source_manifest.json` records original filenames and SHA-256 hashes. Original
files are not modified. This release does not certify model convergence or
reproduce the published/evaluated numbers without the original data and weights.

## Fold Counts and Stages

- The binary research configuration retains three folds, 16 maximum classifier
  epochs, and 50 maximum Stage 2a epochs.
- The current four-class source defaults to five folds and 18 maximum epochs.
  Previously inspected four-class result files came from a three-fold run.
  Changing the source does not regenerate those results. Use `--folds 3` only
  when intentionally matching that historical configuration.
- Stage 2b is optional and disabled in the hybrid baseline. The inspected final
  response-localization results are from Stage 2a, not Stage 2b.
- Binary per-epoch metrics use threshold 0.5. Final fold metrics restore the
  best validation-balanced-accuracy checkpoint and use a training-set threshold.
  The last point of a training curve is not necessarily the reported score.

## Evaluation Limits

The comparison and final-pipeline coarse maps differ in feature bank, fusion,
and normalization. Do not compare their coarse values as though they were an
identical baseline. The final exports include inference on all available
samples; distinguish these descriptive outputs from held-out performance.
Validation-based model selection is not an independent external test.

Some baseline utilities compute PBS references or channel statistics from the
full dataset. Audit each call site before calling an evaluation leakage-free.
Repeated measurements require group-aware splitting by the experimental unit;
image-level folds alone do not establish independent batch generalization.

Use real repeated evaluations or a justified sample-level resampling procedure
for uncertainty. This repository contains no fabricated error bars.

## Feature and Attribution Semantics

The feature maps are physically motivated image descriptors, not 29 separately
validated physical laws. High-frequency content can contain both response and
noise; low-frequency content can contain both background and broad responses.

Feature formulas differ between historical modules. For example, the separate
four-class script uses `diff * diff` for its squared-change channel. Review the
implementation before copying feature definitions between workflows.

The binary pipeline fits a LightGBM surrogate to classifier probabilities and
uses TreeSHAP. In this source, the surrogate inputs are per-feature spatial
means; do not describe them as mean/std/95th-percentile summaries without
changing and verifying the implementation. The four-class workflow uses a
separate gradient-based attribution implementation.

The full pipeline can initialize classification from a baseline localization
checkpoint. Check provenance of any initialization checkpoint and its training
data before claiming an independent cross-validation estimate.

## Environment

The observed requirements document the packaging machine, not necessarily the
environment of historical training. Full training and end-to-end result
reproduction were not run during packaging. GPU determinism is not guaranteed.
