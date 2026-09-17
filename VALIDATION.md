# Release Validation

Validation date: 2026-09-16. Platform: Windows, Python 3.11.

## Passed

- The three included trained checkpoints load strictly with the packaged
  network definitions and produce finite CPU outputs on synthetic inputs.
  Network tensors were verified unchanged during checkpoint reserialization.

- All nine research scripts and the Raspberry Pi deployment server compile and parse.
- All nine command-line entry points exit successfully with `--help`.
- Five standard-library release tests pass: script inventory, syntax/local
  imports, deployment-server syntax, absence of CJK text, and absence of
  personal Windows paths in the research scripts.
- Binary 29-feature extraction on synthetic 128 x 128 paired images is exactly
  equal to the unchanged original implementation; output is finite and has
  shape (29, 128, 128).
- Final-pipeline coarse top1, top3-weighted, top5-RMS, support-fraction, and
  z-score arrays exactly match the original implementation on synthetic inputs.
- Four-class feature extraction exactly matches the original implementation.
- Binary classifier, localization network, and four-class ResNet18 accept
  original state dictionaries and produce exactly equal CPU outputs on
  synthetic inputs after loading identical randomly initialized weights.
  Output shapes are (1,), (1, 1, 128, 128), and (1, 4), respectively.

## Not Performed

- No training, fine-tuning, or end-to-end data processing was repeated.
- No trained research checkpoint was evaluated on experimental samples during
  packaging; included-checkpoint checks used synthetic inputs only.
- No fresh dependency installation or Linux/GPU runtime validation was run.
- No Raspberry Pi camera, mobile-browser, or on-device model inference test was
  performed during packaging.
- No scientific claim, concentration label, or historical metric was validated
  by the synthetic-input checks.
- GitHub Actions configuration is included, but no GitHub upload or remote CI
  execution was performed.

These checks establish source-package consistency, not full experimental
reproducibility. See `docs/REPRODUCIBILITY.md` for known interpretation limits.
