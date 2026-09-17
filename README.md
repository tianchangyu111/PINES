# Fluorescence Response Learning

Research workflows for paired near-infrared fluorescence images: engineered
image features, binary classification, weakly supervised response localization,
gas-domain transfer, and four-class classification.

This English-only release contains nine research scripts, three existing
inference checkpoints, and a Raspberry Pi deployment interface. No retraining
was performed for packaging. No raw images, experimental result tables, or
illustrative/simulated error bars are included.

## Included Models

| File | Task |
| --- | --- |
| `models/binary_classifier.pt` | PBS/analyte classification |
| `models/concentration_classifier.pt` | Four-class classification |
| `models/response_localizer.pt` | Stage 2a response localization |

Network weights and inference normalization statistics are preserved. Duplicate
localization weights, training-split indices, and personal path metadata are
omitted. See [model loading](docs/MODELS.md) for architectures and normalization.

## Workflow

```text
Before/after images -> illumination correction -> feature maps
  -> binary classifier -> encoder-weight transfer -> response localizer
  -> response maps and descriptive metrics
```

The binary classifier uses a custom ResNet18-style encoder, not an unmodified
torchvision ResNet18. The separate four-class script uses torchvision ResNet18
with a 29-channel input. Do not interchange their checkpoints.

## Installation

Use Python 3.11 and an isolated environment. Run commands from the repository root.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CUDA-specific PyTorch installation, select the appropriate official
PyTorch distribution for your hardware before installing the requirements.
`requirements-observed.txt` records versions in the packaging environment; it
is not a verified lockfile for the historical training runs.

## Scripts

| Script in `scripts/` | Purpose |
| --- | --- |
| `resnet_pipeline.py` | Full binary ResNet workflow, localization, surrogate SHAP, and exports |
| `localization_comparison.py` | ResNet/CNN/TinyUNet localization comparison and training utilities |
| `feature_classifier_comparison.py` | Classification comparisons using 29-channel features |
| `gas_transfer.py` | Hybrid classifier/localizer transfer and shared feature/network definitions |
| `transfer_learning.py` | Standalone transfer workflow and shared image preprocessing |
| `hybrid_pipeline.py` | Handcrafted/LightGBM baseline, localization, and export utilities |
| `resnet_inference.py` | Inference on additional paired images using saved ResNet models |
| `lightgbm_inference.py` | Inference using a saved hybrid LightGBM/localizer pipeline |
| `concentration_classifier.py` | Four-class training and attribution with configurable data paths |

Keep all nine files together: some scripts import functions from the others.

## Raspberry Pi Deployment

The device-facing Flask server and smartphone interface are provided in
[`deployment/raspberry_pi/`](deployment/raspberry_pi/README.md). They support
camera control, media capture, LED-field calibration, paired-image detection,
and response-map display. The deployment README documents the expected device
layout, model filenames, installation steps, and optional four-class result
contract.

## Data Layout

```text
data/
  led_field/       # illumination images
  susbtrat/        # before images (historical spelling retained intentionally)
  pbs/            # PBS after images
  analyte/        # analyte after images
```

Before/after pairs use matching filenames. Supply your own data. See
[data preparation](docs/DATA.md) for gas, unknown-image, and four-class layouts.

## Example Commands

Inspect options without starting training:

```bash
python scripts/resnet_pipeline.py --help
python scripts/localization_comparison.py --help
python scripts/concentration_classifier.py --help
```

Run the baseline, then the full ResNet pipeline using the baseline output:

```bash
python scripts/hybrid_pipeline.py --data-dir data --output-dir outputs/hybrid
python scripts/resnet_pipeline.py --data-dir data --output-dir outputs/resnet --base-output-dir outputs/hybrid
python scripts/localization_comparison.py --data-dir data --output-dir outputs/localization_comparison --base-output-dir outputs/hybrid
```

Gas transfer requires compatible saved source models:

```bash
python scripts/gas_transfer.py --data-dir data/gas --base-output-dir outputs/hybrid --output-dir outputs/gas_transfer
```

Four-class training:

```bash
python scripts/concentration_classifier.py --zip-root data/concentration_archives --split-root data/concentration --output-dir outputs/concentration_5fold --folds 5
```

Unknown-image inference:

```bash
python scripts/resnet_inference.py --zip-dir data/unknown_archives --work-data-dir data/unknown_extracted --model-dir models --output-dir outputs/unknown_resnet
```

Commands describe entry points, not proof that historical results can be
reproduced from the source alone. Read [reproducibility notes](docs/REPRODUCIBILITY.md)
before interpreting outputs. Existing output directories may be overwritten;
use a new output directory for each run.

## Checks

```bash
python -m unittest discover -s tests -v
python -m compileall -q scripts deployment/raspberry_pi
```

The standard-library checks validate syntax, English-only release text, local
import closure, and the expected nine-script inventory. They do not train models.
See `VALIDATION.md` for checks actually performed during packaging.

## Publication and Licensing

No publication DOI, authorship, or open-source license is invented by this
release. See [licensing notes](LICENSE_NOTES.md). Confirm rights and choose a
license before advertising the repository as open source. No GitHub upload is
performed by creating this archive.
