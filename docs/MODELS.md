# Included Inference Checkpoints

The three files are existing trained weights, not new training runs. Filenames
describe their tasks. Aggregate cross-validation metrics must not be presented
as an independently measured score for an individual checkpoint.

| File | Network class | Normalization keys |
| --- | --- | --- |
| `binary_classifier.pt` | `feature_classifier_comparison.ResNet29Classifier(29)` | `channel_mean`, `channel_std` |
| `concentration_classifier.pt` | `concentration_classifier.ResNet29FourClass()` | `mean`, `std` |
| `response_localizer.pt` | `gas_transfer.ResNet18_29(29)` | `channel_mean`, `channel_std` |

All checkpoints contain `model_state`. The localizer also contains `pbs_mu29`
and `pbs_std29` for PBS-reference map generation. The four-class checkpoint
contains `class_names`; use that stored order rather than inferring order from
a plot. Consult the concentration-label limitations in `DATA.md`.

## Binary Classification and Localization

```bash
python scripts/resnet_inference.py --model-dir models --zip-dir data/unknown_archives --work-data-dir data/unknown_extracted --output-dir outputs/unknown_resnet
```

The inference entry point supports these simplified filenames and the legacy
training-output layout. Input images still require the same preprocessing and
paired-image conventions. A `.pt` file does not embed an entire standalone app.

## Four-Class Model Loading

Run from the repository root with trusted local checkpoint files only:

```python
import sys
sys.path.insert(0, "scripts")
import numpy as np
import torch
from concentration_classifier import ResNet29FourClass, extract_29, read_file_gray

checkpoint = torch.load("models/concentration_classifier.pt", map_location="cpu", weights_only=False)
model = ResNet29FourClass().eval()
model.load_state_dict(checkpoint["model_state"], strict=True)
before = read_file_gray("before.png")
after = read_file_gray("after.png")
features = extract_29(before, after)
mean = np.asarray(checkpoint["mean"], dtype=np.float32).reshape(29, 1, 1)
std = np.asarray(checkpoint["std"], dtype=np.float32).reshape(29, 1, 1)
x = torch.from_numpy(((features - mean) / (std + 1e-6)).astype(np.float32)[None])
with torch.inference_mode():
    probabilities = model(x).softmax(dim=1)[0]
print(dict(zip(checkpoint["class_names"], probabilities.tolist())))
```

The two historical localization checkpoint files contain identical network
weights. Only one copy is distributed. Source and released SHA-256 digests are
recorded in `models/model_manifest.json`. Checkpoints were reserialized to omit
personal paths and unused training-split indices; network tensors were verified
unchanged.
