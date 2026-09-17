# Raspberry Pi Deployment Interface

This directory contains the PINES browser interface and Flask camera server used
for Raspberry Pi acquisition, remote control, LED-field calibration, paired-image
analysis, and response-map visualization.

## Files

| File | Purpose |
| --- | --- |
| `cam_server.py` | Flask API, camera control, media capture, binary response classification, and response localization |
| `index.html` | Smartphone-compatible control interface and result display |
| `requirements.txt` | Python packages used by the deployment server |

The research scripts in `../../scripts/` remain the authoritative source for
training and offline evaluation. The deployment code is a device-oriented
interface and should not be used as a substitute for the documented training
pipelines.

## Raspberry Pi Layout

The server expects the following paths:

```text
~/camera_web/
  cam_server.py
  templates/
    index.html
  final_classifier/
    best_classifier.pt
  stage2a_refine.pt
  photos/
  videos/
  detection/
```

From the repository root, install the included binary classifier and response
localizer using their deployment filenames:

```bash
mkdir -p ~/camera_web/templates ~/camera_web/final_classifier
cp deployment/raspberry_pi/cam_server.py ~/camera_web/cam_server.py
cp deployment/raspberry_pi/index.html ~/camera_web/templates/index.html
cp models/binary_classifier.pt ~/camera_web/final_classifier/best_classifier.pt
cp models/response_localizer.pt ~/camera_web/stage2a_refine.pt
mkdir -p ~/camera_web/photos ~/camera_web/videos ~/camera_web/detection
```

The binary and localization checkpoint structures match the model classes used
by `cam_server.py`. Do not rename the concentration checkpoint as either of
these files: it uses a different network and preprocessing pipeline.

## Installation

Camera utilities and FFmpeg are system dependencies. Their package names vary
with Raspberry Pi OS releases.

```bash
cd ~/camera_web
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r /path/to/repository/deployment/raspberry_pi/requirements.txt
python cam_server.py
```

Install a Raspberry Pi-compatible PyTorch and torchvision build separately if
the default pip wheels do not support the device. Open
`http://<raspberry-pi-ip>:5000` from a phone connected to the same Wi-Fi network
or hotspot.

## Detection Workflow

1. Capture or select an LED reference image and run LED calibration.
2. Select matched before and after images.
3. Select corresponding regions of interest.
4. Run detection.
5. Review the classification confidence and AI-enhanced response map.

For four-class concentration output, the detection endpoint can return the
following fields:

```json
{
  "concentration_label": "10uM",
  "concentration_confidence": 0.93
}
```

`class_label` or `label` may be used instead of `concentration_label`. Accepted
class values are `PBS`, `1uM`, `10uM`, and `100uM`. The interface displays the
concentration only for the three positive classes and suppresses it for PBS.
The included `models/concentration_classifier.pt` is documented in
`../../docs/MODELS.md` and uses its dedicated feature extraction and
normalization pipeline.

## Device Notes

- The code supports `rpicam-*` and legacy `libcamera-*` command names.
- Default media paths are under `/home/cholab/camera_web/`.
- Long exposures temporarily stop the preview stream during still capture.
- Review filesystem permissions before running the server as a service.
- The Flask development server is intended for controlled research networks,
  not direct exposure to the public internet.
