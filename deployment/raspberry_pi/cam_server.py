from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import threading
import time
import os
import datetime
import subprocess
import signal
import io
import shutil
from threading import Event
import json
import psutil
import pathlib
import logging
import re
from functools import wraps
import queue
import weakref
import select  
import numpy as np
import cv2
import base64

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from scipy.ndimage import gaussian_filter, uniform_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)


recording_process = None  # ffmpeg muxer process
recording_cam_process = None  # rpicam-vid/libcamera-vid producer
recording_lock = threading.Lock()
camera_lock = threading.Lock()
capture_in_progress = False


_timelapse_running = False
_timelapse_stop_event = threading.Event()
_timelapse_thread = None
_timelapse_frame_count = 0
_timelapse_lock = threading.Lock()
TIMELAPSE_FRAME_DIR = '/home/cholab/camera_web/timelapse_frames'

HLS_ENABLED = False  

hls_lock = threading.Lock()
hls_process_cam = None
hls_process_mux = None


class PerformanceCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def get(self, key):
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.access_order.append(key)


cache = PerformanceCache()

PHOTO_CIRCLE_RADIUS_RATIO = 0.46
PHOTO_FILENAME_RE = re.compile(r'^image-(\d+)\.(png|jpg|jpeg)$', re.IGNORECASE)


def _next_photo_paths(save_dir):
    """, image-1.png / image-2.png."""
    next_index = 1
    try:
        for name in os.listdir(save_dir):
            match = PHOTO_FILENAME_RE.match(name)
            if match:
                next_index = max(next_index, int(match.group(1)) + 1)
    except FileNotFoundError:
        pass

    stem = f'image-{next_index}'
    return (
        os.path.join(save_dir, f'{stem}_raw.jpg'),
        os.path.join(save_dir, f'{stem}.png'),
    )
# ============================================================

# ============================================================
if TORCH_AVAILABLE:
    
    class _BasicBlock(nn.Module):
        def __init__(self, in_ch, out_ch, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(out_ch)
            self.shortcut = (
                nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                              nn.BatchNorm2d(out_ch))
                if stride != 1 or in_ch != out_ch else nn.Identity()
            )
        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)), inplace=True)
            return F.relu(self.bn2(self.conv2(out)) + self.shortcut(x), inplace=True)

    def _make_layer(in_ch, out_ch, n, stride):
        lrs = [_BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, n):
            lrs.append(_BasicBlock(out_ch, out_ch))
        return nn.Sequential(*lrs)

    
    class ResNet18_29Ch_Staged(nn.Module):
        """
        Stage1: mode='cls'  → [B,2]  logits
        Stage2: mode='refine' + aux [B,8,H,W] → [B,1,H,W]  image logit
        """
        def __init__(self, in_ch=29, aux_ch=8, fuse_ch=64):
            super().__init__()
            self.in_ch  = in_ch
            self.aux_ch = aux_ch
            self.conv1   = nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False)
            self.bn1     = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            self.layer1  = _make_layer(64,  64,  2, stride=1)
            self.layer2  = _make_layer(64,  128, 2, stride=2)
            self.layer3  = _make_layer(128, 256, 2, stride=2)
            self.layer4  = _make_layer(256, 512, 2, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.cls_head = nn.Sequential(nn.Dropout(0.25), nn.Linear(512, 2))
            self.reduce2 = nn.Sequential(
                nn.Conv2d(128, fuse_ch, 1, bias=False), nn.BatchNorm2d(fuse_ch), nn.ReLU(inplace=True))
            self.reduce3 = nn.Sequential(
                nn.Conv2d(256, fuse_ch, 1, bias=False), nn.BatchNorm2d(fuse_ch), nn.ReLU(inplace=True))
            self.reduce4 = nn.Sequential(
                nn.Conv2d(512, fuse_ch, 1, bias=False), nn.BatchNorm2d(fuse_ch), nn.ReLU(inplace=True))
            fuse_in = in_ch + aux_ch + 3 * fuse_ch
            self.refine_fuse = nn.Sequential(
                nn.Conv2d(fuse_in, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            )
            self.refine_head = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1),
            )

        def _extract(self, x):
            h = F.relu(self.bn1(self.conv1(x)), inplace=True)
            h = self.maxpool(h)
            h = self.layer1(h)
            f2 = self.layer2(h)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)
            return torch.flatten(self.avgpool(f4), 1), f2, f3, f4

        def _refine_from_feats(self, x, aux, f2, f3, f4):
            H, W = x.shape[-2:]
            u2 = F.interpolate(self.reduce2(f2), (H, W), mode='bilinear', align_corners=False)
            u3 = F.interpolate(self.reduce3(f3), (H, W), mode='bilinear', align_corners=False)
            u4 = F.interpolate(self.reduce4(f4), (H, W), mode='bilinear', align_corners=False)
            z  = torch.cat([x, aux, u2, u3, u4], dim=1)
            return self.refine_head(self.refine_fuse(z))

        def forward(self, x, aux=None, mode='cls'):
            if mode == 'cls':
                feat_vec, *_ = self._extract(x)
                return self.cls_head(feat_vec)
            elif mode == 'refine':
                if aux is None:
                    raise ValueError('refine mode requires aux')
                _, f2, f3, f4 = self._extract(x)
                return self._refine_from_feats(x, aux, f2, f3, f4)
            else:
                raise ValueError(f'unknown mode: {mode}')

LED_MAP_PATH = os.path.expanduser('~/camera_web/led_map.npy')

BIOSENSOR_MODEL_PATH = os.path.expanduser('~/camera_web/biosensor_net.pth')
BIOSENSOR_MODEL_PATH_29CH = os.path.expanduser('~/camera_web/biosensor_29ch.pt')
DETECTION_DIR = os.path.expanduser('~/camera_web/detection')
os.makedirs(DETECTION_DIR, exist_ok=True)

led_map = None
try:
    if os.path.exists(LED_MAP_PATH):
        led_map = np.load(LED_MAP_PATH).astype(np.float32)
        logger.info(f"LED correction map loaded: shape={led_map.shape}")
except Exception as _e:
    logger.warning(f"Failed to load LED correction map: {_e}")

biosensor_model = None
biosensor_ch_mean = None   
biosensor_ch_std  = None   
biosensor_pbs_mu  = None   
biosensor_pbs_std = None   
BIOSENSOR_MODE = 'none'    # 'v29' | 'none'

SEPARATE_CLASSIFIER_PATH = os.path.expanduser('~/camera_web/final_classifier/best_classifier.pt')
SEPARATE_REFINER_PATH = os.path.expanduser('~/camera_web/stage2a_refine.pt')
_REPOSITORY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_CONCENTRATION_MODEL_CANDIDATES = [
    os.path.expanduser(os.environ.get('PINES_CONCENTRATION_MODEL', '')),
    os.path.expanduser('~/camera_web/concentration_classifier.pt'),
    os.path.join(_REPOSITORY_ROOT, 'models', 'concentration_classifier.pt'),
]
CONCENTRATION_MODEL_PATH = next(
    (path for path in _CONCENTRATION_MODEL_CANDIDATES if path and os.path.isfile(path)),
    None,
)
separate_classifier_model = None
separate_refine_model = None
separate_classifier_ch_mean = None
separate_classifier_ch_std = None
separate_refine_ch_mean = None
separate_refine_ch_std = None
concentration_model = None
concentration_ch_mean = None
concentration_ch_std = None
concentration_class_names = None
SEPARATE_FEATURE_NAMES = [
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

if TORCH_AVAILABLE:
    class _ResidualBlockNoProj(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
            )

        def forward(self, x):
            return F.relu(self.block(x) + x, inplace=True)

    class _ResNet29Backbone(nn.Module):
        def __init__(self, in_ch=29):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.layer1 = nn.Sequential(
                _ResidualBlockNoProj(64),
                _ResidualBlockNoProj(64),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                _ResidualBlockNoProj(128),
                _ResidualBlockNoProj(128),
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                _ResidualBlockNoProj(256),
            )
            self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.dec3 = _ResidualBlockNoProj(128)
            self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.dec2 = _ResidualBlockNoProj(64)
            
            self.head = nn.Conv2d(64, 1, 1)

        def forward(self, x):
            x1 = self.layer1(self.stem(x))
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            y = self.up3(x3) + x2
            y = self.dec3(y)
            y = self.up2(y) + x1
            y = self.dec2(y)
            return x1, x2, x3, y

    class ResNet29BinaryClassifier(nn.Module):
        def __init__(self, in_ch=29):
            super().__init__()
            self.backbone = _ResNet29Backbone(in_ch=in_ch)
            self.fc = nn.Linear(256, 1)

        def forward(self, x):
            _, _, x3, _ = self.backbone(x)
            pooled = torch.flatten(F.adaptive_avg_pool2d(x3, 1), 1)
            return self.fc(pooled)

    class ResNet29Stage2A(nn.Module):
        def __init__(self, in_ch=29):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.layer1 = nn.Sequential(
                _ResidualBlockNoProj(64),
                _ResidualBlockNoProj(64),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                _ResidualBlockNoProj(128),
                _ResidualBlockNoProj(128),
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                _ResidualBlockNoProj(256),
            )
            self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.dec3 = _ResidualBlockNoProj(128)
            self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.dec2 = _ResidualBlockNoProj(64)
            self.head = nn.Conv2d(64, 1, 1)

        def forward(self, x):
            x1 = self.layer1(self.stem(x))
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            y = self.up3(x3) + x2
            y = self.dec3(y)
            y = self.up2(y) + x1
            y = self.dec2(y)
            return self.head(y)

    class ResNet29FourClass(nn.Module):
        def __init__(self, n_classes=4):
            super().__init__()
            self.net = models.resnet18(weights=None)
            self.net.conv1 = nn.Conv2d(
                29, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.net.fc = nn.Linear(self.net.fc.in_features, n_classes)

        def forward(self, x):
            return self.net(x)

    if os.path.exists(SEPARATE_CLASSIFIER_PATH) and os.path.exists(SEPARATE_REFINER_PATH):
        try:
            cls_ckpt = torch.load(SEPARATE_CLASSIFIER_PATH, map_location='cpu', weights_only=False)
            refine_ckpt = torch.load(SEPARATE_REFINER_PATH, map_location='cpu', weights_only=False)

            _cls_model = ResNet29BinaryClassifier(in_ch=29).eval()
            _cls_model.load_state_dict(cls_ckpt['model_state'])
            separate_classifier_model = _cls_model
            separate_classifier_ch_mean = np.asarray(cls_ckpt['channel_mean'], dtype=np.float32)
            separate_classifier_ch_std = np.asarray(cls_ckpt['channel_std'], dtype=np.float32)

            _refine_model = ResNet29Stage2A(in_ch=29).eval()
            _refine_model.load_state_dict(refine_ckpt['model_state'])
            separate_refine_model = _refine_model
            separate_refine_ch_mean = np.asarray(refine_ckpt['channel_mean'], dtype=np.float32)
            separate_refine_ch_std = np.asarray(refine_ckpt['channel_std'], dtype=np.float32)

            BIOSENSOR_MODE = 'separate_v2026'
            logger.info(
                "Loaded29: classifier=%s refine=%s",
                SEPARATE_CLASSIFIER_PATH,
                SEPARATE_REFINER_PATH,
            )
        except Exception as _e:
            logger.warning(f"Failed to load the separate 29-feature models; falling back to the legacy model: {_e}")

    
    _model_path = None
    if BIOSENSOR_MODE == 'none' and os.path.exists(BIOSENSOR_MODEL_PATH_29CH):
        _model_path = BIOSENSOR_MODEL_PATH_29CH
    elif BIOSENSOR_MODE == 'none' and os.path.exists(BIOSENSOR_MODEL_PATH):
        
        logger.warning("Only a legacy model file was found. Place the 29-channel model at biosensor_29ch.pt.")

    if _model_path:
        try:
            ckpt = torch.load(_model_path, map_location='cpu', weights_only=False)
            
            _fuse_w = ckpt['model_state']['refine_fuse.0.weight']
            _aux_ch = int(_fuse_w.shape[1]) - 29 - 192  # fuse_in - in_ch - 3*fuse_ch
            _model = ResNet18_29Ch_Staged(in_ch=29, aux_ch=_aux_ch).eval()
            _model.load_state_dict(ckpt['model_state'])
            biosensor_model    = _model
            biosensor_ch_mean  = ckpt['channel_mean'].astype(np.float32)
            biosensor_ch_std   = ckpt['channel_std'].astype(np.float32)
            
            _pbs_mu = ckpt.get('pbs_mu_px')
            _pbs_std = ckpt.get('pbs_std_px')
            biosensor_pbs_mu  = np.array(_pbs_mu).astype(np.float32)  if _pbs_mu  is not None else None
            biosensor_pbs_std = np.array(_pbs_std).astype(np.float32) if _pbs_std is not None else None
            BIOSENSOR_MODE = 'v29'
            has_pbs = biosensor_pbs_mu is not None
            logger.info(f"29-channel model loaded: {_model_path}; aux_ch={_aux_ch}; PBS baseline={'available' if has_pbs else 'unavailable (self-normalized)'}")
        except Exception as _e:
            logger.warning(f"Failed to load the 29-channel model: {_e}")

    if CONCENTRATION_MODEL_PATH:
        try:
            concentration_ckpt = torch.load(
                CONCENTRATION_MODEL_PATH, map_location='cpu', weights_only=False
            )
            concentration_class_names = [
                str(name) for name in concentration_ckpt['class_names']
            ]
            _concentration_model = ResNet29FourClass(
                n_classes=len(concentration_class_names)
            ).eval()
            _concentration_model.load_state_dict(
                concentration_ckpt['model_state'], strict=True
            )
            concentration_model = _concentration_model
            concentration_ch_mean = np.asarray(
                concentration_ckpt['mean'], dtype=np.float32
            ).reshape(29, 1, 1)
            concentration_ch_std = np.asarray(
                concentration_ckpt['std'], dtype=np.float32
            ).reshape(29, 1, 1)
            logger.info(
                "Loaded four-class concentration model: %s; classes=%s",
                CONCENTRATION_MODEL_PATH,
                concentration_class_names,
            )
        except Exception as _e:
            concentration_model = None
            logger.warning(f"Failed to load the four-class concentration model: {_e}")
    else:
        logger.info(
            "Four-class concentration model not found; set PINES_CONCENTRATION_MODEL "
            "or place concentration_classifier.pt in ~/camera_web"
        )

def log_execution_time(func):
    """:execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} execution time: {execution_time:.3f} s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed (elapsed: {execution_time:.3f} s): {e}")
            raise
    return wrapper

class CameraStreamer:
    def __init__(self):
        self.process = None
        self.running = False
        
        self.desired_width = 640
        self.desired_height = 480
        self.desired_fps = 10  
        
        self.desired_exposure_sec = None
        self.desired_gain = None
        self.startup_warmup = True  
        
        self.af_window = None
        
        self.af_mode = 'auto'

        
        self.frame_queue = queue.Queue(maxsize=10)  
        self.frame_reader_thread = None  
        self.last_frame_time = 0
        self.frame_interval = 1.0 / self.desired_fps
        
    def kill_camera_processes(self):
        """available - """
        try:
            camera_processes = [
                'rpicam-vid', 'rpicam-still',
                'libcamera-vid', 'libcamera-still', 'libcamera-hello',
                'raspivid', 'raspistill'
            ]
            killed_count = 0
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if any(cam_proc in ' '.join(proc.info['cmdline'] or []) for cam_proc in camera_processes):
                        logger.info(f"Terminating process: {proc.info['pid']} - {proc.info['name']}")
                        proc.kill()
                        proc.wait(timeout=2)  
                        killed_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    pass
            
            if killed_count > 0:
                time.sleep(1)  
                logger.info(f"Terminated {killed_count} camera processes")
        except Exception as e:
            logger.error(f"Error while cleaning up camera processes: {e}")
    
    def _frame_reader_loop(self):
        """: - """
        try:
            chunk_size = 8192
            frame_data = b''
            start_marker = b'\xff\xd8'
            end_marker = b'\xff\xd9'
            max_frame_size = 512 * 1024
            
            while self.running and self.process:
                try:
                    
                    if not self.running:
                        break
                    
                    
                    chunk = None
                    try:
                        
                        self.process.stdout.fileno()  
                        
                        chunk = self.process.stdout.read(chunk_size)
                    except (OSError, IOError, ValueError):
                        
                        break
                    
                    if not chunk:
                        time.sleep(0.01)  
                        if not self.running:  
                            break
                        continue
                    
                    frame_data += chunk
                    
                    
                    start_pos = frame_data.find(start_marker)
                    if start_pos >= 0:
                        frame_data = frame_data[start_pos:]
                        end_pos = frame_data.find(end_marker)
                        if end_pos >= 0:
                            frame = frame_data[:end_pos + 2]
                            frame_data = frame_data[end_pos + 2:]
                            
                            
                            try:
                                self.frame_queue.put_nowait(frame)
                            except queue.Full:
                                try:
                                    self.frame_queue.get_nowait()  
                                    self.frame_queue.put_nowait(frame)
                                except queue.Empty:
                                    pass
                    
                    if len(frame_data) > max_frame_size:
                        frame_data = b''  
                        
                except Exception as e:
                    logger.error(f"Background reader error: {e}")
                    break
        except Exception as e:
            logger.error(f"Background frame-reader exception: {e}")
        finally:
            logger.info("Background frame reader exited")
    
    @log_execution_time
    def check_camera_available(self):
        """ (v4l2,libcamera)- """
        
        cache_key = "camera_device"
        cached_result = cache.get(cache_key)
        if cached_result and time.time() - cached_result.get('timestamp', 0) < 30:  
            return cached_result['device']
        
        try:
            import subprocess
            import os

            
            video_devices = ['/dev/video0', '/dev/video1', '/dev/video13', '/dev/video14']
            for device in video_devices:
                if os.path.exists(device):
                    try:
                        result = subprocess.run(['v4l2-ctl', '--device', device, '--info'],
                                               capture_output=True, text=True, timeout=2)  
                        if result.returncode == 0 and 'Video Capture' in result.stdout:
                            if 'Driver name      : unicam' in result.stdout or 'Card type        : unicam' in result.stdout:
                                logger.info(f"Detected CSI/unicam device {device},using libcamera-vid for preview")
                                cache.set(cache_key, {'device': 'libcamera', 'timestamp': time.time()})
                                return 'libcamera'
                            logger.info(f"Found v4l2 camera: {device}")
                            cache.set(cache_key, {'device': device, 'timestamp': time.time()})
                            return device
                    except Exception:
                        continue

            
            try:
                result = subprocess.run(['cam', '--list'], capture_output=True, text=True, timeout=3)  
                if result.returncode == 0 and 'Available cameras:' in result.stdout:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'External camera' in line or 'Internal camera' in line:
                            parts = line.split(':')
                            if len(parts) >= 2:
                                camera_id = parts[0].strip()
                                logger.info(f"Found libcamera camera: {camera_id}")
                                cache.set(cache_key, {'device': camera_id, 'timestamp': time.time()})
                                return camera_id
            except Exception:
                pass

            
            for device in video_devices:
                if os.path.exists(device):
                    cache.set(cache_key, {'device': device, 'timestamp': time.time()})
                    return device

            return False
        except Exception as e:
            logger.error(f"Camera check failed: {e}")
            return False
        
    @log_execution_time
    def start_stream(self):
        """Start the camera stream process - """
        try:
            if self.running:
                return True
            
            logger.info("Cleaning up camera processes...")
            self.kill_camera_processes()
            time.sleep(1)  
            
            camera_device = self.check_camera_available()
            if not camera_device:
                logger.error("Camera unavailable; preview cannot start")
                self.running = False
                return False
            
            logger.info(f"Using camera device: {camera_device}")
            
            
            if camera_device == 'libcamera' or camera_device.isdigit() or camera_device.startswith('/base/'):
                libcamera_app = shutil.which('rpicam-vid') or shutil.which('libcamera-vid')
                if not libcamera_app:
                    logger.error('rpicam-vid or libcamera-vid executable was not found')
                    return False
                
                
                cmd = [
                    libcamera_app,
                    '--codec', 'mjpeg',
                    '-n',  
                    '-t', '0',  
                    '--width', str(self.desired_width),
                    '--height', str(self.desired_height),
                    '--framerate', str(self.desired_fps),
                    '--quality', '85',  
                    '--flush',  
                    '-o', '-'
                ]
                
                try:
                    if self.desired_exposure_sec and float(self.desired_exposure_sec) > 0:
                        shutter_us = max(1000, int(float(self.desired_exposure_sec) * 1_000_000))
                        cmd += ['--shutter', str(shutter_us)]
                        logger.info(f"Applying exposure: {self.desired_exposure_sec}s = {shutter_us}us")
                except Exception as e:
                    logger.warning(f"Failed to apply exposure: {e}")
                try:
                    if self.desired_gain and float(self.desired_gain) >= 1.0:
                        gain_val = f"{float(self.desired_gain):.2f}"
                        cmd += ['--gain', gain_val]
                        logger.info(f"Applying gain: {self.desired_gain} = {gain_val}")
                except Exception as e:
                    logger.warning(f"Failed to apply gain: {e}")
                    
                                    
                try:
                    if self.af_window:
                        x, y, w, h = self.af_window
                        x = max(0.0, min(1.0, float(x)))
                        y = max(0.0, min(1.0, float(y)))
                        w = max(0.02, min(1.0, float(w)))
                        h = max(0.02, min(1.0, float(h)))
                        cmd += ['--autofocus-window', f'{x:.3f},{y:.3f},{w:.3f},{h:.3f}']
                        mode = 'auto' if (self.af_mode or 'auto') == 'auto' else 'continuous'
                        cmd += ['--autofocus-mode', mode]
                        logger.info(f"Applying tap-to-focus window: x={x:.3f}, y={y:.3f}, w={w:.3f}, h={h:.3f}, mode={mode}")
                except Exception as e:
                    logger.warning(f"Failed to apply tap-to-focus parameters: {e}")

                
                logger.info(f"Starting camera stream: {' '.join(cmd)}")
                try:
                    self.process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=0
                    )
                except Exception as e:
                    logger.error(f"Failed to start process: {e}")
                    return False
            else:
                
                cmd = [
                    'ffmpeg',
                    '-f', 'v4l2',
                    '-input_format', 'mjpeg',
                    '-framerate', str(self.desired_fps),
                    '-video_size', f'{self.desired_width}x{self.desired_height}',
                    '-i', camera_device,
                    '-f', 'mjpeg',
                    '-q:v', '3',  
                    '-r', str(self.desired_fps),
                    '-'
                ]
                
                logger.info(f"Starting camera stream: {' '.join(cmd)}")
                try:
                    self.process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=0
                    )
                except Exception as e:
                    logger.error(f"Failed to start process: {e}")
                    return False

            
            time.sleep(2)  
            if self.process.poll() is not None:
                try:
                    err = (self.process.stderr.read() or b'').decode(errors='ignore')
                    logger.error(f"Process exited with error: {err[:300]}")
                except Exception:
                    pass
                return False

            
            try:
                import select
                ready, _, _ = select.select([self.process.stdout], [], [], 1)  
                if ready:
                    self.running = True
                    logger.info("Camera stream started")
                    
                    self.frame_reader_thread = threading.Thread(target=self._frame_reader_loop, daemon=True)
                    self.frame_reader_thread.start()
                    logger.info("Background frame reader started")
                    
                    if self.startup_warmup:
                        logger.info("Warming up and waiting for stream stabilization...")
                        time.sleep(3)  
                        self.startup_warmup = False
                    return True
                else:
                    logger.warning("No camera output detected")
                    return False
            except Exception as e:
                logger.error(f"Output check failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start camera stream: {e}")
            return False
            
    def stop_stream(self):
        """Stop the camera stream - """
        if self.running and self.process:
            try:
                logger.info("Stopping camera stream...")
                
                self.running = False
                
                
                try:
                    self.process.terminate()
                    self.process.wait(timeout=3)  
                except subprocess.TimeoutExpired:
                    logger.warning("Force-stopping camera stream...")
                    self.process.kill()
                    self.process.wait()
                except Exception as e:
                    logger.error(f"Failed to stop camera stream: {e}")
            except Exception as e:
                logger.error(f"Exception while stopping: {e}")
            finally:
                
                if self.frame_reader_thread and self.frame_reader_thread.is_alive():
                    logger.info("Waiting for the background frame reader to exit...")
                    self.frame_reader_thread.join(timeout=2)
                
                
                try:
                    while not self.frame_queue.empty():
                        self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                
                self.process = None
                logger.info("Camera stream stopped")
                time.sleep(0.5)  
            
    def read_frame(self):
        """Read a frame from the camera stream - """
        if not self.running:
            return None
            
        
        current_time = time.time()
        if current_time - self.last_frame_time < self.frame_interval:
            return None
        
        try:
            
            try:
                frame = self.frame_queue.get(timeout=1.0)
                self.last_frame_time = current_time
                return frame
            except queue.Empty:
                
                return None
        except Exception as e:
            logger.error(f"Frame-read error: {e}")
            return None


camera = CameraStreamer()

def get_json_safe(default=None):
    """Robust JSON body parsing to avoid BadRequest on malformed/empty input."""
    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            # try raw data
            try:
                raw = (request.data or b'').decode('utf-8', errors='ignore').strip()
                if raw:
                    import json as _json
                    return _json.loads(raw)
            except Exception:
                pass
        return data if data is not None else (default or {})
    except Exception:
        return default or {}

def get_request_data() -> dict:
    """Merge JSON body, form fields and query params into one dict."""
    data = get_json_safe({}) or {}
    try:
        # Merge form
        if hasattr(request, 'form') and request.form:
            for k in request.form.keys():
                data.setdefault(k, request.form.get(k))
        # Merge query
        if hasattr(request, 'args') and request.args:
            for k in request.args.keys():
                data.setdefault(k, request.args.get(k))
    except Exception:
        pass
    return data

def generate_frames():
    """Generate frames for video streaming - """
    global camera, capture_in_progress
    
    consecutive_errors = 0
    max_errors = 30  
    restart_delay = 3  
    startup_delay = True
    frame_count = 0
    last_error_log = 0
    
    while True:
        try:
            with camera_lock:
                if not camera.running:
                    if capture_in_progress:
                        error_frame = create_error_frame("Capturing photo...")
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
                        time.sleep(0.2)
                        continue
                    logger.info("Camera stream is not running; attempting restart...")
                    if not camera.start_stream():
                        error_frame = create_error_frame("Camera unavailable")
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
                        time.sleep(restart_delay)
                        continue
                
                
                if startup_delay:
                    logger.info("Waiting for camera stream stabilization...")
                    time.sleep(2)  
                    startup_delay = False
            
            frame = camera.read_frame()
            frame_count += 1
            
            if frame and len(frame) > 1000:
                consecutive_errors = 0
                
                headers = (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n'
                )
                yield headers + frame + b'\r\n'
            else:
                consecutive_errors += 1
                
                current_time = time.time()
                if current_time - last_error_log > 10:  
                    logger.warning(f"Frame read failed; consecutive errors: {consecutive_errors}, total frames: {frame_count}")
                    last_error_log = current_time
                
                
                if consecutive_errors >= max_errors:
                    logger.warning("Too many consecutive frame-read failures; restarting the camera stream")
                    if not capture_in_progress:
                        with camera_lock:
                            camera.stop_stream()
                            time.sleep(restart_delay)
                            camera.start_stream()
                    consecutive_errors = 0
                    frame_count = 0
                
                
                time.sleep(0.05)  
                
        except Exception as e:
            logger.error(f"Frame-generation error: {e}")
            consecutive_errors += 1
            if consecutive_errors >= max_errors:
                logger.warning("Repeated frame-generation errors; restarting the camera stream")
                if not capture_in_progress:
                    with camera_lock:
                        camera.stop_stream()
                        time.sleep(restart_delay)
                        camera.start_stream()
                consecutive_errors = 0
                frame_count = 0
            time.sleep(0.05)  

def create_error_frame(message):
    """JPEG"""
    return b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9'


def _mask_center_circle_gray(img_bgr_or_gray):
    """, float32  image,."""
    if img_bgr_or_gray is None:
        raise ValueError("Input image is empty")

    if len(img_bgr_or_gray.shape) == 2:
        gray = img_bgr_or_gray.astype(np.float32)
    else:
        gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY).astype(np.float32)

    h, w = gray.shape
    cy = h // 2
    cx = w // 2
    radius = max(1, int(min(h, w) * PHOTO_CIRCLE_RADIUS_RATIO))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, thickness=-1)
    masked = np.where(mask > 0, gray, 0.0).astype(np.float32)
    return masked


def _find_coating_roi(gray):
    """ coating ,failed None."""
    h, w = gray.shape
    search_h = max(1, int(h * 0.55))
    search = gray[:search_h, :]

    
    
    upper_h = max(1, int(h * 0.24))
    upper = search[:upper_h, :]
    upper_blur = cv2.GaussianBlur(upper, (0, 0), 2)
    upper_thresh = float(np.percentile(upper_blur, 96.0))
    _, upper_mask = cv2.threshold(
        upper_blur.astype(np.uint8),
        int(np.clip(upper_thresh, 1, 255)),
        255,
        cv2.THRESH_BINARY,
    )
    upper_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    upper_mask = cv2.morphologyEx(upper_mask, cv2.MORPH_CLOSE, upper_kernel, iterations=2)

    upper_contours, _ = cv2.findContours(upper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    upper_best = None
    upper_best_score = -1.0
    for contour in upper_contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if area < 180:
            continue

        aspect = bw / max(bh, 1)
        if not (0.45 <= aspect <= 2.2):
            continue
        if area > upper_h * w * 0.12:
            continue

        contour_area = cv2.contourArea(contour)
        fill_ratio = contour_area / max(area, 1)
        if fill_ratio < 0.22:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.045 * peri, True)
        rect_bonus = 1.45 if 4 <= len(approx) <= 8 else 1.0
        center_y = y + bh * 0.5
        upper_bonus = 1.55 if center_y < upper_h * 0.72 else 1.0
        compact_bonus = 1.2 if aspect <= 1.45 else 0.95
        center_x = x + bw * 0.5
        center_bonus = max(0.55, 1.35 - abs(center_x - w * 0.5) / max(w * 0.25, 1))
        brightness = float(np.mean(upper[y:y + bh, x:x + bw]))
        score = area * fill_ratio * rect_bonus * upper_bonus * compact_bonus * center_bonus * max(brightness, 1.0)
        if score > upper_best_score:
            upper_best_score = score
            upper_best = (x, y, bw, bh)

    if upper_best is not None:
        return upper_best

    blurred = cv2.GaussianBlur(search, (0, 0), 3)
    bright_p = float(np.percentile(blurred, 99.2))
    mean_v = float(blurred.mean())
    std_v = float(blurred.std())
    threshold = max(mean_v + std_v * 2.0, bright_p * 0.65)
    _, mask = cv2.threshold(blurred.astype(np.uint8), int(np.clip(threshold, 1, 255)), 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1.0
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if area < 300:
            continue

        aspect = bw / max(bh, 1)
        if not (0.8 <= aspect <= 3.5):
            continue

        contour_area = cv2.contourArea(contour)
        fill_ratio = contour_area / max(area, 1)
        if fill_ratio < 0.28:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        rect_bonus = 1.35 if 4 <= len(approx) <= 8 else 1.0
        center_y = y + bh * 0.5
        upper_bonus = 1.45 if center_y < h * 0.22 else 1.0
        compact_bonus = 1.15 if aspect <= 2.4 else 0.85
        brightness = float(np.mean(search[y:y + bh, x:x + bw]))
        score = area * fill_ratio * rect_bonus * upper_bonus * compact_bonus * max(brightness, 1.0)
        if score > best_score:
            best_score = score
            best = (x, y, bw, bh)

    return best


def _find_bright_region_roi(gray):
    """ coating ,."""
    h, w = gray.shape
    search_h = max(1, int(h * 0.7))
    search = gray[:search_h, :]
    blurred = cv2.GaussianBlur(search, (0, 0), 5)
    peak_y, peak_x = np.unravel_index(np.argmax(blurred), blurred.shape)

    row_profile = blurred.mean(axis=1)
    col_profile = blurred.mean(axis=0)
    row_threshold = max(float(row_profile.mean() + row_profile.std() * 1.0), float(row_profile[peak_y] * 0.55))
    col_threshold = max(float(col_profile.mean() + col_profile.std() * 1.0), float(col_profile[peak_x] * 0.45))

    top = peak_y
    while top > 0 and row_profile[top] >= row_threshold:
        top -= 1
    bottom = peak_y
    while bottom < search.shape[0] - 1 and row_profile[bottom] >= row_threshold:
        bottom += 1

    left = peak_x
    while left > 0 and col_profile[left] >= col_threshold:
        left -= 1
    right = peak_x
    while right < search.shape[1] - 1 and col_profile[right] >= col_threshold:
        right += 1

    bw = max(40, right - left + 1)
    bh = max(28, bottom - top + 1)
    x = max(0, left)
    y = max(0, top)
    if bw * bh < 400:
        return None
    return (x, y, bw, bh)


def _build_presentable_coating_view(gray):
    """ coating , image."""
    h, w = gray.shape
    roi = _find_coating_roi(gray)
    use_bright_fallback = roi is None
    if use_bright_fallback:
        roi = _find_bright_region_roi(gray)
    if roi is None:
        return _mask_center_circle_gray(gray)

    x, y, bw, bh = roi
    roi_cx = x + bw / 2.0
    roi_cy = y + bh / 2.0
    target_cy = int(h * 0.6) if use_bright_fallback else h // 2

    
    shift_x = (w / 2.0) - roi_cx
    shift_y = target_cy - roi_cy
    translate_m = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(gray, translate_m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    base = shifted * 0.42

    
    pad_x = max(70, int(bw * 1.18))
    pad_top = max(56, int(bh * 1.32))
    pad_bottom = max(86, int(bh * 1.9))

    x1 = max(0, x - pad_x)
    x2 = min(w, x + bw + pad_x)
    y1 = max(0, y - pad_top)
    y2 = min(h, y + bh + pad_bottom)

    crop = gray[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return _mask_center_circle_gray(gray)

    
    p_low, p_high = np.percentile(crop, [10, 99.8])
    if p_high > p_low:
        crop = np.clip((crop - p_low) * (255.0 / (p_high - p_low)), 0, 255)
    crop = np.power(crop / 255.0, 1.08) * 255.0

    ch, cw = crop.shape
    target_w = min(int(w * 0.82), max(int(cw * 1.9), int(bw * 8.2)))
    target_h = min(int(h * 0.72), max(int(ch * 1.9), int(bh * 9.0)))
    scale = min(target_w / max(cw, 1), target_h / max(ch, 1))
    scale = max(scale, 1.0)

    resized_w = max(1, int(cw * scale))
    resized_h = max(1, int(ch * scale))
    zoomed = cv2.resize(crop, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)

    canvas = base.copy()
    target_cx = w // 2
    dst_x1 = max(0, target_cx - resized_w // 2)
    dst_y1 = max(0, target_cy - resized_h // 2)
    dst_x2 = min(w, dst_x1 + resized_w)
    dst_y2 = min(h, dst_y1 + resized_h)

    src_x1 = max(0, -(target_cx - resized_w // 2))
    src_y1 = max(0, -(target_cy - resized_h // 2))
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)

    overlay = zoomed[src_y1:src_y2, src_x1:src_x2]
    soft_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(soft_mask, (dst_x1, dst_y1), (dst_x2, dst_y2), 255, thickness=-1)
    soft_mask = cv2.GaussianBlur(soft_mask, (0, 0), 26).astype(np.float32) / 255.0

    canvas_region = canvas[dst_y1:dst_y2, dst_x1:dst_x2]
    mask_region = soft_mask[dst_y1:dst_y2, dst_x1:dst_x2][..., None]
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = (
        canvas_region * (1.0 - mask_region[..., 0]) + overlay * mask_region[..., 0]
    )

    return np.clip(canvas, 0, 255).astype(np.float32)


def _save_processed_gray_photo(src_path, dst_path):
    """ image, coating  PNG ."""
    raw = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if raw is None:
        raise RuntimeError(f"Cannot read the source photo: {src_path}")

    if len(raw.shape) == 2:
        gray = raw.astype(np.float32)
    else:
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32)

    processed_gray = _build_presentable_coating_view(gray)
    save_arr = np.clip(processed_gray, 0, 255).astype(np.uint8)
    if not cv2.imwrite(dst_path, save_arr):
        raise RuntimeError(f"Cannot write the processed photo: {dst_path}")
    return processed_gray

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    headers = {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Connection': 'keep-alive',
        'X-Content-Type-Options': 'nosniff'
    }
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace;boundary=frame',
        headers=headers,
        direct_passthrough=True
    )

@app.route('/start', methods=['POST'])
@log_execution_time
def start_recording():
    global recording_process, recording_cam_process, hls_process_cam, hls_process_mux
    
    with recording_lock:
        
        with hls_lock:
            try:
                if hls_process_cam or hls_process_mux:
                    logger.info("Stopping HLS before recording")
                    try:
                        if hls_process_cam and hls_process_cam.poll() is None:
                            hls_process_cam.terminate()
                    except Exception:
                        pass
                    try:
                        if hls_process_mux and hls_process_mux.poll() is None:
                            hls_process_mux.terminate()
                    except Exception:
                        pass
                    hls_process_cam = None
                    hls_process_mux = None
            except Exception:
                pass
        if recording_process is not None:
            return jsonify({'status': 'error', 'message': 'Recording is already active'}), 400
            
        try:
            data = get_request_data()
            quality = data.get('quality', '1080p')
            framerate = int(data.get('framerate', 30))
            bitrate = data.get('bitrate', 10)
            # Read exposure/gain from request (UI sliders), fall back to preview settings
            exposure_sec = data.get('exposure', None)
            gain_val = data.get('gain', None)
            if exposure_sec is None:
                exposure_sec = camera.desired_exposure_sec
            if gain_val is None:
                gain_val = camera.desired_gain
            if exposure_sec is not None:
                exposure_sec = max(0.001, float(exposure_sec))
            if gain_val is not None:
                gain_val = max(1.0, float(gain_val))

            if quality == '480p':
                width, height = 854, 480
            elif quality == '720p':
                width, height = 1280, 720
            else:  # 1080p
                width, height = 1920, 1080

            # rpicam-vid caps shutter to 1/framerate.
            # rpicam-vid minimum supported framerate is 1fps (fractional fps not supported).
            # So max effective exposure for regular recording is 1s.
            # For exposures >1s, use the Timelapse feature instead.
            framerate = float(framerate)
            if exposure_sec and exposure_sec > 0:
                max_fps_for_exposure = max(1.0, 1.0 / exposure_sec)
                if framerate > max_fps_for_exposure:
                    logger.info(f"Lowering framerate from {framerate} to {max_fps_for_exposure:.2f} to fit exposure {exposure_sec}s")
                    framerate = max_fps_for_exposure

            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            save_dir = '/home/cholab/camera_web/videos'
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(save_dir, f'video_{timestamp}.mp4')

            
            with camera_lock:
                logger.info("Stopping preview before recording")
                camera.stop_stream()
                camera.kill_camera_processes()  
                time.sleep(3)  

            
            camera_device = camera.check_camera_available()
            if not camera_device:
                return jsonify({'status': 'error', 'message': 'No available camera was detected; recording cannot start'}), 400

            
            if camera_device == 'libcamera' or camera_device.isdigit() or camera_device.startswith('/base/'):
                
                cam_app = shutil.which('rpicam-vid') or shutil.which('libcamera-vid')
                if not cam_app:
                    return jsonify({'status': 'error', 'message': 'rpicam-vid or libcamera-vid was not found; recording is unavailable'}), 500
                cam_cmd = [
                    cam_app,
                    '--codec', 'h264',
                    '-t', '0',
                    '--width', str(width),
                    '--height', str(height),
                    '--framerate', f'{framerate:.2f}',
                ]
                if exposure_sec and exposure_sec > 0:
                    cam_cmd += ['--shutter', str(max(1000, int(exposure_sec * 1_000_000)))]
                if gain_val and gain_val >= 1.0:
                    cam_cmd += ['--gain', f"{gain_val:.2f}"]
                cam_cmd += [
                    '-o', '-'
                ]
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-thread_queue_size', '512',
                    '-f', 'h264',
                    '-i', 'pipe:0',
                    '-f', 'lavfi',
                    '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-shortest',
                    '-movflags', '+faststart',
                    '-y',
                    filename
                ]
                try:
                    recording_cam_process = subprocess.Popen(cam_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
                    recording_process = subprocess.Popen(ffmpeg_cmd, stdin=recording_cam_process.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    
                    recording_cam_process.stdout.close()
                except Exception as e:
                    if recording_cam_process:
                        try:
                            recording_cam_process.terminate()
                        except Exception:
                            pass
                    recording_process = None
                    recording_cam_process = None
                    return jsonify({'status': 'error', 'message': f'Failed to start recording: {e}'}), 500
            else:
                
                v4l2_candidates = [
                    [
                    'ffmpeg',
                        '-f', 'v4l2',
                        '-input_format', 'mjpeg',
                        '-framerate', str(framerate),
                        '-video_size', f'{width}x{height}',
                        '-i', camera_device,
                    '-f', 'lavfi',
                    '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
                    '-c:v', 'libx264',
                    '-profile:v', 'baseline',
                    '-level', '3.1',
                    '-pix_fmt', 'yuv420p',
                    '-b:v', f'{bitrate}M',
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-shortest',
                    '-movflags', '+faststart',
                    '-preset', 'veryfast',
                    '-y',
                    filename
                    ],
                    [
                    'ffmpeg',
                    '-f', 'v4l2',
                        '-input_format', 'yuyv422',
                    '-framerate', str(framerate),
                    '-video_size', f'{width}x{height}',
                    '-i', camera_device,
                    '-f', 'lavfi',
                    '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
                    '-c:v', 'libx264',
                    '-profile:v', 'baseline',
                    '-level', '3.1',
                    '-pix_fmt', 'yuv420p',
                    '-b:v', f'{bitrate}M',
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-shortest',
                    '-movflags', '+faststart',
                    '-preset', 'veryfast',
                    '-y',
                    filename
                    ]
                ]
                last_err = ''
                for cmd in v4l2_candidates:
                    logger.info(f"Recording command: {' '.join(cmd)}")
                    recording_process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    time.sleep(2)  
                    if recording_process.poll() is None:
                        break
                    else:
                        last_err = (recording_process.stderr.read() or b'').decode(errors='ignore')
                        logger.warning(f"Recording pipeline failed: {last_err[:300]}")
                        recording_process = None
                if recording_process is None:
                    return jsonify({'status': 'error', 'message': f'Failed to start recording: {last_err}'}), 500
            time.sleep(2)  
            if recording_process.poll() is None:
                logger.info(f"Recording video to: {filename}")
                return jsonify({'status': 'success', 'message': f'Recording started: {os.path.basename(filename)}'})
            else:
                stderr_output = recording_process.stderr.read().decode()
                logger.error(f"Failed to start recording: {stderr_output}")
                try:
                    if recording_cam_process:
                        recording_cam_process.terminate()
                except Exception:
                    pass
                recording_process = None
                recording_cam_process = None
                with camera_lock:
                    time.sleep(2)  
                    camera.start_stream()
                return jsonify({'status': 'error', 'message': f'Failed to start recording: {stderr_output}'}), 500
                
        except Exception as e:
            logger.error(f"Recording-start exception: {e}")
            recording_process = None
            with camera_lock:
                camera.start_stream()
            return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/stop', methods=['POST'])
@log_execution_time
def stop_recording():
    global recording_process, recording_cam_process
    
    with recording_lock:
        if recording_process is None:
            return jsonify({'status': 'error', 'message': 'No recording is currently active'}), 400
            
        try:
            
            if recording_cam_process and recording_cam_process.poll() is None:
                try:
                    recording_cam_process.terminate()
                    recording_cam_process.wait(timeout=5)  
                except Exception:
                    try:
                        recording_cam_process.kill()
                    except Exception:
                        pass
            recording_cam_process = None
            
            if recording_process and recording_process.poll() is None:
                try:
                    stdout, stderr = recording_process.communicate(timeout=15)  
                except subprocess.TimeoutExpired:
                    recording_process.kill()
                    recording_process.wait()
            logger.info("Recording stopped")
            recording_process = None
            with camera_lock:
                logger.info("Restarting camera stream")
                time.sleep(1)  
                camera.start_stream()
            return jsonify({'status': 'success', 'message': 'Recording stopped and saved'})
                
        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            recording_process = None
            with camera_lock:
                camera.start_stream()
            return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/capture', methods=['POST'])
@log_execution_time
def capture_photo():
    """: image."""
    global hls_process_cam, hls_process_mux, capture_in_progress
    try:
        
        with hls_lock:
            try:
                if hls_process_cam and hls_process_cam.poll() is None:
                    hls_process_cam.terminate()
            except Exception:
                pass
            try:
                if hls_process_mux and hls_process_mux.poll() is None:
                    hls_process_mux.terminate()
            except Exception:
                pass
            hls_process_cam = None
            hls_process_mux = None
        data = get_request_data()
        quality = str(data.get('quality', '1080p'))
        exposure_sec = data.get('exposure', 0.05)  
        camera_gain = data.get('gain', 1.0)

        capture_size_map = {
            '480p': (854, 480),
            '720p': (1280, 720),
            '1080p': (1920, 1080),
        }
        capture_width, capture_height = capture_size_map.get(quality, (1920, 1080))
        
        
        exposure_sec = max(0.001, min(30.0, float(exposure_sec)))
        camera_gain = max(1.0, min(16.0, float(camera_gain)))
        
        save_dir = '/home/cholab/camera_web/photos'
        os.makedirs(save_dir, exist_ok=True)
        raw_filename, filename = _next_photo_paths(save_dir)
        
        logger.info(f"Capture settings: exposure={exposure_sec} s, gain={camera_gain}")
        capture_in_progress = True
        
        
        with camera_lock:
            camera_was_running = camera.running
            if camera_was_running:
                logger.info("Stopping camera stream")
                camera.stop_stream()
                camera.kill_camera_processes()  
                time.sleep(3)  
        
        try:
            
            camera_device = camera.check_camera_available()
            if not camera_device:
                logger.warning("Camera unavailable; using the test image")
                
                test_src = '/home/cholab/camera_web/test.jpg'
                try:
                    if os.path.exists(test_src):
                        shutil.copyfile(test_src, raw_filename)
                        _save_processed_gray_photo(raw_filename, filename)
                        try:
                            os.remove(raw_filename)
                        except Exception:
                            pass
                        file_size = os.path.getsize(filename) / 1024
                        logger.info(f"Test image saved: {filename} ({file_size:.1f}KB)")
                        return jsonify({
                            'status': 'success',
                            'message': f'Test image saved: {os.path.basename(filename)}',
                            'filename': os.path.basename(filename),
                            'size': f'{file_size:.1f}KB',
                            'exposure_sec': exposure_sec,
                            'gain': camera_gain
                        })
                    else:
                        return jsonify({'status': 'error', 'message': 'Test image test.jpg was not found'}), 500
                except Exception as e:
                    return jsonify({'status': 'error', 'message': f'Test-image processing failed: {e}'}), 500
            
            
            if camera_device == 'libcamera' or camera_device.isdigit() or camera_device.startswith('/base/'):
                still_app = shutil.which('rpicam-still') or shutil.which('libcamera-still')
                if not still_app:
                    return jsonify({'status': 'error', 'message': 'rpicam-still or libcamera-still was not found; photo capture is unavailable'}), 500
                shutter_us = max(1000, int(float(exposure_sec) * 1_000_000))
                gain_val = max(1.0, float(camera_gain))
                cmd = [
                    still_app,
                    '-o', raw_filename,
                    '--width', str(capture_width),
                    '--height', str(capture_height),
                    '--shutter', str(shutter_us),
                    '--gain', f'{gain_val:.2f}',
                    '-t', '1'
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=max(20, exposure_sec * 2 + 5))  
                if result.returncode != 0:
                    return jsonify({'status': 'error', 'message': f'Photo capture failed: {result.stderr}'}), 500
                
                err_msg = ''
            else:
                
                photo_candidates = [
                    [
                        'ffmpeg',
                        '-f', 'v4l2',
                        '-input_format', 'mjpeg',
                        '-framerate', '15',
                        '-video_size', f'{capture_width}x{capture_height}',
                        '-i', camera_device,
                        '-frames:v', '1',
                        '-q:v', '2',
                        '-y',
                        raw_filename
                    ],
                    [
                    'ffmpeg',
                    '-f', 'v4l2',
                        '-input_format', 'yuyv422',
                        '-framerate', '15',
                        '-video_size', f'{capture_width}x{capture_height}',
                    '-i', camera_device,
                    '-frames:v', '1',
                    '-q:v', '2',
                    '-y',
                    raw_filename
                ]
                ]
                cmd = photo_candidates[0]
            
            logger.info(f"Photo command: {' '.join(cmd)}")
            
            
            timeout_duration = max(20, exposure_sec * 2 + 10)  
            
            
            if 'photo_candidates' in locals():
                err_msg = ''
                for attempt_cmd in photo_candidates:
                    result = subprocess.run(
                        attempt_cmd,
                        capture_output=True,
                        text=True,
                        timeout=timeout_duration
                    )
                    if result.returncode == 0:
                        cmd = attempt_cmd
                        break
                    else:
                        err_msg = (result.stderr or '').strip()
            
            if os.path.exists(raw_filename):
                time.sleep(0.5)  

                _save_processed_gray_photo(raw_filename, filename)
                try:
                    os.remove(raw_filename)
                except Exception:
                    pass

                if os.path.exists(filename) and os.path.getsize(filename) > 1000:
                    file_size = os.path.getsize(filename) / 1024
                    logger.info(f"Photo saved: {filename} ({file_size:.1f}KB)")
                    
                    return jsonify({
                        'status': 'success', 
                        'message': f'Photo saved: {os.path.basename(filename)}',
                        'filename': os.path.basename(filename),
                        'size': f'{file_size:.1f}KB',
                        'exposure_sec': exposure_sec,
                        'gain': camera_gain
                    })
                else:
                    return jsonify({'status': 'error', 'message': 'Photo file was not created'}), 500
            else:
                logger.error(f"Photo capture failed: {err_msg}")
                return jsonify({'status': 'error', 'message': f'Photo capture failed: {err_msg}'}), 500
        
        finally:
            capture_in_progress = False
            
            with camera_lock:
                if camera_was_running:
                    logger.info("Restarting camera stream")
                    time.sleep(3)  
                    camera.start_stream()
            
            
    except subprocess.TimeoutExpired:
        return jsonify({'status': 'error', 'message': f'Photo capture timed out after {timeout_duration} seconds'}), 500
    except Exception as e:
        logger.error(f"Photo-capture exception: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/status')
def get_status():
    """ - """
    with recording_lock:
        is_recording = recording_process is not None
    
    storage_info = None
    try:
        home_path = '/home/cholab'
        total, used, free = shutil.disk_usage(home_path)
        storage_info = {
            'total': total,
            'used': used,
            'free': free
        }
    except Exception as e:
        logger.error(f"Failed to obtain storage information: {e}")
    
    
    camera_online = True
    try:
        if camera.running:
            camera_online = True
        else:
            
            test_cmd = ['libcamera-hello', '--list-cameras']
            result = subprocess.run(test_cmd, capture_output=True, timeout=3)  
            camera_online = result.returncode == 0 and 'No cameras available' not in result.stderr.decode()
    except Exception:
        camera_online = False
    
    return jsonify({
        'recording': is_recording,
        'camera_online': camera_online,
        'storage': storage_info
    })

@app.route('/set_preview', methods=['POST'])
@log_execution_time
def set_preview():
    """ - """
    try:
        data = get_request_data()
        width = int(data.get('width', 640))
        height = int(data.get('height', 480))
        fps = int(data.get('fps', 15))
        exposure_sec = data.get('exposure')
        gain = data.get('gain')
        
        logger.info(f"set_preview received parameters: width={width}, height={height}, fps={fps}, exposure={exposure_sec}, gain={gain}")

        
        width = max(160, min(1920, width))
        height = max(120, min(1080, height))
        fps = max(5, min(60, fps))

        with camera_lock:
            camera.desired_width = width
            camera.desired_height = height
            camera.desired_fps = fps
            
            camera.frame_interval = 1.0 / fps
            try:
                camera.desired_exposure_sec = float(exposure_sec) if exposure_sec is not None else None
                logger.info(f"Exposure set to {camera.desired_exposure_sec} s")
            except Exception as e:
                camera.desired_exposure_sec = None
                logger.warning(f"Failed to set exposure: {e}")
            try:
                camera.desired_gain = float(gain) if gain is not None else None
                logger.info(f"Gain set to {camera.desired_gain}")
            except Exception as e:
                camera.desired_gain = None
                logger.warning(f"Failed to set gain: {e}")
            
            camera.stop_stream()
            time.sleep(0.5)  
            success = camera.start_stream()
            if not success:
                logger.error("Failed to restart camera stream")

        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Failed to set preview parameters: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
@app.route('/focus', methods=['POST'])
@log_execution_time
def set_focus_window():
    """
    :(cx, cy), autofocus-mode=auto
     rpicam-vid/libcamera-vid available;v4l2 .
    """
    try:
        data = get_request_data() or {}
        cx = float(data.get('x'))
        cy = float(data.get('y'))
        box = data.get('box')  
        if box and isinstance(box, (list, tuple)) and len(box) == 2:
            bw, bh = float(box[0]), float(box[1])
        else:
            bw, bh = 0.2, 0.2  

        
        x = max(0.0, min(1.0, cx - bw / 2.0))
        y = max(0.0, min(1.0, cy - bh / 2.0))
        if x + bw > 1.0: x = 1.0 - bw
        if y + bh > 1.0: y = 1.0 - bh

        with camera_lock:
            camera.af_window = (x, y, bw, bh)
            camera.af_mode = 'auto'  

            dev = camera.check_camera_available()
            if not dev:
                return jsonify({'status': 'error', 'message': 'No camera device detected'}), 400
            if not (dev == 'libcamera' or str(dev).isdigit() or str(dev).startswith('/base/')):
                return jsonify({'status': 'error', 'message': 'Tap-to-focus is unavailable for the current video backend; rpicam-vid or libcamera-vid is required'}), 400

            
            camera.stop_stream()
            time.sleep(0.3)
            if not camera.start_stream():
                return jsonify({'status': 'error', 'message': 'Failed to apply the focus window because the stream could not be started'}), 500

        return jsonify({'status': 'success', 'x': x, 'y': y, 'w': bw, 'h': bh})
    except Exception as e:
        logger.error(f'Failed to set tap-to-focus: {e}')
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/photos/<path:subpath>')
def serve_photo(subpath):
    base_dir = os.path.abspath('/home/cholab/camera_web/photos')
    safe_path = os.path.abspath(os.path.join(base_dir, subpath))
    if os.path.commonpath([base_dir, safe_path]) != base_dir:
        return jsonify({'status': 'error', 'message': 'Invalid path'}), 400
    return send_from_directory(os.path.dirname(safe_path), os.path.basename(safe_path))

@app.route('/videos/<path:subpath>')
def serve_video(subpath):
    """ ()"""
    base_dir = os.path.abspath('/home/cholab/camera_web/videos')
    safe_path = os.path.abspath(os.path.join(base_dir, subpath))
    if os.path.commonpath([base_dir, safe_path]) != base_dir:
        return jsonify({'status': 'error', 'message': 'Invalid path'}), 400
    directory = os.path.dirname(safe_path)
    fname = os.path.basename(safe_path)
    return send_from_directory(directory, fname)

@app.route('/files')
def get_files():
    """ ()- """
    try:
        def walk_collect(root, exts):
            items = []
            for dirpath, dirnames, filenames in os.walk(root):
                for fn in filenames:
                    if fn.lower().endswith(exts):
                        fp = os.path.join(dirpath, fn)
                        try:
                            size = os.path.getsize(fp)
                            mtime = os.path.getmtime(fp)
                            rel = os.path.relpath(fp, root)
                            items.append({'name': rel.replace('\\','/'), 'size': size, 'modified': mtime})
                        except Exception as e:
                            logger.warning(f"Failed to obtain file information {fp}: {e}")
            return items

        video_dir = '/home/cholab/camera_web/videos'
        photo_dir = '/home/cholab/camera_web/photos'

        videos = walk_collect(video_dir, ('.mp4', '.h264', '.avi', '.mov', '.mkv')) if os.path.exists(video_dir) else []
        photos = walk_collect(photo_dir, ('.jpg', '.jpeg', '.png')) if os.path.exists(photo_dir) else []

        videos.sort(key=lambda x: x['modified'], reverse=True)
        photos.sort(key=lambda x: x['modified'], reverse=True)

        return jsonify({'videos': videos[:10], 'photos': photos[:10]})
    except Exception as e:
        logger.error(f"Failed to obtain media list: {e}")
        return jsonify({'videos': [], 'photos': []}), 500


def _ensure_hls_dir() -> str:
    out_dir = '/home/cholab/camera_web/videos/hls'
    try:
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return out_dir

@app.route('/hls/start', methods=['POST'])
@log_execution_time
def hls_start():
    """HLS:rpicam-vid/libcamera-vid(h264)->ffmpeg - """
    global hls_process_cam, hls_process_mux
    try:
        with hls_lock:
            if hls_process_mux is not None:
                return jsonify({'status': 'success'})

            out_dir = _ensure_hls_dir()

            
            with camera_lock:
                camera.stop_stream()
                camera.kill_camera_processes()
                time.sleep(0.5)  

            cam_app = shutil.which('rpicam-vid') or shutil.which('libcamera-vid')
            if not cam_app:
                return jsonify({'status': 'error', 'message': 'rpicam-vid or libcamera-vid was not found'}), 500

            cam_cmd = [
                cam_app,
                '--codec', 'h264',
                '-n', '-t', '0',
                '--width', str(camera.desired_width),
                '--height', str(camera.desired_height),
                '--framerate', str(camera.desired_fps),
                '-o', '-'
            ]
            try:
                if camera.desired_exposure_sec and float(camera.desired_exposure_sec) > 0:
                    cam_cmd += ['--shutter', str(max(1000, int(float(camera.desired_exposure_sec)*1_000_000)))]
            except Exception:
                pass
            try:
                if camera.desired_gain and float(camera.desired_gain) >= 1.0:
                    cam_cmd += ['--gain', f"{float(camera.desired_gain):.2f}"]
            except Exception:
                pass

            out_index = os.path.join(out_dir, 'index.m3u8')
            hls_cmd = [
                'ffmpeg',
                '-thread_queue_size', '1024',
                '-f', 'h264', '-i', 'pipe:0',
                '-c:v', 'copy',
                '-hls_time', '1',
                '-hls_list_size', '6',
                '-hls_flags', 'delete_segments+append_list+independent_segments+split_by_time',
                '-hls_segment_type', 'mpegts',
                '-y', out_index
            ]

            hls_process_cam = subprocess.Popen(cam_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            hls_process_mux = subprocess.Popen(hls_cmd, stdin=hls_process_cam.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(1)  
            return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Failed to start HLS: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/hls/stop', methods=['POST'])
@log_execution_time
def hls_stop():
    """HLS MJPEG  - """
    global hls_process_cam, hls_process_mux
    try:
        with hls_lock:
            try:
                if hls_process_cam and hls_process_cam.poll() is None:
                    hls_process_cam.terminate()
            except Exception:
                pass
            try:
                if hls_process_mux and hls_process_mux.poll() is None:
                    hls_process_mux.terminate()
            except Exception:
                pass
            hls_process_cam = None
            hls_process_mux = None

        with camera_lock:
            time.sleep(0.5)  
            camera.start_stream()
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Failed to stop HLS: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/hls/<path:filename>')
def hls_files(filename):
    base = os.path.abspath(_ensure_hls_dir())
    safe = os.path.abspath(os.path.join(base, filename))
    if os.path.commonpath([base, safe]) != base:
        return jsonify({'status': 'error', 'message': 'Invalid path'}), 400
    return send_from_directory(os.path.dirname(safe), os.path.basename(safe))

# ============================================================

# ============================================================

def _apply_led_correction(img_gray, lmap):
    """LED.img_gray: float32 H×W,lmap: float32 H×W."""
    if lmap is None:
        return img_gray.astype(np.float32)
    lm = lmap
    if lm.shape != img_gray.shape:
        lm = cv2.resize(lm, (img_gray.shape[1], img_gray.shape[0]),
                        interpolation=cv2.INTER_LINEAR)
    return img_gray.astype(np.float32) / np.maximum(lm, 1e-6)


def _crop_roi(img, roi_xywh):
    """ROI, image."""
    h, w = img.shape[:2]
    x, y, rw, rh = roi_xywh
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + rw), min(h, y + rh)
    if x2 > x1 and y2 > y1:
        return img[y1:y2, x1:x2], (x1, y1, x2, y2)
    return img, (0, 0, w, h)




def _local_std(x, win):
    mean  = uniform_filter(x, size=win)
    mean2 = uniform_filter(x * x, size=win)
    return np.sqrt(np.maximum(mean2 - mean * mean, 0.0))

def _speckle_contrast(x, win):
    mu = uniform_filter(x, size=win)
    sd = _local_std(x, win)
    return sd / np.maximum(mu, 1e-6)

def _local_corr_drop(a, b, win):
    ma  = uniform_filter(a, size=win)
    mb  = uniform_filter(b, size=win)
    sa  = _local_std(a, win)
    sb  = _local_std(b, win)
    cov = uniform_filter((a - ma) * (b - mb), size=win)
    corr = np.clip(cov / np.maximum(sa * sb, 1e-6), -1, 1)
    return 1.0 - corr

def extract_feature_bank_29_legacy(before_corr, after_corr):
    """29, biosensor_29ch.pt."""
    b = before_corr.astype(np.float32)
    a = after_corr.astype(np.float32)

    log_ratio     = np.log(a + 1.0) - np.log(b + 1.0)
    relative_diff = (a - b) / np.maximum(b, 1e-3)
    sym_ratio     = (a - b) / np.maximum(a + b, 1e-3)
    feats = [log_ratio, relative_diff, sym_ratio]

    for sig in [1, 2, 4, 8, 12]:
        feats.append(gaussian_filter(log_ratio, sigma=sig))
    for sig in [1, 2, 4, 8, 12]:
        feats.append(gaussian_filter(relative_diff, sigma=sig))
    for s1, s2 in [(1, 2), (2, 4), (4, 8), (8, 12)]:
        feats.append(gaussian_filter(log_ratio, s1) - gaussian_filter(log_ratio, s2))
    for w in [7, 13, 21]:
        feats.append(uniform_filter(a, size=w) - uniform_filter(b, size=w))
    for w in [7, 13, 21]:
        feats.append(_local_std(a, w) - _local_std(b, w))
    for w in [7, 13]:
        feats.append(_speckle_contrast(a, w) - _speckle_contrast(b, w))

    low_a  = gaussian_filter(a, sigma=4)
    low_b  = gaussian_filter(b, sigma=4)
    feats.append(np.abs(a - low_a) - np.abs(b - low_b))  # high_freq diff
    feats.append(np.abs(low_a) - np.abs(low_b))           # low_freq diff

    for w in [7, 13]:
        feats.append(_local_corr_drop(a, b, w))

    return np.stack(feats, axis=0).astype(np.float32)  # [29,128,128]


def _local_contrast(x, win):
    mu = uniform_filter(x, size=win)
    sd = _local_std(x, win)
    return sd / np.maximum(mu, 1e-6)


def extract_feature_bank_29_separate(before_corr, after_corr):
    """29, resnet29_full_parity.py ."""
    b = before_corr.astype(np.float32)
    a = after_corr.astype(np.float32)

    diff = a - b
    abs_diff = np.abs(diff)
    log1p_diff = np.log1p(np.maximum(a, 0.0)) - np.log1p(np.maximum(b, 0.0))
    ratio_minus1 = a / np.maximum(b, 1e-3) - 1.0
    relative_diff_clip = np.clip(diff / np.maximum(b, 1e-3), -3.0, 3.0)

    gauss_diff_s1 = gaussian_filter(a, sigma=1) - gaussian_filter(b, sigma=1)
    gauss_diff_s3 = gaussian_filter(a, sigma=3) - gaussian_filter(b, sigma=3)
    gauss_diff_s6 = gaussian_filter(a, sigma=6) - gaussian_filter(b, sigma=6)

    local_std_delta_w3 = _local_std(a, 3) - _local_std(b, 3)
    local_std_delta_w7 = _local_std(a, 7) - _local_std(b, 7)
    local_std_delta_w15 = _local_std(a, 15) - _local_std(b, 15)

    local_contrast_delta_w5 = _local_contrast(a, 5) - _local_contrast(b, 5)
    local_contrast_delta_w9 = _local_contrast(a, 9) - _local_contrast(b, 9)

    lowpass_delta = gaussian_filter(a, sigma=8) - gaussian_filter(b, sigma=8)
    highpass_delta = (a - gaussian_filter(a, sigma=2)) - (b - gaussian_filter(b, sigma=2))
    highpass_energy_delta = (
        np.square(a - gaussian_filter(a, sigma=2))
        - np.square(b - gaussian_filter(b, sigma=2))
    )

    bandpass_diff_s1 = (
        gaussian_filter(a, sigma=1) - gaussian_filter(a, sigma=2)
        - gaussian_filter(b, sigma=1) + gaussian_filter(b, sigma=2)
    )
    bandpass_diff_s2 = (
        gaussian_filter(a, sigma=2) - gaussian_filter(a, sigma=4)
        - gaussian_filter(b, sigma=2) + gaussian_filter(b, sigma=4)
    )
    bandpass_diff_s4 = (
        gaussian_filter(a, sigma=4) - gaussian_filter(a, sigma=8)
        - gaussian_filter(b, sigma=4) + gaussian_filter(b, sigma=8)
    )

    before_raw = b
    after_raw = a
    before_gauss_s2 = gaussian_filter(b, sigma=2)
    after_gauss_s2 = gaussian_filter(a, sigma=2)
    local_std_diff_w3 = np.abs(_local_std(a, 3) - _local_std(b, 3))
    local_std_diff_w7 = np.abs(_local_std(a, 7) - _local_std(b, 7))
    gauss_absdiff_delta = gaussian_filter(abs_diff, sigma=1) - abs_diff
    square_delta = np.square(a) - np.square(b)
    gauss_absdiff_s3 = gaussian_filter(abs_diff, sigma=3)
    positive_diff = np.maximum(diff, 0.0)

    feats = [
        diff,
        abs_diff,
        log1p_diff,
        ratio_minus1,
        relative_diff_clip,
        gauss_diff_s1,
        gauss_diff_s3,
        gauss_diff_s6,
        local_std_delta_w3,
        local_std_delta_w7,
        local_std_delta_w15,
        local_contrast_delta_w5,
        local_contrast_delta_w9,
        lowpass_delta,
        highpass_delta,
        highpass_energy_delta,
        bandpass_diff_s1,
        bandpass_diff_s2,
        bandpass_diff_s4,
        before_raw,
        after_raw,
        before_gauss_s2,
        after_gauss_s2,
        local_std_diff_w3,
        local_std_diff_w7,
        gauss_absdiff_delta,
        square_delta,
        gauss_absdiff_s3,
        positive_diff,
    ]
    return np.stack(feats, axis=0).astype(np.float32)


def extract_feature_bank_29_concentration(before, after):
    """Match the feature order and operations used to train concentration_classifier.pt."""
    before = before.astype(np.float32)
    after = after.astype(np.float32)
    diff = after - before
    abs_diff = np.abs(diff)
    low_before = gaussian_filter(before, 8)
    low_after = gaussian_filter(after, 8)
    high_before = before - low_before
    high_after = after - low_after

    def local_std(x, win):
        mean = uniform_filter(x, size=win)
        mean2 = uniform_filter(x * x, size=win)
        return np.sqrt(np.maximum(mean2 - mean * mean, 0))

    def local_contrast(x, win):
        return local_std(x, win) / (uniform_filter(x, size=win) + 1e-3)

    feats = [
        diff,
        abs_diff,
        np.log1p(abs_diff) * np.sign(diff),
        after / (before + 1e-3) - 1.0,
        np.clip(diff / (before + 1e-3), -2, 2),
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


def _normalize01(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)


def _build_response_maps(z_stack):
    """z_stack [29,H,W] →  image."""
    sort_idx = np.argsort(z_stack, axis=0)[::-1]
    sort_val = np.take_along_axis(z_stack, sort_idx, axis=0)
    top1    = sort_val[0].copy()
    top3w   = 0.6 * sort_val[0] + 0.3 * sort_val[1] + 0.1 * sort_val[2]
    top5rms = np.sqrt(np.mean(sort_val[:5] ** 2, axis=0))
    support = (z_stack > 2.5).sum(axis=0).astype(np.float32)
    return top1, top3w, top5rms, support


def _build_aux(before_corr, after_corr, feat_stack, pbs_mu, pbs_std):
    """
     aux  [aux_ch, H, W] (aux_ch :v3=6, v4=8).
    pbs_mu/pbs_std:  checkpoint PBSbaseline (None,).
    """
    if pbs_mu is not None and pbs_std is not None:
        pm, ps = pbs_mu, pbs_std
        if pm.shape != feat_stack.shape:
            pm = np.stack([cv2.resize(pm[i], (feat_stack.shape[2], feat_stack.shape[1]),
                           interpolation=cv2.INTER_LINEAR) for i in range(29)], axis=0)
            ps = np.stack([cv2.resize(ps[i], (feat_stack.shape[2], feat_stack.shape[1]),
                           interpolation=cv2.INTER_LINEAR) for i in range(29)], axis=0)
        z_stack = (feat_stack - pm) / np.maximum(ps, 1e-6)
    else:
        
        mu = feat_stack.mean(axis=(1, 2), keepdims=True)
        sd = feat_stack.std(axis=(1, 2), keepdims=True)
        z_stack = (feat_stack - mu) / (sd + 1e-6)

    top1, top3w, top5rms, support = _build_response_maps(z_stack)
    spread13 = _normalize01(np.abs(_normalize01(top1) - _normalize01(top3w)))
    spread35 = _normalize01(np.abs(_normalize01(top5rms) - _normalize01(top3w)))

    
    aux_ch = biosensor_model.aux_ch if biosensor_model is not None else 6
    if aux_ch == 8:
        channels = [
            _normalize01(top1), _normalize01(top3w), _normalize01(top5rms),
            _normalize01(support), spread13, spread35,
            _normalize01(before_corr), _normalize01(after_corr)
        ]
    else:
        channels = [
            _normalize01(top1), _normalize01(top3w), _normalize01(top5rms),
            _normalize01(support), spread13, spread35
        ]

    return np.stack(channels, axis=0).astype(np.float32)  # [aux_ch, 128, 128]


def _grab_gray_from_stream(timeout=3.0):
    """MJPEG,float32None."""
    try:
        frame_bytes = camera.frame_queue.get(timeout=timeout)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img.astype(np.float32)
    except Exception as e:
        logger.warning(f"Failed to capture a frame from the stream: {e}")
    return None


def _capture_frame_to_path(save_path, width=640, height=480,
                            exposure_sec=0.05, gain=1.0):
    """ →  → ,save_path."""
    with camera_lock:
        camera_was_running = camera.running
        if camera_was_running:
            camera.stop_stream()
            camera.kill_camera_processes()
            time.sleep(2)
    try:
        camera_device = camera.check_camera_available()
        if not camera_device:
            raise RuntimeError("No camera device detected")
        if (camera_device == 'libcamera' or
                str(camera_device).isdigit() or
                str(camera_device).startswith('/base/')):
            still_app = shutil.which('rpicam-still') or shutil.which('libcamera-still')
            if not still_app:
                raise RuntimeError("rpicam-still or libcamera-still was not found")
            shutter_us = max(1000, int(float(exposure_sec) * 1_000_000))
            cmd = [still_app, '-o', save_path,
                   '--width', str(width), '--height', str(height),
                   '--shutter', str(shutter_us),
                   '--gain', f'{max(1.0, float(gain)):.2f}',
                   '-t', '1']
        else:
            cmd = ['ffmpeg', '-f', 'v4l2', '-input_format', 'mjpeg',
                   '-framerate', '15', '-video_size', f'{width}x{height}',
                   '-i', camera_device,
                   '-frames:v', '1', '-q:v', '2', '-y', save_path]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=max(20, exposure_sec * 2 + 10))
        if result.returncode != 0:
            raise RuntimeError(f"Photo command failed: {result.stderr[:300]}")
        if not os.path.exists(save_path) or os.path.getsize(save_path) < 100:
            raise RuntimeError("The output file was not created or is too small")
    finally:
        with camera_lock:
            if camera_was_running:
                time.sleep(2)
                camera.start_stream()


def _load_gray(path):
    """ image float32  image."""
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _to_png_b64(gray_or_bgr):
    arr = np.clip(gray_or_bgr, 0, 255).astype(np.uint8)
    _, buf = cv2.imencode('.png', arr)
    return base64.b64encode(buf.tobytes()).decode('utf-8')


def _normalize_feat_stack(feat_stack, ch_mean, ch_std):
    return ((feat_stack - ch_mean[:, None, None]) / np.maximum(ch_std[:, None, None], 1e-6)).astype(np.float32)


# ============================================================

# ============================================================

@app.route('/capture_led_frame', methods=['POST'])
@log_execution_time
def capture_led_frame():
    """Grab a single live frame for ROI selection before LED calibration."""
    try:
        if not camera.running:
            return jsonify({'status': 'error',
                            'message': 'Camera not streaming. Start preview first.'}), 400
        img = _grab_gray_from_stream(timeout=5.0)
        if img is None:
            return jsonify({'status': 'error',
                            'message': 'Failed to grab frame from stream.'}), 500
        save_path = os.path.join(DETECTION_DIR, 'led_frame.png')
        cv2.imwrite(save_path, img.astype(np.uint8))
        _, buf = cv2.imencode('.png', img.astype(np.uint8))
        img_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
        h, w = img.shape
        return jsonify({'status': 'success', 'image': img_b64, 'width': w, 'height': h})
    except Exception as e:
        logger.error(f"capture_led_frame error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/calibrate_led', methods=['POST'])
@log_execution_time
def calibrate_led():
    """Capture N frames, crop to ROI if provided, build LED flat-field correction map."""
    global led_map
    try:
        data = get_request_data() or {}
        roi_data = data.get('roi') or {}
        roi_xywh = None
        if roi_data and int(roi_data.get('w', 0)) > 0 and int(roi_data.get('h', 0)) > 0:
            roi_xywh = (int(roi_data.get('x', 0)), int(roi_data.get('y', 0)),
                        int(roi_data.get('w', 0)), int(roi_data.get('h', 0)))

        photo_file = data.get('photo_filename')
        if photo_file:
            # Load from existing photo file
            photo_path = os.path.join('/home/cholab/camera_web/photos',
                                      os.path.basename(photo_file))
            raw = cv2.imread(photo_path)
            if raw is None:
                return jsonify({'status': 'error',
                                'message': f'Cannot read photo: {photo_file}'}), 400
            img = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if roi_xywh:
                fh, fw = img.shape
                x, y, rw, rh = roi_xywh
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(fw, x + rw), min(fh, y + rh)
                if x2 > x1 and y2 > y1:
                    img = img[y1:y2, x1:x2]
            led_map = img / (np.max(img) + 1e-8)
            np.save(LED_MAP_PATH, led_map)
            roi_info = f' ROI={roi_xywh}' if roi_xywh else ''
            logger.info(f"LED map saved from file: {photo_file}, shape={led_map.shape}{roi_info}")
            return jsonify({'status': 'success',
                            'message': f'LED calibration done from {os.path.basename(photo_file)}{roi_info}',
                            'shape': list(led_map.shape)})

        # Fallback: no photo provided, so LED calibration must grab frames from the live preview.
        if not camera.running:
            return jsonify({'status': 'error',
                            'message': 'Camera not streaming. Start preview first.'}), 400
        n_frames = max(1, int(data.get('n_frames', 5)))
        images = []
        for i in range(n_frames):
            img = _grab_gray_from_stream(timeout=5.0)
            if img is None:
                return jsonify({'status': 'error',
                                'message': f'Failed to grab frame {i+1}. Ensure camera is streaming.'}), 500
            if roi_xywh:
                fh, fw = img.shape
                x, y, rw, rh = roi_xywh
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(fw, x + rw), min(fh, y + rh)
                if x2 > x1 and y2 > y1:
                    img = img[y1:y2, x1:x2]
            images.append(img)
            time.sleep(0.3)
        stack = np.stack(images, axis=0)
        avg = np.mean(stack, axis=0)
        led_map = avg / (np.max(avg) + 1e-8)
        np.save(LED_MAP_PATH, led_map)
        roi_info = f' ROI={roi_xywh}' if roi_xywh else ''
        logger.info(f"LED map saved from stream, shape={led_map.shape}{roi_info}")
        return jsonify({'status': 'success',
                        'message': f'LED calibration done ({n_frames} frames){roi_info}',
                        'shape': list(led_map.shape)})
    except Exception as e:
        logger.error(f"calibrate_led error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/capture_before', methods=['POST'])
@log_execution_time
def capture_before():
    """before →  →  → base64ROI."""
    try:
        os.makedirs(DETECTION_DIR, exist_ok=True)
        data = get_request_data() or {}
        exposure_sec = float(data.get('exposure', 0.05))
        gain = float(data.get('gain', 1.0))
        save_path = os.path.join(DETECTION_DIR, 'before.png')
        _capture_frame_to_path(save_path, width=640, height=480,
                               exposure_sec=exposure_sec, gain=gain)
        img = cv2.imread(save_path)
        if img is None:
            return jsonify({'status': 'error', 'message': 'Failed to read captured image'}), 500
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(save_path, gray)
        _, buf = cv2.imencode('.png', gray)
        img_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
        h, w = gray.shape
        return jsonify({'status': 'success', 'image': img_b64, 'width': w, 'height': h})
    except Exception as e:
        logger.error(f"capture_before failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/detect', methods=['POST'])
@log_execution_time
def detect():
    """
    Prefer the separate 2026 models:
      classifier(best_classifier.pt) → PBS / Analyte
      refiner(stage2a_refine.pt)     → analyte  image
    Fall back to the legacy staged model when necessary.
    """
    if not TORCH_AVAILABLE:
        return jsonify({'status': 'error',
                        'message': 'PyTorch not installed'}), 500
    if not SCIPY_AVAILABLE:
        return jsonify({'status': 'error',
                        'message': 'SciPy is not installed. Run: pip3 install scipy'}), 500
    if BIOSENSOR_MODE not in ('separate_v2026', 'v29'):
        return jsonify({'status': 'error',
                        'message': 'No detection model is available. Provide best_classifier.pt and stage2a_refine.pt, or the legacy biosensor_29ch.pt'}), 500
    try:
        data = get_request_data() or {}
        roi_data = data.get('roi') or {}
        before_roi_data = data.get('before_roi') or roi_data
        after_roi_data = data.get('after_roi') or before_roi_data
        before_roi_xywh = (
            int(before_roi_data.get('x', 0)),
            int(before_roi_data.get('y', 0)),
            int(before_roi_data.get('w', 640)),
            int(before_roi_data.get('h', 480)),
        )
        after_roi_xywh = (
            int(after_roi_data.get('x', 0)),
            int(after_roi_data.get('y', 0)),
            int(after_roi_data.get('w', before_roi_xywh[2])),
            int(after_roi_data.get('h', before_roi_xywh[3])),
        )

        
        before_file = data.get('before_filename')
        if before_file:
            before_gray = _load_gray(os.path.join('/home/cholab/camera_web/photos', os.path.basename(before_file)))
            if before_gray is None:
                return jsonify({'status': 'error',
                                'message': f'Cannot read the before image: {before_file}'}), 400
        else:
            before_path = os.path.join(DETECTION_DIR, 'before.png')
            if not os.path.exists(before_path):
                return jsonify({'status': 'error',
                                'message': 'The before image is missing. Capture or select a before image first'}), 400
            before_gray = cv2.imread(before_path, cv2.IMREAD_GRAYSCALE)
            if before_gray is None:
                return jsonify({'status': 'error', 'message': 'Cannot read the before image'}), 500
            before_gray = before_gray.astype(np.float32)

        
        after_file = data.get('after_filename')
        if after_file:
            after_gray = _load_gray(os.path.join('/home/cholab/camera_web/photos', os.path.basename(after_file)))
            if after_gray is None:
                return jsonify({'status': 'error',
                                'message': f'Cannot read the after image: {after_file}'}), 400
        else:
            after_gray = _grab_gray_from_stream(timeout=3.0)
            if after_gray is None:
                tmp_path = os.path.join(DETECTION_DIR, '_after_tmp.jpg')
                try:
                    _capture_frame_to_path(tmp_path, width=640, height=480)
                    after_gray = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)
                    if after_gray is None:
                        return jsonify({'status': 'error',
                                        'message': 'Cannot capture the after frame'}), 500
                    after_gray = after_gray.astype(np.float32)
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

        
        if before_gray.shape != after_gray.shape:
            before_gray = cv2.resize(
                before_gray, (after_gray.shape[1], after_gray.shape[0]),
                interpolation=cv2.INTER_LINEAR).astype(np.float32)

        
        after_roi_display, _ = _crop_roi(after_gray, after_roi_xywh)

        
        before_cropped, _ = _crop_roi(before_gray, before_roi_xywh)
        after_cropped,  _ = _crop_roi(after_gray,  after_roi_xywh)

        before_128 = cv2.resize(_apply_led_correction(before_cropped, led_map),
                                (128, 128), interpolation=cv2.INTER_LINEAR)
        after_128  = cv2.resize(_apply_led_correction(after_cropped,  led_map),
                                (128, 128), interpolation=cv2.INTER_LINEAR)
        if BIOSENSOR_MODE == 'separate_v2026':
            feat_stack = extract_feature_bank_29_separate(before_128, after_128)
            cls_norm = np.clip(
                _normalize_feat_stack(
                    feat_stack, separate_classifier_ch_mean, separate_classifier_ch_std
                ),
                -5.0,
                5.0,
            )
            cls_xt = torch.from_numpy(cls_norm[None])
            with torch.no_grad():
                cls_logit = separate_classifier_model(cls_xt)
                prob_analyte = float(torch.sigmoid(cls_logit).cpu().numpy().reshape(-1)[0])
            prob_pbs = float(1.0 - prob_analyte)
            label = 'analyte' if prob_analyte >= 0.5 else 'pbs'

            refine_norm = np.clip(
                _normalize_feat_stack(
                    feat_stack, separate_refine_ch_mean, separate_refine_ch_std
                ),
                -5.0,
                5.0,
            )
            refine_xt = torch.from_numpy(refine_norm[None])
            with torch.no_grad():
                heatmap_128 = torch.sigmoid(separate_refine_model(refine_xt)).cpu().numpy()[0, 0]
        else:
            feat_stack = extract_feature_bank_29_legacy(before_128, after_128)
            x_norm = np.clip(
                _normalize_feat_stack(feat_stack, biosensor_ch_mean, biosensor_ch_std),
                -5.0,
                5.0,
            )
            xt = torch.from_numpy(x_norm[None])

            with torch.no_grad():
                cls_logits = biosensor_model(xt, mode='cls')
                probs = torch.softmax(cls_logits, dim=1)[0].tolist()
            prob_pbs = float(probs[0])
            prob_analyte = float(probs[1])
            label = 'analyte' if prob_analyte >= prob_pbs else 'pbs'

            aux = _build_aux(before_128, after_128, feat_stack, biosensor_pbs_mu, biosensor_pbs_std)
            auxt = torch.from_numpy(aux[None])
            with torch.no_grad():
                heatmap_128 = torch.sigmoid(biosensor_model(xt, aux=auxt, mode='refine')).cpu().numpy()[0, 0]

        concentration_label = None
        concentration_confidence = None
        if concentration_model is not None:
            try:
                concentration_before = cv2.resize(
                    before_cropped, (128, 128), interpolation=cv2.INTER_AREA
                )
                concentration_after = cv2.resize(
                    after_cropped, (128, 128), interpolation=cv2.INTER_AREA
                )
                concentration_features = extract_feature_bank_29_concentration(
                    concentration_before, concentration_after
                )
                concentration_norm = (
                    (concentration_features - concentration_ch_mean)
                    / (concentration_ch_std + 1e-6)
                ).astype(np.float32)
                with torch.no_grad():
                    concentration_probs = torch.softmax(
                        concentration_model(torch.from_numpy(concentration_norm[None])),
                        dim=1,
                    )[0]
                concentration_index = int(torch.argmax(concentration_probs).item())
                concentration_label = concentration_class_names[concentration_index]
                concentration_confidence = round(
                    float(concentration_probs[concentration_index].item()), 4
                )
            except Exception as _e:
                logger.warning(f"Four-class concentration inference failed: {_e}")

        oh, ow = after_roi_display.shape[:2]
        heatmap_disp = cv2.resize(heatmap_128, (ow, oh), interpolation=cv2.INTER_LINEAR)
        hm_lo, hm_hi = np.percentile(heatmap_disp, [5, 99.5])
        if hm_hi > hm_lo:
            heatmap_disp = np.clip((heatmap_disp - hm_lo) / (hm_hi - hm_lo), 0.0, 1.0)
        heatmap_disp = np.power(np.clip(heatmap_disp, 0.0, 1.0), 0.9)
        heatmap_color = cv2.applyColorMap(
            (np.clip(heatmap_disp, 0.0, 1.0) * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        after_bgr = cv2.cvtColor(
            np.clip(after_roi_display, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
        )
        overlay = cv2.addWeighted(after_bgr, 0.56, heatmap_color, 0.44, 0)

        return jsonify({
            'status': 'success',
            'label': label,
            'prob_analyte': round(prob_analyte, 4),
            'prob_pbs': round(prob_pbs, 4),
            'concentration_label': concentration_label,
            'concentration_confidence': concentration_confidence,
            'cam_image': _to_png_b64(overlay),
            'model_backend': BIOSENSOR_MODE,
        })

    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


def cleanup():
    """ - """
    global recording_process, camera, hls_process_cam, hls_process_mux
    
    logger.info("Cleaning up resources...")
    
    
    with recording_lock:
        if recording_process:
            try:
                recording_process.terminate()
                recording_process.wait(timeout=3)  
            except:
                try:
                    recording_process.kill()
                    recording_process.wait()
                except:
                    pass
            recording_process = None
    
    
    with camera_lock:
        camera.stop_stream()
        camera.kill_camera_processes()

    
    if HLS_ENABLED:
        with hls_lock:
            try:
                if hls_process_cam and hls_process_cam.poll() is None:
                    hls_process_cam.terminate()
            except Exception:
                pass
            try:
                if hls_process_mux and hls_process_mux.poll() is None:
                    hls_process_mux.terminate()
            except Exception:
                pass
            hls_process_cam = None
            hls_process_mux = None
    
    logger.info("Resource cleanup complete")

# ============================================================
# Timelapse recorder
# ============================================================

def _timelapse_worker(exposure_sec, gain, output_fps):
    """Background thread: loop rpicam-still captures, then assemble into MP4."""
    global _timelapse_running, _timelapse_frame_count

    # Each run gets its own timestamped directory — frames are never overwritten
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    frame_dir = os.path.join(TIMELAPSE_FRAME_DIR, f'run_{timestamp}')
    os.makedirs(frame_dir, exist_ok=True)

    still_app = shutil.which('rpicam-still') or shutil.which('libcamera-still')
    if not still_app:
        logger.error("Timelapse: rpicam-still not found")
        with _timelapse_lock:
            _timelapse_running = False
        return

    shutter_us = max(1000, int(float(exposure_sec) * 1_000_000))
    gain_val = max(1.0, float(gain))
    frame_idx = 0
    timeout_s = max(20, exposure_sec * 2 + 5)

    logger.info(f"Timelapse started: exposure={exposure_sec}s gain={gain_val} output_fps={output_fps} dir={frame_dir}")

    while not _timelapse_stop_event.is_set():
        frame_path = os.path.join(frame_dir, f'frame_{frame_idx:04d}.jpg')
        cmd = [
            still_app,
            '-o', frame_path,
            '--width', '1920',
            '--height', '1080',
            '--shutter', str(shutter_us),
            '--gain', f'{gain_val:.2f}',
            '-t', '1',
            '--nopreview',
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
            if result.returncode == 0 and os.path.exists(frame_path):
                frame_idx += 1
                with _timelapse_lock:
                    _timelapse_frame_count = frame_idx
                logger.info(f"Timelapse frame {frame_idx} captured")
            else:
                logger.warning(f"Timelapse frame failed: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            logger.warning("Timelapse: capture timed out, skipping frame")
        except Exception as e:
            logger.error(f"Timelapse capture error: {e}")
            break

    # Assemble into MP4
    logger.info(f"Timelapse: assembling {frame_idx} frames at {output_fps} fps")
    if frame_idx >= 2:
        save_dir = '/home/cholab/camera_web/videos'
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f'timelapse_{timestamp}.mp4')
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-r', str(output_fps),
            '-i', os.path.join(frame_dir, 'frame_%04d.jpg'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            out_path,
        ]
        try:
            subprocess.run(ffmpeg_cmd, capture_output=True, timeout=120)
            logger.info(f"Timelapse saved: {out_path}")
        except Exception as e:
            logger.error(f"Timelapse ffmpeg error: {e}")
    else:
        logger.warning("Timelapse: fewer than 2 frames, skipping assembly")

    # Restart preview
    with camera_lock:
        try:
            camera.start_stream()
        except Exception:
            pass

    with _timelapse_lock:
        _timelapse_running = False
    logger.info("Timelapse worker done")


@app.route('/start_timelapse', methods=['POST'])
def start_timelapse():
    global _timelapse_running, _timelapse_thread, _timelapse_frame_count

    with _timelapse_lock:
        if _timelapse_running:
            return jsonify({'status': 'error', 'message': 'Timelapse already running'}), 400

    data = get_request_data()
    exposure_sec = max(0.001, min(30.0, float(data.get('exposure', 0.05))))
    gain = max(1.0, min(16.0, float(data.get('gain', 1.0))))
    output_fps = max(1, min(60, int(data.get('output_fps', 10))))

    # Stop preview stream so camera is free
    with camera_lock:
        camera.stop_stream()
        camera.kill_camera_processes()
        time.sleep(2)

    with _timelapse_lock:
        _timelapse_running = True
        _timelapse_frame_count = 0
        _timelapse_stop_event.clear()

    _timelapse_thread = threading.Thread(
        target=_timelapse_worker,
        args=(exposure_sec, gain, output_fps),
        daemon=True
    )
    _timelapse_thread.start()
    return jsonify({'status': 'success', 'message': 'Timelapse started'})


@app.route('/stop_timelapse', methods=['POST'])
def stop_timelapse():
    global _timelapse_running
    with _timelapse_lock:
        if not _timelapse_running:
            return jsonify({'status': 'error', 'message': 'No timelapse running'}), 400
    _timelapse_stop_event.set()
    return jsonify({'status': 'success', 'message': 'Stopping timelapse, assembling video...'})


@app.route('/timelapse_status', methods=['GET'])
def timelapse_status():
    with _timelapse_lock:
        return jsonify({
            'running': _timelapse_running,
            'frame_count': _timelapse_frame_count,
        })


import atexit
atexit.register(cleanup)

if __name__ == '__main__':
    try:
        logger.info("Starting the camera-control server...")
        logger.info("Open http://<raspberry-pi-ip>:5000")
        
        
        try:
            import psutil
        except ImportError:
            logger.error("Install psutil: pip install psutil")
            exit(1)
            
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Stop signal received")
        cleanup()
    except Exception as e:
        logger.error(f"Server error: {e}")
        cleanup()
