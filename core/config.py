import sys
from pathlib import Path
from typing import Literal

# ---------------- Paths ----------------

# Packaged (PyInstaller) build: models, insightface pack and source_faces live next to the .exe,
# so the whole folder can be copied to another PC.
FROZEN = getattr(sys, "frozen", False)
BASE_DIR = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
SOURCE_FACES_DIR = BASE_DIR / "source_faces"


ANALYSIS_MODEL = "buffalo_l"

SWAPPER_MODEL = (
    MODELS_DIR / "inswapper_128.onnx"
)  # higher model inswapper_128_fp16.onnx

# "mixed": convolutions in FP16 (tensor cores), everything else FP32. ~2.3x faster swap on
# RTX 4000 with output within 2/255 of FP32. "fp32": original model.
SWAPPER_PRECISION: Literal["mixed", "fp32"] = "mixed"

ENHANCER_MODEL = MODELS_DIR / "GFPGANv1.4.onnx"
ENHANCE_WEIGHT = 0.6  # blend strength: 0 = full GFPGAN, 1 = original face
# GFPGAN after the swap. Off by default: ~60 ms per face on an RTX 4000.
ENHANCE_ENABLED = False
ENHANCER_PRECISION: Literal["mixed", "fp32"] = "mixed"  # mixed: 1.3x faster, output within PSNR 60 dB

# Temporal stabilization: One-Euro smoothing of face landmarks between video frames.
# -56% landmark jitter, -15..19% flicker inside the swapped face, no visible lag, ~free.
TEMPORAL_SMOOTHING = True

# Occlusion mask (XSeg): hands, mics, cups, phones in front of the face keep their own pixels.
# ~26 ms per face on RTX 4000 and no visible change without occluders, so off by default.
OCCLUSION_MODEL = MODELS_DIR / "xseg_2.onnx"
OCCLUSION_MASK_ENABLED = False

# insightface packs (buffalo_l): a bundled copy next to the app wins over ~/.insightface
_BUNDLED_INSIGHTFACE = BASE_DIR / "insightface" / "models"
INSIGHTFACE_DIR = _BUNDLED_INSIGHTFACE if (_BUNDLED_INSIGHTFACE / "buffalo_l").is_dir() else Path.home() / ".insightface" / "models"
INSIGHTFACE_ROOT = INSIGHTFACE_DIR.parent  # FaceAnalysis(root=...) expects <root>/models/<pack>
BUFFALO_L_DIR = INSIGHTFACE_DIR / "buffalo_l"

# ---------------- Application Settings ----------------

# Camera settings
DEFAULT_CAMERA_INDEX = 0
CAMERA_WIDTH = 1280  # Razer Kiyo: 30 real FPS at 720p; app swaps at ~18 FPS (640x480: 20.8, 1080p: 13.0)
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Face detection settings
FACE_DETECTION_SIZE = (640, 640)  # source image analysis
# Detector input size for live frames. (480, 480) is ~36% faster (15.2 -> 9.7 ms on RTX 4000) and
# still finds ~30 px faces, but the 5 keypoints shift up to ~2% of face width vs 640; check for
# swap jitter on a live camera before lowering it. (320, 320) misses faces under ~40 px.
LIVE_DETECTION_SIZE = (640, 640)
FACE_CONFIDENCE_THRESHOLD = 0.5

# Advanced Masking / Swapping
OPACITY = 1.0
MOUTH_MASK_ENABLED = False
MOUTH_MASK_SIZE = 0.0
POISSON_BLEND_ENABLED = False
EYES_MASK_ENABLED = False
EYES_MASK_SIZE = 0.0
EYEBROWS_MASK_ENABLED = False
EYEBROWS_MASK_SIZE = 0.0
MASK_FEATHER_RATIO = 10
MASK_DOWN_SIZE = 1.0
SHARPNESS = 0.0
ENABLE_INTERPOLATION = False
INTERPOLATION_WEIGHT = 0.2
