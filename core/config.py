from pathlib import Path
from typing import Literal

# ---------------- Paths ----------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


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

INSIGHTFACE_DIR = Path.home() / ".insightface" / "models"
BUFFALO_L_DIR = INSIGHTFACE_DIR / "buffalo_l"

# ---------------- Application Settings ----------------

# Camera settings
DEFAULT_CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Face detection settings
FACE_DETECTION_SIZE = (640, 640)
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
