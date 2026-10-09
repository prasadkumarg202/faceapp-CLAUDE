"""Mask builders must not crash when a face is partly or fully outside the frame."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import onnxruntime as ort  # noqa: E402

ort.preload_dlls()

import core.config as config  # noqa: E402
from core.engine import face_masking  # noqa: E402
from core.engine.face_swapper import swap_face  # noqa: E402
from core.face_analyser import get_face_analyser, get_live_face_analyser  # noqa: E402

FRAME = np.zeros((480, 640, 3), np.uint8)


@pytest.fixture(scope="module")
def face():
    img = cv2.resize(cv2.imread(str(ROOT / "assets" / "vijay.jpg")), (640, 480))
    return get_live_face_analyser().get(img)[0]


class ShiftedFace:
    def __init__(self, face, dx, dy):
        self.landmark_2d_106 = face.landmark_2d_106 + [dx, dy]
        self.kps = face.kps + [dx, dy]
        self.bbox = face.bbox + [dx, dy, dx, dy]


SHIFTS = {"left": (-400, 0), "right": (400, 0), "top": (0, -300), "bottom": (0, 300), "gone": (2000, 2000)}
BUILDERS = ["create_face_mask", "create_lower_mouth_mask", "create_eyes_mask", "create_eyebrows_mask"]


@pytest.mark.parametrize("shift", SHIFTS)
@pytest.mark.parametrize("builder", BUILDERS)
def test_mask_builders_survive_off_frame_faces(face, shift, builder):
    result = getattr(face_masking, builder)(ShiftedFace(face, *SHIFTS[shift]), FRAME)
    mask = result[0] if isinstance(result, tuple) else result
    assert mask.shape == FRAME.shape[:2]


@pytest.mark.parametrize("shift", SHIFTS)
def test_swap_with_all_masks_survives_off_frame_faces(face, shift, monkeypatch):
    for flag in ("MOUTH_MASK_ENABLED", "EYES_MASK_ENABLED", "EYEBROWS_MASK_ENABLED", "POISSON_BLEND_ENABLED"):
        monkeypatch.setattr(config, flag, True)
    source = get_face_analyser().get(cv2.imread(str(ROOT / "assets" / "elon_musk.jpg")))[0]
    out = swap_face(source, ShiftedFace(face, *SHIFTS[shift]), FRAME.copy())
    assert out.shape == FRAME.shape


def test_seamless_blending_keeps_chin_tone(face, monkeypatch):
    """Poisson cloning used to pull colours from the jaw edge into the chin (white/grey patch)."""
    img = cv2.resize(cv2.imread(str(ROOT / "assets" / "vijay.jpg")), (640, 480))
    source = get_face_analyser().get(cv2.imread(str(ROOT / "assets" / "elon_musk.jpg")))[0]
    lm = face.landmark_2d_106
    y0, y1 = int(lm[52:72, 1].max()) + 5, int(lm[0:33, 1].max()) - 5
    x0, x1 = int(lm[52:72, 0].min()), int(lm[52:72, 0].max())

    def chin(im):
        return cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2LAB)[..., 0].mean()

    monkeypatch.setattr(config, "POISSON_BLEND_ENABLED", False)
    plain = chin(swap_face(source, face, img.copy()))
    monkeypatch.setattr(config, "POISSON_BLEND_ENABLED", True)
    seamless = chin(swap_face(source, face, img.copy()))
    assert abs(seamless - plain) < 2.0, f"seamless blending shifted chin brightness by {seamless - plain:+.1f}"


def test_jaw_clip_leaves_neck_and_shirt_untouched(face, monkeypatch):
    """Below the jawline (neck, collar) must keep the camera's pixels; the face is still swapped."""
    img = cv2.resize(cv2.imread(str(ROOT / "assets" / "vijay.jpg")), (640, 480))
    source = get_face_analyser().get(cv2.imread(str(ROOT / "assets" / "elon_musk.jpg")))[0]
    monkeypatch.setattr(config, "JAW_CLIP_ENABLED", True)
    out = swap_face(source, face, img.copy())
    jaw = face.landmark_2d_106[0:33]
    fw = float(np.ptp(jaw[:, 0]))
    y = int(jaw[:, 1].max() + 0.08 * fw)
    x0, x1 = int(jaw[:, 0].min()), int(jaw[:, 0].max())
    below = (slice(y, min(y + 40, img.shape[0])), slice(x0, x1))
    assert np.abs(out[below].astype(int) - img[below].astype(int)).mean() < 0.5
    eyes = slice(int(face.kps[:2, 1].min()) - 10, int(face.kps[:2, 1].max()) + 10)
    assert cv2.absdiff(out[eyes], img[eyes]).mean() > 3, "face was not swapped"


def test_jaw_clip_without_landmarks_is_a_no_op():
    from core.engine.jaw_clip import restore_below_jaw

    class NoLandmarks:
        landmark_2d_106 = None

    a, b = np.zeros((10, 10, 3), np.uint8), np.full((10, 10, 3), 9, np.uint8)
    assert restore_below_jaw(a, b, NoLandmarks()) is b


def test_camera_texture_restores_real_skin_detail(face, monkeypatch):
    """The upscaled swap is far smoother than real skin; the camera's fine detail is put back."""
    img = cv2.resize(cv2.imread(str(ROOT / "assets" / "vijay.jpg")), (1280, 960))  # upscale like a 720p face
    big = get_live_face_analyser().get(img)[0]
    source = get_face_analyser().get(cv2.imread(str(ROOT / "assets" / "elon_musk.jpg")))[0]

    def texture(im):
        x1, y1, x2, y2 = big.bbox.astype(int)
        patch = im[y1 + (y2 - y1) // 10:y1 + (y2 - y1) // 4, x1 + (x2 - x1) // 3:x2 - (x2 - x1) // 3]
        return cv2.Laplacian(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()

    monkeypatch.setattr(config, "DETAIL_TRANSFER_ENABLED", False)
    plain = swap_face(source, big, img.copy())
    monkeypatch.setattr(config, "DETAIL_TRANSFER_ENABLED", True)
    textured = swap_face(source, big, img.copy())
    assert texture(textured) > 3 * texture(plain), "camera texture was not restored"
    assert np.array_equal(textured[:, :40], img[:, :40]), "background far from the face changed"
