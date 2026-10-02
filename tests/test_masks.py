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
