"""Occlusion mask (XSeg): objects in front of the face keep their own pixels."""

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
from core.engine import occlusion  # noqa: E402
from core.engine.face_swapper import swap_face  # noqa: E402
from core.face_analyser import get_face_analyser, get_live_face_analyser  # noqa: E402

TARGET = cv2.resize(cv2.imread(str(ROOT / "assets" / "vijay.jpg")), (640, 480))
pytestmark = pytest.mark.skipif(not config.OCCLUSION_MODEL.exists(), reason="occlusion model not downloaded")


@pytest.fixture(scope="module")
def source_face():
    return get_face_analyser().get(cv2.imread(str(ROOT / "assets" / "elon_musk.jpg")))[0]


def box_over_mouth(img):
    """A dark, non-skin object covering the mouth: the case BiSeNet face parsing missed."""
    face = get_live_face_analyser().get(img)[0]
    out = img.copy()
    (lx, ly), (rx, ry) = face.kps[3], face.kps[4]
    w = rx - lx
    box = (int(lx + 0.1 * w), int(min(ly, ry) - 0.35 * w), int(rx - 0.1 * w), int(max(ly, ry) + 0.35 * w))
    cv2.rectangle(out, box[:2], box[2:], (60, 40, 30), -1)
    return out, box


def phone_at_cheek(img):
    face = get_live_face_analyser().get(img)[0]
    out = img.copy()
    x1, y1, x2, y2 = face.bbox.astype(int)
    w = x2 - x1
    box = (x2 - w // 4, y1 + w // 4, min(x2 + 40, img.shape[1]), y2)
    cv2.rectangle(out, box[:2], box[2:], (210, 205, 200), -1)
    return out, box


def _object_and_face_change(img, box, source_face):
    face = get_live_face_analyser().get(img)[0]
    out = swap_face(source_face, face, img.copy())
    inner = (slice(box[1] + 4, box[3] - 4), slice(box[0] + 4, box[2] - 4))
    eyes = slice(int(face.kps[:2, 1].min()) - 10, int(face.kps[:2, 1].max()) + 10)
    return cv2.absdiff(out[inner], img[inner]).mean(), cv2.absdiff(out[eyes], img[eyes]).mean()


@pytest.mark.parametrize("make, painted_without_mask", [(box_over_mouth, True), (phone_at_cheek, False)])
def test_object_in_front_of_face_is_kept(monkeypatch, source_face, make, painted_without_mask):
    img, box = make(TARGET)
    monkeypatch.setattr(config, "OCCLUSION_MASK_ENABLED", False)
    painted, _ = _object_and_face_change(img, box, source_face)
    monkeypatch.setattr(config, "OCCLUSION_MASK_ENABLED", True)
    kept, face_change = _object_and_face_change(img, box, source_face)
    if painted_without_mask:  # proves the test can see the problem it guards against
        assert painted > 5, f"without the mask the object should be painted over ({painted:.1f})"
    assert kept < 2, f"object in front of the face was painted over ({kept:.1f})"
    assert face_change > 3, "the rest of the face was not swapped"


def test_unobstructed_face_interior_is_unchanged_by_mask(monkeypatch, source_face):
    """Without occluders the mask only trims the outer ring (hair, ears, jaw edge), not the face."""
    face = get_live_face_analyser().get(TARGET)[0]
    monkeypatch.setattr(config, "OCCLUSION_MASK_ENABLED", False)
    a = swap_face(source_face, face, TARGET.copy())
    monkeypatch.setattr(config, "OCCLUSION_MASK_ENABLED", True)
    b = swap_face(source_face, face, TARGET.copy())
    x1, y1, x2, y2 = face.bbox
    mx, my = 0.2 * (x2 - x1), 0.2 * (y2 - y1)
    inner = (slice(int(y1 + my), int(y2 - my)), slice(int(x1 + mx), int(x2 - mx)))
    assert np.abs(a[inner].astype(int) - b[inner].astype(int)).mean() < 0.5


def test_missing_model_is_skipped(monkeypatch, source_face, tmp_path):
    monkeypatch.setattr(config, "OCCLUSION_MODEL", tmp_path / "missing.onnx")
    monkeypatch.setattr(occlusion, "_SESSION", None)
    monkeypatch.setattr(occlusion, "_UNAVAILABLE", False)
    monkeypatch.setattr(config, "OCCLUSION_MASK_ENABLED", True)
    face = get_live_face_analyser().get(TARGET)[0]
    out = swap_face(source_face, face, TARGET.copy())
    assert cv2.absdiff(out, TARGET).mean() > 1
