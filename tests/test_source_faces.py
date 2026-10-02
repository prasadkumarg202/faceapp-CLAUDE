"""Multi-photo source identity."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import onnxruntime as ort  # noqa: E402

ort.preload_dlls()

from core.engine.face_swapper import swap_face  # noqa: E402
from core.face_analyser import get_face_analyser, get_live_face_analyser  # noqa: E402
from core.source_faces import build_source_identity  # noqa: E402

ASSETS = ROOT / "assets"
USER_SET = Path("C:/deep-live-cam/source_faces")


def test_single_photo_identity_equals_that_photo():
    analyser = get_face_analyser()
    identity = build_source_identity([ASSETS / "elon_musk.jpg"], analyser)
    direct = analyser.get(cv2.imread(str(ASSETS / "elon_musk.jpg")))[0]
    assert len(identity.used) == 1
    assert float(identity.face.normed_embedding @ direct.normed_embedding) > 0.999


def test_unreadable_and_faceless_photos_are_skipped(tmp_path):
    (tmp_path / "broken.jpg").write_bytes(b"not an image")
    cv2.imwrite(str(tmp_path / "blank.png"), np.zeros((200, 200, 3), np.uint8))
    identity = build_source_identity([tmp_path, ASSETS / "vijay.jpg"], get_face_analyser())
    status = {p.path.name: p.status for p in identity.photos}
    assert status == {"blank.png": "no face", "broken.jpg": "unreadable", "vijay.jpg": "used"}


def test_no_usable_photo_raises(tmp_path):
    cv2.imwrite(str(tmp_path / "blank.png"), np.zeros((200, 200, 3), np.uint8))
    with pytest.raises(ValueError):
        build_source_identity([tmp_path], get_face_analyser())


def test_different_person_is_dropped_as_outlier(tmp_path):
    """Three photos of one person plus one of another: the other person is dropped."""
    img = cv2.imread(str(ASSETS / "elon_musk.jpg"))
    for i, (flip, scale) in enumerate([(False, 1.0), (True, 1.0), (False, 0.8)]):
        v = cv2.flip(img, 1) if flip else img
        cv2.imwrite(str(tmp_path / f"elon_{i}.jpg"), cv2.resize(v, None, fx=scale, fy=scale))
    cv2.imwrite(str(tmp_path / "other.jpg"), cv2.imread(str(ASSETS / "vijay.jpg")))
    identity = build_source_identity([tmp_path], get_face_analyser())
    status = {p.path.name: p.status for p in identity.photos}
    assert status["other.jpg"] == "outlier"
    assert all(status[f"elon_{i}.jpg"] == "used" for i in range(3))


@pytest.mark.skipif(not USER_SET.is_dir(), reason="user's photo set not on this machine")
def test_user_photo_set_and_swap():
    identity = build_source_identity([USER_SET], get_face_analyser())
    status = {p.path.name: p.status for p in identity.photos}
    assert status.get("right_45.jpeg") == "outlier"
    assert len(identity.used) >= 7
    frame = cv2.resize(cv2.imread(str(ASSETS / "vijay.jpg")), (640, 480))
    out = swap_face(identity.face, get_live_face_analyser().get(frame)[0], frame.copy())
    assert cv2.absdiff(out, frame).mean() > 1
