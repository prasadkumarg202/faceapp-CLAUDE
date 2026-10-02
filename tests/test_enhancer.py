"""GFPGAN enhancer: ROI paste-back equivalence, mixed precision quality, non-blocking warm-up."""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import onnxruntime as ort  # noqa: E402

ort.preload_dlls()

import core.config as config  # noqa: E402
from core.engine import face_enhancer as fe  # noqa: E402
from core.engine.face_swapper import detect_and_swap  # noqa: E402
from core.face_analyser import get_face_analyser, get_live_face_analyser  # noqa: E402

TARGET = cv2.resize(cv2.imread(str(ROOT / "assets" / "vijay.jpg")), (640, 480))

pytestmark = pytest.mark.skipif(not config.ENHANCER_MODEL.exists(), reason="GFPGAN model not downloaded")


def reference_paste_back(frame, enhanced_face, affine_matrix, output_size, weight):
    """The original full-frame implementation, kept as the reference."""
    h, w = frame.shape[:2]
    inv_matrix = cv2.invertAffineTransform(affine_matrix)
    inv_restored = cv2.warpAffine(enhanced_face, inv_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    face_mask = fe._edge_mask(output_size, weight)
    face_mask_3c = np.stack([face_mask] * 3, axis=-1)
    inv_mask = cv2.warpAffine(face_mask_3c, inv_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    inv_mask = np.clip(inv_mask, 0.0, 1.0)
    result = frame.astype(np.float32) * (1.0 - inv_mask) + inv_restored.astype(np.float32) * inv_mask
    return np.clip(result, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def face():
    return get_live_face_analyser().get(TARGET)[0]


@pytest.mark.parametrize("dx", [0, -250, 250])
def test_roi_paste_back_matches_reference(face, dx):
    frame = cv2.warpAffine(TARGET, np.float32([[1, 0, dx], [0, 1, 0]]), (640, 480))
    kps = face.kps.astype(np.float32) + [dx, 0]
    aligned, M = fe._align_face(frame, kps, 512)
    enhanced = cv2.GaussianBlur(aligned, (0, 0), 3)  # any image works for paste-back
    ours = fe._paste_back(frame, enhanced, M, 512, 0.6)
    ref = reference_paste_back(frame, enhanced, M, 512, 0.6)
    diff = np.abs(ours.astype(int) - ref.astype(int))
    assert diff.max() <= 2 and (diff > 1).sum() <= diff.size * 1e-4


def test_mixed_precision_enhancer_matches_fp32(face):
    session = fe.get_face_enhancer()
    assert ".mixed" in session._model_path
    aligned, _ = fe._align_face(TARGET, face.kps.astype(np.float32), 512)
    x = fe._preprocess_face(aligned)
    fp32 = ort.InferenceSession(str(config.ENHANCER_MODEL), providers=["CUDAExecutionProvider"])
    name = fp32.get_inputs()[0].name
    a = fe._postprocess_face(fp32.run(None, {name: x})[0]).astype(float)
    b = fe._postprocess_face(session.run(None, {name: x})[0]).astype(float)
    assert b.mean() > 20
    assert 10 * np.log10(255**2 / max(np.mean((a - b) ** 2), 1e-9)) > 45


def test_pipeline_streams_until_enhancer_ready_then_enhances(monkeypatch):
    monkeypatch.setattr(config, "ENHANCE_ENABLED", True)
    monkeypatch.setattr(fe, "_ENHANCER", None)
    monkeypatch.setattr(fe, "_WARMING", type(fe._WARMING)())
    source = get_face_analyser().get(cv2.imread(str(ROOT / "assets" / "elon_musk.jpg")))[0]
    analyser = get_live_face_analyser()

    start = time.perf_counter()
    unenhanced, _ = detect_and_swap(source, TARGET.copy(), analyser)
    assert time.perf_counter() - start < 1.0, "first frame blocked on enhancer loading"

    deadline = time.time() + 120
    while not fe.is_enhancer_ready() and time.time() < deadline:
        time.sleep(0.2)
    assert fe.is_enhancer_ready()
    enhanced, _ = detect_and_swap(source, TARGET.copy(), analyser)
    assert cv2.absdiff(enhanced, unenhanced).mean() > 0.5, "enhancer had no effect"
