"""Smoke tests for the face-swap pipeline. Requires the models to be downloaded."""

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
from core.face_analyser import get_face_analyser  # noqa: E402
from core.engine.face_swapper import swap_face, detect_and_swap  # noqa: E402


@pytest.fixture(scope="module")
def analyser():
    return get_face_analyser()


@pytest.fixture(scope="module")
def source_face(analyser):
    faces = analyser.get(cv2.imread(str(ROOT / "assets" / "elon_musk.jpg")))
    assert faces, "no face found in source asset"
    return faces[0]


@pytest.fixture(scope="module")
def target():
    return cv2.resize(cv2.imread(str(ROOT / "assets" / "vijay.jpg")), (640, 480))


@pytest.fixture(autouse=True)
def reset_config():
    saved = {k: getattr(config, k) for k in dir(config) if k.isupper()}
    yield
    for k, v in saved.items():
        setattr(config, k, v)


def test_cuda_provider_available():
    assert "CUDAExecutionProvider" in ort.get_available_providers()


def test_detects_target_face(analyser, target):
    faces = analyser.get(target)
    assert len(faces) == 1
    face = faces[0]
    assert face.kps.shape == (5, 2)
    assert face.landmark_2d_106 is not None


def test_swap_changes_face_region_only(analyser, source_face, target):
    face = analyser.get(target)[0]
    result = swap_face(source_face, face, target.copy())
    assert result.shape == target.shape and result.dtype == np.uint8

    x1, y1, x2, y2 = face.bbox.astype(int)
    diff = cv2.absdiff(result, target).max(axis=2)
    inside = diff[max(y1, 0):y2, max(x1, 0):x2]
    assert inside.mean() > 5, "face region barely changed"

    # The swapper pastes back its whole aligned crop, which extends well past the
    # detector bbox (forehead/hair), so allow a margin proportional to face size.
    outside = diff.copy()
    pad = int(0.6 * max(x2 - x1, y2 - y1))
    outside[max(y1 - pad, 0):y2 + pad, max(x1 - pad, 0):x2 + pad] = 0
    assert outside.max() == 0, "pixels far from the face were modified"


@pytest.mark.parametrize("flag", ["MOUTH_MASK_ENABLED", "EYES_MASK_ENABLED", "EYEBROWS_MASK_ENABLED", "POISSON_BLEND_ENABLED"])
def test_masks_run(analyser, source_face, target, flag):
    setattr(config, flag, True)
    face = analyser.get(target)[0]
    result = swap_face(source_face, face, target.copy())
    assert result.shape == target.shape and result.dtype == np.uint8


def test_detect_and_swap_no_face_returns_frame(source_face):
    blank = np.zeros((480, 640, 3), np.uint8)
    out, count = detect_and_swap(source_face, blank, get_face_analyser())
    assert count == 0
    assert np.array_equal(out, blank)


def test_opacity_zero_returns_input(source_face, target):
    config.OPACITY = 0
    out, count = detect_and_swap(source_face, target.copy(), get_face_analyser())
    assert np.array_equal(out, target)


def test_live_analyser_loads_only_needed_models(target):
    from core.face_analyser import get_live_face_analyser
    live = get_live_face_analyser()
    assert set(live.models) == {"detection", "landmark_2d_106"}
    face = live.get(target)[0]
    assert face.kps.shape == (5, 2) and face.landmark_2d_106 is not None


def test_live_analyser_swap_matches_full_analyser(analyser, source_face, target):
    """Swapping with the light analyser's face gives the same result as with the full one."""
    from core.face_analyser import get_live_face_analyser
    full = swap_face(source_face, analyser.get(target)[0], target.copy())
    live = swap_face(source_face, get_live_face_analyser().get(target)[0], target.copy())
    assert np.abs(full.astype(int) - live.astype(int)).max() <= 1


def test_all_sessions_on_gpu(analyser):
    from core.face_analyser import get_live_face_analyser
    from core.engine.face_swapper import get_face_swapper
    from core.runtime import ACTIVE_PROVIDERS, gpu_status
    get_live_face_analyser()
    get_face_swapper()
    assert ACTIVE_PROVIDERS, "no sessions registered"
    assert gpu_status()[0] == "GPU", ACTIVE_PROVIDERS


def test_mixed_precision_swapper_matches_fp32(analyser, source_face, target):
    """Guards against FP16 overflow (a full FP16 conversion outputs a black face)."""
    import insightface
    from core.engine.face_swapper import get_face_swapper
    from core.runtime import get_providers

    live_face = analyser.get(target)[0]
    swapper = get_face_swapper()
    assert ".mixed" in swapper.model_file, swapper.model_file
    fp32 = insightface.model_zoo.get_model(str(config.SWAPPER_MODEL), providers=get_providers())
    a = fp32.get(target, live_face, source_face, paste_back=False)[0].astype(float)
    b = swapper.get(target, live_face, source_face, paste_back=False)[0].astype(float)
    assert b.mean() > 20, "mixed-precision output is (near) black"
    psnr = 10 * np.log10(255**2 / max(np.mean((a - b) ** 2), 1e-9))
    assert psnr > 45, f"PSNR {psnr:.1f} dB"


def _variants(img):
    h, w = img.shape[:2]
    yield "centered", img
    for name, (dx, dy) in {"off_left": (-0.3, 0), "off_right": (0.3, 0), "off_top": (0, -0.3), "off_bottom": (0, 0.3)}.items():
        M = np.float32([[1, 0, dx * w], [0, 1, dy * h]])
        yield name, cv2.warpAffine(img, M, (w, h))
    small = cv2.resize(img, (w // 3, h // 3))
    canvas = np.zeros_like(img)
    canvas[100:100 + small.shape[0], 150:150 + small.shape[1]] = small
    yield "small", canvas
    yield "rotated", cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), 25, 1.0), (w, h))


def test_roi_paste_back_matches_insightface(analyser, source_face, target):
    from core.engine.face_swapper import get_face_swapper
    from core.engine.paste_back import paste_back

    swapper = get_face_swapper()
    checked = 0
    for name, img in _variants(target):
        faces = analyser.get(img)
        if not faces:
            continue
        reference = swapper.get(img, faces[0], source_face, paste_back=True)
        bgr_fake, M = swapper.get(img, faces[0], source_face, paste_back=False)
        ours = paste_back(img, bgr_fake, M)
        # cv2.blendLinear rounds where insightface truncates, so +-1 is expected; allow a
        # handful of +-2 values from float rounding at the blend edge.
        diff = np.abs(reference.astype(int) - ours.astype(int))
        assert diff.max() <= 2, f"{name}: max diff {diff.max()}"
        assert (diff > 1).sum() <= diff.size * 1e-4, f"{name}: {(diff > 1).sum()} values differ by >1"
        checked += 1
    assert checked >= 5


def test_mixed_models_output_float32():
    from core.engine.model_optimizer import ensure_mixed_precision_model
    from core.runtime import get_providers
    for fp32 in (config.SWAPPER_MODEL, config.ENHANCER_MODEL):
        if not fp32.exists():
            continue
        session = ort.InferenceSession(str(ensure_mixed_precision_model(fp32)), providers=get_providers())
        assert all(o.type == "tensor(float)" for o in session.get_outputs()), fp32.name


@pytest.mark.parametrize("zoom", [1.6, 2.2, 3.0])
def test_face_too_close_to_camera_is_still_detected_and_swapped(source_face, target, zoom):
    """A face overflowing the frame is missed by the detector; the padded retry must find it."""
    from core.face_analyser import get_live_face_analyser

    live = get_live_face_analyser()
    f = live.get(target)[0]
    cx, cy = (f.bbox[:2] + f.bbox[2:]) / 2
    M = np.float32([[zoom, 0, 320 - zoom * cx], [0, zoom, 260 - zoom * cy]])
    close = cv2.warpAffine(target, M, (640, 480))
    assert not live._analyser.get(close), "plain detector unexpectedly finds the close face"
    faces = live.get(close)
    assert len(faces) == 1
    x1, y1, x2, y2 = faces[0].bbox
    assert x1 < 320 < x2 and y1 < 260 < y2, "bbox not mapped back to frame coordinates"
    out, count = detect_and_swap(source_face, close.copy(), live)
    assert count == 1 and cv2.absdiff(out, close).mean() > 2
