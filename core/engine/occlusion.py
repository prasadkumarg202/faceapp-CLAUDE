"""Occlusion mask (XSeg) to keep hands, microphones, cups, phones in front of the face.

inswapper pastes back its whole aligned crop, so anything in front of the face gets face texture
painted over it. XSeg predicts which pixels of the aligned face are visible face (1) versus
occluder (0); the swap is then blended only onto visible face.

Model choice, measured on sample and webcam frames with synthetic occluders:
  - BiSeNet face parsing keeps objects at the face edge but labels an object over the mouth as
    "skin" (87%), so it misses the most common case. Rejected.
  - XSeg (FaceFusion xseg_1/2/3): xseg_2 excluded every object (0.00) while keeping the face (1.00)
    in all cases; xseg_1 missed the over-the-mouth box on a webcam frame (0.82).
Cost: ~26 ms per face on a Quadro RTX 4000 (mixed precision gives no speedup), so it is optional.
"""

import logging
import threading
from pathlib import Path

import cv2
import numpy as np
from insightface.utils import face_align

import core.config as config
from core.runtime import get_providers, log_throttled, register_session

logger = logging.getLogger(__name__)

_SIZE = 256
_SESSION = None
_LOCK = threading.Lock()
_UNAVAILABLE = False


def get_occlusion_model():
    """The XSeg session, or None if the model is missing or fails to load."""
    global _SESSION, _UNAVAILABLE
    if _SESSION is not None or _UNAVAILABLE:
        return _SESSION
    with _LOCK:
        if _SESSION is None and not _UNAVAILABLE:
            path = Path(config.OCCLUSION_MODEL)
            if not path.exists():
                logger.info("Occlusion model not downloaded (Models tab); objects in front of the face won't be kept")
                _UNAVAILABLE = True
                return None
            try:
                import onnxruntime

                _SESSION = onnxruntime.InferenceSession(str(path), providers=get_providers())
                register_session("occlusion", _SESSION)
            except Exception as e:
                logger.warning("Occlusion model unavailable: %s", e)
                _UNAVAILABLE = True
    return _SESSION


def visible_face_mask(frame, face):
    """Mask of the visible (non-occluded) face in frame coordinates, as (mask HxW float32, roi), or None."""
    session = get_occlusion_model()
    if session is None:
        return None
    crop, M = face_align.norm_crop2(frame, face.kps, _SIZE)
    x = (crop.astype(np.float32) * np.float32(1 / 255))[None]  # NHWC, BGR as FaceFusion feeds it
    try:
        mask = session.run(None, {session.get_inputs()[0].name: x})[0][0, :, :, 0]
    except Exception as e:
        log_throttled(logger, "occlusion", "Occlusion mask failed: %s", e)
        return None
    np.clip(mask, 0, 1, out=mask)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.5)  # soften the edge of the occluder

    # Warp back only over the region the aligned crop covers
    h, w = frame.shape[:2]
    IM = cv2.invertAffineTransform(M)
    corners = np.array([[0, 0], [_SIZE, 0], [0, _SIZE], [_SIZE, _SIZE]], np.float32) @ IM[:, :2].T + IM[:, 2]
    x0, y0 = max(int(corners[:, 0].min()) - 1, 0), max(int(corners[:, 1].min()) - 1, 0)
    x1, y1 = min(int(np.ceil(corners[:, 0].max())) + 2, w), min(int(np.ceil(corners[:, 1].max())) + 2, h)
    if x1 <= x0 or y1 <= y0:
        return None
    IM[0, 2] -= x0
    IM[1, 2] -= y0
    # Outside the crop the swap paste-back is already zero, so treat it as "face" (no change)
    roi_mask = cv2.warpAffine(mask, IM, (x1 - x0, y1 - y0), borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
    return roi_mask, (x0, y0, x1, y1)


def keep_occluders(original, swapped, face):
    """Restore `original` pixels wherever XSeg sees something in front of `face`."""
    result = visible_face_mask(original, face)
    if result is None:
        return swapped
    mask, (x0, y0, x1, y1) = result
    out = swapped.copy()
    out[y0:y1, x0:x1] = cv2.blendLinear(
        np.ascontiguousarray(swapped[y0:y1, x0:x1]), np.ascontiguousarray(original[y0:y1, x0:x1]), mask, 1.0 - mask
    )
    return out
