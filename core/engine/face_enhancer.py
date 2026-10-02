import cv2
import logging
import threading
import numpy as np
from pathlib import Path

import onnxruntime
import core.config as config
from core.config import ENHANCER_MODEL, ENHANCE_WEIGHT
from core.runtime import get_providers, register_session, log_throttled
from core.engine.model_optimizer import ensure_mixed_precision_model

logger = logging.getLogger(__name__)

_ENHANCER = None
_LOCK = threading.Lock()
_WARMING = threading.Event()

FFHQ_TEMPLATE_512 = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.19366],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)

def get_face_enhancer() -> onnxruntime.InferenceSession:
    global _ENHANCER
    with _LOCK:
        if _ENHANCER is None:
            if not ENHANCER_MODEL.exists():
                raise FileNotFoundError(f"ONNX Model not found at {ENHANCER_MODEL}")

            providers = get_providers()
            model_path = str(ENHANCER_MODEL)
            if getattr(config, "ENHANCER_PRECISION", "fp32") == "mixed" and "CUDAExecutionProvider" in providers:
                model_path = str(ensure_mixed_precision_model(ENHANCER_MODEL) or ENHANCER_MODEL)
            logger.info("Enhancer model: %s", Path(model_path).name)
            session_options = onnxruntime.SessionOptions()
            session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

            _ENHANCER = onnxruntime.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=providers,
            )
            register_session("enhancer", _ENHANCER)
    return _ENHANCER

def is_enhancer_ready() -> bool:
    return _ENHANCER is not None


def warm_up_async():
    """Load (and on first use convert) the enhancer in the background so video doesn't stall."""
    if _ENHANCER is not None or _WARMING.is_set():
        return
    _WARMING.set()

    def load():
        try:
            get_face_enhancer()
        except Exception as e:
            logger.warning("Face enhancer unavailable: %s", e)

    threading.Thread(target=load, name="enhancer-loader", daemon=True).start()


def _align_face(frame: np.ndarray, landmarks_5: np.ndarray, output_size: int) -> tuple:
    scale = output_size / 512.0
    template = FFHQ_TEMPLATE_512 * scale
    affine_matrix, _ = cv2.estimateAffinePartial2D(landmarks_5, template, method=cv2.LMEDS)
    if affine_matrix is None:
        return None, None
    aligned_face = cv2.warpAffine(
        frame, affine_matrix, (output_size, output_size),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132)
    )
    return aligned_face, affine_matrix

def _edge_mask(output_size: int, weight: float) -> np.ndarray:
    face_mask = np.ones((output_size, output_size), dtype=np.float32)
    border = max(1, int(output_size * 0.05))
    ramp_up = np.linspace(0.0, 1.0, border, dtype=np.float32)
    ramp_down = np.linspace(1.0, 0.0, border, dtype=np.float32)

    face_mask[:border, :] *= ramp_up[:, None]
    face_mask[-border:, :] *= ramp_down[:, None]
    face_mask[:, :border] *= ramp_up[None, :]
    face_mask[:, -border:] *= ramp_down[None, :]

    if weight < 1.0:
        face_mask *= weight
    return face_mask


def _paste_back(frame: np.ndarray, enhanced_face: np.ndarray, affine_matrix: np.ndarray, output_size: int, weight: float) -> np.ndarray:
    """Blend the enhanced face back, processing only the region the aligned face covers."""
    h, w = frame.shape[:2]
    inv_matrix = cv2.invertAffineTransform(affine_matrix)

    corners = np.array([[0, 0], [output_size, 0], [0, output_size], [output_size, output_size]], np.float32)
    pts = corners @ inv_matrix[:, :2].T + inv_matrix[:, 2]
    x0, y0 = max(int(np.floor(pts[:, 0].min())) - 1, 0), max(int(np.floor(pts[:, 1].min())) - 1, 0)
    x1, y1 = min(int(np.ceil(pts[:, 0].max())) + 2, w), min(int(np.ceil(pts[:, 1].max())) + 2, h)
    result = frame.copy()
    if x1 <= x0 or y1 <= y0:
        return result

    inv_roi = inv_matrix.copy()
    inv_roi[0, 2] -= x0
    inv_roi[1, 2] -= y0
    size = (x1 - x0, y1 - y0)
    restored = cv2.warpAffine(enhanced_face, inv_roi, size, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpAffine(_edge_mask(output_size, weight), inv_roi, size, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    np.clip(mask, 0.0, 1.0, out=mask)

    roi = np.ascontiguousarray(frame[y0:y1, x0:x1])
    result[y0:y1, x0:x1] = cv2.blendLinear(restored, roi, mask, 1.0 - mask)
    return result

def _preprocess_face(aligned_face: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = rgb / 255.0
    rgb = (rgb - 0.5) / 0.5
    chw = np.transpose(rgb, (2, 0, 1))
    return np.expand_dims(chw, axis=0)

def _postprocess_face(output: np.ndarray) -> np.ndarray:
    # The mixed-precision model returns float16; numpy float16 math is ~4x slower than float32
    face = np.squeeze(output).astype(np.float32, copy=False)
    face = np.transpose(face, (1, 2, 0))
    face = (face + 1.0) / 2.0
    face = np.clip(face * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

def enhance_faces(frame: np.ndarray, faces: list, frame_size: tuple) -> np.ndarray:
    session = get_face_enhancer()
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    try:
        align_size = int(input_info.shape[2])
        if align_size <= 0:
            align_size = 512
    except (ValueError, TypeError, IndexError):
        align_size = 512

    result_frame = frame.copy()

    for face in faces:
        if not hasattr(face, "kps") or face.kps is None:
            continue

        landmarks_5 = face.kps.astype(np.float32)
        if landmarks_5.shape[0] < 5:
            continue

        aligned_face, affine_matrix = _align_face(result_frame, landmarks_5, output_size=align_size)
        if aligned_face is None or affine_matrix is None:
            continue

        try:
            input_tensor = _preprocess_face(aligned_face)
            output_tensor = session.run(None, {input_name: input_tensor})[0]
            enhanced_bgr = _postprocess_face(output_tensor)

            eh, ew = enhanced_bgr.shape[:2]
            if eh != align_size or ew != align_size:
                enhanced_bgr = cv2.resize(enhanced_bgr, (align_size, align_size), interpolation=cv2.INTER_LANCZOS4)

            result_frame = _paste_back(result_frame, enhanced_bgr, affine_matrix, output_size=align_size, weight=ENHANCE_WEIGHT)
        except Exception as e:
            log_throttled(logger, "enhance", "Error enhancing face: %s", e)
            continue

    return result_frame
