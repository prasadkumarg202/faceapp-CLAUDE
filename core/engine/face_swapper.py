import logging
import cv2
import numpy as np
import insightface
import threading
from core.config import SWAPPER_MODEL
import core.config as config
from core.runtime import get_providers, register_session, log_throttled
from core.engine.model_optimizer import ensure_mixed_precision_model
from core.engine.paste_back import paste_back
from core.engine.gpu_processing import gpu_add_weighted, gpu_resize, gpu_sharpen
from core.engine.face_masking import (
    create_face_mask,
    create_lower_mouth_mask,
    apply_mask_area,
    create_eyes_mask,
    create_eyebrows_mask
)

logger = logging.getLogger(__name__)

POISSON_MASK_SHRINK = 0.08

swapper_ = None
THREAD_LOCK = threading.Lock()
PREVIOUS_FRAME_RESULT = None

def get_face_swapper():
    global swapper_
    with THREAD_LOCK:
        if swapper_ is None:
            model_path = SWAPPER_MODEL
            providers = get_providers()
            if getattr(config, "SWAPPER_PRECISION", "fp32") == "mixed" and "CUDAExecutionProvider" in providers:
                model_path = ensure_mixed_precision_model(SWAPPER_MODEL) or SWAPPER_MODEL
            swapper_ = insightface.model_zoo.get_model(str(model_path), providers=providers)
            logger.info("Swapper model: %s", model_path.name)
            register_session("swapper", swapper_.session)
    return swapper_

def swap_face(source_face, target_face, frame):
    face_swapper = get_face_swapper()
    if face_swapper is None or source_face is None or target_face is None:
        return frame

    opacity = getattr(config, "OPACITY", 1.0)
    mouth_mask_enabled = getattr(config, "MOUTH_MASK_ENABLED", False)
    eyes_mask_enabled = getattr(config, "EYES_MASK_ENABLED", False)
    eyebrows_mask_enabled = getattr(config, "EYEBROWS_MASK_ENABLED", False)

    occlusion_mask_enabled = getattr(config, "OCCLUSION_MASK_ENABLED", False)
    jaw_clip_enabled = getattr(config, "JAW_CLIP_ENABLED", False)
    original_frame = frame.copy() if (opacity < 1.0 or mouth_mask_enabled or eyes_mask_enabled or eyebrows_mask_enabled or occlusion_mask_enabled or jaw_clip_enabled) else frame

    temp_frame = frame
    if temp_frame.dtype != np.uint8:
        temp_frame = np.clip(temp_frame, 0, 255).astype(np.uint8)

    if not temp_frame.flags['C_CONTIGUOUS']:
        temp_frame = np.ascontiguousarray(temp_frame)

    try:
        bgr_fake, M = face_swapper.get(temp_frame, target_face, source_face, paste_back=False)
        swapped_frame_raw = paste_back(temp_frame, bgr_fake, M)
        if swapped_frame_raw is None or not isinstance(swapped_frame_raw, np.ndarray):
            return original_frame
        if swapped_frame_raw.shape != temp_frame.shape:
            swapped_frame_raw = gpu_resize(swapped_frame_raw, (temp_frame.shape[1], temp_frame.shape[0]))
        swapped_frame = np.clip(swapped_frame_raw, 0, 255).astype(np.uint8)
    except Exception as e:
        log_throttled(logger, "swap", "Error during swap: %s", e)
        return original_frame

    # Neck, collar and shirt below the jawline keep their original pixels
    if jaw_clip_enabled:
        from core.engine.jaw_clip import restore_below_jaw
        swapped_frame = restore_below_jaw(original_frame, swapped_frame, target_face)

    # Objects in front of the face (hands, mics, cups) keep their original pixels
    if occlusion_mask_enabled:
        from core.engine.occlusion import keep_occluders
        swapped_frame = keep_occluders(original_frame, swapped_frame, target_face)

    # Masking
    face_mask = None
    if mouth_mask_enabled or eyes_mask_enabled or eyebrows_mask_enabled or getattr(config, "POISSON_BLEND_ENABLED", False):
        face_mask = create_face_mask(target_face, original_frame)

    if mouth_mask_enabled:
        mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon = create_lower_mouth_mask(target_face, original_frame)
        if mouth_cutout is not None and mouth_box != (0,0,0,0):
            swapped_frame = apply_mask_area(swapped_frame, mouth_cutout, mouth_box, face_mask, lower_lip_polygon)

    if eyes_mask_enabled:
        mask, cutout, box, polygon = create_eyes_mask(target_face, original_frame)
        if cutout is not None and box != (0,0,0,0):
            swapped_frame = apply_mask_area(swapped_frame, cutout, box, face_mask, polygon)

    if eyebrows_mask_enabled:
        mask, cutout, box, polygon = create_eyebrows_mask(target_face, original_frame)
        if cutout is not None and box != (0,0,0,0):
            swapped_frame = apply_mask_area(swapped_frame, cutout, box, face_mask, polygon)

    # Poisson Blending
    if getattr(config, "POISSON_BLEND_ENABLED", False):
        if face_mask is not None:
            # Poisson cloning takes its colours from the mask boundary. The face mask reaches the
            # jaw edge, where the lighter neck/collar/wall bled into the chin as a white patch
            # (chin brightness 39.8 -> 46.4 on webcam frames). Shrinking the mask by 8% of the
            # face width keeps the boundary on facial skin (-> 42.8; 15% gave no further gain).
            jaw = target_face.landmark_2d_106[0:33]
            shrink = max(3, int(POISSON_MASK_SHRINK * float(np.ptp(jaw[:, 0]))))
            face_mask = cv2.erode(face_mask, np.ones((shrink, shrink), np.uint8))
            y_indices, x_indices = np.where(face_mask > 0)
            if len(x_indices) > 0 and len(y_indices) > 0:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
                src_crop = swapped_frame[y_min : y_max + 1, x_min : x_max + 1]
                mask_crop = face_mask[y_min : y_max + 1, x_min : x_max + 1]
                try:
                    swapped_frame = cv2.seamlessClone(src_crop, original_frame, mask_crop, center, cv2.NORMAL_CLONE)
                except Exception as e:
                    log_throttled(logger, "poisson", "Poisson blending failed: %s", e)

    if opacity >= 1.0:
        return swapped_frame.astype(np.uint8)

    final_swapped_frame = gpu_add_weighted(original_frame.astype(np.uint8), 1 - opacity, swapped_frame.astype(np.uint8), opacity, 0)
    return final_swapped_frame.astype(np.uint8)


def apply_post_processing(current_frame, swapped_face_bboxes):
    global PREVIOUS_FRAME_RESULT
    processed_frame = current_frame.copy()
    sharpness_value = getattr(config, "SHARPNESS", 0.0)
    if sharpness_value > 0.0 and swapped_face_bboxes:
        height, width = processed_frame.shape[:2]
        for bbox in swapped_face_bboxes:
            if not hasattr(bbox, '__iter__') or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            try:
                 x1, y1 = max(0, int(x1)), max(0, int(y1))
                 x2, y2 = min(width, int(x2)), min(height, int(y2))
            except ValueError:
                continue

            if x2 <= x1 or y2 <= y1:
                continue

            face_region = processed_frame[y1:y2, x1:x2]
            if face_region.size == 0:
                continue

            try:
                sharpened_region = gpu_sharpen(face_region, strength=sharpness_value, sigma=3)
                processed_frame[y1:y2, x1:x2] = sharpened_region
            except cv2.error:
                pass


    enable_interpolation = getattr(config, "ENABLE_INTERPOLATION", False)
    interpolation_weight = getattr(config, "INTERPOLATION_WEIGHT", 0.2)
    final_frame = processed_frame

    if enable_interpolation and 0 < interpolation_weight < 1:
        if PREVIOUS_FRAME_RESULT is not None and PREVIOUS_FRAME_RESULT.shape == processed_frame.shape and PREVIOUS_FRAME_RESULT.dtype == processed_frame.dtype:
            try:
                 final_frame = gpu_add_weighted(PREVIOUS_FRAME_RESULT, 1.0 - interpolation_weight, processed_frame, interpolation_weight, 0)
                 final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)
            except cv2.error:
                 final_frame = processed_frame
                 PREVIOUS_FRAME_RESULT = None
            PREVIOUS_FRAME_RESULT = final_frame.copy()
        else:
            PREVIOUS_FRAME_RESULT = processed_frame.copy()
    else:
         PREVIOUS_FRAME_RESULT = None

    return final_frame

def detect_and_swap(source_face, frame, face_analyser):
    if getattr(config, "OPACITY", 1.0) == 0:
        global PREVIOUS_FRAME_RESULT
        PREVIOUS_FRAME_RESULT = None
        return frame, 0

    faces = face_analyser.get(frame)
    face_count = len(faces)

    processed_frame = frame
    swapped_face_bboxes = []

    if face_count > 0:
        current_swap_target = processed_frame.copy()
        for face in faces:
            current_swap_target = swap_face(source_face, face, current_swap_target)
            if face is not None and hasattr(face, "bbox") and face.bbox is not None:
                swapped_face_bboxes.append(face.bbox.astype(int))
        processed_frame = current_swap_target

        if getattr(config, "ENHANCE_ENABLED", False):
            from core.engine.face_enhancer import enhance_faces, is_enhancer_ready, warm_up_async
            if is_enhancer_ready():
                processed_frame = enhance_faces(processed_frame, faces, None)
            else:
                warm_up_async()  # keep streaming unenhanced until it has loaded

    final_frame = apply_post_processing(processed_frame, swapped_face_bboxes)
    return final_frame, face_count
