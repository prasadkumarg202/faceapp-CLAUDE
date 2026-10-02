import logging
import cv2
import numpy as np
from typing import Any
import core.config as config
from core.runtime import log_throttled
from core.engine.gpu_processing import gpu_gaussian_blur, gpu_resize

logger = logging.getLogger(__name__)

Face = Any
Frame = np.ndarray

def apply_color_transfer(source, target):
    """
    Apply color transfer from target to source image using LAB color space.
    Uses float32 throughout for performance (sufficient precision for 8-bit images).
    """
    # Convert to float32 [0,1] range for proper LAB conversion
    source_f32 = source.astype(np.float32) / 255.0
    target_f32 = target.astype(np.float32) / 255.0

    source_lab = cv2.cvtColor(source_f32, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_f32, cv2.COLOR_BGR2LAB)

    source_mean, source_std = cv2.meanStdDev(source_lab)
    target_mean, target_std = cv2.meanStdDev(target_lab)

    source_mean = source_mean.reshape(1, 1, 3).astype(np.float32)
    source_std = np.maximum(source_std.reshape(1, 1, 3), 1e-6).astype(np.float32)
    target_mean = target_mean.reshape(1, 1, 3).astype(np.float32)
    target_std = target_std.reshape(1, 1, 3).astype(np.float32)

    result_lab = (source_lab - source_mean) * (target_std / source_std) + target_mean
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    return np.clip(result_bgr * 255.0, 0, 255).astype(np.uint8)

def create_face_mask(face: Face, frame: Frame) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        landmarks = landmarks.astype(np.int32)
        right_side_face = landmarks[0:16]
        left_side_face = landmarks[17:32]

        padding = int(
            np.linalg.norm(right_side_face[0] - left_side_face[-1]) * 0.05
        )

        face_outline = landmarks[0:33]
        hull = cv2.convexHull(face_outline)
        center = np.mean(face_outline, axis=0, dtype=np.float32)
        hull_pts = hull.reshape(-1, 2).astype(np.float32)
        directions = hull_pts - center
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        directions /= norms
        hull_padded = (hull_pts + directions * padding).astype(np.int32)

        cv2.fillConvexPoly(mask, hull_padded, 255)
        mask = gpu_gaussian_blur(mask, (5, 5), 3)

    return mask

def create_lower_mouth_mask(
    face: Face, frame: Frame
) -> tuple[np.ndarray, np.ndarray | None, tuple, np.ndarray | None]:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mouth_cutout = None
    lower_lip_polygon = None
    mouth_box = (0,0,0,0)

    landmarks = face.landmark_2d_106
    if landmarks is not None:
        lower_lip_order = list(range(52, 72))
        if max(lower_lip_order) >= landmarks.shape[0]:
            return mask, mouth_cutout, mouth_box, lower_lip_polygon

        lower_lip_landmarks = landmarks[lower_lip_order].astype(np.float32)
        center = np.mean(lower_lip_landmarks, axis=0)

        mouth_mask_size = getattr(config, "MOUTH_MASK_SIZE", 0.0)
        expansion_factor = 1 + (mouth_mask_size / 100.0) * 2.5
        offsets = lower_lip_landmarks - center
        chin_bias = 1 + (mouth_mask_size / 100.0) * 1.5
        scale_y = np.where(offsets[:, 1] > 0, expansion_factor * chin_bias, expansion_factor)
        expanded_landmarks = lower_lip_landmarks.copy()
        expanded_landmarks[:, 0] = center[0] + offsets[:, 0] * expansion_factor
        expanded_landmarks[:, 1] = center[1] + offsets[:, 1] * scale_y
        expanded_landmarks = expanded_landmarks.astype(np.int32)

        min_x, min_y = np.min(expanded_landmarks, axis=0)
        max_x, max_y = np.max(expanded_landmarks, axis=0)

        padding = int((max_x - min_x) * 0.1)
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(frame.shape[1], max_x + padding)
        max_y = min(frame.shape[0], max_y + padding)

        if max_x - min_x < 2 or max_y - min_y < 2:  # mouth outside the frame
            return mask, None, (0, 0, 0, 0), None

        mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        polygon_relative_to_roi = expanded_landmarks - [min_x, min_y]
        cv2.fillPoly(mask_roi, [polygon_relative_to_roi], 255)
        mask_roi = gpu_gaussian_blur(mask_roi, (15, 15), 5)
        mask[min_y:max_y, min_x:max_x] = mask_roi

        mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()
        lower_lip_polygon = expanded_landmarks
        mouth_box = (min_x, min_y, max_x, max_y)

    return mask, mouth_cutout, mouth_box, lower_lip_polygon

def create_eyes_mask(face: Face, frame: Frame) -> tuple[np.ndarray, np.ndarray | None, tuple, np.ndarray | None]:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    eyes_cutout = None
    eyes_polygon = None
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        left_eye = landmarks[87:96]
        right_eye = landmarks[33:42]

        left_eye_center = np.mean(left_eye, axis=0).astype(np.int32)
        right_eye_center = np.mean(right_eye, axis=0).astype(np.int32)

        def get_eye_dimensions(eye_points):
            x_coords = eye_points[:, 0]
            y_coords = eye_points[:, 1]
            width = int((np.max(x_coords) - np.min(x_coords)) * (1 + getattr(config, "MASK_DOWN_SIZE", 1.0) * getattr(config, "EYES_MASK_SIZE", 0.0)))
            height = int((np.max(y_coords) - np.min(y_coords)) * (1 + getattr(config, "MASK_DOWN_SIZE", 1.0) * getattr(config, "EYES_MASK_SIZE", 0.0)))
            return width, height

        left_width, left_height = get_eye_dimensions(left_eye)
        right_width, right_height = get_eye_dimensions(right_eye)

        padding = int(max(left_width, right_width) * 0.2)

        min_x = min(left_eye_center[0] - left_width//2, right_eye_center[0] - right_width//2) - padding
        max_x = max(left_eye_center[0] + left_width//2, right_eye_center[0] + right_width//2) + padding
        min_y = min(left_eye_center[1] - left_height//2, right_eye_center[1] - right_height//2) - padding
        max_y = max(left_eye_center[1] + left_height//2, right_eye_center[1] + right_height//2) + padding

        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(frame.shape[1], max_x)
        max_y = min(frame.shape[0], max_y)
        if max_x - min_x < 2 or max_y - min_y < 2:  # eyes outside the frame
            return mask, None, (0, 0, 0, 0), None

        mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)

        left_center = (left_eye_center[0] - min_x, left_eye_center[1] - min_y)
        right_center = (right_eye_center[0] - min_x, right_eye_center[1] - min_y)

        left_axes = (left_width//2, left_height//2)
        right_axes = (right_width//2, right_height//2)

        cv2.ellipse(mask_roi, left_center, left_axes, 0, 0, 360, 255, -1)
        cv2.ellipse(mask_roi, right_center, right_axes, 0, 0, 360, 255, -1)
        mask_roi = gpu_gaussian_blur(mask_roi, (15, 15), 5)
        mask[min_y:max_y, min_x:max_x] = mask_roi
        eyes_cutout = frame[min_y:max_y, min_x:max_x].copy()

        def create_ellipse_points(center, axes):
            t = np.linspace(0, 2*np.pi, 32)
            x = center[0] + axes[0] * np.cos(t)
            y = center[1] + axes[1] * np.sin(t)
            return np.column_stack((x, y)).astype(np.int32)

        left_points = create_ellipse_points((left_eye_center[0], left_eye_center[1]), (left_width//2, left_height//2))
        right_points = create_ellipse_points((right_eye_center[0], right_eye_center[1]), (right_width//2, right_height//2))
        eyes_polygon = np.vstack([left_points, right_points])

    return mask, eyes_cutout, (min_x, min_y, max_x, max_y), eyes_polygon

def create_curved_eyebrow(points):
    if len(points) >= 5:
        sorted_idx = np.argsort(points[:, 0])
        sorted_points = points[sorted_idx]
        x_min, y_min = np.min(sorted_points, axis=0)
        x_max, y_max = np.max(sorted_points, axis=0)
        width = x_max - x_min
        height = y_max - y_min
        num_points = 50
        x = np.linspace(x_min, x_max, num_points)
        coeffs = np.polyfit(sorted_points[:, 0], sorted_points[:, 1], 2)
        y = np.polyval(coeffs, x)
        top_offset = height * 0.5
        bottom_offset = height * 0.2
        top_curve = y - top_offset
        bottom_curve = y + bottom_offset
        end_points = 5
        start_x = np.linspace(x[0] - width * 0.15, x[0], end_points)
        end_x = np.linspace(x[-1], x[-1] + width * 0.15, end_points)
        start_curve = np.column_stack((
            start_x,
            np.linspace(bottom_curve[0], top_curve[0], end_points)
        ))
        end_curve = np.column_stack((
            end_x,
            np.linspace(bottom_curve[-1], top_curve[-1], end_points)
        ))
        contour_points = np.vstack([
            start_curve,
            np.column_stack((x, top_curve)),
            end_curve,
            np.column_stack((x[::-1], bottom_curve[::-1]))
        ])
        center = np.mean(contour_points, axis=0)
        vectors = contour_points - center
        padded_points = center + vectors * 1.2
        return padded_points
    return points

def create_eyebrows_mask(face: Face, frame: Frame) -> tuple[np.ndarray, np.ndarray | None, tuple, np.ndarray | None]:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    eyebrows_cutout = None
    eyebrows_polygon = None
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        left_eyebrow = landmarks[97:105].astype(np.float32)
        right_eyebrow = landmarks[43:51].astype(np.float32)

        all_points = np.vstack([left_eyebrow, right_eyebrow])
        padding_factor = getattr(config, "EYEBROWS_MASK_SIZE", 0.0)
        min_x = np.min(all_points[:, 0]) - 25 * padding_factor
        max_x = np.max(all_points[:, 0]) + 25 * padding_factor
        min_y = np.min(all_points[:, 1]) - 20 * padding_factor
        max_y = np.max(all_points[:, 1]) + 15 * padding_factor

        min_x = max(0, int(min_x))
        min_y = max(0, int(min_y))
        max_x = min(frame.shape[1], int(max_x))
        max_y = min(frame.shape[0], int(max_y))
        if max_x - min_x < 2 or max_y - min_y < 2:  # eyebrows outside the frame
            return mask, None, (0, 0, 0, 0), None

        mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)

        try:
            left_local = left_eyebrow - [min_x, min_y]
            right_local = right_eyebrow - [min_x, min_y]

            left_shape = create_curved_eyebrow(left_local)
            right_shape = create_curved_eyebrow(right_local)

            mask_roi = gpu_gaussian_blur(mask_roi, (21, 21), 7)
            mask_roi = gpu_gaussian_blur(mask_roi, (11, 11), 3)
            mask_roi = gpu_gaussian_blur(mask_roi, (5, 5), 1)
            mask_roi = cv2.normalize(mask_roi, None, 0, 255, cv2.NORM_MINMAX)

            mask[min_y:max_y, min_x:max_x] = mask_roi
            eyebrows_cutout = frame[min_y:max_y, min_x:max_x].copy()

            eyebrows_polygon = np.vstack([
                left_shape + [min_x, min_y],
                right_shape + [min_x, min_y]
            ]).astype(np.int32)

        except Exception:
            left_local = left_eyebrow - [min_x, min_y]
            right_local = right_eyebrow - [min_x, min_y]
            cv2.fillPoly(mask_roi, [left_local.astype(np.int32)], 255)
            cv2.fillPoly(mask_roi, [right_local.astype(np.int32)], 255)
            mask_roi = gpu_gaussian_blur(mask_roi, (21, 21), 7)
            mask[min_y:max_y, min_x:max_x] = mask_roi
            eyebrows_cutout = frame[min_y:max_y, min_x:max_x].copy()
            eyebrows_polygon = np.vstack([left_eyebrow, right_eyebrow]).astype(np.int32)

    return mask, eyebrows_cutout, (min_x, min_y, max_x, max_y), eyebrows_polygon

def apply_mask_area(
    frame: np.ndarray,
    cutout: np.ndarray,
    box: tuple,
    face_mask: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray:
    min_x, min_y, max_x, max_y = box
    box_width = max_x - min_x
    box_height = max_y - min_y

    if (
        cutout is None
        or box_width is None
        or box_height is None
        or face_mask is None
        or polygon is None
    ):
        return frame

    try:
        resized_cutout = gpu_resize(cutout, (box_width, box_height))
        roi = frame[min_y:max_y, min_x:max_x]

        if roi.shape != resized_cutout.shape:
            resized_cutout = gpu_resize(
                resized_cutout, (roi.shape[1], roi.shape[0])
            )

        color_corrected_area = apply_color_transfer(resized_cutout, roi)

        polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8)

        if len(polygon) > 50:
            mid_point = len(polygon) // 2
            left_points = polygon[:mid_point] - [min_x, min_y]
            right_points = polygon[mid_point:] - [min_x, min_y]
            cv2.fillPoly(polygon_mask, [left_points], 255)
            cv2.fillPoly(polygon_mask, [right_points], 255)
        else:
            adjusted_polygon = polygon - [min_x, min_y]
            cv2.fillPoly(polygon_mask, [adjusted_polygon], 255)

        polygon_mask = gpu_gaussian_blur(polygon_mask, (21, 21), 7)

        mask_feather_ratio = getattr(config, "MASK_FEATHER_RATIO", 10)
        feather_amount = min(
            30,
            box_width // mask_feather_ratio,
            box_height // mask_feather_ratio,
        )
        if feather_amount % 2 == 0:
            feather_amount += 1

        feathered_mask = cv2.GaussianBlur(
            polygon_mask.astype(np.float32), (feather_amount, feather_amount), 0
        )
        max_val = feathered_mask.max()
        if max_val > 1e-6:
            feathered_mask *= np.float32(1.0 / max_val)

        feathered_mask = cv2.GaussianBlur(feathered_mask, (5, 5), 1)

        face_mask_roi = face_mask[min_y:max_y, min_x:max_x]
        combined_mask = feathered_mask * (face_mask_roi.astype(np.float32) * np.float32(1.0 / 255.0))

        combined_mask_3ch = combined_mask[:, :, np.newaxis]
        inv_mask = np.float32(1.0) - combined_mask_3ch
        blended = (
            color_corrected_area * combined_mask_3ch + roi * inv_mask
        ).astype(np.uint8)

        face_mask_f32 = face_mask_roi[:, :, np.newaxis].astype(np.float32) * np.float32(1.0 / 255.0)
        face_mask_3channel = np.broadcast_to(face_mask_f32, blended.shape)
        final_blend = blended * face_mask_3channel + roi * (np.float32(1.0) - face_mask_3channel)

        frame[min_y:max_y, min_x:max_x] = final_blend.astype(np.uint8)
    except Exception as e:
        log_throttled(logger, "mask", "Error applying mask area: %s", e)

    return frame
