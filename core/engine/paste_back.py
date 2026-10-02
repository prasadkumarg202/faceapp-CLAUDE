"""Region-of-interest version of insightface's INSwapper paste-back.

INSwapper.get(paste_back=True) warps the 128px swapped crop back onto the frame by running three
full-frame warpAffines, a full-frame np.where, a large erode, a Gaussian blur and a float32
blend over the whole image; it also builds a `fake_diff` mask that is never used. This module
produces the same output but only processes the bounding box of the warped crop, padded by
enough margin that erode/blur see the same zero neighbourhood as in the full-frame version.
"""

import cv2
import numpy as np


def _warped_bounds(IM, crop_size, frame_w, frame_h, margin):
    corners = np.array([[0, 0], [crop_size, 0], [0, crop_size], [crop_size, crop_size]], dtype=np.float32)
    pts = corners @ IM[:, :2].T + IM[:, 2]
    x0 = max(int(np.floor(pts[:, 0].min())) - margin, 0)
    y0 = max(int(np.floor(pts[:, 1].min())) - margin, 0)
    x1 = min(int(np.ceil(pts[:, 0].max())) + margin + 1, frame_w)
    y1 = min(int(np.ceil(pts[:, 1].max())) + margin + 1, frame_h)
    return x0, y0, x1, y1


def paste_back(frame, bgr_fake, M):
    """Blend the swapped crop `bgr_fake` (aligned with affine `M`) into a copy of `frame`."""
    h, w = frame.shape[:2]
    crop = bgr_fake.shape[0]
    IM = cv2.invertAffineTransform(M)

    # Mask size as computed by insightface (extent of the warped crop), before clipping to the frame
    # it is approximated from the transformed corners; recomputed exactly below from the ROI.
    scale = float(np.sqrt(abs(np.linalg.det(IM[:, :2]))))
    approx_size = int(crop * scale * 1.5)
    margin = max(approx_size // 10, 10) + 2 * max(approx_size // 20, 5) + 4

    x0, y0, x1, y1 = _warped_bounds(IM, crop, w, h, margin)
    if x1 <= x0 or y1 <= y0:
        return frame.copy()

    IM_roi = IM.copy()
    IM_roi[0, 2] -= x0
    IM_roi[1, 2] -= y0
    roi_size = (x1 - x0, y1 - y0)

    fake_roi = cv2.warpAffine(bgr_fake, IM_roi, roi_size, borderValue=0.0)
    white = np.full((crop, crop), 255, dtype=np.float32)
    mask = cv2.warpAffine(white, IM_roi, roi_size, borderValue=0.0)
    mask[mask > 20] = 255

    _, _, bw, bh = cv2.boundingRect((mask == 255).astype(np.uint8))
    if bw == 0 or bh == 0:
        return frame.copy()
    mask_size = int(np.sqrt((bh - 1) * (bw - 1)))  # same extent as insightface's max - min

    k = max(mask_size // 10, 10)
    mask = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1)
    k = max(mask_size // 20, 5)
    mask = cv2.GaussianBlur(mask, (2 * k + 1, 2 * k + 1), 0)
    mask *= np.float32(1.0 / 255)

    # cv2.blendLinear: multithreaded uint8 blend, ~10x faster than the numpy float32 version.
    out = frame.copy()
    target_roi = np.ascontiguousarray(frame[y0:y1, x0:x1])
    out[y0:y1, x0:x1] = cv2.blendLinear(fake_roi, target_roi, mask, 1.0 - mask)
    return out
