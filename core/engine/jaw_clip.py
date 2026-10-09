"""Keep everything below the jawline (neck, collar, shirt) untouched by the swap.

inswapper pastes back its whole aligned square crop. When the face sits low in the frame that
square reaches below the chin, and the model's version of the neck/collar is blended in: a light,
shirt-coloured haze on the chin and neck. Cutting the swap off along the 106-point jaw contour
removes it. Measured on 8 webcam frames (Razer Kiyo, 720p): mean change below the jaw
2.31 -> 0.02 (XSeg occlusion mask: 0.05, at ~30 ms/frame), for about a millisecond.
"""

import cv2
import numpy as np

JAW_DROP = 0.02  # move the cut slightly below the detected jaw line (fraction of face width)
FEATHER = 0.04  # softness of the edge (fraction of face width)


def restore_below_jaw(original, swapped, face):
    """Return `swapped` with the pixels below the jaw contour of `face` taken from `original`."""
    lm = getattr(face, "landmark_2d_106", None)
    if lm is None:
        return swapped
    jaw = lm[0:33].astype(np.float32)  # face contour: ear -> chin -> ear
    face_w = float(np.ptp(jaw[:, 0]))
    if face_w < 8:
        return swapped
    h, w = original.shape[:2]
    jaw = jaw + [0, JAW_DROP * face_w]
    jaw = jaw[np.argsort(jaw[:, 0])]

    # Work only on the band around the jaw where the paste-back can reach below it
    k = max(3, int(face_w * FEATHER) | 1)
    x0 = max(int(jaw[:, 0].min() - face_w * 0.6), 0)
    x1 = min(int(jaw[:, 0].max() + face_w * 0.6), w)
    y0 = max(int(jaw[:, 1].min()) - k, 0)
    y1 = min(int(jaw[:, 1].max() + face_w * 0.8), h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return swapped

    # Region above the jaw (keeps the swap); the polygon is closed along the top of the band
    pts = jaw - [x0, y0]
    poly = np.vstack([[pts[0, 0], 0], pts, [pts[-1, 0], 0]]).astype(np.int32)
    keep = np.zeros((y1 - y0, x1 - x0), np.float32)
    cv2.fillPoly(keep, [poly], 1.0)
    # Left/right of the jaw contour stay as pasted (cheeks/ears are handled by the paste mask)
    keep[:, : max(int(pts[:, 0].min()), 0)] = 1.0
    keep[:, int(pts[:, 0].max()) + 1:] = 1.0
    keep = cv2.GaussianBlur(keep, (k, k), 0)

    out = swapped.copy()
    out[y0:y1, x0:x1] = cv2.blendLinear(
        np.ascontiguousarray(swapped[y0:y1, x0:x1]), np.ascontiguousarray(original[y0:y1, x0:x1]), keep, 1.0 - keep
    )
    return out
