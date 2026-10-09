"""Give the swapped face the camera's own fine skin texture.

inswapper outputs a 128x128 face. At 720p the face is 200+ px wide, so the swap is upscaled ~2x
and looks smoother than the real skin around it; where the two meet (e.g. across the forehead)
the swap shows as a visible smooth "layer". This replaces the swap's finest detail band with the
camera's: out = blur(swapped) + (camera - blur(camera)). Where nothing was swapped,
swapped == camera, so the result is exactly the camera image.

Measured on 6 webcam frames (Razer Kiyo, 720p), forehead texture vs real skin (Laplacian
variance ratio, 1.0 = same): plain swap 0.04, sharpening 0.08, GFPGAN 0.13, detail transfer 0.96;
identity similarity to the source 0.917 -> 0.900.
"""

import cv2
import numpy as np

SIGMA = 1.2  # only the finest detail (pores, grain); larger values bring back the camera's features
PAD = 0.3  # region around the face box that the paste-back can reach (fraction of face width)


def apply_camera_texture(camera, swapped, face):
    """Return `swapped` with the face region's finest detail taken from `camera`."""
    h, w = camera.shape[:2]
    x1, y1, x2, y2 = face.bbox
    pad = PAD * max(x2 - x1, y2 - y1)
    X0, Y0 = max(int(x1 - pad), 0), max(int(y1 - pad), 0)
    X1, Y1 = min(int(x2 + pad), w), min(int(y2 + pad), h)
    if X1 - X0 < 4 or Y1 - Y0 < 4:
        return swapped

    cam = camera[Y0:Y1, X0:X1]
    swp = swapped[Y0:Y1, X0:X1]
    # int16 avoids clipping the signed detail band before it is added back
    detail = cam.astype(np.int16) - cv2.GaussianBlur(cam, (0, 0), SIGMA).astype(np.int16)
    base = cv2.GaussianBlur(swp, (0, 0), SIGMA).astype(np.int16)

    out = swapped.copy()
    out[Y0:Y1, X0:X1] = np.clip(base + detail, 0, 255).astype(np.uint8)
    return out
