import cv2
import insightface
import threading

from core.config import ANALYSIS_MODEL, FACE_CONFIDENCE_THRESHOLD, FACE_DETECTION_SIZE, LIVE_DETECTION_SIZE
from core.runtime import get_providers, register_session
from download_models import check_model_status

FACE_ANALYSER_ = None
LIVE_FACE_ANALYSER_ = None
LOCK_ = threading.Lock()

# Target frames only need the detector (bbox + 5 keypoints used by the swapper) and the
# 106-point landmarks used by the masks. Recognition, gender/age and 3D landmarks are only
# needed for the source face, so the per-frame analyser skips them.
LIVE_MODULES = ["detection", "landmark_2d_106"]

# When a face is very close to the camera it overflows the frame and the detector misses it,
# so the unswapped face is shown. Re-detecting on a frame padded with a black border makes the
# face smaller and fully contained. Measured on synthetic close-ups: a 50% border finds faces
# zoomed 1.6-3.0x (no detection without it); a 25% border only handles 1.6x.
CLOSE_FACE_PAD_RATIO = 0.5


class LiveFaceAnalyser:
    """Wraps the live FaceAnalysis with a padded re-detection when nothing is found."""

    def __init__(self, analyser):
        self._analyser = analyser

    def __getattr__(self, name):
        return getattr(self._analyser, name)

    def get(self, frame, *args, **kwargs):
        faces = self._analyser.get(frame, *args, **kwargs)
        if faces or CLOSE_FACE_PAD_RATIO <= 0:
            return faces
        pad = int(max(frame.shape[:2]) * CLOSE_FACE_PAD_RATIO)
        padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        faces = self._analyser.get(padded, *args, **kwargs)
        for face in faces:
            # Map back to frame coordinates; points outside the frame are fine for alignment
            face.bbox = face.bbox - pad
            face.kps = face.kps - pad
            if face.get("landmark_2d_106") is not None:
                face.landmark_2d_106 = face.landmark_2d_106 - pad
        return faces


def _create_analyser(allowed_modules, label, det_size):
    is_downloaded, _, _ = check_model_status(ANALYSIS_MODEL)
    if not is_downloaded:
        raise Exception(f"The '{ANALYSIS_MODEL}' model is not loaded/downloaded. Please go to the Models tab to download it first.")

    analyser = insightface.app.FaceAnalysis(
        name=ANALYSIS_MODEL, allowed_modules=allowed_modules, providers=get_providers()
    )
    analyser.prepare(ctx_id=0, det_thresh=FACE_CONFIDENCE_THRESHOLD, det_size=det_size)
    for task, model in analyser.models.items():
        register_session(f"{label}/{task}", model.session)
    return analyser


def get_face_analyser():
    """Full analyser (includes the recognition embedding). Use for the source face."""
    global FACE_ANALYSER_

    if FACE_ANALYSER_ is None:
        with LOCK_:
            if FACE_ANALYSER_ is None:
                FACE_ANALYSER_ = _create_analyser(None, "analyser", FACE_DETECTION_SIZE)

    return FACE_ANALYSER_


def get_live_face_analyser():
    """Lightweight analyser for target frames: detection + 106-point landmarks only."""
    global LIVE_FACE_ANALYSER_

    if LIVE_FACE_ANALYSER_ is None:
        with LOCK_:
            if LIVE_FACE_ANALYSER_ is None:
                LIVE_FACE_ANALYSER_ = LiveFaceAnalyser(_create_analyser(LIVE_MODULES, "live", LIVE_DETECTION_SIZE))

    return LIVE_FACE_ANALYSER_
