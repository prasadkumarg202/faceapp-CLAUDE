"""Temporal stabilization of face landmarks for live video.

Detector landmarks wobble by ~0.8 px per frame from sensor noise even when the head is still,
which makes the swapped face shimmer. A One-Euro filter (Casiez et al. 2012) smooths strongly
when still and follows quickly when moving.

Measured on a 60-frame noisy webcam sequence (min_cutoff=1.0, beta=0.3):
  landmark jitter 0.78 -> 0.34 px/frame (-56%), landmark error 1.08 -> 0.99 px,
  output flicker inside the face -19% (still head) / -15% (slow motion),
  a sudden 40 px move lags at most 1.5 px and catches up within one frame.
"""

import time

import numpy as np

MIN_CUTOFF = 1.0
BETA = 0.3
D_CUTOFF = 1.0
# A face whose centre jumps further than this (fraction of face width) is treated as a new face
MAX_JUMP = 0.5
# Forget a track after this many seconds without a detection
RESET_AFTER = 0.5


class OneEuroFilter:
    def __init__(self, min_cutoff=MIN_CUTOFF, beta=BETA, d_cutoff=D_CUTOFF):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = None

    @staticmethod
    def _alpha(cutoff, dt):
        r = 2 * np.pi * cutoff * dt
        return r / (r + 1)

    def __call__(self, x, dt):
        if self.x is None:
            self.x, self.dx = x.copy(), np.zeros_like(x)
            return x
        dx = (x - self.x) / dt
        self.dx = self.dx + self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        self.x = self.x + self._alpha(cutoff, dt) * (x - self.x)
        return self.x.copy()


class _Track:
    def __init__(self):
        self.kps = OneEuroFilter()
        self.bbox = OneEuroFilter()
        self.center = None
        self.width = None


class StabilizedAnalyser:
    """Wraps a live face analyser; smooths each face's bbox and 5 keypoints across frames.

    Faces are matched to tracks by nearest centre. One instance per video session.
    """

    def __init__(self, analyser, clock=time.perf_counter):
        self._analyser = analyser
        self._clock = clock
        self._tracks = []
        self._last_time = None

    def __getattr__(self, name):
        return getattr(self._analyser, name)

    def reset(self):
        self._tracks = []
        self._last_time = None

    def get(self, frame, *args, **kwargs):
        faces = self._analyser.get(frame, *args, **kwargs)
        now = self._clock()
        dt = None if self._last_time is None else now - self._last_time
        self._last_time = now
        if dt is None or dt > RESET_AFTER:
            self._tracks = []
        dt = min(max(dt or 1 / 30, 1 / 120), RESET_AFTER)

        unused = list(self._tracks)
        tracks = []
        smoothed = []
        for face in faces:
            centre = (face.bbox[:2] + face.bbox[2:]) / 2
            width = float(face.bbox[2] - face.bbox[0])
            track = None
            if unused:
                best = min(unused, key=lambda t: np.linalg.norm(t.center - centre))
                if np.linalg.norm(best.center - centre) <= MAX_JUMP * max(width, best.width):
                    track = best
                    unused.remove(best)
            if track is None:
                track = _Track()
            track.center, track.width = centre, width

            out = face.__class__(face)  # insightface Face is a dict subclass: shallow copy
            out["kps"] = track.kps(face.kps.astype(np.float64), dt).astype(np.float32)
            out["bbox"] = track.bbox(face.bbox.astype(np.float64), dt).astype(np.float32)
            smoothed.append(out)
            tracks.append(track)
        self._tracks = tracks
        return smoothed
