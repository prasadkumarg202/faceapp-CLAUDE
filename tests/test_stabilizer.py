"""Temporal stabilization of face landmarks."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.engine.stabilizer import StabilizedAnalyser  # noqa: E402


class FakeFace(dict):
    __getattr__ = dict.get


class FakeAnalyser:
    """Returns one face whose position follows `path`, plus detector noise."""

    def __init__(self, path, noise=0.8, seed=0):
        self.path, self.noise, self.i = path, noise, 0
        self.rng = np.random.default_rng(seed)
        self.base = np.array([[280, 200], [360, 200], [320, 240], [290, 280], [350, 280]], np.float32)

    def get(self, frame):
        x, y = self.path(self.i)
        self.i += 1
        kps = self.base + [x, y] + self.rng.normal(0, self.noise, (5, 2))
        bbox = np.array([250 + x, 150 + y, 390 + x, 320 + y], np.float32)
        return [FakeFace(kps=kps.astype(np.float32), bbox=bbox)]


def run(path, n, smooth=True, fps=15):
    clock = iter(np.arange(n) / fps)
    analyser = FakeAnalyser(path)
    stab = StabilizedAnalyser(analyser, clock=lambda: next(clock)) if smooth else analyser
    return np.array([stab.get(None)[0].kps for _ in range(n)])


def truth(path, n):
    base = FakeAnalyser(path).base
    return np.array([base + path(i) for i in range(n)])


def test_still_head_jitter_is_reduced():
    path = lambda i: (0.0, 0.0)  # noqa: E731
    raw = run(path, 60, smooth=False)
    smooth = run(path, 60)
    jitter = lambda k: np.linalg.norm(np.diff(k, axis=0), axis=2).mean()  # noqa: E731
    assert jitter(smooth) < 0.5 * jitter(raw)


def test_fast_move_is_followed_without_lag():
    step = lambda i: (0.0 if i < 10 else 40.0, 0.0)  # noqa: E731
    clock = iter(np.arange(30) / 15)
    analyser = FakeAnalyser(step, noise=0.0)
    stab = StabilizedAnalyser(analyser, clock=lambda: next(clock))
    out = np.array([stab.get(None)[0].kps for _ in range(30)])
    lag = np.abs(out[:, 0, 0] - truth(step, 30)[:, 0, 0])
    assert lag[12:].max() < 1.0, "still lagging 2 frames after a fast move"


def test_new_face_far_away_is_not_smoothed_towards_old_one():
    jump = lambda i: (0.0 if i < 5 else 300.0, 0.0)  # noqa: E731
    out = run(jump, 7)
    assert abs(out[5, 0, 0] - (280 + 300)) < 3, "track was not reset for a face that jumped"


def test_gap_resets_tracks():
    times = iter([0.0, 0.066, 0.133, 2.0])  # 2 s without frames before the last one
    analyser = FakeAnalyser(lambda i: (0.0, 0.0) if i < 3 else (20.0, 0.0), noise=0.0)
    stab = StabilizedAnalyser(analyser, clock=lambda: next(times))
    for _ in range(3):
        stab.get(None)
    assert abs(stab.get(None)[0].kps[0, 0] - 300) < 1e-3
