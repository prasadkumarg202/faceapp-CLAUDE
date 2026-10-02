"""VideoThread behaviour against a fake 30 fps camera (no hardware needed)."""

import sys
import time
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import onnxruntime as ort  # noqa: E402

ort.preload_dlls()

from PyQt6.QtCore import QCoreApplication, Qt  # noqa: E402

from app import video_thread as vt  # noqa: E402
from core.face_analyser import get_face_analyser  # noqa: E402

TARGET = cv2.resize(cv2.imread(str(ROOT / "assets" / "vijay.jpg")), (640, 480))


def stamp(frame, n):
    """Encode the frame number in the top-left 4 pixels (outside the face)."""
    frame[0, 0:4, 0] = [(n >> s) & 255 for s in (0, 8, 16, 24)]
    return frame


def read_stamp(frame):
    return sum(int(v) << s for v, s in zip(frame[0, 0:4, 0], (0, 8, 16, 24), strict=True))


class FakeCapture:
    def __init__(self, *args, **kwargs):
        self.n = 0
        self.next_time = time.perf_counter()

    def isOpened(self):
        return True

    def set(self, *args):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480}.get(prop, 30)

    def read(self):
        self.next_time += 1 / 30
        delay = self.next_time - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        self.n += 1
        return True, stamp(TARGET.copy(), self.n)

    def release(self):
        pass


@pytest.fixture(scope="module")
def qapp():
    return QCoreApplication.instance() or QCoreApplication([])


@pytest.fixture(scope="module")
def source_face():
    return get_face_analyser().get(cv2.imread(str(ROOT / "assets" / "elon_musk.jpg")))[0]


times = []  # emit timestamps of the most recent run_thread call


def run_thread(monkeypatch, qapp, seconds, swap, source_face):
    times.clear()
    monkeypatch.setattr(vt.cv2, "VideoCapture", FakeCapture)
    thread = vt.VideoThread(0)
    thread.set_source_face(source_face)
    thread.enable_swap(swap)
    frames, fps = [], []

    def on_frame(f):
        frames.append(f.copy())
        times.append(time.perf_counter())
        thread.frame_displayed()  # act as a GUI that draws instantly

    # Direct connections: collect in the emitting thread, no event loop needed
    thread.frame_ready.connect(on_frame, Qt.ConnectionType.DirectConnection)
    thread.fps_update.connect(fps.append, Qt.ConnectionType.DirectConnection)
    thread.start()
    time.sleep(seconds)
    thread.stop()
    return frames, fps


def test_no_repeated_or_out_of_order_frames(monkeypatch, qapp, source_face):
    frames, _ = run_thread(monkeypatch, qapp, 4, True, source_face)
    stamps = [read_stamp(f) for f in frames]
    # Thresholds are deliberately loose: the GPU may be shared with another app during tests
    assert len(stamps) > 10, f"only {len(stamps)} frames processed"
    assert all(b > a for a, b in zip(stamps, stamps[1:], strict=False)), "repeated or out-of-order frame"


def test_no_unswapped_frames_while_swap_enabled(monkeypatch, qapp, source_face):
    frames, _ = run_thread(monkeypatch, qapp, 3, True, source_face)
    assert frames
    for f in frames:
        raw = stamp(TARGET.copy(), read_stamp(f))
        assert cv2.absdiff(f, raw).mean() > 1, "an unswapped camera frame was displayed"


def test_reported_fps_is_processing_rate(monkeypatch, qapp, source_face):
    frames, fps = run_thread(monkeypatch, qapp, 4, True, source_face)
    reported = fps[-1]
    # Compare against the actual rate over the counter's own window (last 30 frames), not the
    # whole run, which includes model warm-up
    window = times[-31:]
    actual = (len(window) - 1) / (window[-1] - window[0])
    assert reported < 30 * 0.95 or actual > 27, "FPS reports camera rate instead of processing rate"
    assert abs(reported - actual) / actual < 0.25, f"reported {reported:.1f} vs actual {actual:.1f}"


def test_camera_glitch_does_not_stop_thread(monkeypatch, qapp, source_face):
    class GlitchyCapture(FakeCapture):
        def read(self):
            ok, frame = super().read()
            return (False, None) if 30 <= self.n < 40 else (ok, frame)

    monkeypatch.setattr(vt.cv2, "VideoCapture", GlitchyCapture)
    thread = vt.VideoThread(0)
    errors, frames = [], []
    thread.error_occurred.connect(errors.append, Qt.ConnectionType.DirectConnection)
    thread.frame_ready.connect(lambda f: (frames.append(read_stamp(f)), thread.frame_displayed()),
                               Qt.ConnectionType.DirectConnection)
    thread.start()
    time.sleep(3)
    thread.stop()
    assert not errors, errors
    assert frames and max(frames) > 40, "processing did not resume after the glitch"


def test_duplicate_camera_frames_are_skipped(monkeypatch, qapp, source_face):
    class PaddingCapture(FakeCapture):
        """Delivers every frame twice, like a 15 fps sensor padded to 30 fps."""
        def read(self):
            ok, frame = super().read()
            return ok, stamp(TARGET.copy(), (self.n + 1) // 2)

    monkeypatch.setattr(vt.cv2, "VideoCapture", PaddingCapture)
    thread = vt.VideoThread(0)
    stamps = []
    thread.frame_ready.connect(lambda f: (stamps.append(read_stamp(f)), thread.frame_displayed()),
                               Qt.ConnectionType.DirectConnection)
    thread.start()
    time.sleep(3)
    thread.stop()
    assert thread.duplicate_frames > 20
    assert len(stamps) == len(set(stamps)), "a duplicate frame was processed twice"


def test_slow_gui_does_not_queue_frames(monkeypatch, qapp, source_face):
    """If the GUI never finishes drawing, only one frame is emitted; processing continues."""
    monkeypatch.setattr(vt.cv2, "VideoCapture", FakeCapture)
    thread = vt.VideoThread(0)
    emitted, fps = [], []
    thread.frame_ready.connect(emitted.append, Qt.ConnectionType.DirectConnection)
    thread.fps_update.connect(fps.append, Qt.ConnectionType.DirectConnection)
    thread.start()
    time.sleep(2)
    thread.stop()
    assert len(emitted) == 1
    assert len(fps) > 10, "processing stalled while display was busy"


def test_virtual_camera_receives_every_processed_frame(monkeypatch, qapp, source_face):
    class FakeVirtualCam:
        def __init__(self):
            self.frames = []

        def send(self, frame):
            self.frames.append(read_stamp(frame))

    monkeypatch.setattr(vt.cv2, "VideoCapture", FakeCapture)
    thread = vt.VideoThread(0)
    thread.set_source_face(source_face)
    thread.enable_swap(True)
    cam = FakeVirtualCam()
    thread.set_virtual_camera(cam)
    fps = []
    thread.fps_update.connect(fps.append, Qt.ConnectionType.DirectConnection)
    thread.start()
    time.sleep(3)
    thread.stop()
    # display never acknowledged, yet the virtual camera still gets every processed frame
    assert len(cam.frames) > 10 and len(cam.frames) == len(set(cam.frames))
