"""Remembered settings between sessions, and 16:9 virtual-camera output that follows the face."""

import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import onnxruntime as ort  # noqa: E402

ort.preload_dlls()

from PyQt6.QtWidgets import QApplication  # noqa: E402

from app import video_thread as vt  # noqa: E402
from app.video_thread import widescreen_crop_top  # noqa: E402
from test_video_thread import FakeCapture  # noqa: E402

# Keep a reference: a QApplication that is garbage-collected takes the process down with it
# when the next widget is created.
APP = QApplication.instance() or QApplication([])


# --- crop maths -------------------------------------------------------------------------------

def test_crop_centred_without_face():
    assert widescreen_crop_top(480, 360, None) == 60


def test_crop_clamped_to_frame():
    assert widescreen_crop_top(480, 360, 20) == 0      # face near the top
    assert widescreen_crop_top(480, 360, 470) == 120   # face near the bottom


def test_crop_follows_face_smoothly():
    first = widescreen_crop_top(480, 360, 250)
    assert first == pytest.approx(250 - 0.45 * 360)
    moved = widescreen_crop_top(480, 360, 200, previous_top=first)
    assert first > moved > widescreen_crop_top(480, 360, 200)  # moves part of the way


# --- video thread output ---------------------------------------------------------------------

def test_virtual_camera_gets_16x9_frames_with_face_in_view(monkeypatch):
    class FakeVirtualCam:
        def __init__(self):
            self.shapes, self.face_offsets = [], []

        def send(self, frame):
            self.shapes.append(frame.shape)

    monkeypatch.setattr(vt.cv2, "VideoCapture", FakeCapture)
    thread = vt.VideoThread(0)
    cam = FakeVirtualCam()
    thread.set_virtual_camera(cam, crop_height=360)
    thread.start()
    deadline = time.time() + 30  # model loading takes a few seconds in a fresh process
    while len(cam.shapes) < 10 and time.time() < deadline:
        time.sleep(0.1)
    face_y, top = thread.face_center_y, thread._crop_top
    thread.stop()
    assert cam.shapes and all(s == (360, 640, 3) for s in cam.shapes)
    assert face_y is not None and top is not None
    assert top <= face_y <= top + 360, "face outside the 16:9 crop"


# --- remembered settings -----------------------------------------------------------------------

@pytest.fixture
def make_window(tmp_path, monkeypatch):
    settings = tmp_path / "settings.json"
    monkeypatch.setenv("DEEPFACENET_SETTINGS", str(settings))
    windows = []

    def make(initial=None):
        if initial is not None:
            settings.write_text(json.dumps(initial))
        from app.deepfake_app import DeepfakeApp
        w = DeepfakeApp()
        w._restore_session()  # normally run by a 0 ms timer once the event loop starts
        windows.append(w)
        return w

    yield make, settings
    for w in windows:
        w.close()  # waits for the startup camera scan thread
        w.deleteLater()


def test_session_restores_photos_mode_and_widescreen(make_window):
    make, _ = make_window
    w = make({"source_paths": [str(ROOT / "assets" / "vijay.jpg")], "mode": "Quality", "widescreen": False})
    assert w.source_face is not None
    assert w.mode_combo.currentText() == "Quality"
    assert not w.widescreen_checkbox.isChecked()


def test_user_choices_are_saved(make_window):
    make, settings = make_window
    w = make({})
    w.mode_combo.setCurrentText("Quality")
    w.widescreen_checkbox.setChecked(False)
    saved = json.loads(settings.read_text())
    assert saved["mode"] == "Quality" and saved["widescreen"] is False


def test_stop_camera_does_not_forget_choices(make_window):
    make, settings = make_window
    w = make({"virtual_camera": True, "face_swap": True})
    w.stop_camera()  # resets the controls, which must not be saved as the user's choice
    saved = json.loads(settings.read_text()) if settings.exists() else {}
    assert saved.get("virtual_camera") is True and saved.get("face_swap") is True


def test_missing_photos_are_ignored(make_window):
    make, _ = make_window
    w = make({"source_paths": ["C:/does/not/exist.jpg"]})
    assert w.source_face is None
