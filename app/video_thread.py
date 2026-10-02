"""Video capture and processing thread for Qt application

Two loops run concurrently:

  * capture (this QThread): reads the camera as fast as it delivers and keeps only the newest
    frame. Stale frames are dropped, so processing never falls behind the camera.
  * processing worker: takes the newest frame, runs detect+swap (or detection boxes when swap
    is off), sends it to the virtual camera and emits it for display. Only processed frames are
    emitted, so the display shows no repeated frames and never shows an unswapped frame while
    swap is enabled. A new frame is only emitted once the GUI has drawn the previous one
    (frame_displayed), so a slow GUI drops frames instead of queueing them.

Many webcams pad a low sensor rate (e.g. 15 fps in dim light) to 30 fps by repeating frames
bit-for-bit; those duplicates are dropped at capture so the GPU doesn't process them twice.

fps_update reports processed frames per second, not the camera rate.
"""

import logging
import threading
import time
import traceback

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from core.face_analyser import get_live_face_analyser
from core.engine.face_swapper import detect_and_swap, get_face_swapper
import core.config as config

logger = logging.getLogger(__name__)

# Consecutive failed camera reads tolerated before giving up (~1-2 s at 30 fps)
MAX_READ_FAILURES = 30


class VideoThread(QThread):
    """Worker thread for video capture and face swapping"""

    frame_ready = pyqtSignal(np.ndarray)
    fps_update = pyqtSignal(float)
    face_count_update = pyqtSignal(int)
    error_occurred = pyqtSignal(str)
    virtual_cam_error = pyqtSignal(str)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.swap_enabled = False
        self.mouth_mask_enabled = False
        self.source_face = None
        self.cap = None
        self.face_analyser = None

        self._frame_cond = threading.Condition()
        self._latest_frame = None
        self._worker = None
        self.capture_fps = 0.0
        self.processing_ms = 0.0
        self.duplicate_frames = 0

        self._display_ready = threading.Event()
        self._display_ready.set()
        self._virtual_cam = None
        self._virtual_cam_lock = threading.Lock()

    def set_source_face(self, source_face):
        """Set the source face for swapping"""
        self.source_face = source_face

    def enable_swap(self, enabled):
        """Enable or disable face swapping"""
        self.swap_enabled = enabled

    def enable_mouth_mask(self, enabled):
        """Enable or disable mouth masking"""
        self.mouth_mask_enabled = enabled

    def set_virtual_camera(self, camera):
        """Send processed frames to a pyvirtualcam.Camera from the worker thread (None to stop)"""
        with self._virtual_cam_lock:
            self._virtual_cam = camera

    def frame_displayed(self):
        """Called by the GUI after drawing a frame; allows the next frame to be emitted"""
        self._display_ready.set()

    def run(self):
        """Capture loop: keep the newest camera frame available for the processing worker"""
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.error_occurred.emit("Failed to open camera")
                return

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

            self.running = True
            self._worker = threading.Thread(target=self._process_loop, name="frame-processor", daemon=True)
            self._worker.start()

            capture_counter = FPSCounter()
            failures = 0
            previous_sample = None
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    failures += 1
                    if failures >= MAX_READ_FAILURES:
                        self.error_occurred.emit("Camera stopped sending frames")
                        break
                    time.sleep(0.02)
                    continue
                failures = 0

                # Exact-duplicate check on a subsample: padded frames are bit-identical copies
                sample = frame[::16, ::16]
                if previous_sample is not None and np.array_equal(sample, previous_sample):
                    self.duplicate_frames += 1
                    continue
                previous_sample = sample.copy()
                self.capture_fps = capture_counter.update()

                with self._frame_cond:
                    self._latest_frame = frame  # overwrite: an unprocessed older frame is dropped
                    self._frame_cond.notify()

        except Exception as e:
            self.error_occurred.emit(f"Video thread error: {str(e)}")
        finally:
            self.running = False
            with self._frame_cond:
                self._frame_cond.notify_all()
            if self._worker is not None:
                self._worker.join(timeout=5)
            self.cleanup()

    def _load_models(self):
        """Load models up front so the first detected face doesn't freeze the video for ~1 s"""
        try:
            self.face_analyser = get_live_face_analyser()
        except Exception as e:
            logger.warning("Face analyser unavailable: %s", e)
            self.face_analyser = None
            return
        try:
            get_face_swapper()
        except Exception as e:
            logger.warning("Face swapper unavailable: %s", e)

    def _process_loop(self):
        """Processing worker: always process the newest frame, emit only processed frames"""
        self._load_models()
        processed_counter = FPSCounter()

        while self.running:
            with self._frame_cond:
                while self._latest_frame is None and self.running:
                    self._frame_cond.wait(timeout=0.5)
                frame, self._latest_frame = self._latest_frame, None
            if frame is None:
                continue

            start = time.perf_counter()
            try:
                if self.swap_enabled and self.source_face is not None and self.face_analyser is not None:
                    frame, face_count = detect_and_swap(self.source_face, frame, self.face_analyser)
                elif self.face_analyser is not None:
                    faces = self.face_analyser.get(frame)
                    face_count = len(faces)
                    for face in faces:
                        bbox = face.bbox.astype(int)
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                else:
                    face_count = 0
            except Exception as e:
                # Don't emit the raw frame: while swap is on that would show the unswapped face
                logger.error("[Face Swap Error] %s\n%s", e, traceback.format_exc())
                continue
            self.processing_ms = (time.perf_counter() - start) * 1000

            self._send_to_virtual_camera(frame)
            if self._display_ready.is_set():
                self._display_ready.clear()
                self.frame_ready.emit(frame)
            self.face_count_update.emit(face_count)
            fps = processed_counter.update()
            if fps > 0:
                self.fps_update.emit(fps)

    def _send_to_virtual_camera(self, frame):
        with self._virtual_cam_lock:
            camera = self._virtual_cam
            if camera is None:
                return
            try:
                camera.send(frame)
            except Exception as e:
                self._virtual_cam = None
                self.virtual_cam_error.emit(str(e))

    def stop(self):
        """Stop the video thread"""
        self.running = False
        with self._frame_cond:
            self._frame_cond.notify_all()
        self.wait()

    def cleanup(self):
        """Release resources"""
        if self.cap is not None:
            self.cap.release()


class FPSCounter:
    """Simple FPS counter"""

    def __init__(self, avg_frames=30):
        self.avg_frames = avg_frames
        self.frame_times = []
        self.last_time = cv2.getTickCount()

    def update(self):
        """Update and return current FPS"""
        current_time = cv2.getTickCount()
        time_diff = (current_time - self.last_time) / cv2.getTickFrequency()
        self.last_time = current_time

        self.frame_times.append(time_diff)
        if len(self.frame_times) > self.avg_frames:
            self.frame_times.pop(0)

        if len(self.frame_times) > 0:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_time if avg_time > 0 else 0
        return 0
