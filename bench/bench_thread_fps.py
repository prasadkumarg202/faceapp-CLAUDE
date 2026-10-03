"""Throughput of the real VideoThread on a synthetic 30 fps camera (swap on, enhancer on/off).

Usage: python -m bench.bench_thread_fps --seconds 30 [--enhance]
"""

import argparse
import sys
import time

import bench.common  # noqa: F401
from bench.common import ROOT, latency_stats

import cv2
import onnxruntime as ort

ort.preload_dlls()

from PyQt6.QtCore import QCoreApplication, Qt  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
from test_video_thread import FakeCapture  # noqa: E402

from app import video_thread as vt  # noqa: E402
import core.config as config  # noqa: E402
from core.face_analyser import get_face_analyser  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--enhance", action="store_true")
    args = parser.parse_args()

    QCoreApplication.instance() or QCoreApplication([])
    vt.cv2.VideoCapture = FakeCapture
    config.ENHANCE_ENABLED = args.enhance
    if args.enhance:
        from core.engine.face_enhancer import get_face_enhancer
        get_face_enhancer()
    source = get_face_analyser().get(cv2.imread(str(ROOT / "assets" / "elon_musk.jpg")))[0]

    thread = vt.VideoThread(0)
    thread.set_source_face(source)
    thread.enable_swap(True)
    times, latency = [], []

    def on_frame(_):
        times.append(time.perf_counter())
        latency.append(thread.processing_ms)
        if hasattr(thread, "frame_displayed"):
            thread.frame_displayed()

    thread.frame_ready.connect(on_frame, Qt.ConnectionType.DirectConnection)
    thread.start()
    time.sleep(5)  # warm-up
    times.clear()
    latency.clear()
    time.sleep(args.seconds)
    thread.stop()
    fps = (len(times) - 1) / (times[-1] - times[0])
    lat = latency_stats(latency)
    print(f"> enhance={args.enhance}  {fps:.1f} FPS  latency p50 {lat['p50_ms']} ms  p95 {lat['p95_ms']} ms  frames {len(times)}")


if __name__ == "__main__":
    main()
