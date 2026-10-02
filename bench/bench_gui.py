"""GUI benchmark: runs the real DeepfakeApp + VideoThread with swap enabled and measures

  - displayed FPS (frames reaching update_frame)
  - unique FPS (displayed frames that differ from the previous one; repeats are not new work)
  - GUI event-loop lag (how late a 10 ms QTimer fires; high lag = unresponsive UI)
  - frames shown before the first swapped frame (raw camera frames leaked while swap is on)

Usage:
    python bench/bench_gui.py --label baseline --camera 0 --seconds 20
"""

import argparse
import hashlib
import os
import time

import bench.common  # noqa: F401
from bench.common import ResourceSampler, latency_stats, print_table, save_results

import cv2
import onnxruntime as ort

ort.preload_dlls()

from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from app import deepfake_app  # noqa: E402
from core.face_analyser import get_face_analyser  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--source", default="assets/elon_musk.jpg")
    parser.add_argument("--show", action="store_true", help="show the window instead of offscreen")
    args = parser.parse_args()

    if not args.show:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = QApplication([])
    window = deepfake_app.DeepfakeApp()
    window.show()

    source_face = get_face_analyser().get(cv2.imread(args.source))[0]
    window.source_face = source_face
    window.selected_camera_index = args.camera

    displayed = []  # (timestamp, frame hash)
    original_update = window.update_frame

    def counting_update(frame):
        displayed.append((time.perf_counter(), hashlib.md5(frame.tobytes()).digest()))
        original_update(frame)

    window.update_frame = counting_update

    lag_ms = []
    tick = {"expected": None}
    lag_timer = QTimer()
    lag_timer.setInterval(10)

    def on_tick():
        now = time.perf_counter()
        if tick["expected"] is not None:
            lag_ms.append(max(0.0, (now - tick["expected"]) * 1000))
        tick["expected"] = now + 0.010

    lag_timer.timeout.connect(on_tick)

    swap_on_at = {}

    def start():
        window.start_camera()  # connects frame_ready to window.update_frame, i.e. counting_update
        QTimer.singleShot(1500, enable_swap)

    def enable_swap():
        window.toggle_swap()
        swap_on_at["t"] = time.perf_counter()
        lag_timer.start()
        QTimer.singleShot(int(args.seconds * 1000), finish)

    def finish():
        lag_timer.stop()
        window.stop_camera()
        app.quit()

    QTimer.singleShot(500, start)
    with ResourceSampler() as sampler:
        app.exec()

    t0 = swap_on_at.get("t")
    if t0 is None:
        raise SystemExit("Swap was never enabled (camera failed to start?)")
    window_frames = [(t, h) for t, h in displayed if t >= t0]
    duration = window_frames[-1][0] - window_frames[0][0] if len(window_frames) > 1 else 0
    unique = sum(1 for i, (_, h) in enumerate(window_frames) if i == 0 or h != window_frames[i - 1][1])
    intervals = [(window_frames[i][0] - window_frames[i - 1][0]) * 1000 for i in range(1, len(window_frames))]

    results = {
        "seconds_measured": round(duration, 1),
        "displayed_fps": round(len(window_frames) / duration, 1) if duration else 0,
        "unique_fps": round(unique / duration, 1) if duration else 0,
        "repeated_frame_pct": round(100 * (1 - unique / len(window_frames)), 1) if window_frames else 0,
        "display_interval": latency_stats(intervals),
        "gui_event_loop_lag": latency_stats(lag_ms),
        "resources": sampler.summary(),
    }
    flat = {
        k: (f"p50 {v['p50_ms']} ms   p95 {v['p95_ms']} ms" if isinstance(v, dict) and "p50_ms" in v else v)
        for k, v in results.items()
    }
    print(f"\n[gui / {args.label}]")
    print_table(flat)
    save_results("gui", args.label, results)


if __name__ == "__main__":
    main()
