"""Soak test: run VideoThread with swap enabled on a synthetic 30 fps camera and track memory.

Usage: python -m bench.bench_memory --minutes 3 [--enhance]
"""

import argparse
import os
import sys
import time

import bench.common  # noqa: F401
from bench.common import save_results

import cv2
import numpy as np
import onnxruntime as ort
import psutil

ort.preload_dlls()

from PyQt6.QtCore import QCoreApplication, Qt  # noqa: E402

sys.path.insert(0, str(bench.common.ROOT / "tests"))
from test_video_thread import FakeCapture  # noqa: E402

from app import video_thread as vt  # noqa: E402
import core.config as config  # noqa: E402
from core.face_analyser import get_face_analyser  # noqa: E402


def gpu_mem_mb():
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return float(out.strip().splitlines()[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=3)
    parser.add_argument("--enhance", action="store_true")
    args = parser.parse_args()

    QCoreApplication.instance() or QCoreApplication([])
    config.ENHANCE_ENABLED = args.enhance
    vt.cv2.VideoCapture = FakeCapture
    source = get_face_analyser().get(cv2.imread(str(bench.common.ROOT / "assets" / "elon_musk.jpg")))[0]

    thread = vt.VideoThread(0)
    thread.set_source_face(source)
    thread.enable_swap(True)
    frames = [0]

    def on_frame(_):
        frames[0] += 1
        thread.frame_displayed()

    thread.frame_ready.connect(on_frame, Qt.ConnectionType.DirectConnection)
    proc = psutil.Process(os.getpid())
    thread.start()

    samples = []  # (seconds, rss MB, vram MB, frames)
    start = time.time()
    while time.time() - start < args.minutes * 60:
        time.sleep(10)
        samples.append((round(time.time() - start), proc.memory_info().rss / 2**20, gpu_mem_mb(), frames[0]))
        t, rss, vram, n = samples[-1]
        print(f"t={t:4d}s  rss={rss:7.1f} MB  vram={vram:6.0f} MB  frames={n}", flush=True)
    thread.stop()

    # Growth after warm-up (first minute excluded): slope in MB per minute
    steady = [s for s in samples if s[0] >= 60] or samples
    t = np.array([s[0] for s in steady]) / 60
    rss = np.array([s[1] for s in steady])
    vram = np.array([s[2] for s in steady])
    slope_rss = float(np.polyfit(t, rss, 1)[0]) if len(t) > 2 else None
    slope_vram = float(np.polyfit(t, vram, 1)[0]) if len(t) > 2 else None
    print(f"\nsteady-state growth: RSS {slope_rss:+.2f} MB/min, VRAM {slope_vram:+.2f} MB/min")
    save_results("memory", "enhance" if args.enhance else "swap",
                 {"samples": samples, "rss_mb_per_min": slope_rss, "vram_mb_per_min": slope_vram})


if __name__ == "__main__":
    main()
