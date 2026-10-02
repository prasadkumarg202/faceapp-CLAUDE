"""Shared helpers for the benchmark scripts: GPU/memory sampling and result output."""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "bench" / "results"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ResourceSampler:
    """Samples GPU utilisation / VRAM (nvidia-smi) and process RSS in the background."""

    def __init__(self, interval_ms=200):
        self.interval_ms = interval_ms
        self.gpu_util = []
        self.vram_mb = []
        self.rss_mb = []
        self._proc = None
        self._stop = threading.Event()
        self._threads = []

    def __enter__(self):
        try:
            self._proc = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                    f"-lms={self.interval_ms}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self._threads.append(threading.Thread(target=self._read_gpu, daemon=True))
        except FileNotFoundError:
            self._proc = None
        self._threads.append(threading.Thread(target=self._read_rss, daemon=True))
        for t in self._threads:
            t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
        for t in self._threads:
            t.join(timeout=2)

    def _read_gpu(self):
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            try:
                util, mem = (float(v) for v in line.strip().split(","))
            except ValueError:
                continue
            self.gpu_util.append(util)
            self.vram_mb.append(mem)

    def _read_rss(self):
        proc = psutil.Process(os.getpid())
        while not self._stop.is_set():
            self.rss_mb.append(proc.memory_info().rss / 2**20)
            time.sleep(self.interval_ms / 1000)

    def summary(self):
        def stats(values):
            if not values:
                return None
            return {"mean": round(float(np.mean(values)), 1), "max": round(float(np.max(values)), 1)}

        return {
            "gpu_util_pct": stats(self.gpu_util),
            "vram_mb": stats(self.vram_mb),
            "rss_mb": stats(self.rss_mb),
            "rss_growth_mb": round(self.rss_mb[-1] - self.rss_mb[0], 1) if len(self.rss_mb) > 1 else None,
        }


def latency_stats(samples_ms):
    if not samples_ms:
        return None
    arr = np.asarray(samples_ms)
    return {
        "n": len(arr),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "mean_ms": round(float(arr.mean()), 2),
    }


def save_results(name, label, data):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}_{label}.json"
    data = {"label": label, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **data}
    path.write_text(json.dumps(data, indent=2))
    print(f"\nSaved {path.relative_to(ROOT)}")
    return path


def print_table(rows):
    width = max(len(k) for k in rows)
    for key, value in rows.items():
        print(f"  {key:<{width}}  {value}")
