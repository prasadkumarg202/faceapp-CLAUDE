"""Experiment: inswapper inference latency under different ONNX Runtime provider configurations."""

import sys
import time

import bench.common  # noqa: F401
from bench.common import latency_stats

import numpy as np
import onnxruntime as ort

ort.preload_dlls()
MODEL = "models/inswapper_128.onnx"

CONFIGS = {
    "cuda_default": [("CUDAExecutionProvider", {})],
    "cuda_heuristic": [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"})],
    "cuda_exhaustive": [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "EXHAUSTIVE"})],
    "cuda_tf32off_default_stream": [("CUDAExecutionProvider", {"do_copy_in_default_stream": True, "use_tf32": False})],
    "tensorrt_fp16": [
        ("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                                       "trt_engine_cache_path": "models/trt_cache"}),
        ("CUDAExecutionProvider", {}),
    ],
}


def run(name, providers, n=60):
    try:
        sess = ort.InferenceSession(MODEL, providers=providers + [("CPUExecutionProvider", {})])
    except Exception as e:
        print(f"{name:30s} FAILED to create: {str(e).splitlines()[0][:120]}")
        return
    feeds = {"target": np.random.rand(1, 3, 128, 128).astype(np.float32),
             "source": np.random.rand(1, 512).astype(np.float32)}
    t0 = time.perf_counter()
    sess.run(None, feeds)
    first = (time.perf_counter() - t0) * 1000
    for _ in range(10):
        sess.run(None, feeds)
    samples = []
    for _ in range(n):
        t = time.perf_counter()
        sess.run(None, feeds)
        samples.append((time.perf_counter() - t) * 1000)
    s = latency_stats(samples)
    print(f"{name:30s} active={sess.get_providers()[0]:28s} first={first:8.0f}ms  p50={s['p50_ms']:6.2f}ms  p95={s['p95_ms']:6.2f}ms")


for name in (sys.argv[1:] or CONFIGS):
    run(name, CONFIGS[name])
