"""Pipeline benchmark: per-stage latency on a static image, or end-to-end FPS on a live camera.

Usage:
    python bench/bench_pipeline.py --label baseline                 # static image
    python bench/bench_pipeline.py --label baseline --camera 0      # live camera
"""

import argparse
import time

import bench.common  # noqa: F401  (adds repo root to sys.path)
from bench.common import ResourceSampler, latency_stats, print_table, save_results

import cv2
import numpy as np
import onnxruntime as ort

ort.preload_dlls()

from core.face_analyser import get_face_analyser  # noqa: E402
from core.engine.face_swapper import swap_face, detect_and_swap  # noqa: E402
import core.config as config  # noqa: E402


def get_live_analyser():
    """The analyser used for target frames (P0-3 may introduce a lighter one)."""
    try:
        from core.face_analyser import get_live_face_analyser
        return get_live_face_analyser()
    except ImportError:
        return get_face_analyser()


def timed(fn, n, warmup=5):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return latency_stats(samples)


def bench_static(args, source_face):
    target = cv2.resize(cv2.imread(args.target), (640, 480))
    analyser = get_live_analyser()
    faces = analyser.get(target)
    if not faces:
        raise SystemExit("No face in target image")

    results = {}
    with ResourceSampler() as sampler:
        results["analysis"] = timed(lambda: analyser.get(target), args.frames)
        results["swap_face"] = timed(lambda: swap_face(source_face, faces[0], target.copy()), args.frames)
        config.MOUTH_MASK_ENABLED = True
        results["swap_face+mouth_mask"] = timed(
            lambda: swap_face(source_face, faces[0], target.copy()), args.frames
        )
        config.MOUTH_MASK_ENABLED = False
        results["detect_and_swap"] = timed(
            lambda: detect_and_swap(source_face, target.copy(), analyser), args.frames
        )
        if config.ENHANCER_MODEL.exists():
            from core.engine.face_enhancer import get_face_enhancer
            get_face_enhancer()
            config.ENHANCE_ENABLED = True
            results["detect_and_swap+enhance"] = timed(
                lambda: detect_and_swap(source_face, target.copy(), analyser), args.frames
            )
            config.ENHANCE_ENABLED = False
    results["resources"] = sampler.summary()
    results["max_fps_from_p50"] = round(1000 / results["detect_and_swap"]["p50_ms"], 1)
    return results


def bench_camera(args, source_face):
    analyser = get_live_analyser()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

    open_info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "reported_fps": cap.get(cv2.CAP_PROP_FPS),
        "backend": cap.getBackendName(),
    }

    # Capture-only: what the camera delivers with no processing.
    for _ in range(10):
        cap.read()
    read_ms = []
    start = time.perf_counter()
    for _ in range(args.frames):
        t = time.perf_counter()
        ok, _ = cap.read()
        read_ms.append((time.perf_counter() - t) * 1000)
        if not ok:
            raise SystemExit("Camera read failed")
    capture_fps = args.frames / (time.perf_counter() - start)

    # Sequential capture + detect + swap: the processing ceiling.
    stage = {"read": [], "detect_and_swap": [], "with_face": []}
    faces_seen = 0
    with ResourceSampler() as sampler:
        start = time.perf_counter()
        for _ in range(args.frames):
            t0 = time.perf_counter()
            ok, frame = cap.read()
            t1 = time.perf_counter()
            _, count = detect_and_swap(source_face, frame, analyser)
            t2 = time.perf_counter()
            stage["read"].append((t1 - t0) * 1000)
            stage["detect_and_swap"].append((t2 - t1) * 1000)
            faces_seen += count > 0
            if count > 0:
                stage["with_face"].append((t2 - t1) * 1000)
        processed_fps = args.frames / (time.perf_counter() - start)
    cap.release()

    return {
        "camera": open_info,
        "capture_only_fps": round(capture_fps, 1),
        "capture_read": latency_stats(read_ms),
        "processed_fps": round(processed_fps, 1),
        "stage_read": latency_stats(stage["read"]),
        "stage_detect_and_swap": latency_stats(stage["detect_and_swap"]),
        "frames_with_face_pct": round(100 * faces_seen / args.frames, 1),
        # Latency on frames that actually contained a face (the swap path); comparable across
        # runs even when the subject moves out of view.
        "stage_detect_and_swap_face_frames": latency_stats(stage["with_face"]),
        "max_fps_face_frames": round(1000 / np.percentile(stage["with_face"], 50), 1) if stage["with_face"] else None,
        "resources": sampler.summary(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="name for the results file, e.g. baseline")
    parser.add_argument("--source", default="assets/elon_musk.jpg")
    parser.add_argument("--target", default="assets/vijay.jpg")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()

    source_face = get_face_analyser().get(cv2.imread(args.source))[0]
    if args.camera is None:
        name, results = "static", bench_static(args, source_face)
    else:
        name, results = "camera", bench_camera(args, source_face)

    flat = {}
    for key, value in results.items():
        if isinstance(value, dict) and "p50_ms" in value:
            flat[key] = f"p50 {value['p50_ms']} ms   p95 {value['p95_ms']} ms"
        else:
            flat[key] = value
    print(f"\n[{name} / {args.label}]")
    print_table(flat)
    save_results(name, args.label, results)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()
