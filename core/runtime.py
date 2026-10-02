"""ONNX Runtime setup: provider selection and a record of which provider each model really uses.

ONNX Runtime silently falls back to CPU when CUDA cannot be loaded (for example when the
CPU-only `onnxruntime` package shadows `onnxruntime-gpu`). Every session created by the app
is registered here so the fallback is logged and visible in the GUI.
"""

import logging
import threading

import onnxruntime

logger = logging.getLogger(__name__)

_PRELOADED = False
_LOCK = threading.Lock()
ACTIVE_PROVIDERS: dict[str, str] = {}  # model name -> first provider of its session


def preload_cuda_dlls():
    """Load the CUDA/cuDNN DLLs shipped with onnxruntime-gpu[cuda,cudnn] before any session is created."""
    global _PRELOADED
    with _LOCK:
        if _PRELOADED:
            return
        _PRELOADED = True
        if hasattr(onnxruntime, "preload_dlls"):
            try:
                onnxruntime.preload_dlls()
            except Exception as e:
                logger.warning("onnxruntime.preload_dlls() failed: %s", e)


def get_providers():
    """Preferred execution providers, best first, limited to what this onnxruntime build has."""
    preload_cuda_dlls()
    available = onnxruntime.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    if "CUDAExecutionProvider" not in providers:
        logger.warning(
            "CUDAExecutionProvider is not available (available: %s) - inference will run on CPU. "
            "If onnxruntime-gpu is installed, uninstall the CPU-only 'onnxruntime' package.",
            available,
        )
    return providers


def register_session(name, session):
    """Record and log the provider a session actually uses; warn if it fell back to CPU."""
    active = session.get_providers()[0] if session.get_providers() else "unknown"
    ACTIVE_PROVIDERS[name] = active
    if active == "CPUExecutionProvider" and "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        logger.warning("%s fell back to CPU even though CUDA is available", name)
    else:
        logger.info("%s running on %s", name, active)


_last_logged: dict[str, float] = {}


def log_throttled(log, key, message, *args, interval=5.0):
    """Log at most once per `interval` seconds per key (for errors that can repeat every frame)."""
    import time

    now = time.monotonic()
    if now - _last_logged.get(key, 0.0) >= interval:
        _last_logged[key] = now
        log.warning(message, *args)


def gpu_status():
    """Short summary for the GUI: ('GPU' | 'CPU' | 'Mixed' | None, details)."""
    if not ACTIVE_PROVIDERS:
        return None, "no models loaded"
    on_gpu = [n for n, p in ACTIVE_PROVIDERS.items() if p != "CPUExecutionProvider"]
    details = ", ".join(f"{n}: {p.replace('ExecutionProvider', '')}" for n, p in ACTIVE_PROVIDERS.items())
    if len(on_gpu) == len(ACTIVE_PROVIDERS):
        return "GPU", details
    if not on_gpu:
        return "CPU", details
    return "Mixed", details
