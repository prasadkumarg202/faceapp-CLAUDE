import argparse
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

try:
    from core import config

    CONFIG_LOADED = True
except ImportError:
    CONFIG_LOADED = False

MODELS = {
    "inswapper_128.onnx": {
        "url": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        "size": 554253681,
        "type": "swapper",
        "required": True,
        "description": "Face swap model (FP32; a faster mixed-precision copy is built locally)",
        "location": "models",
    },
    "GFPGANv1.4.onnx": {
        "url": "https://huggingface.co/neurobytemind/GFPGANv1.4.onnx/resolve/main/GFPGANv1.4.onnx",
        "size": 340256686,
        "type": "enhancer",
        "required": False,
        "description": "Face enhancement model (ONNX)",
        "location": "models",
    },
    "buffalo_l": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "size": 288621354,
        "type": "analyser",
        "required": True,
        "description": "Face detection & analysis pack",
        "location": "insightface",
        "expected_files": [
            "1k3d68.onnx",
            "2d106det.onnx",
            "det_10g.onnx",
            "genderage.onnx",
            "w600k_r50.onnx",
        ],
    },
}

# A file this far from the expected size is treated as truncated or wrong (e.g. an error page)
SIZE_TOLERANCE = 0.01
RETRIES = 8


class DownloadCancelled(Exception):
    pass


def get_model_path(model_name):
    """Get the full path where a model should be located."""
    info = MODELS.get(model_name)
    if not info:
        return None

    if info["location"] == "insightface":
        return Path.home() / ".insightface" / "models" / model_name
    else:
        base = config.MODELS_DIR if CONFIG_LOADED else Path("models")
        return base / model_name


def _size_ok(actual, expected):
    return not expected or abs(actual - expected) / expected <= SIZE_TOLERANCE


def check_model_status(model_name):
    """Check if a model is downloaded and complete. Returns (is_downloaded, path, current_size)."""
    info = MODELS.get(model_name)
    if not info:
        return False, None, 0

    path = get_model_path(model_name)

    if info["location"] == "insightface":
        # Check if directory exists with expected files
        expected = info.get("expected_files", [])
        if path.is_dir() and all((path / f).exists() for f in expected):
            total_size = sum((path / f).stat().st_size for f in expected)
            return True, path, total_size
        return False, path, 0
    else:
        if path.exists():
            size = path.stat().st_size
            # A truncated file (cancelled download) must not report as ready
            return _size_ok(size, info.get("size")), path, size
        return False, path, 0


def format_size(size_bytes):
    """Format bytes to human readable string."""
    if size_bytes == 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def download_to(url, dest, expected_size=None, progress=None, is_cancelled=None):
    """Download `url` to `dest` safely.

    Writes to `<dest>.part`, resumes it after dropped connections (HTTP Range), and only renames
    it to `dest` once the size matches. `progress(done, total)` is called as data arrives;
    `is_cancelled()` aborts the download (the .part file is kept so it can resume later).
    """
    dest = Path(dest)
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    total = expected_size or 0
    for attempt in range(1, RETRIES + 1):
        done = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={done}-"} if done else {}
        try:
            with requests.get(url, stream=True, headers=headers, timeout=30) as response:
                if response.status_code == 416:  # already complete
                    break
                response.raise_for_status()
                if done and response.status_code != 206:  # server ignored Range: start over
                    done = 0
                length = int(response.headers.get("content-length", 0))
                total = done + length if length else total
                with open(part, "ab" if done else "wb") as f:
                    for chunk in response.iter_content(1 << 20):
                        if is_cancelled and is_cancelled():
                            raise DownloadCancelled()
                        f.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, total)
            break
        except DownloadCancelled:
            raise
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as e:
            if attempt == RETRIES:
                raise
            print(f"Connection problem ({e.__class__.__name__}), resuming in {2 * attempt}s...")
            time.sleep(2 * attempt)

    size = part.stat().st_size
    if total and size != total:
        raise IOError(f"Incomplete download of {dest.name}: {size} of {total} bytes")
    if not _size_ok(size, expected_size):
        part.unlink()
        raise IOError(
            f"{dest.name} is {format_size(size)} but should be about {format_size(expected_size)} "
            "- the download link may require a login or have moved"
        )
    part.replace(dest)
    return dest


def fetch_model(model_name, progress=None, is_cancelled=None, status=None):
    """Download one model from MODELS into place (zip packs are extracted). Returns its path."""
    info = MODELS[model_name]
    dest = get_model_path(model_name)

    if info["location"] == "insightface":
        zip_path = dest.parent / f"{model_name}.zip"
        download_to(info["url"], zip_path, info.get("size"), progress, is_cancelled)
        if status:
            status("Extracting...")
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            # Flatten: the pack may or may not contain a top-level folder
            for member in z.namelist():
                if member.endswith(".onnx"):
                    (dest / Path(member).name).write_bytes(z.read(member))
        zip_path.unlink(missing_ok=True)
        ok, _, _ = check_model_status(model_name)
        if not ok:
            raise IOError(f"{model_name} pack is missing expected files after extraction")
    else:
        download_to(info["url"], dest, info.get("size"), progress, is_cancelled)
    return dest


def get_required_models(args):
    """Determine which models to download based on args."""
    if args.model:
        return [m for m in MODELS if m in args.model]
    if args.all:
        return list(MODELS.keys())
    return [m for m, info in MODELS.items() if info["required"]]


def main():
    parser = argparse.ArgumentParser(
        description="Download required models for Deep-Face-Net"
    )
    parser.add_argument(
        "--all", action="store_true", help="Download all available models"
    )
    parser.add_argument("--model", nargs="+", help="Download specific model(s) by name")
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for m, info in MODELS.items():
            ok, _, _ = check_model_status(m)
            print(f" - {m:22s} {'[ready]' if ok else '[missing]':10s} {info['description']}")
        return

    models_to_download = get_required_models(args)
    print(f"Targets: {', '.join(models_to_download)}")

    failed = []
    for name in models_to_download:
        ok, path, _ = check_model_status(name)
        if ok:
            print(f"{name} already downloaded. Skipping.")
            continue
        print(f"Downloading {name}...")
        bar = tqdm(desc=name, unit="iB", unit_scale=True, unit_divisor=1024)

        def progress(done, total, bar=bar):
            bar.total = total or None
            bar.n = done
            bar.refresh()

        try:
            fetch_model(name, progress=progress)
        except Exception as e:
            failed.append(name)
            print(f"\nFailed to download {name}: {e}")
        finally:
            bar.close()

    print("\nDownload process completed." if not failed else f"\nFailed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
