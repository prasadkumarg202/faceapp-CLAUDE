"""Finish the portable build: copy models next to dist/DeepFaceNet/DeepFaceNet.exe.

Usage (after `pyinstaller DeepFaceNet.spec`):
    python tools/package_portable.py

Result: dist/DeepFaceNet/ can be copied to another Windows PC as-is.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DIST = ROOT / "dist" / "DeepFaceNet"

README = """Deep Face Net - portable build
==============================

Start: double-click DeepFaceNet.exe (a log window opens with it; closing it closes the app).

Needs on the other PC
---------------------
- Windows 10/11 64-bit
- NVIDIA GPU with a recent driver (R580 or newer; CUDA 13). Without an NVIDIA GPU the app
  still runs, but on the CPU and very slowly. The status bar shows GPU or CPU ONLY.
- For Meet/Zoom output: OBS Studio installed (it provides "OBS Virtual Camera").
  OBS itself does not need to be open.

Source photos
-------------
Put photos of the face you want into the source_faces folder, then use Select Source Image
(it opens that folder) and select them all (Ctrl+A). Several photos of one person work best.
Only use photos of someone who has agreed to it.

Folders
-------
models/        face swap, enhancer and occlusion models (fast versions pre-built)
insightface/   face detection pack (buffalo_l)
source_faces/  your source photos
_internal/     program files - do not change

Your settings (last photos, mode, virtual camera) are saved per Windows user in
%USERPROFILE%\\.deepfacenet_settings.json and logs in %USERPROFILE%\\.deepfacenet\\.
"""


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    print(f"  {dst.relative_to(DIST)}")


def main():
    from core import config

    if not (DIST / "DeepFaceNet.exe").exists():
        raise SystemExit("dist/DeepFaceNet/DeepFaceNet.exe not found - run: pyinstaller DeepFaceNet.spec")

    print("Copying models:")
    for name in ("inswapper_128.onnx", "GFPGANv1.4.onnx", "xseg_2.onnx"):
        src = config.MODELS_DIR / name
        if src.exists():
            copy(src, DIST / "models" / name)
    # Pre-converted fast (mixed-precision) models: no first-run conversion on the other PC.
    # Copied after the originals so their timestamps stay newer (the app rebuilds stale ones).
    for mixed in sorted(config.MODELS_DIR.glob("*.mixed-v*.onnx")):
        copy(mixed, DIST / "models" / mixed.name)

    print("Copying face detection pack:")
    copy(config.INSIGHTFACE_DIR / "buffalo_l", DIST / "insightface" / "models" / "buffalo_l")

    (DIST / "source_faces").mkdir(exist_ok=True)
    (DIST / "source_faces" / "PUT_PHOTOS_HERE.txt").write_text(
        "Put photos of the source face here (several photos of one person work best).\n", encoding="utf-8"
    )
    (DIST / "README.txt").write_text(README, encoding="utf-8")
    print("  source_faces/, README.txt")

    size = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file())
    print(f"\nDone: {DIST}  ({size / 2**30:.1f} GB) - copy this whole folder to the other PC.")


if __name__ == "__main__":
    main()
