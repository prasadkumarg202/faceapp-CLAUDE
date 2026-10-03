# PyInstaller spec: portable folder build (dist/DeepFaceNet/DeepFaceNet.exe).
# Models are not baked in; tools/package_portable.py copies them next to the exe.
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules
from pathlib import Path
import site

sp = Path(next(p for p in site.getsitepackages() if p.endswith("site-packages")))
nvidia_bins = [(str(dll), f"nvidia/{dll.parent.parent.name}/bin") for dll in sp.glob("nvidia/*/bin/*.dll")]

a = Analysis(
    ["run.py"],
    pathex=["."],
    binaries=nvidia_bins + collect_dynamic_libs("onnxruntime") + collect_dynamic_libs("pyvirtualcam"),
    # insightface's get_object() looks in <_MEIPASS>/objects when frozen (meanshape_68.pkl etc.)
    datas=[("assets", "assets"), (str(sp / "insightface" / "data" / "objects"), "objects")]
    + collect_data_files("insightface") + collect_data_files("onnxruntime"),
    hiddenimports=collect_submodules("insightface") + collect_submodules("app") + collect_submodules("core")
    + ["onnxconverter_common", "pyvirtualcam", "download_models"],
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "torch", "tensorflow"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="DeepFaceNet",
    console=True,  # keeps a log window; closing it closes the app
    icon=None, upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="DeepFaceNet")
