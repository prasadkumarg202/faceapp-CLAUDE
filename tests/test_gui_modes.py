"""Fast / Quality / Custom mode selector."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import onnxruntime as ort  # noqa: E402

ort.preload_dlls()

from PyQt6.QtWidgets import QApplication  # noqa: E402

from core import config  # noqa: E402


@pytest.fixture(scope="module")
def window():
    from app.deepfake_app import DeepfakeApp

    app = QApplication.instance() or QApplication([])
    w = DeepfakeApp()
    w.enhance_checkbox.setEnabled(True)
    yield w
    w.close()
    app.processEvents()


def test_presets_set_options(window, monkeypatch):
    monkeypatch.setattr(config, "ENHANCE_ENABLED", False)
    monkeypatch.setattr(config, "OCCLUSION_MASK_ENABLED", False)
    window.mode_combo.setCurrentText("Quality")
    assert config.ENHANCE_ENABLED and config.OCCLUSION_MASK_ENABLED
    window.mode_combo.setCurrentText("Fast")
    assert not config.ENHANCE_ENABLED and not config.OCCLUSION_MASK_ENABLED


def test_manual_change_shows_custom_or_matching_preset(window):
    window.mode_combo.setCurrentText("Fast")
    window.occlusion_checkbox.setChecked(True)
    assert window.mode_combo.currentText() == "Custom"
    window.enhance_checkbox.setChecked(True)
    assert window.mode_combo.currentText() == "Quality"
    window.mode_combo.setCurrentText("Fast")
