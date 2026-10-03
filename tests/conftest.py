"""Test-wide setup: never read or write the user's real app settings."""

import os
import tempfile
from pathlib import Path

_settings = Path(tempfile.mkdtemp(prefix="deepfacenet-test-")) / "settings.json"
os.environ["DEEPFACENET_SETTINGS"] = str(_settings)


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_config():
    """GUI tests flip settings in core.config (e.g. Mode: Quality turns the enhancer on); undo after each test."""
    import core.config as config

    saved = {k: getattr(config, k) for k in dir(config) if k.isupper()}
    yield
    for k, v in saved.items():
        setattr(config, k, v)
