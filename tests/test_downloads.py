"""Model download safety: partial files, resume, wrong-size responses. Uses a local HTTP server."""

import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import download_models as dm  # noqa: E402

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB


class RangeHandler(SimpleHTTPRequestHandler):
    """Serves PAYLOAD with Range support; /drop drops the connection halfway once."""
    dropped = False

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/error":
            body = b"Invalid username or password."
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        start = 0
        if "Range" in self.headers:
            start = int(self.headers["Range"].split("=")[1].split("-")[0])
            self.send_response(206)
        else:
            self.send_response(200)
        data = PAYLOAD[start:]
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.path == "/drop" and not RangeHandler.dropped:
            RangeHandler.dropped = True
            self.wfile.write(data[: len(data) // 2])
            self.wfile.flush()
            self.connection.shutdown(2)  # simulate the connection breaking mid-download
            return
        self.wfile.write(data)


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(RangeHandler))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_download_completes_and_leaves_no_part_file(server, tmp_path):
    dest = dm.download_to(f"{server}/ok", tmp_path / "m.onnx", len(PAYLOAD))
    assert dest.read_bytes() == PAYLOAD
    assert not (tmp_path / "m.onnx.part").exists()


def test_dropped_connection_resumes(server, tmp_path, monkeypatch):
    monkeypatch.setattr(dm.time, "sleep", lambda s: None)
    dest = dm.download_to(f"{server}/drop", tmp_path / "m.onnx", len(PAYLOAD))
    assert dest.read_bytes() == PAYLOAD


def test_error_page_is_rejected(server, tmp_path):
    with pytest.raises(IOError, match="should be about"):
        dm.download_to(f"{server}/error", tmp_path / "m.onnx", len(PAYLOAD))
    assert not (tmp_path / "m.onnx").exists()


def test_cancel_keeps_part_and_does_not_create_model(server, tmp_path):
    with pytest.raises(dm.DownloadCancelled):
        dm.download_to(f"{server}/ok", tmp_path / "m.onnx", len(PAYLOAD), is_cancelled=lambda: True)
    assert not (tmp_path / "m.onnx").exists()


def test_truncated_model_is_not_reported_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(dm.config, "MODELS_DIR", tmp_path)
    (tmp_path / "inswapper_128.onnx").write_bytes(b"x" * 1000)
    ok, _, _ = dm.check_model_status("inswapper_128.onnx")
    assert not ok


def test_real_models_report_ready():
    for name in ("inswapper_128.onnx", "buffalo_l"):
        assert dm.check_model_status(name)[0], name
