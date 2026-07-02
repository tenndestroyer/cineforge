"""GUI host: stdlib http.server bound to 127.0.0.1, serving the single offline page
and the JSON API. No build step, no CDN, no telemetry."""

from __future__ import annotations

import http.server
import json
import urllib.parse
import webbrowser
from pathlib import Path

from ..config import Config
from ..logging_setup import get_logger
from .api import GuiApi

STATIC = Path(__file__).parent / "static"
_log = get_logger("cineforge.gui")


def serve(cfg: Config, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    api = GuiApi(cfg)
    index_html = (STATIC / "index.html").read_bytes()

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, status: int, ctype: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", index_html)
                return
            if parsed.path.startswith("/api/"):
                query = urllib.parse.parse_qs(parsed.query)
                status, ctype, payload = api.handle("GET", parsed.path, query, None)
                self._send(status, ctype, payload)
                return
            self._send(404, "text/plain; charset=utf-8", b"not found")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            try:
                length = int(self.headers.get("Content-Length", "0") or 0)
            except ValueError:
                length = 0
            if length < 0:
                self._send(400, "text/plain; charset=utf-8", b"bad content-length")
                return
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw.decode("utf-8")) if raw else {}
            except (UnicodeDecodeError, json.JSONDecodeError):
                body = {}
            status, ctype, payload = api.handle("POST", parsed.path, urllib.parse.parse_qs(parsed.query), body)
            self._send(status, ctype, payload)

        def log_message(self, *args) -> None:  # silence default stderr spam
            return

    httpd = http.server.ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    _log.info("Cineforge GUI at %s (Ctrl+C to stop)", url)
    print(f"Cineforge GUI: {url}")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - headless is fine
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
