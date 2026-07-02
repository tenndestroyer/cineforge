"""A thin, dependency-free client for the local ComfyUI instance (127.0.0.1).

Uses only urllib so it works before `requests` is guaranteed present. All heavy
model inference is executed by submitting API-format workflow graphs to ComfyUI;
adapters build the graph, this client submits and waits.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from ..errors import BackendError


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188", timeout: float = 10.0) -> None:
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    # ---- low-level ----
    def _get(self, path: str, timeout: float | None = None) -> Any:
        req = urllib.request.Request(f"{self.base}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post(self, path: str, payload: dict) -> Any:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}{path}", data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ---- public ----
    def is_reachable(self) -> bool:
        try:
            self._get("/system_stats", timeout=3.0)
            return True
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def submit(self, workflow: dict) -> str:
        """Queue an API-format workflow; return its prompt id."""
        try:
            resp = self._post("/prompt", {"prompt": workflow})
        except (urllib.error.URLError, OSError) as e:
            raise BackendError(f"ComfyUI submit failed: {e}") from e
        pid = resp.get("prompt_id")
        if not pid:
            raise BackendError(f"ComfyUI did not return a prompt_id: {resp}")
        return pid

    def wait(self, prompt_id: str, poll: float = 1.0, max_wait: float = 3600.0) -> dict:
        """Poll /history until the prompt completes; return its outputs dict."""
        waited = 0.0
        while waited < max_wait:
            try:
                hist = self._get(f"/history/{prompt_id}")
            except (urllib.error.URLError, OSError) as e:
                raise BackendError(f"ComfyUI history poll failed: {e}") from e
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status", {})
                if status.get("completed") or entry.get("outputs"):
                    return entry.get("outputs", {})
                if status.get("status_str") == "error":
                    raise BackendError(f"ComfyUI run errored: {status}")
            time.sleep(poll)
            waited += poll
        raise BackendError(f"ComfyUI run {prompt_id} timed out after {max_wait}s")

    def interrupt(self) -> None:
        try:
            self._post("/interrupt", {})
        except (urllib.error.URLError, OSError):
            pass
