"""Local VLM judge (Qwen2-VL / LLaVA via Ollama) for the Consistency + QA agents.

This is the local replacement for ViMax's cloud MLLM consistency check: it scores a
candidate frame against a character's canonical reference for identity similarity and
prompt adherence, and captions frames for QA — all offline.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ...errors import BackendError, NotInstalledError
from ...models.registry import register
from ..base import Backend


def _b64(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


class OllamaVLM:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = "qwen2.5vl:7b", timeout: float = 300.0) -> None:
        self.base = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _generate(self, prompt: str, images: list[str], fmt: str | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "images": [_b64(p) for p in images],
            "stream": False,
        }
        if fmt:
            payload["format"] = fmt
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}/api/generate", data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8")).get("response", "")
        except (urllib.error.URLError, OSError) as e:
            raise NotInstalledError(
                f"Ollama VLM not reachable at {self.base} (model {self.model}). "
                f"`ollama pull {self.model}` or run `cineforge doctor`."
            ) from e

    def caption(self, image: str | Path) -> str:
        return self._generate("Describe this image in one detailed sentence.", [str(image)])

    def score(self, image: str | Path, reference: str | Path, criteria: str = "same character identity") -> float:
        """Return an identity/adherence score in [0,1] comparing image to reference."""
        prompt = (
            f"Image 1 is a reference of a character. Image 2 is a new frame. On a scale of 0.0 to 1.0, "
            f"how well does image 2 match the reference for: {criteria}? Consider face, hair, outfit, "
            f'colors, and proportions. Respond as JSON: {{"score": <float 0-1>, "reason": "<short>"}}.'
        )
        raw = self._generate(prompt, [str(reference), str(image)], fmt="json")
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise BackendError(f"VLM returned non-object JSON: {raw[:160]!r}")
            return max(0.0, min(1.0, float(data.get("score", 0.0))))
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as e:
            raise BackendError(f"VLM returned an unscoreable response: {raw[:160]!r}") from e


@register("vlm", subsystem="llm", license_id="vlm")
class VlmBackend(Backend):
    subsystem = "llm"

    def estimate_vram(self, request: Any) -> float:
        return 8.0

    def generate(self, request: Any):  # pragma: no cover - not the vlm entry point
        raise BackendError("Use OllamaVLM.score/caption for the vlm judge, not generate().")

    def make_vlm(self) -> OllamaVLM:
        return OllamaVLM(self.cfg.ollama_url)
