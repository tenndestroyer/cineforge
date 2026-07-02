"""Local LLM handle for the text agents, backed by Ollama (127.0.0.1:11434).

`OllamaLLM` is the helper the agents actually use (chat / json). `OllamaBackend`
is a thin registry entry so the model matrix + doctor can see 'ollama' as the llm
subsystem; its generate() is intentionally unused (agents call chat/json instead).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from ...errors import BackendError, NotInstalledError
from ...models.registry import register
from ..base import Backend


class OllamaLLM:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = "qwen2.5:7b",
                 timeout: float = 300.0, num_ctx: int = 8192) -> None:
        self.base = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.num_ctx = num_ctx  # explicit context window; Ollama's model default is often small

    def available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3.0):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def _chat(self, messages: list[dict], fmt: str | None = None, temperature: float = 0.7) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx},
        }
        if fmt:
            payload["format"] = fmt
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}/api/chat", data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError) as e:
            raise NotInstalledError(
                f"Ollama not reachable at {self.base} (model {self.model}). "
                f"Install Ollama and `ollama pull {self.model}`, or run `cineforge doctor`."
            ) from e
        return (body.get("message") or {}).get("content", "")

    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        return self._chat(messages, temperature=temperature)

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.7) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self._chat(msgs, temperature=temperature)

    def json(self, prompt: str, system: str | None = None, temperature: float = 0.3) -> dict:
        """Ask for strict JSON and parse it. Raises BackendError on unparseable output."""
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        raw = self._chat(msgs, fmt="json", temperature=temperature)
        return _parse_json(raw)


def _parse_json(raw: str) -> dict:
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # tolerate a fenced ```json block or trailing prose
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise BackendError(f"LLM did not return valid JSON: {raw[:200]!r}")


# --- registry entry (so matrix/doctor see the llm subsystem) ---
@register("ollama", subsystem="llm", license_id="ollama")
class OllamaBackend(Backend):
    subsystem = "llm"

    def estimate_vram(self, request: Any) -> float:
        return float(self.choice.min_vram_gb) if self.choice else 6.0

    def generate(self, request: Any):  # pragma: no cover - not the llm entry point
        raise BackendError("Use OllamaLLM.chat/json for the llm subsystem, not generate().")

    def make_llm(self) -> OllamaLLM:
        model = self.choice.variant if self.choice and self.choice.variant else "qwen2.5:7b"
        return OllamaLLM(self.cfg.ollama_url, model)
