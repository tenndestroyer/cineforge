"""Kokoro-82M — Apache-2.0 no-clone narration fallback (~2-3 GB, CPU-capable).

Safe floor when no voice cloning is needed (narrator lines, placeholder VO).
"""

from __future__ import annotations

from ...models.registry import register
from ..base import ComfyBackend


@register("kokoro", subsystem="voice", license_id="kokoro")
class Kokoro(ComfyBackend):
    subsystem = "voice"
    default_vram = 3.0
    required_nodes = ("ComfyUI-TTS-Audio-Suite",)
    required_weights = ("voice/kokoro/",)

    def capabilities(self) -> set[str]:
        return set()
