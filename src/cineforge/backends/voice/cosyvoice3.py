"""CosyVoice3 — Apache-2.0 dialect / cross-lingual voices.

Runs concurrently with Chatterbox for multi-character casts (distinct engines reduce
same-voice bleed across characters).
"""

from __future__ import annotations

from ...models.registry import register
from ..base import CAP_CLONING, ComfyBackend


@register("cosyvoice3", subsystem="voice", license_id="cosyvoice3")
class CosyVoice3(ComfyBackend):
    subsystem = "voice"
    default_vram = 8.0
    required_nodes = ("ComfyUI-TTS-Audio-Suite",)
    required_weights = ("voice/cosyvoice3/",)

    def capabilities(self) -> set[str]:
        return {CAP_CLONING}
