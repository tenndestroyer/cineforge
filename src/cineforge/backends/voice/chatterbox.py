"""Chatterbox Multilingual v3 — the DEFAULT TTS (MIT, 25 languages).

exaggeration / cfg_weight emotion sliders; a Turbo sibling for speed. Voice cloning
from a reference clip. Emits an inaudible PerTh watermark (disclosed). Runs via the
ComfyUI TTS-Audio-Suite nodes.
"""

from __future__ import annotations

from ...models.registry import register
from ..base import CAP_CLONING, ComfyBackend


@register("chatterbox", subsystem="voice", license_id="chatterbox")
class Chatterbox(ComfyBackend):
    subsystem = "voice"
    default_vram = 6.0
    required_nodes = ("ComfyUI-TTS-Audio-Suite",)
    required_weights = ("voice/chatterbox/",)

    def capabilities(self) -> set[str]:
        return {CAP_CLONING}
