"""Higgs Audio v2 — Apache-2.0 hero-character voice (~5.8B).

Zero-shot cloning at high quality; realistically needs 18-24 GB, so it's the 24 GB+
tier's license-safe cloning engine.
"""

from __future__ import annotations

from ...models.registry import register
from ..base import CAP_CLONING, ComfyBackend


@register("higgs2", subsystem="voice", license_id="higgs2")
class HiggsAudio2(ComfyBackend):
    subsystem = "voice"
    default_vram = 24.0
    required_nodes = ("ComfyUI-TTS-Audio-Suite",)
    required_weights = ("voice/higgs2/",)

    def capabilities(self) -> set[str]:
        return {CAP_CLONING}
