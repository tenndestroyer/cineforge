"""IndexTTS-2 — OPT-IN max-emotion TTS (bilibili Index license, non-commercial).

Timbre/emotion disentanglement + duration control — ideal for dub-to-lipsync timing.
Non-commercial without written authorization (100M MAU / RMB1B trigger) -> gated.
"""

from __future__ import annotations

from ...models.registry import register
from ..base import CAP_CLONING, CAP_DURATION_CONTROL, ComfyBackend


@register("indextts2", subsystem="voice", license_id="indextts2")
class IndexTTS2(ComfyBackend):
    subsystem = "voice"
    default_vram = 8.0
    required_nodes = ("ComfyUI-TTS-Audio-Suite",)
    required_weights = ("voice/indextts2/",)

    def capabilities(self) -> set[str]:
        return {CAP_CLONING, CAP_DURATION_CONTROL}
