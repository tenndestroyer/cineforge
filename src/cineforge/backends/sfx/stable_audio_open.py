"""Stable Audio Open Small — text-to-ambience beds (Stability Community license).

~11s 44.1 kHz loops / room-tone. Tiny footprint. Free for individuals and businesses
under $1M revenue. Layered under the foley then mixed. This is the Safe-mode SFX path
(the video-conditioned foley model is gated).
"""

from __future__ import annotations

from ...models.registry import register
from ..base import ComfyBackend


@register("stable_audio_open", subsystem="sfx", license_id="stable_audio_open")
class StableAudioOpen(ComfyBackend):
    subsystem = "sfx"
    default_vram = 4.0
    required_nodes = ("ComfyUI-StableAudioOpen",)
    required_weights = ("sfx/stable_audio_open/",)

    def capabilities(self) -> set[str]:
        return set()
