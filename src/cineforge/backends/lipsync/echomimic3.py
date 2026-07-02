"""EchoMimicV3 — the DEFAULT lip-sync backend (Apache-2.0, 1.3B).

Flash variant runs 6.5-12 GB; stylized-face capable; audio_guidance_scale 1.8-2.0; a
no-mask manual-region mode handles non-human faces.
"""

from __future__ import annotations

from ...models.registry import register
from ..base import CAP_STYLIZED_FACE, ComfyBackend


@register("echomimic3", subsystem="lipsync", license_id="echomimic3")
class EchoMimicV3(ComfyBackend):
    subsystem = "lipsync"
    default_vram = 7.0
    required_nodes = ("ComfyUI-EchoMimic",)
    required_weights = ("lipsync/echomimic3/",)

    def capabilities(self) -> set[str]:
        return {CAP_STYLIZED_FACE}
