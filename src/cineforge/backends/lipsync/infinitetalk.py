"""InfiniteTalk — OPT-IN stylized lip-sync (Apache-2.0, 14B).

Stronger explicit anime/stylized-domain claim than EchoMimic; 16-24 GB (GGUF Q4 pushes
to 8-12 GB with tradeoffs). Use for stylized hero shots.
"""

from __future__ import annotations

from ...models.registry import register
from ..base import CAP_STYLIZED_FACE, ComfyBackend


@register("infinitetalk", subsystem="lipsync", license_id="infinitetalk")
class InfiniteTalk(ComfyBackend):
    subsystem = "lipsync"
    default_vram = 16.0
    required_nodes = ("ComfyUI-WanVideoWrapper", "ComfyUI-GGUF")
    required_weights = ("lipsync/infinitetalk/",)

    def capabilities(self) -> set[str]:
        return {CAP_STYLIZED_FACE}
