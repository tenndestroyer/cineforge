"""SeedVR2-7B — DEFAULT upscale/restore (Apache-2.0).

One-step temporally-consistent DiT restorer. GGUF (cmeka/SeedVR2-GGUF) + BlockSwap +
VAE tiling runs on 8 GB (slow: ~6-8 min per 1 min of 1080p — disclosed). Two-stage:
restore before upscale.
"""

from __future__ import annotations

from ...models.registry import register
from ..base import CAP_UPSCALE, ComfyBackend


@register("seedvr2", subsystem="enhance", license_id="seedvr2")
class SeedVR2(ComfyBackend):
    subsystem = "enhance"
    default_vram = 8.0
    required_nodes = ("ComfyUI-SeedVR2", "ComfyUI-GGUF")
    required_weights = ("enhance/seedvr2/",)

    def capabilities(self) -> set[str]:
        return {CAP_UPSCALE}
