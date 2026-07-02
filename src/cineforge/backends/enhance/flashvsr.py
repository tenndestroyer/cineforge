"""FlashVSR — Apache-2.0 long-footage video super-resolution upscaler."""

from __future__ import annotations

from ...models.registry import register
from ..base import CAP_UPSCALE, ComfyBackend


@register("flashvsr", subsystem="enhance", license_id="flashvsr")
class FlashVSR(ComfyBackend):
    subsystem = "enhance"
    default_vram = 12.0
    required_nodes = ("ComfyUI-FlashVSR",)
    required_weights = ("enhance/flashvsr/",)

    def capabilities(self) -> set[str]:
        return {CAP_UPSCALE}
