"""GMFSS_union — anime-tuned frame interpolation (maintained RIFE successor for 2D)."""

from __future__ import annotations

from ...models.registry import register
from ..base import CAP_INTERPOLATE, ComfyBackend


@register("gmfss_union", subsystem="enhance", license_id="gmfss_union")
class GMFSSUnion(ComfyBackend):
    subsystem = "enhance"
    default_vram = 4.0
    required_nodes = ("ComfyUI-Frame-Interpolation",)
    required_weights = ("enhance/gmfss/",)

    def capabilities(self) -> set[str]:
        return {CAP_INTERPOLATE}
