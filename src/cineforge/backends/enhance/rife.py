"""Practical-RIFE — frame interpolation (MIT; cite hzwer, not academic RIFE).

Smooths fps and masks lip jitter. Runs via ComfyUI-Frame-Interpolation.
"""

from __future__ import annotations

from ...models.registry import register
from ..base import CAP_INTERPOLATE, ComfyBackend


@register("rife", subsystem="enhance", license_id="rife")
class RIFE(ComfyBackend):
    subsystem = "enhance"
    default_vram = 4.0
    required_nodes = ("ComfyUI-Frame-Interpolation",)
    required_weights = ("enhance/rife/",)

    def capabilities(self) -> set[str]:
        return {CAP_INTERPOLATE}
