"""Qwen-Image-Edit — Apache-2.0 multi-reference identity lock.

Fused multi-person reference with drift reduction — the license-SAFE consistency
image engine that feeds locked keyframes into the video stage.
"""

from __future__ import annotations

from ...models.registry import register
from ..base import CAP_MULTI_REFERENCE, ComfyBackend


@register("qwen_image_edit", subsystem="image", license_id="qwen_image_edit")
class QwenImageEdit(ComfyBackend):
    subsystem = "image"
    default_vram = 16.0
    required_weights = ("image/qwen_image_edit/",)

    def capabilities(self) -> set[str]:
        return {CAP_MULTI_REFERENCE}
