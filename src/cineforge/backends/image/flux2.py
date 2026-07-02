"""FLUX.2 — OPT-IN character-sheet keyframe backend (non-commercial, gated).

Best-documented multi-reference (~10 images) primitive for identity-locked character
model sheets. FLUX.2-dev license is non-commercial -> blocked in Safe mode; gated repo
needs an HF_TOKEN to download.
"""

from __future__ import annotations

from ...models.registry import register
from ..base import CAP_MULTI_REFERENCE, ComfyBackend


@register("flux2", subsystem="image", license_id="flux2")
class Flux2(ComfyBackend):
    subsystem = "image"
    default_vram = 16.0
    required_nodes = ("ComfyUI-GGUF",)
    required_weights = ("image/flux2/",)

    def capabilities(self) -> set[str]:
        return {CAP_MULTI_REFERENCE}
