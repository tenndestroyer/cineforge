"""Z-Image — CO-DEFAULT keyframe image backend (Apache-2.0).

Turbo for fast previews (top open-weights on the arena, ~6B, <16 GB), Base for the
high-fidelity final pass. Draft-then-final pattern feeds locked hero keyframes to the
video stage.
"""

from __future__ import annotations

from ...models.registry import register
from ..base import ComfyBackend


@register("zimage", subsystem="image", license_id="zimage")
class ZImage(ComfyBackend):
    subsystem = "image"
    default_vram = 12.0
    required_nodes = ()
    required_weights = ("image/zimage/",)

    def capabilities(self) -> set[str]:
        return set()
