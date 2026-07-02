"""Wan 2.2 — the DEFAULT video backend (Apache-2.0).

T2V-A14B / I2V-A14B / TI2V-5B (the 5B is the only top-3 model native on 8 GB). The
primary consistency path is image-to-video off a locked keyframe. Largest 3rd-party
LoRA ecosystem for stylized/3D-cartoon looks. Fully unrestricted commercially.
"""

from __future__ import annotations

from ...models.registry import register
from ..base import CAP_IMG2VID, ComfyBackend


@register("wan22", subsystem="video", license_id="wan22")
class Wan22(ComfyBackend):
    subsystem = "video"
    default_vram = 16.0
    required_nodes = ("ComfyUI-WanVideoWrapper",)
    required_weights = ("video/wan22/",)

    def capabilities(self) -> set[str]:
        return {CAP_IMG2VID}
