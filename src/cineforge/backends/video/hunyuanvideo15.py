"""HunyuanVideo-1.5 — low-VRAM / max-realism video (Tencent, territory-gated).

8.3B params; best raw image quality and most physically-grounded motion of the top
three, and the lightest-weight (fits ~10-14 GB at FP8 + CPU offload) — so it is the
strong pick for the 8-12 GB tier or as a "max realism" fallback. No native audio.
License excludes EU/UK/South Korea and has a 100M MAU cap -> gated.
"""

from __future__ import annotations

from ...models.registry import register
from ..base import CAP_IMG2VID, ComfyBackend


@register("hunyuanvideo15", subsystem="video", license_id="hunyuanvideo15")
class HunyuanVideo15(ComfyBackend):
    subsystem = "video"
    default_vram = 10.0
    required_nodes = ("ComfyUI-HunyuanVideoWrapper",)
    required_weights = ("video/hunyuanvideo15/",)

    def capabilities(self) -> set[str]:
        return {CAP_IMG2VID}
