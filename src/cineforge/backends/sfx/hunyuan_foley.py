"""HunyuanVideo-Foley — DEFAULT video-to-audio foley/SFX (Tencent, territory-gated).

48 kHz stereo, video-conditioned sync (Synchformer). XXL 20 GB / XL 16 GB with offload
tiers. Territory-gated (excludes EU/UK/KR) + 100M MAU -> gated, requires consent.
"""

from __future__ import annotations

from ...models.registry import register
from ..base import CAP_VIDEO_CONDITIONED, ComfyBackend


@register("hunyuan_foley", subsystem="sfx", license_id="hunyuan_foley")
class HunyuanFoley(ComfyBackend):
    subsystem = "sfx"
    default_vram = 16.0
    required_nodes = ("ComfyUI-HunyuanVideoFoley",)
    required_weights = ("sfx/hunyuan_foley/",)

    def capabilities(self) -> set[str]:
        return {CAP_VIDEO_CONDITIONED}
