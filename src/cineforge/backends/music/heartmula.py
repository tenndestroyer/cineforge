"""HeartMuLa (oss-3B) — Apache-2.0 music runner-up.

4-bit fits 16 GB; sequential offload down to ~8 GB. A viable low-VRAM alternate and an
A/B "second opinion" against ACE-Step.
"""

from __future__ import annotations

from ...models.registry import register
from ..base import CAP_INSTRUMENTAL, CAP_VOCALS, ComfyBackend


@register("heartmula", subsystem="music", license_id="heartmula")
class HeartMuLa(ComfyBackend):
    subsystem = "music"
    default_vram = 8.0
    required_weights = ("music/heartmula/",)

    def capabilities(self) -> set[str]:
        return {CAP_VOCALS, CAP_INSTRUMENTAL}
