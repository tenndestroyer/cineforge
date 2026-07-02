"""ACE-Step 1.5 — the DEFAULT music backend (MIT).

base 3.5B for 8-12 GB; XL-SFT for the final pass; XL-Turbo for auditions. Exact
duration control, structured lyric tags, repaint mode, and per-show theme LoRA. Native
ComfyUI support. Songs-with-vocals and instrumental score.
"""

from __future__ import annotations

from ...models.registry import register
from ..base import CAP_DURATION_CONTROL, CAP_INSTRUMENTAL, CAP_VOCALS, ComfyBackend


@register("acestep15", subsystem="music", license_id="acestep15")
class AceStep15(ComfyBackend):
    subsystem = "music"
    default_vram = 8.0
    required_nodes = ("ComfyUI-ACE-Step",)
    required_weights = ("music/acestep15/",)

    def capabilities(self) -> set[str]:
        return {CAP_VOCALS, CAP_INSTRUMENTAL, CAP_DURATION_CONTROL}
