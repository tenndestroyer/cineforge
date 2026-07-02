"""MuseTalk 1.5 — VRAM-floor photoreal lip-sync fallback (MIT, ~4 GB)."""

from __future__ import annotations

from ...models.registry import register
from ..base import ComfyBackend


@register("musetalk", subsystem="lipsync", license_id="musetalk")
class MuseTalk(ComfyBackend):
    subsystem = "lipsync"
    default_vram = 4.0
    required_nodes = ("ComfyUI-MuseTalk",)
    required_weights = ("lipsync/musetalk/",)

    def capabilities(self) -> set[str]:
        return set()
