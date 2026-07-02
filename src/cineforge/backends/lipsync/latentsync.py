"""LatentSync 1.6 — photoreal-only lip-sync fallback (Apache-2.0).

Best raw LSE-C sync score, but a HARD NO for cartoon faces (its face detector fails on
non-photoreal faces). Offered only for photoreal characters.
"""

from __future__ import annotations

from ...models.registry import register
from ..base import ComfyBackend


@register("latentsync", subsystem="lipsync", license_id="latentsync")
class LatentSync(ComfyBackend):
    subsystem = "lipsync"
    default_vram = 18.0
    required_nodes = ("ComfyUI-LatentSyncWrapper",)
    required_weights = ("lipsync/latentsync/",)

    def capabilities(self) -> set[str]:
        return set()  # NOT stylized-face capable
