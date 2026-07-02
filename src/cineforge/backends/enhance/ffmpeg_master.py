"""ffmpeg mastering — two-pass loudnorm, sidechain duck, LUT grade, grain, mux.

A tool, not a model. This adapter checks ffmpeg is present and exposes the mastering
primitive the `master` pipeline stage uses (two-pass loudnorm measure->apply, never
single-pass; sidechain-duck music under dialogue; 3D-LUT grade; subtle grain at export
res; stream-copy mux). Targets -14 LUFS / -1.5 dBTP (streaming) or -23 (broadcast).
"""

from __future__ import annotations

import shutil
from typing import Any

from ...errors import NotInstalledError
from ...models.registry import register
from ..base import Backend, Result


@register("ffmpeg_master", subsystem="enhance", license_id="ffmpeg_master")
class FFmpegMaster(Backend):
    subsystem = "enhance"

    def estimate_vram(self, request: Any) -> float:
        return 0.0  # CPU tool

    @staticmethod
    def has_ffmpeg() -> bool:
        return shutil.which("ffmpeg") is not None

    def generate(self, request: Any) -> Result:
        if not self.has_ffmpeg():
            raise NotInstalledError("ffmpeg is not on PATH; install it (setup does this) to master audio/video.")
        # The real two-pass master runs in pipeline._run_master over the full timeline,
        # not as a per-shot generate. Kept here so the tool is registry-visible + doctor-checkable.
        raise NotInstalledError("ffmpeg mastering runs in the 'master' stage, not per-shot generate().")
