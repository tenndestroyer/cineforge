"""Map VRAM to a model-selection tier.

  low  : 8-15 GB   (short, softer, lower-res clips; long gen times — honest about it)
  mid  : 16-31 GB  (RTX 4090-class; the primary target)
  high : 32 GB+     (RTX 5090 / workstation; native 4K, concurrent hero models)

Sub-8GB still classifies as 'low' but the GUI surfaces a strong "expect slow /
degraded" warning (see backend_select warnings).
"""

from __future__ import annotations

from .detect import GpuInfo

LOW_MIN = 8.0
MID_MIN = 16.0
HIGH_MIN = 32.0


def classify_vram(vram_gb: float) -> str:
    if vram_gb >= HIGH_MIN:
        return "high"
    if vram_gb >= MID_MIN:
        return "mid"
    return "low"


def classify_tier(gpu: GpuInfo) -> str:
    return classify_vram(gpu.vram_gb)
