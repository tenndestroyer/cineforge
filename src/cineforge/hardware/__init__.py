"""Hardware detection, VRAM tiering, and runtime/backend selection."""

from .backend_select import BackendPlan, select_backend
from .detect import GpuInfo, detect_gpus, primary_gpu
from .vram_tier import classify_tier, classify_vram

__all__ = [
    "BackendPlan",
    "GpuInfo",
    "classify_tier",
    "classify_vram",
    "detect_gpus",
    "primary_gpu",
    "select_backend",
]
