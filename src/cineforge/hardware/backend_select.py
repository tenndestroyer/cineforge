"""Choose the torch runtime + quant preference for the detected hardware.

Encodes the two facts the research flagged as easy to get catastrophically wrong:
  1. Blackwell / RTX 50-series (compute cap 12.x, sm_120): stable torch still lacks
     sm_120 kernels, so a naive cu128-stable install silently CPU-falls-back or throws
     a kernel error. We route it to a nightly channel by default.
  2. AMD/ROCm: GGUF is broken/unusably slow on ROCm (esp. Windows). Prefer fp8/
     safetensors, never GGUF. DirectML is a degraded last resort with no real VRAM
     management.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from .detect import GpuInfo, primary_gpu


@dataclass
class BackendPlan:
    runtime: str            # 'cuda' | 'rocm' | 'directml' | 'cpu'
    torch_channel: str      # 'cu128-stable' | 'cu128-nightly' | 'rocm' | 'directml' | 'cpu'
    quant_pref: str         # 'bf16' | 'fp8' | 'gguf' | 'fp16'
    warnings: list[str] = field(default_factory=list)
    gpu: GpuInfo | None = None


# RTX 50-series marketing names -> Blackwell (sm_120). Needed when detection came
# from nvidia-smi (no compute capability) rather than torch.
_BLACKWELL_NAME_TOKENS = ("5050", "5060", "5070", "5080", "5090", "blackwell",
                          "rtx pro 6000", "b100", "b200", "gb100", "gb200")


def _is_blackwell(cc: str) -> bool:
    # sm_100 (datacenter B100/B200 = cc 10.x) and sm_120 (consumer RTX 50 = cc 12.x)
    # are both Blackwell. There is no non-Blackwell NVIDIA compute cap >= 10.
    try:
        return int(cc.split(".")[0]) >= 10
    except (ValueError, IndexError):
        return False


def _looks_blackwell(gpu: GpuInfo) -> bool:
    if _is_blackwell(gpu.compute_cap):
        return True
    name = gpu.name.lower()
    return any(tok in name for tok in _BLACKWELL_NAME_TOKENS)


def select_backend(gpus: list[GpuInfo] | None = None) -> BackendPlan:
    gpu = primary_gpu(gpus)
    warnings: list[str] = []

    if gpu.vendor == "nvidia":
        if _looks_blackwell(gpu):
            channel = "cu128-nightly"
            warnings.append(
                "Blackwell (sm_120) detected: using a NIGHTLY torch build. Stable torch "
                "still lacks sm_120 kernels and would silently fall back to CPU. Re-check "
                "monthly — this is the fastest-moving fact in the stack."
            )
        else:
            channel = "cu128-stable"
        quant = "bf16" if gpu.vram_gb >= 24 else "fp8"
        if gpu.vram_gb < 8:
            warnings.append("Under 8 GB VRAM: expect low-res, short clips and long render times.")
        return BackendPlan("cuda", channel, quant, warnings, gpu)

    if gpu.vendor == "amd":
        if sys.platform == "win32":
            warnings.append(
                "AMD on Windows -> DirectML (ROCm has no Windows wheels). It works, but it's a "
                "degraded path with no real VRAM management; expect slower renders."
            )
            return BackendPlan("directml", "directml", "fp16", warnings, gpu)
        warnings.append(
            "AMD/ROCm (Linux): using fp8/safetensors, NOT GGUF (broken/slow on ROCm). Use a "
            "ROCm-supported card (RX 7000/9000 / W7000)."
        )
        return BackendPlan("rocm", "rocm", "fp8", warnings, gpu)

    if gpu.backend_hint == "directml":
        warnings.append(
            "DirectML fallback: degraded last resort with no real VRAM management. "
            "Renders will be slow; prefer an NVIDIA or ROCm-supported AMD GPU."
        )
        return BackendPlan("directml", "directml", "fp16", warnings, gpu)

    warnings.append(
        "No supported GPU detected — CPU only. Image/voice/music are viable but slow; "
        "video generation is impractical on CPU."
    )
    return BackendPlan("cpu", "cpu", "fp16", warnings, gpu)
