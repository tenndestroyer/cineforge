"""Detect available GPUs and normalize them to GpuInfo.

Order of attempts: torch CUDA/ROCm (best signal) -> nvidia-smi -> torch-directml
presence -> CPU fallback. Every path is wrapped so a missing dependency never
crashes detection (this runs before torch is guaranteed installed).
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class GpuInfo:
    vendor: str            # 'nvidia' | 'amd' | 'intel' | 'unknown' | 'cpu'
    name: str
    vram_gb: float
    compute_cap: str = ""  # e.g. '8.9' (Ada), '9.0' (Hopper), '12.0' (Blackwell)
    backend_hint: str = "cpu"  # 'cuda' | 'rocm' | 'directml' | 'cpu'
    index: int = 0
    extra: dict = field(default_factory=dict)


def _from_torch() -> list[GpuInfo]:
    try:
        import torch
    except ImportError:
        return []
    try:
        if not torch.cuda.is_available():
            return []
    except Exception:
        return []
    is_rocm = bool(getattr(torch.version, "hip", None))
    vendor = "amd" if is_rocm else "nvidia"
    hint = "rocm" if is_rocm else "cuda"
    gpus: list[GpuInfo] = []
    try:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append(
                GpuInfo(
                    vendor=vendor,
                    name=p.name,
                    vram_gb=round(p.total_memory / (1024 ** 3), 1),
                    compute_cap=f"{p.major}.{p.minor}" if not is_rocm else "",
                    backend_hint=hint,
                    index=i,
                )
            )
    except Exception:
        return []
    return gpus


def _from_nvidia_smi() -> list[GpuInfo]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    gpus: list[GpuInfo] = []
    for i, line in enumerate(out.stdout.strip().splitlines()):
        if "," not in line:
            continue
        name, mem = (x.strip() for x in line.split(",", 1))
        try:
            vram = round(float(mem) / 1024, 1)  # MiB -> GiB
        except ValueError:
            vram = 0.0
        gpus.append(GpuInfo("nvidia", name, vram, backend_hint="cuda", index=i))
    return gpus


def _from_directml() -> list[GpuInfo]:
    try:
        import torch_directml  # noqa: F401
    except ImportError:
        return []
    # DirectML doesn't expose reliable VRAM; mark unknown so tiering stays conservative.
    return [GpuInfo("unknown", "DirectML device", 0.0, backend_hint="directml")]


def detect_gpus() -> list[GpuInfo]:
    for source in (_from_torch, _from_nvidia_smi, _from_directml):
        gpus = source()
        if gpus:
            return gpus
    return [GpuInfo("cpu", "CPU (no GPU detected)", 0.0, backend_hint="cpu")]


def primary_gpu(gpus: list[GpuInfo] | None = None) -> GpuInfo:
    if gpus is None:
        gpus = detect_gpus()
    if not gpus:  # explicit empty list -> honest CPU fallback, not silent re-detection
        return GpuInfo("cpu", "CPU (no GPU detected)", 0.0, backend_hint="cpu")
    return max(gpus, key=lambda g: g.vram_gb)
