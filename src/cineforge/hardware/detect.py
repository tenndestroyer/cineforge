"""Detect available GPUs and normalize them to GpuInfo.

Order of attempts: torch CUDA/ROCm (best signal) -> nvidia-smi -> torch-directml
presence -> CPU fallback. Every path is wrapped so a missing dependency never
crashes detection (this runs before torch is guaranteed installed).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
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


def _vendor_from_name(name: str) -> tuple[str, str]:
    """(vendor, backend_hint) from a GPU name string, platform-aware."""
    low = name.lower()
    if "nvidia" in low or "geforce" in low or "quadro" in low or "tesla" in low:
        return "nvidia", "cuda"
    if "amd" in low or "radeon" in low or "advanced micro" in low:
        return "amd", ("rocm" if sys.platform != "win32" else "directml")
    if "intel" in low or "arc" in low:
        return "intel", ("directml" if sys.platform == "win32" else "cpu")
    return "unknown", "directml" if sys.platform == "win32" else "cpu"


def _from_wmi() -> list[GpuInfo]:
    """Windows: identify the GPU vendor/name via WMI even before torch is installed.
    NOTE: WMI AdapterRAM is a 32-bit field that caps ~4 GB, so VRAM here is only a
    lower bound (tiering stays conservative until torch reports the real number)."""
    if sys.platform != "win32":
        return []
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    try:
        data = json.loads(out.stdout or "[]")
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    gpus: list[GpuInfo] = []
    for i, d in enumerate(data):
        name = (d.get("Name") or "").strip()
        if not name:
            continue
        vendor, hint = _vendor_from_name(name)
        ram = d.get("AdapterRAM") or 0
        vram = round(ram / (1024 ** 3), 1) if ram and ram > 0 else 0.0
        gpus.append(GpuInfo(vendor, name, vram, backend_hint=hint, index=i))
    discrete = [g for g in gpus if g.vendor in ("nvidia", "amd")]
    return discrete or gpus


def _from_lspci() -> list[GpuInfo]:
    """Linux: identify the GPU vendor/name via lspci before torch is installed."""
    if sys.platform == "win32" or not shutil.which("lspci"):
        return []
    try:
        out = subprocess.run(["lspci"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    gpus: list[GpuInfo] = []
    for line in out.stdout.splitlines():
        if any(k in line for k in ("VGA compatible controller", "3D controller", "Display controller")):
            desc = line.split(": ", 1)[-1][:80]
            vendor, hint = _vendor_from_name(desc)
            if vendor in ("nvidia", "amd", "intel"):
                gpus.append(GpuInfo(vendor, desc, 0.0, backend_hint=hint))
    discrete = [g for g in gpus if g.vendor in ("nvidia", "amd")]
    return discrete or gpus


def _from_directml() -> list[GpuInfo]:
    try:
        import torch_directml  # noqa: F401
    except ImportError:
        return []
    # DirectML doesn't expose reliable VRAM; mark unknown so tiering stays conservative.
    return [GpuInfo("unknown", "DirectML device", 0.0, backend_hint="directml")]


def detect_gpus() -> list[GpuInfo]:
    # torch (real VRAM) first, then vendor-id fallbacks that work before torch is
    # installed (nvidia-smi, Windows WMI, Linux lspci), then DirectML, then CPU.
    for source in (_from_torch, _from_nvidia_smi, _from_wmi, _from_lspci, _from_directml):
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
