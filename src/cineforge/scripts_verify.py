"""Doctor checks — the body of `cineforge doctor` (and scripts/verify_install.py).

Each check returns {name, ok, detail}. Designed to run even before torch is installed
(imports are guarded) so it can diagnose a half-finished setup.
"""

from __future__ import annotations

import shutil
import sys

from .config import Config
from .hardware import classify_vram, detect_gpus, primary_gpu, select_backend
from .models.comfy_client import ComfyClient
from .models.licenses import LicenseGate
from .models.matrix import ModelMatrix
from .models.registry import BackendRegistry


def _check(name: str, ok: bool, detail: str) -> dict:
    return {"name": name, "ok": ok, "detail": detail}


def run_doctor(cfg: Config) -> list[dict]:
    checks: list[dict] = []

    # Python version (torch has no wheels for 3.13/3.14)
    v = sys.version_info
    py_ok = (3, 10) <= (v.major, v.minor) <= (3, 12)
    checks.append(_check("python", py_ok, f"{v.major}.{v.minor}.{v.micro} "
                                          f"({'ok' if py_ok else 'need 3.10-3.12; torch has no 3.13/3.14 wheels'})"))

    # torch + CUDA
    try:
        import torch

        cuda = torch.cuda.is_available()
        detail = f"torch {torch.__version__}, cuda_available={cuda}"
        if cuda:
            try:
                detail += f", device={torch.cuda.get_device_name(0)}"
            except Exception:
                pass
        checks.append(_check("torch", True, detail))
    except ImportError:
        checks.append(_check("torch", False, "torch not installed — run setup"))

    # hardware detection + plan
    try:
        from .hardware import primary_gpu, select_backend

        plan = select_backend()
        g = plan.gpu
        checks.append(_check("gpu", g is not None and g.vendor != "cpu",
                             f"{g.name} ({g.vram_gb} GB) -> {plan.runtime}/{plan.torch_channel}, quant {plan.quant_pref}"))
        _ = primary_gpu
    except Exception as e:  # noqa: BLE001
        checks.append(_check("gpu", False, f"detection error: {e}"))

    # ComfyUI reachable (only up during a render, so 'not reachable' is not fatal)
    reachable = ComfyClient(cfg.comfy_url).is_reachable()
    checks.append(_check("comfyui", True,
                         f"reachable at {cfg.comfy_url}" if reachable
                         else f"not running at {cfg.comfy_url} (expected unless a render is active)"))

    # Ollama reachable (needed for the planning agents)
    try:
        from .backends.llm.ollama import OllamaLLM

        ok = OllamaLLM(cfg.ollama_url).available()
        checks.append(_check("ollama", ok,
                             f"reachable at {cfg.ollama_url}" if ok
                             else f"not reachable at {cfg.ollama_url} — install Ollama + pull a model"))
    except Exception as e:  # noqa: BLE001
        checks.append(_check("ollama", False, str(e)))

    # ffmpeg
    checks.append(_check("ffmpeg", shutil.which("ffmpeg") is not None,
                         "found" if shutil.which("ffmpeg") else "not on PATH (needed for mastering)"))

    # data files load
    try:
        matrix = ModelMatrix.load(cfg.data_dir / "model_matrix.json")
        gate = LicenseGate.load(cfg.data_dir / "licenses.json")
        checks.append(_check("data_files", True,
                             f"model_matrix ({len(matrix.subsystems())} subsystems) + "
                             f"licenses ({len(gate.models)} models) loaded"))
    except Exception as e:  # noqa: BLE001
        checks.append(_check("data_files", False, str(e)))
        return checks

    # every model in the matrix has a registered adapter
    missing = []
    for sub in matrix.subsystems():
        for mode in ("safe", "research"):
            try:
                c = matrix.resolve(sub, "mid", mode)
                if not BackendRegistry.has(c.model_id):
                    missing.append(c.model_id)
            except Exception:  # noqa: BLE001
                pass
    missing = sorted(set(missing))
    checks.append(_check("registry", not missing,
                         "all matrix models have adapters" if not missing else f"missing adapters: {missing}"))

    return checks


def install_status(cfg: Config) -> dict:
    """Structured, GUI-friendly install status: per-component ok/detail, an overall
    `ready` flag (all critical components satisfied), GPU/tier, and token state."""
    comps: list[dict] = []

    def add(key: str, label: str, ok: bool, detail: str, critical: bool = True) -> None:
        comps.append({"key": key, "label": label, "ok": bool(ok), "detail": detail, "critical": critical})

    v = sys.version_info
    add("python", "Python 3.10-3.12", (3, 10) <= (v.major, v.minor) <= (3, 12),
        f"{v.major}.{v.minor}.{v.micro}", critical=False)

    torch_ok, torch_detail = False, "not installed - run install"
    try:
        import torch

        torch_ok = True
        cuda = torch.cuda.is_available()
        torch_detail = f"torch {torch.__version__} (cuda={cuda})"
    except ImportError:
        pass
    add("torch", "PyTorch", torch_ok, torch_detail)

    plan = select_backend()
    g = plan.gpu
    add("gpu", "GPU acceleration", bool(g and g.vendor != "cpu"),
        f"{g.name} -> {plan.runtime}/{plan.torch_channel}" if g else "none", critical=False)

    comfy_ok = (cfg.comfy_dir / "main.py").is_file()
    add("comfyui", "ComfyUI engine", comfy_ok, "installed" if comfy_ok else "not installed (Install downloads it)")

    from .backends.llm.ollama import OllamaLLM

    ol = OllamaLLM(cfg.ollama_url).available()
    add("ollama", "Ollama (agent brain)", ol, "running" if ol else "not reachable - install from ollama.com")

    ff = shutil.which("ffmpeg") is not None
    add("ffmpeg", "FFmpeg (audio/video)", ff, "found" if ff else "not on PATH (Install adds it)")

    matrix = ModelMatrix.load(cfg.data_dir / "model_matrix.json")
    tier = cfg.tier_override or classify_vram(primary_gpu(detect_gpus()).vram_gb)
    needed, present = [], 0
    for sub in matrix.subsystems():
        c = matrix.resolve(sub, tier, "safe")
        if not c.repo or c.repo.startswith("ollama:"):
            continue
        needed.append(sub)
        d = cfg.models_dir / sub / c.model_id
        if d.is_dir() and any(d.iterdir()):
            present += 1
    total = len(needed)
    add("weights", "Model weights", total > 0 and present == total, f"{present}/{total} downloaded (tier {tier})")

    ready = all(c["ok"] for c in comps if c["critical"])
    return {
        "ready": ready,
        "components": comps,
        "tier": tier,
        "gpu": {"name": g.name, "vendor": g.vendor, "vram_gb": g.vram_gb, "runtime": plan.runtime} if g else {},
        "has_hf_token": bool(cfg.hf_token),
        "needs_hf_token": cfg.license_mode == "research",
        "platform": sys.platform,
    }
