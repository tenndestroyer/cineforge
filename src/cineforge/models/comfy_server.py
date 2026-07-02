"""Launch and manage the local ComfyUI server as a subprocess.

The render backends submit workflow graphs to ComfyUI over HTTP; something has to
actually *start* ComfyUI. `ensure_running` does that on demand (idempotent): if it's
already reachable it's a no-op, otherwise it launches `ComfyUI/main.py` with the bundled
Python, adds `--directml` for AMD/Intel-on-Windows, points ComfyUI at our weights via
extra_model_paths.yaml, and waits until the server answers.

If ComfyUI isn't installed yet (no main.py), it returns False so callers raise a clear
NotInstalledError telling the user to run setup.
"""

from __future__ import annotations

import subprocess
import sys
import time

from ..logging_setup import get_logger
from .comfy_client import ComfyClient

_log = get_logger("cineforge.comfy")
_PROC: subprocess.Popen | None = None


def _python_exe(cfg) -> str:
    for cand in (
        cfg.repo_root / "python_embeded" / "python.exe",
        cfg.repo_root / ".venv" / "Scripts" / "python.exe",
        cfg.repo_root / ".venv" / "bin" / "python",
    ):
        if cand.exists():
            return str(cand)
    return sys.executable


def _port(cfg) -> int:
    try:
        return int(cfg.comfy_url.rsplit(":", 1)[1].split("/")[0])
    except (ValueError, IndexError):
        return 8188


def is_running(cfg) -> bool:
    return ComfyClient(cfg.comfy_url).is_reachable()


def installed(cfg) -> bool:
    return (cfg.comfy_dir / "main.py").is_file()


def _write_extra_model_paths(cfg) -> None:
    """Point ComfyUI at our models_store so it can find downloaded weights. Best-effort;
    per-model file placement is refined as each backend graph is wired."""
    yaml_path = cfg.comfy_dir / "extra_model_paths.yaml"
    if yaml_path.exists():
        return
    base = cfg.models_dir.as_posix()
    yaml_path.write_text(
        "cineforge:\n"
        f"  base_path: {base}\n"
        "  checkpoints: image\n"
        "  diffusion_models: |\n"
        "    video\n"
        "    image\n"
        "  text_encoders: text_encoders\n"
        "  clip: text_encoders\n"
        "  vae: vae\n"
        "  loras: loras\n"
        "  upscale_models: enhance\n",
        encoding="utf-8",
    )


def ensure_running(cfg, plan=None, timeout: float = 180.0) -> bool:
    """Return True once ComfyUI is reachable. Launches it if needed. False if ComfyUI
    isn't installed."""
    global _PROC
    if is_running(cfg):
        return True
    if not installed(cfg):
        return False

    if plan is None:
        from ..hardware import select_backend

        plan = select_backend()
    _write_extra_model_paths(cfg)
    args = [_python_exe(cfg), str(cfg.comfy_dir / "main.py"),
            "--listen", "127.0.0.1", "--port", str(_port(cfg))]
    if plan is not None and getattr(plan, "runtime", "") == "directml":
        args.append("--directml")
    _log.info("launching ComfyUI: %s", " ".join(args))
    try:
        _PROC = subprocess.Popen(args, cwd=str(cfg.comfy_dir),
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:  # pragma: no cover
        _log.error("failed to launch ComfyUI: %s", e)
        return False

    waited = 0.0
    while waited < timeout:
        if is_running(cfg):
            _log.info("ComfyUI is up at %s", cfg.comfy_url)
            return True
        if _PROC.poll() is not None:  # process died
            _log.error("ComfyUI exited during startup (code %s)", _PROC.returncode)
            return False
        time.sleep(2.0)
        waited += 2.0
    return is_running(cfg)


def stop() -> None:
    global _PROC
    if _PROC and _PROC.poll() is None:
        _PROC.terminate()
        try:
            _PROC.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            _PROC.kill()
    _PROC = None
