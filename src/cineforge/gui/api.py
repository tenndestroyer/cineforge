"""JSON API bridging the GUI page to the coordinator + state. No external calls."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from .. import __version__
from ..config import Config
from ..config import _slug as slug
from ..hardware import classify_vram, detect_gpus, primary_gpu, select_backend
from ..logging_setup import EventLog
from ..models.matrix import ModelMatrix
from ..state import Project, store


class GuiApi:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.events = EventLog()
        self.matrix = ModelMatrix.load(cfg.data_dir / "model_matrix.json")
        self._thread: threading.Thread | None = None
        self._run_lock = threading.Lock()
        self._install_thread: threading.Thread | None = None
        self._install_lock = threading.Lock()

    # ---- data ----
    def _tier(self) -> tuple[str, Any, Any]:
        gpus = detect_gpus()
        prim = primary_gpu(gpus)
        plan = select_backend(gpus)
        tier = self.cfg.tier_override or classify_vram(prim.vram_gb)
        return tier, prim, plan

    def status(self) -> dict:
        tier, prim, plan = self._tier()
        return {
            "version": __version__,
            "license_mode": self.cfg.license_mode,
            "tier": tier,
            "gpu": {"name": prim.name, "vram_gb": prim.vram_gb, "vendor": prim.vendor},
            "runtime": plan.runtime,
            "torch_channel": plan.torch_channel,
            "quant": plan.quant_pref,
            "warnings": plan.warnings,
            "projects": self.list_projects(),
            "running": bool(self._thread and self._thread.is_alive()),
        }

    def models(self) -> dict:
        tier, _, _ = self._tier()
        out = {}
        for sub in self.matrix.subsystems():
            c = self.matrix.resolve(sub, tier, self.cfg.license_mode)
            out[sub] = {"model": c.model_id, "variant": c.variant, "quant": c.quant,
                        "min_vram_gb": c.min_vram_gb, "license": c.license_id}
        return {"tier": tier, "license_mode": self.cfg.license_mode, "subsystems": out}

    def list_projects(self) -> list[str]:
        d = self.cfg.projects_dir
        if not d.is_dir():
            return []
        return sorted(p.name for p in d.iterdir() if (p / "project.json").is_file())

    def get_project(self, name: str) -> dict:
        return store.load(self.cfg.project_dir(name)).to_dict()

    def new_project(self, idea: str, name: str = "", style: str = "stylized 3D cartoon",
                    overwrite: bool = False) -> dict:
        import datetime as _dt

        if not idea.strip():
            return {"ok": False, "error": "idea is required"}
        name = name or slug(idea)[:40] or "project"
        pdir = self.cfg.project_dir(name)
        if store.project_json_path(pdir).is_file() and not overwrite:
            return {"ok": False, "error": f"project {name!r} already exists (pass overwrite to replace it)"}
        project = Project(name=name, idea=idea, style=style, created=_dt.date.today().isoformat())
        self.cfg.ensure_dirs()
        store.save(project, pdir)
        self.events.emit("info", "project", f"created project {name!r}")
        return {"ok": True, "name": name}

    def start_run(self, name: str, resume: bool = False) -> dict:
        def _run() -> None:
            from ..pipeline import Coordinator

            try:
                project = store.load(self.cfg.project_dir(name))
                coord = Coordinator(self.cfg, self.events)
                coord.resume(project) if resume else coord.run(project)
            except Exception as e:  # noqa: BLE001 - surface to the events feed, don't crash the server
                self.events.emit("error", "run", f"{type(e).__name__}: {e}")

        # Atomic check-then-start so two concurrent request threads can't both launch a run.
        with self._run_lock:
            if self._thread and self._thread.is_alive():
                return {"ok": False, "error": "a run is already in progress"}
            self._thread = threading.Thread(target=_run, name=f"run-{name}", daemon=True)
            self._thread.start()
        return {"ok": True}

    def ack(self, name: str, model_id: str) -> dict:
        project = store.load(self.cfg.project_dir(name))
        if model_id not in project.license_acks:
            project.license_acks.append(model_id)
            store.save(project, self.cfg.project_dir(name))
        self.events.emit("info", "license", f"acknowledged {model_id} for {name}")
        return {"ok": True, "acks": project.license_acks}

    # ---- install / onboarding ----
    def install_status(self) -> dict:
        from ..scripts_verify import install_status as _status

        st = _status(self.cfg)
        st["installing"] = bool(self._install_thread and self._install_thread.is_alive())
        return st

    def save_keys(self, hf_token: str) -> dict:
        keys_path = self.cfg.repo_root / "keys.env"
        lines = []
        if keys_path.is_file():
            lines = [ln for ln in keys_path.read_text(encoding="utf-8").splitlines()
                     if not ln.strip().startswith("HF_TOKEN=")]
        lines.append(f"HF_TOKEN={hf_token.strip()}")
        keys_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.cfg.hf_token = hf_token.strip()
        self.events.emit("info", "keys", "saved HF_TOKEN to keys.env")
        return {"ok": True, "has_hf_token": bool(hf_token.strip())}

    def start_install(self) -> dict:
        def _run() -> None:
            self.events.emit("stage", "install", "Starting installer — downloads several GB (torch, ComfyUI, models)...")
            if sys.platform == "win32":
                cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                       "-File", str(self.cfg.repo_root / "setup.ps1")]
            else:
                cmd = ["bash", str(self.cfg.repo_root / "setup.sh")]
            try:
                proc = subprocess.Popen(cmd, cwd=str(self.cfg.repo_root), stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                        errors="replace", bufsize=1)
                for line in proc.stdout:  # stream installer output live into the events feed
                    line = line.rstrip()
                    if line.strip():
                        self.events.emit("info", "install", line)
                proc.wait()
                if proc.returncode == 0:
                    self.events.emit("info", "install", "Install complete — you're ready to render.", pct=100.0)
                else:
                    self.events.emit("error", "install",
                                     f"Installer exited with code {proc.returncode}. See docs/TROUBLESHOOTING.md")
            except Exception as e:  # noqa: BLE001 - surface to the feed, don't crash the server
                self.events.emit("error", "install", f"{type(e).__name__}: {e}")

        with self._install_lock:
            if self._install_thread and self._install_thread.is_alive():
                return {"ok": False, "error": "install already running"}
            self._install_thread = threading.Thread(target=_run, name="install", daemon=True)
            self._install_thread.start()
        return {"ok": True}

    # ---- project settings / script editing ----
    def update_settings(self, name: str, settings: dict) -> dict:
        pdir = self.cfg.project_dir(name)
        project = store.load(pdir)
        rp = project.render_plan
        if settings.get("target_shots") is not None:
            rp["target_shots"] = max(1, int(settings["target_shots"]))
        if settings.get("target_scenes") is not None:
            rp["target_scenes"] = max(1, int(settings["target_scenes"]))
        q = dict(rp.get("quality") or {})
        for k in ("width", "height", "steps", "best_of_n"):
            if settings.get(k) is not None:
                q[k] = int(settings[k])
        if settings.get("cfg") is not None:
            q["cfg"] = float(settings["cfg"])
        if settings.get("checkpoint"):
            q["checkpoint"] = settings["checkpoint"]
        rp["quality"] = q
        store.save(project, pdir)
        self.events.emit("info", "settings", f"updated settings for {name}")
        return {"ok": True}

    def update_script(self, name: str, shots: list) -> dict:
        pdir = self.cfg.project_dir(name)
        project = store.load(pdir)
        by_id = {sh.id: sh for sh in project.all_shots()}
        n = 0
        for s in shots or []:
            sh = by_id.get(s.get("id"))
            if not sh:
                continue
            if s.get("keyframe_prompt") is not None:
                sh.keyframe_prompt = s["keyframe_prompt"]
            if s.get("description") is not None:
                sh.description = s["description"]
            n += 1
        store.save(project, pdir)
        self.events.emit("info", "script", f"updated {n} shot(s) for {name}")
        return {"ok": True, "updated": n}

    def rerender(self, name: str) -> dict:
        """Re-render keyframes with the current prompts + settings, keeping the script."""
        pdir = self.cfg.project_dir(name)
        project = store.load(pdir)
        for sh in project.all_shots():
            sh.takes = [t for t in sh.takes if t.kind != "keyframe"]
            sh.status = "pending"
        project.stage = "storyboard"  # resume -> producing (applies settings) -> keyframes
        store.save(project, pdir)
        return self.start_run(name, resume=True)

    def download_checkpoint(self, model: str) -> dict:
        repos = {"sdxl": ("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors")}
        if model not in repos:
            return {"ok": False, "error": f"unknown model {model!r}"}
        repo, fn = repos[model]

        def _run() -> None:
            self.events.emit("stage", "download", f"Downloading {model} ({fn}, several GB)...")
            try:
                from huggingface_hub import hf_hub_download

                dest = self.cfg.comfy_dir / "models" / "checkpoints"
                dest.mkdir(parents=True, exist_ok=True)
                hf_hub_download(repo_id=repo, filename=fn, local_dir=str(dest), token=self.cfg.hf_token or None)
                self.events.emit("info", "download", f"{model} ready — select it under Quality and Re-render.")
            except Exception as e:  # noqa: BLE001
                self.events.emit("error", "download", f"{type(e).__name__}: {e}")

        with self._install_lock:
            if self._install_thread and self._install_thread.is_alive():
                return {"ok": False, "error": "a download/install is already running"}
            self._install_thread = threading.Thread(target=_run, name=f"dl-{model}", daemon=True)
            self._install_thread.start()
        return {"ok": True}

    # ---- gallery ----
    def project_frames(self, name: str) -> dict:
        project = store.load(self.cfg.project_dir(name))
        frames = []
        for sh in project.all_shots():
            for kind in ("video", "enhance", "keyframe"):
                t = sh.accepted_take(kind)
                if t and Path(t.path).is_file():
                    frames.append({"shot": sh.id, "kind": kind, "path": t.path,
                                   "video": Path(t.path).suffix.lower() in (".webm", ".mp4")})
                    break
        return {"frames": frames}

    def serve_file(self, path: str) -> tuple[int, str, bytes]:
        out = (self.cfg.comfy_dir / "output").resolve()
        try:
            rp = Path(path).resolve()
            rp.relative_to(out)  # only serve files from the ComfyUI output dir
        except (ValueError, OSError):
            return 403, "text/plain; charset=utf-8", b"forbidden"
        if not rp.is_file():
            return 404, "text/plain; charset=utf-8", b"not found"
        ctypes = {".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp",
                  ".webm": "video/webm", ".mp4": "video/mp4"}
        return 200, ctypes.get(rp.suffix.lower(), "application/octet-stream"), rp.read_bytes()

    # ---- routing ----
    def handle(self, method: str, path: str, query: dict, body: dict | None) -> tuple[int, str, bytes]:
        # binary file serving bypasses the JSON wrapper
        if method == "GET" and path == "/api/file":
            return self.serve_file((query.get("path", [""]) or [""])[0])
        try:
            data = self._route(method, path, query, body or {})
            return 200, "application/json; charset=utf-8", _json(data)
        except FileNotFoundError as e:
            return 404, "application/json", _json({"error": str(e)})
        except Exception as e:  # noqa: BLE001
            return 400, "application/json", _json({"error": f"{type(e).__name__}: {e}"})

    def _route(self, method: str, path: str, query: dict, body: dict) -> Any:
        one = lambda k, d="": (query.get(k, [d])[0])  # noqa: E731
        if method == "GET" and path == "/api/status":
            return self.status()
        if method == "GET" and path == "/api/models":
            return self.models()
        if method == "GET" and path == "/api/events":
            return {"events": self.events.since(int(one("since", "0") or 0))}
        if method == "GET" and path == "/api/projects":
            return {"projects": self.list_projects()}
        if method == "GET" and path == "/api/project":
            name = one("name").strip()
            if not name:
                raise ValueError("query parameter 'name' is required")
            return self.get_project(name)
        if method == "POST" and path == "/api/new":
            return self.new_project(body.get("idea", ""), body.get("name", ""), body.get("style", "stylized 3D cartoon"))
        if method == "POST" and path == "/api/run":
            return self.start_run(body.get("name", ""), resume=bool(body.get("resume")))
        if method == "POST" and path == "/api/ack":
            return self.ack(body.get("name", ""), body.get("model_id", ""))
        if method == "GET" and path == "/api/install_status":
            return self.install_status()
        if method == "POST" and path == "/api/save_keys":
            return self.save_keys(body.get("hf_token", ""))
        if method == "POST" and path == "/api/install":
            return self.start_install()
        if method == "POST" and path == "/api/project/settings":
            return self.update_settings(body.get("name", ""), body.get("settings", {}))
        if method == "POST" and path == "/api/project/script":
            return self.update_script(body.get("name", ""), body.get("shots", []))
        if method == "POST" and path == "/api/project/rerender":
            return self.rerender(body.get("name", ""))
        if method == "POST" and path == "/api/download_checkpoint":
            return self.download_checkpoint(body.get("model", ""))
        if method == "GET" and path == "/api/project/frames":
            return self.project_frames(one("name"))
        raise FileNotFoundError(f"no route: {method} {path}")


def _json(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")
