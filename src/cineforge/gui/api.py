"""JSON API bridging the GUI page to the coordinator + state. No external calls."""

from __future__ import annotations

import json
import threading
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

    # ---- routing ----
    def handle(self, method: str, path: str, query: dict, body: dict | None) -> tuple[int, str, bytes]:
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
        raise FileNotFoundError(f"no route: {method} {path}")


def _json(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")
