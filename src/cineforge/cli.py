"""Cineforge command-line interface.

    cineforge new "<idea>" [--name N] [--style S]   scaffold a project
    cineforge run <name>                            run the full pipeline
    cineforge resume <name>                         resume after a crash/interrupt
    cineforge gui [--port P] [--no-browser]         launch the local GUI
    cineforge doctor                                verify torch/CUDA/ComfyUI/weights
    cineforge models                                show the model matrix for THIS gpu
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys

from .config import Config
from .config import _slug as slug
from .errors import CineforgeError
from .hardware import classify_vram, detect_gpus, primary_gpu, select_backend
from .models.matrix import ModelMatrix
from .state import Project, store


def _today() -> str:
    return _dt.date.today().isoformat()


def cmd_new(cfg: Config, args) -> int:
    name = args.name or slug(args.idea)[:40] or "project"
    project = Project(name=name, idea=args.idea, style=args.style, created=_today())
    project_dir = cfg.project_dir(name)
    cfg.ensure_dirs()
    store.save(project, project_dir)
    print(f"Created project {name!r} at {project_dir}")
    print(f"Next: cineforge run {name}")
    return 0


def _load(cfg: Config, name: str) -> Project:
    return store.load(cfg.project_dir(name))


def cmd_run(cfg: Config, args) -> int:
    from .pipeline import Coordinator

    project = _load(cfg, args.name)
    Coordinator(cfg).run(project)
    print(f"Done. Project state: {store.project_json_path(cfg.project_dir(args.name))}")
    return 0


def cmd_resume(cfg: Config, args) -> int:
    from .pipeline import Coordinator

    project = _load(cfg, args.name)
    Coordinator(cfg).resume(project)
    return 0


def cmd_gui(cfg: Config, args) -> int:
    from .gui.server import serve

    serve(cfg, port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_doctor(cfg: Config, args) -> int:
    from .scripts_verify import run_doctor

    report = run_doctor(cfg)
    ok = all(c["ok"] for c in report)
    for c in report:
        mark = "OK " if c["ok"] else "XX "
        print(f"[{mark}] {c['name']}: {c['detail']}")
    print("\nAll green." if ok else "\nSome checks failed — see docs/TROUBLESHOOTING.md")
    return 0 if ok else 1


def cmd_models(cfg: Config, args) -> int:
    gpus = detect_gpus()
    prim = primary_gpu(gpus)
    plan = select_backend(gpus)
    tier = cfg.tier_override or classify_vram(prim.vram_gb)
    matrix = ModelMatrix.load(cfg.data_dir / "model_matrix.json")

    print(f"GPU: {prim.name} ({prim.vram_gb} GB)  ->  tier {tier}")
    print(f"Runtime: {plan.runtime}/{plan.torch_channel}  quant: {plan.quant_pref}  license mode: {cfg.license_mode}")
    for w in plan.warnings:
        print(f"  ! {w}")
    print(f"\n{'subsystem':<12} {'model':<18} {'variant':<12} {'quant':<10} min_vram")
    print("-" * 66)
    for sub in matrix.subsystems():
        c = matrix.resolve(sub, tier, cfg.license_mode)
        print(f"{sub:<12} {c.model_id:<18} {c.variant:<12} {c.quant:<10} {c.min_vram_gb:g} GB")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cineforge", description="Fully-local AI animated-video studio")
    sub = p.add_subparsers(dest="command", required=True)

    p_new = sub.add_parser("new", help="scaffold a project from an idea")
    p_new.add_argument("idea")
    p_new.add_argument("--name")
    p_new.add_argument("--style", default="stylized 3D cartoon")
    p_new.set_defaults(func=cmd_new)

    p_run = sub.add_parser("run", help="run the pipeline")
    p_run.add_argument("name")
    p_run.set_defaults(func=cmd_run)

    p_res = sub.add_parser("resume", help="resume a project")
    p_res.add_argument("name")
    p_res.set_defaults(func=cmd_resume)

    p_gui = sub.add_parser("gui", help="launch the local GUI")
    p_gui.add_argument("--port", type=int, default=8765)
    p_gui.add_argument("--no-browser", action="store_true")
    p_gui.set_defaults(func=cmd_gui)

    p_doc = sub.add_parser("doctor", help="verify the install")
    p_doc.set_defaults(func=cmd_doctor)

    p_mod = sub.add_parser("models", help="show the resolved model matrix")
    p_mod.set_defaults(func=cmd_models)

    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    cfg = Config.load()
    try:
        return args.func(cfg, args)
    except KeyboardInterrupt:
        print("\nInterrupted. `cineforge resume <name>` to continue.", file=sys.stderr)
        return 130
    except CineforgeError as e:
        print(f"\n{type(e).__name__}: {e}", file=sys.stderr)
        print("See docs/TROUBLESHOOTING.md (or run `cineforge doctor`).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
