"""Coordinator — the single run loop.

Drives STAGES in order: planning agents mutate the project; generation stages resolve
a model (matrix), enforce its license (gate), and render via the registered backend.
Everything checkpoints after each stage so a crash/OOM/interrupt resumes cleanly.

Honest "dry" behavior: on a fresh install (no ComfyUI/weights), generation stages
report exactly what they *would* render with which model, then continue — so the whole
orchestration runs and is inspectable before you commit to a 100 GB weight download.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..agents import Context, build_agents
from ..backends.base import (
    CAP_NATIVE_AUDIO,
    EnhanceRequest,
    FoleyRequest,
    ImageRequest,
    LipsyncRequest,
    MusicRequest,
    SpeechRequest,
    VideoRequest,
)
from ..backends.llm.ollama import OllamaLLM
from ..backends.llm.vlm import OllamaVLM
from ..config import Config
from ..errors import BackendError, ConfigError, NotInstalledError, VRAMError
from ..hardware import classify_vram, detect_gpus, primary_gpu, select_backend
from ..logging_setup import EventLog, get_logger
from ..models.licenses import BLOCKED, REQUIRES_ACK, LicenseGate
from ..models.matrix import ModelMatrix
from ..models.registry import BackendRegistry
from ..state import Project, Take, store
from ..state.project import _new_id
from . import checkpoint
from .stages import STAGES, stage_index

_log = get_logger("cineforge.coordinator")


class Coordinator:
    def __init__(self, cfg: Config, events: EventLog | None = None) -> None:
        self.cfg = cfg
        self.events = events or EventLog()
        self.matrix = ModelMatrix.load(cfg.data_dir / "model_matrix.json")
        self.gate = LicenseGate.load(cfg.data_dir / "licenses.json")
        self.agents = build_agents()

    # ---- context ----
    def build_context(self, project: Project) -> Context:
        gpus = detect_gpus()
        plan = select_backend(gpus)
        prim = primary_gpu(gpus)
        tier = self.cfg.tier_override or classify_vram(prim.vram_gb)
        project.tier = tier

        llm: OllamaLLM | None = OllamaLLM(
            self.cfg.ollama_url, self.matrix.resolve("llm", tier, self.cfg.license_mode).variant or "qwen2.5:7b"
        )
        vlm: OllamaVLM | None = OllamaVLM(self.cfg.ollama_url)
        if not llm.available():
            llm = None
            vlm = None

        return Context(
            project=project,
            config=self.cfg,
            events=self.events,
            llm=llm,
            vlm=vlm,
            tier=tier,
            license_mode=self.cfg.license_mode,
            matrix=self.matrix,
            gate=self.gate,
            plan=plan,
        )

    # ---- run / resume ----
    def run(self, project: Project, from_stage: str | None = None) -> Project:
        self.cfg.ensure_dirs()
        project_dir = self.cfg.project_dir(project.name)
        project_dir.mkdir(parents=True, exist_ok=True)
        ctx = self.build_context(project)

        start = stage_index(from_stage) if from_stage else 0
        if start < 0:
            start = 0
        total = len(STAGES)

        for i, stage in enumerate(STAGES[start:], start=start):
            ctx.stage = stage.name
            self.events.emit("stage", stage.name, f"stage {i + 1}/{total}: {stage.name}",
                             pct=round(i / total * 100, 1))
            completed = self._run_stage(ctx, stage)
            if completed:
                if stage.checkpointed:
                    checkpoint.save(project, project_dir, stage.name)
            else:
                # Stage blocked (e.g. backend not installed) and every downstream stage
                # depends on it. Persist partial progress WITHOUT advancing the stage
                # pointer, then STOP so `resume` re-attempts from exactly here — never
                # letting a later stage checkpoint past the failure.
                store.save(project, project_dir)
                self.events.emit("warn", stage.name,
                                 f"stopped at {stage.name!r} — install the backend(s), then "
                                 f"`cineforge resume {project.name}` to continue.")
                return project

        self.events.emit("info", "done", f"pipeline complete for {project.name!r}", pct=100.0)
        return project

    def resume(self, project: Project) -> Project:
        nxt = checkpoint.next_stage(project)
        if nxt is None:
            self.events.emit("info", "done", "nothing to resume; project already complete")
            return project
        self.events.emit("info", "resume", f"resuming at stage {nxt!r} (last done: {project.stage!r})")
        return self.run(project, from_stage=nxt)

    # ---- stage dispatch ----
    def _run_stage(self, ctx: Context, stage) -> bool:
        """Return True if the stage completed. Generate stages return False when they
        bail early (backend not installed) so the stage is NOT marked done."""
        if stage.kind == "setup":
            self._ingest(ctx)
            return True
        if stage.kind == "agent":
            self.agents[stage.agent].run(ctx)
            return True
        if stage.kind == "generate":
            return self._run_generate(ctx, stage.subsystem)
        if stage.kind == "assemble":
            self._run_master(ctx)
            return True
        raise ConfigError(f"unknown stage kind {stage.kind!r}")  # pragma: no cover

    def _ingest(self, ctx: Context) -> None:
        g = ctx.plan.gpu
        ctx.emit("info", f"GPU: {g.name} ({g.vram_gb} GB) -> tier {ctx.tier}; "
                         f"runtime {ctx.plan.runtime}/{ctx.plan.torch_channel}, quant {ctx.plan.quant_pref}")
        for w in ctx.plan.warnings:
            ctx.emit("warn", w)
        if ctx.llm is None:
            ctx.emit("warn", "Ollama not reachable — planning agents will use defaults or fail. "
                             "Install Ollama + pull a model, then `cineforge doctor`.")

    # ---- generation ----
    def _run_generate(self, ctx: Context, subsystem: str) -> bool:
        """Render a generation stage. Returns True if it finished (all items done, or
        nothing to do, or an intentional skip), False only if it BAILED because the
        backend isn't installed — so resume re-attempts this exact stage."""
        p = ctx.project
        project_dir = self.cfg.project_dir(p.name)
        try:
            choice = self.matrix.resolve(subsystem, ctx.tier, ctx.license_mode)
        except ConfigError as e:
            ctx.emit("warn", f"{subsystem}: no model in matrix ({e}); skipping")
            return True

        verdict = self.gate.check(choice.license_id, ctx.license_mode, p.license_acks)
        if verdict.status == BLOCKED:
            ctx.emit("warn", f"{subsystem}: {choice.model_id} blocked — {verdict.reason}; skipping")
            return True
        if verdict.status == REQUIRES_ACK:
            ctx.emit("warn", f"{subsystem}: {choice.model_id} needs consent — {verdict.reason}; "
                             f"acknowledge in the GUI to enable. Skipping.")
            return True

        try:
            backend = BackendRegistry.get(choice.model_id, ctx.config, choice)
        except BackendError as e:
            ctx.emit("error", f"{subsystem}: {e}")
            return True

        items = self._items_for(ctx, subsystem, backend)
        if not items:
            ctx.emit("info", f"{subsystem}: nothing to generate (already done or no inputs)")
            return True

        ctx.emit("info", f"{subsystem}: {choice.model_id} [{choice.variant} {choice.quant}] — {len(items)} item(s)")
        backend.load()
        done = 0
        try:
            for label, request, store_fn in items:
                try:
                    result = backend.generate(request)
                except NotInstalledError as e:
                    ctx.emit("warn", f"{subsystem}: not installed — {len(items) - done} item(s) pending. {e}")
                    return False   # transient: resume should re-attempt this stage
                except VRAMError as e:
                    ctx.emit("warn", f"{subsystem}: OOM on {label} ({e}); skipping this item")
                    continue
                except BackendError as e:
                    ctx.emit("error", f"{subsystem}: {label} failed ({e})")
                    continue
                store_fn(result)
                done += 1
                # Persist each finished item so a crash/OOM mid-stage resumes without
                # re-rendering completed shots (the whole point of long-form resume).
                store.save(p, project_dir)
        finally:
            backend.unload()
        ctx.emit("info", f"{subsystem}: generated {done}/{len(items)}")
        return True

    def _items_for(self, ctx: Context, subsystem: str, backend) -> list[tuple]:
        """Build (label, request, store_fn) tuples for a generation stage.

        Every branch SKIPS work that already has an accepted take, so a resumed or
        re-run stage only (re)generates the shots/lines/cues that are still missing.
        """
        p = ctx.project
        render = p.render_plan.get("render", {"width": 768, "height": 512, "frames": 97, "fps": 24})
        items: list[tuple] = []

        def add_take(shot, kind, result, accepted=True, extra_meta=None):
            meta = dict(result.meta)
            if extra_meta:
                meta.update(extra_meta)
            shot.takes.append(Take(
                id=_new_id("t", len(shot.takes) + 1), shot_id=shot.id, kind=kind,
                path=result.path, accepted=accepted, score=result.score, meta=meta,
            ))

        if subsystem == "image":
            for shot in p.all_shots():
                if shot.accepted_take("keyframe"):
                    continue
                prompt = shot.keyframe_prompt or shot.description
                refs = [r for c in p.characters for r in c.reference_images]
                req = ImageRequest(prompt=prompt, refs=refs,
                                   width=render["width"], height=render["height"],
                                   steps=render.get("steps"), cfg=render.get("cfg"),
                                   checkpoint=render.get("checkpoint"))
                items.append((f"keyframe {shot.id}", req,
                              lambda res, sh=shot: add_take(sh, "keyframe", res)))

        elif subsystem == "video":
            for shot in p.all_shots():
                if shot.accepted_take("video"):
                    continue
                kf = shot.accepted_take("keyframe")
                req = VideoRequest(
                    prompt=shot.description, keyframe=(kf.path if kf else None),
                    frames=render["frames"], fps=render["fps"],
                    width=render["width"], height=render["height"],
                    duration_s=shot.duration_s,
                    native_audio=backend.supports(CAP_NATIVE_AUDIO),
                )
                items.append((f"video {shot.id}", req, lambda res, sh=shot: add_take(sh, "video", res)))

        elif subsystem == "voice":
            for shot in p.all_shots():
                done_lines = {t.meta.get("line") for t in shot.takes if t.kind == "voice" and t.accepted}
                for j, d in enumerate(shot.dialogue):
                    if j in done_lines:  # this specific line already voiced
                        continue
                    ch = p.character(d.character)
                    prof = ch.voice_profile if ch else {}
                    req = SpeechRequest(
                        text=d.line, emotion=d.emotion,
                        exaggeration=float(prof.get("exaggeration", 0.5)),
                        ref_clip=(ch.voice_ref if ch else None),
                    )
                    items.append((f"voice {shot.id}#{j}", req,
                                  lambda res, sh=shot, jj=j: add_take(sh, "voice", res, extra_meta={"line": jj})))

        elif subsystem == "lipsync":
            for shot in p.all_shots():
                if shot.accepted_take("lipsync"):
                    continue
                vid = shot.accepted_take("video")
                voi = shot.accepted_take("voice")
                if shot.dialogue and vid and voi:
                    req = LipsyncRequest(face_source=vid.path, audio=voi.path)
                    items.append((f"lipsync {shot.id}", req,
                                  lambda res, sh=shot: add_take(sh, "lipsync", res)))

        elif subsystem == "sfx":
            by_shot = {sh.id: sh for sh in p.all_shots()}
            for cue in [c for c in p.audio_cues if c.kind in ("foley", "ambience")]:
                if cue.path:
                    continue
                shot = by_shot.get(cue.meta.get("shot_id"))
                vt = shot.accepted_take("video") if shot else None
                # Video-conditioned foley uses the shot's picture when available.
                req = FoleyRequest(video=(vt.path if vt else ""), prompt=cue.prompt, seconds=cue.duration_s)
                items.append((f"sfx {cue.id}", req, lambda res, cu=cue: _set_cue_path(cu, res)))

        elif subsystem == "music":
            for cue in [c for c in p.audio_cues if c.kind == "music"]:
                if cue.path:
                    continue
                lyrics = cue.meta.get("lyrics")
                req = MusicRequest(style=cue.prompt, lyrics=lyrics, duration_s=cue.duration_s,
                                   instrumental=not lyrics)
                items.append((f"music {cue.id}", req, lambda res, cu=cue: _set_cue_path(cu, res)))

        elif subsystem == "enhance":
            for shot in p.all_shots():
                if shot.accepted_take("enhance"):
                    continue
                vid = shot.accepted_take("video")
                if vid:
                    req = EnhanceRequest(clip=vid.path, interpolate_factor=2)
                    items.append((f"enhance {shot.id}", req, lambda res, sh=shot: add_take(sh, "enhance", res)))

        return items

    # ---- master ----
    def _run_master(self, ctx: Context) -> None:
        p = ctx.project
        project_dir = self.cfg.project_dir(p.name)
        shots = p.all_shots()
        ready = [sh for sh in shots if (sh.accepted_take("enhance") or sh.accepted_take("video"))]
        p.render_plan.setdefault("master", {})
        if not shots:
            p.render_plan["master"]["status"] = "empty"
            ctx.emit("warn", "Master: no shots to assemble")
            return
        if len(ready) < len(shots):
            p.render_plan["master"]["status"] = "pending"
            ctx.emit("warn", f"Master: {len(ready)}/{len(shots)} shots rendered. Install backends "
                             f"(run setup) and re-run to produce the final file.")
            return
        if not shutil.which("ffmpeg"):
            p.render_plan["master"]["status"] = "pending"
            ctx.emit("warn", "Master: all shots ready but ffmpeg is not on PATH; install it to render the final file.")
            return
        try:
            out = self._assemble_master(ctx, project_dir)
        except (OSError, subprocess.SubprocessError) as e:
            p.render_plan["master"]["status"] = "error"
            ctx.emit("error", f"Master assembly failed: {e}")
            return
        p.render_plan["master"]["status"] = "ready"
        p.render_plan["master"]["path"] = str(out)
        ctx.emit("info", f"Master: wrote {out}")

    def _assemble_master(self, ctx: Context, project_dir: Path) -> Path:
        """Concatenate each shot's final clip (enhance take preferred, else video) in
        timeline order and normalize loudness with ffmpeg. Full stem mixing (separate
        music/sfx/dialogue buses) is a later refinement; this produces the assembled cut."""
        p = ctx.project
        by_id = {sh.id: sh for sh in p.all_shots()}
        timeline = p.render_plan.get("timeline") or []
        ordered = [by_id[t["shot_id"]] for t in timeline if t.get("shot_id") in by_id] or p.all_shots()

        def clip_path(sh) -> Path:
            take = sh.accepted_take("enhance") or sh.accepted_take("video")
            path = Path(take.path)
            return path if path.is_absolute() else (project_dir / path)

        list_file = project_dir / "_concat.txt"
        list_file.write_text(
            "".join(f"file '{clip_path(sh).as_posix()}'\n" for sh in ordered), encoding="utf-8"
        )
        target = p.render_plan.get("mix", {}).get("loudness_target_lufs", -14.0)
        out = project_dir / "master.mp4"
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-af", f"loudnorm=I={target}:TP=-1.5:LRA=11",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return out


def _set_cue_path(cue, result) -> None:
    cue.path = result.path
