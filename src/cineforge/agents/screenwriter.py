"""Screenwriter — idea -> script, generated in CHUNKS for long-form.

A single monolithic LLM call cannot reliably produce a coherent ~300-shot (15-20 min)
script on a local 7B model — it blows past the model's reliable output length and
context window. So the Screenwriter works in two chunked passes:

  1. Outline: logline, characters, and a per-scene brief (bounded to MAX_SCENES).
  2. Per scene: fill that scene's shots + dialogue in its own bounded call.

This keeps every request inside the model's context window and lets the whole thing
scale to feature length. The Director later refines shots; Storyboard adds visuals.
"""

from __future__ import annotations

import math

from ..state import Character, Dialogue, Scene, Shot
from ..state.project import _new_id
from .base import Agent, Context

MAX_SCENES = 40
MAX_SHOTS_TOTAL = 600
MAX_SHOTS_PER_SCENE = 60

SYSTEM = (
    "You are a professional children's-animation screenwriter. You write warm, funny, "
    "age-appropriate stories with clear structure, a gentle lesson, and strong visual "
    "moments. You always respond with STRICT JSON and nothing else."
)

OUTLINE_TEMPLATE = """Plan a short animated episode from this idea. Return a scene OUTLINE only.

IDEA: {idea}
STYLE: {style}
Plan about {n_scenes} scenes for roughly {n_shots} shots total.

Return STRICT JSON:
{{
  "logline": "one sentence",
  "characters": [{{"name": "Name", "description": "look + personality in one sentence"}}],
  "scenes": [{{"heading": "INT./EXT. LOCATION - TIME", "description": "what happens (1-2 sentences)", "target_shots": 4}}]
}}"""

SCENE_TEMPLATE = """Write the shots for THIS scene of "{logline}" (style: {style}).

SCENE: {heading}
WHAT HAPPENS: {description}
CHARACTERS: {chars}

Produce up to {max_shots} shots. Each shot is ONE clear visual action (a 4-8 second clip).
Return STRICT JSON:
{{"shots": [{{"description": "what the camera sees", "duration_s": 4,
  "dialogue": [{{"character": "Name", "line": "spoken line", "emotion": "happy|sad|excited|scared|curious|neutral"}}]}}]}}
Use only the listed characters. Keep dialogue short."""


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


class ScreenwriterAgent(Agent):
    name = "screenwriter"

    def run(self, ctx: Context) -> Context:
        llm = ctx.require_llm()
        p = ctx.project
        n_scenes = _clamp(int(p.render_plan.get("target_scenes", ctx.data.get("target_scenes", 3))), 1, MAX_SCENES)
        n_shots = _clamp(int(p.render_plan.get("target_shots", ctx.data.get("target_shots", 12))), 1, MAX_SHOTS_TOTAL)
        ctx.emit("info", f"Outlining ~{n_scenes} scenes (~{n_shots} shots) for: {p.idea!r}")

        outline = llm.json(
            OUTLINE_TEMPLATE.format(idea=p.idea, style=p.style, n_scenes=n_scenes, n_shots=n_shots),
            system=SYSTEM,
        )
        p.logline = outline.get("logline") or p.logline

        # --- characters (from the outline) ---
        name_to_id: dict[str, str] = {}
        p.characters = []
        for i, c in enumerate(outline.get("characters") or [], start=1):
            if not isinstance(c, dict):
                continue
            cid = _new_id("c", i)
            name = c.get("name", f"Character {i}")
            p.characters.append(Character(id=cid, name=name, description=c.get("description", "")))
            name_to_id[name.strip().lower()] = cid

        def resolve_char(name: str) -> str:
            key = (name or "").strip().lower()
            if key in name_to_id:
                return name_to_id[key]
            cid = _new_id("c", len(p.characters) + 1)
            p.characters.append(Character(id=cid, name=name or "Unknown", description=""))
            name_to_id[key] = cid
            ctx.emit("warn", f"dialogue referenced unlisted character {name!r}; added it")
            return cid

        scenes_outline = [s for s in (outline.get("scenes") or []) if isinstance(s, dict)][:n_scenes]
        if not scenes_outline:
            ctx.emit("warn", "outline produced no scenes; creating a single scene from the idea")
            scenes_outline = [{"heading": "SCENE 1", "description": p.idea,
                               "target_shots": min(n_shots, MAX_SHOTS_PER_SCENE)}]

        chars_str = ", ".join(c.name for c in p.characters) or "the characters"
        per_scene_default = max(1, math.ceil(n_shots / len(scenes_outline)))

        # --- per-scene shot generation (chunked) ---
        p.scenes = []
        shot_count = 0
        for si, so in enumerate(scenes_outline, start=1):
            scene = Scene(id=_new_id("s", si), index=si,
                          heading=so.get("heading", ""), description=so.get("description", ""))
            if shot_count >= n_shots:
                p.scenes.append(scene)  # keep the beat even if we hit the shot budget
                continue
            cap = _clamp(int(so.get("target_shots", per_scene_default) or per_scene_default), 1, MAX_SHOTS_PER_SCENE)
            cap = min(cap, n_shots - shot_count)
            try:
                sc = llm.json(
                    SCENE_TEMPLATE.format(logline=p.logline, style=p.style, heading=scene.heading,
                                          description=scene.description, chars=chars_str, max_shots=cap),
                    system=SYSTEM,
                )
            except Exception as e:  # noqa: BLE001 - one bad scene shouldn't sink the script
                ctx.emit("warn", f"scene {si} generation failed ({e}); skipping its shots")
                sc = {}
            for sh in (sc.get("shots") or [])[:cap]:
                if not isinstance(sh, dict):
                    continue
                shot_count += 1
                shot = Shot(id=_new_id("sh", shot_count), scene_id=scene.id, index=shot_count,
                            description=sh.get("description", ""), duration_s=float(sh.get("duration_s", 4) or 4))
                for d in (sh.get("dialogue") or []):
                    if not isinstance(d, dict):
                        continue
                    shot.dialogue.append(Dialogue(character=resolve_char(d.get("character", "")),
                                                  line=d.get("line", ""), emotion=d.get("emotion", "neutral")))
                scene.shots.append(shot)
            p.scenes.append(scene)
            ctx.emit("info", f"scene {si}/{len(scenes_outline)}: {len(scene.shots)} shots ({shot_count} total)")

        p.script = _render_script_text(p)
        ctx.emit("info", f"Script: {len(p.scenes)} scenes, {shot_count} shots, {len(p.characters)} characters")
        return ctx


def _render_script_text(project) -> str:
    lines = [f"LOGLINE: {project.logline}", ""]
    for scene in project.scenes:
        lines.append(f"{scene.index}. {scene.heading}")
        lines.append(scene.description)
        for shot in scene.shots:
            lines.append(f"  [{shot.id}] {shot.description}")
            for d in shot.dialogue:
                name = next((c.name for c in project.characters if c.id == d.character), d.character)
                lines.append(f"      {name} ({d.emotion}): {d.line}")
        lines.append("")
    return "\n".join(lines)
