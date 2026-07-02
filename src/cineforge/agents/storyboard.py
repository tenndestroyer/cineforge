"""Storyboard — shot -> keyframe prompt + pose/style brief for the image stage."""

from __future__ import annotations

from .base import Agent, Context

SYSTEM = (
    "You are a storyboard artist. For each shot you write a vivid image-generation prompt "
    "for the hero keyframe and a short pose/staging brief. STRICT JSON only."
)

TEMPLATE = """Style: {style}
Characters: {chars}

For each shot, write a 'keyframe_prompt' (a detailed prompt for an image model to render
the shot's hero frame, mentioning the character(s) by name and the setting) and a 'pose'
brief (body pose / staging, one line).

SHOTS:
{shots}

Return STRICT JSON:
{{"shots": [{{"id": "sh001", "keyframe_prompt": "...", "pose": "..."}}]}}"""


class StoryboardAgent(Agent):
    name = "storyboard"

    def run(self, ctx: Context) -> Context:
        p = ctx.project
        shots = p.all_shots()
        if not shots:
            return ctx
        chars = ", ".join(c.name for c in p.characters) or "the characters"

        if ctx.llm is None:
            for sh in shots:
                sh.keyframe_prompt = sh.keyframe_prompt or f"{sh.description}, {p.style}"
                sh.pose = sh.pose or "natural staging"
            ctx.emit("info", "no LLM: applied default keyframe prompts")
            return ctx

        listing = "\n".join(f"- {sh.id}: {sh.description} (camera: {sh.camera})" for sh in shots)
        try:
            result = ctx.llm.json(
                TEMPLATE.format(style=p.style, chars=chars, shots=listing), system=SYSTEM
            )
        except Exception as e:  # noqa: BLE001 - best-effort; fall back to default prompts
            ctx.emit("warn", f"storyboard LLM step failed, using default prompts: {e}")
            result = {}
        by_id = {sh.id: sh for sh in shots}
        n = 0
        for entry in result.get("shots") or []:
            if not isinstance(entry, dict):
                continue
            sh = by_id.get(entry.get("id", ""))
            if not sh:
                continue
            sh.keyframe_prompt = entry.get("keyframe_prompt", f"{sh.description}, {p.style}")
            sh.pose = entry.get("pose", "natural staging")
            n += 1
        for sh in shots:
            if not sh.keyframe_prompt:
                sh.keyframe_prompt = f"{sh.description}, {p.style}"
        ctx.emit("info", f"Storyboarded {n}/{len(shots)} shots")
        return ctx
