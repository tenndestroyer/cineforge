"""Director — script -> shot list with camera framing, movement, and pacing.

ViMax's Director lineage. Enriches the draft shots the Screenwriter produced with
concrete camera direction and timing that later stages (music/foley) key cues to.
"""

from __future__ import annotations

from .base import Agent, Context

SYSTEM = (
    "You are a film director planning coverage for an animated short. You choose clear, "
    "purposeful camera framing and movement per shot. Respond with STRICT JSON only."
)

TEMPLATE = """Here is a shot list. For EACH shot id, choose a camera framing and movement
and confirm a duration (seconds). Keep it readable and appropriate for a kids' show.

SHOTS:
{shots}

Return STRICT JSON:
{{"shots": [{{"id": "sh001", "camera": "wide/medium/close-up + angle", "movement": "static/pan/push-in/etc", "duration_s": 4}}]}}"""


class DirectorAgent(Agent):
    name = "director"

    def run(self, ctx: Context) -> Context:
        p = ctx.project
        shots = p.all_shots()
        if not shots:
            ctx.emit("warn", "no shots to direct")
            return ctx
        if ctx.llm is None:
            for sh in shots:
                sh.camera = sh.camera or "medium shot, eye level"
            ctx.emit("info", "no LLM: applied default framing")
            return ctx

        listing = "\n".join(f"- {sh.id}: {sh.description}" for sh in shots)
        try:
            result = ctx.llm.json(TEMPLATE.format(shots=listing), system=SYSTEM)
        except Exception as e:  # noqa: BLE001 - LLM step is best-effort; fall back to defaults
            ctx.emit("warn", f"director LLM step failed, using default framing: {e}")
            result = {}
        by_id = {sh.id: sh for sh in shots}
        covered = 0
        for entry in result.get("shots") or []:
            if not isinstance(entry, dict):
                continue
            sh = by_id.get(entry.get("id", ""))
            if not sh:
                continue
            sh.camera = f"{entry.get('camera', 'medium shot')}, {entry.get('movement', 'static')}"
            if entry.get("duration_s"):
                sh.duration_s = float(entry["duration_s"])
            covered += 1
        for sh in shots:  # fill any the LLM missed
            if not sh.camera:
                sh.camera = "medium shot, eye level, static"
        ctx.emit("info", f"Directed {covered}/{len(shots)} shots")
        return ctx
