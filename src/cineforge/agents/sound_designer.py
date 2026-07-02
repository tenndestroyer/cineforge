"""SoundDesigner — plan foley / ambience / music cues and the mix.

Routes on-screen SFX to the video-conditioned foley backend, room-tone/ambience to
the ambience backend, and songs/score to the music backend, then defines the duck/mix
plan the master stage applies. Uses the LLM to author cue prompts when available.
"""

from __future__ import annotations

from ..state import AudioCue
from ..state.project import _new_id
from .base import Agent, Context

SYSTEM = "You are a sound designer for animation. You suggest concise foley and ambience cues. STRICT JSON only."

TEMPLATE = """For each shot, suggest one foley/SFX prompt (on-screen sounds) and note the ambience
(background) for its scene. Also suggest ONE music brief for the whole piece.

STYLE: {style}
SHOTS:
{shots}

Return STRICT JSON:
{{"shots": [{{"id":"sh001","foley":"...","ambience":"..."}}], "music": {{"style":"...","lyrics": null}}}}"""


class SoundDesignerAgent(Agent):
    name = "sound_designer"

    def run(self, ctx: Context) -> Context:
        p = ctx.project
        shots = p.all_shots()
        p.audio_cues = []
        by_id: dict[str, dict] = {}
        music_brief = {"style": f"gentle {p.style} underscore", "lyrics": None}

        if ctx.llm is not None and shots:
            listing = "\n".join(f"- {sh.id}: {sh.description}" for sh in shots)
            try:
                res = ctx.llm.json(TEMPLATE.format(style=p.style, shots=listing), system=SYSTEM)
                by_id = {e.get("id", ""): e for e in res.get("shots", [])}
                if isinstance(res.get("music"), dict):
                    music_brief.update({k: res["music"].get(k, music_brief.get(k)) for k in ("style", "lyrics")})
            except Exception as e:  # noqa: BLE001
                ctx.emit("warn", f"sound-design LLM step failed, using heuristics: {e}")

        clock = 0.0
        for sh in shots:
            info = by_id.get(sh.id, {})
            foley_prompt = info.get("foley") or f"foley for: {sh.description}"
            p.audio_cues.append(AudioCue(
                id=_new_id("fx", len(p.audio_cues) + 1), kind="foley",
                start_s=round(clock, 2), duration_s=sh.duration_s, prompt=foley_prompt,
                meta={"shot_id": sh.id},
            ))
            amb = info.get("ambience")
            if amb:
                p.audio_cues.append(AudioCue(
                    id=_new_id("amb", len(p.audio_cues) + 1), kind="ambience",
                    start_s=round(clock, 2), duration_s=sh.duration_s, prompt=amb,
                    meta={"shot_id": sh.id},
                ))
            clock += sh.duration_s

        total = round(clock, 2) or 30.0
        p.audio_cues.append(AudioCue(
            id=_new_id("mus", len(p.audio_cues) + 1), kind="music",
            start_s=0.0, duration_s=total, prompt=music_brief["style"],
            meta={"lyrics": music_brief.get("lyrics")},
        ))
        p.render_plan["mix"] = {
            "loudness_target_lufs": -14.0,   # streaming; -23 for broadcast
            "true_peak_dbtp": -1.5,
            "music_duck_db": -9.0,           # sidechain duck under dialogue
        }
        n_foley = sum(1 for c in p.audio_cues if c.kind == "foley")
        ctx.emit("info", f"Planned audio: {n_foley} foley cues + ambience + 1 music bed ({total}s)")
        return ctx
