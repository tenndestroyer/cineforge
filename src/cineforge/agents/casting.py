"""Casting — character bible -> asset + voice plan.

Decides, per character, whether identity should be held by a locally-trained LoRA
(characters that appear a lot) or by multi-reference conditioning (bit parts), and
assigns a voice profile the voice stage will use. Actual LoRA training is a separate
long GPU job (scripts/train_character_lora.py); here we plan and register intent.
"""

from __future__ import annotations

from .base import Agent, Context

LORA_SHOT_THRESHOLD = 3  # appears in >= this many shots -> worth a dedicated LoRA

SYSTEM = "You cast voices and design reference looks for animated characters. STRICT JSON only."

TEMPLATE = """For each character, give a visual 'reference prompt' (for an image model to make a
character sheet) and a voice profile.

CHARACTERS:
{chars}

Return STRICT JSON:
{{"characters": [{{"name": "Name", "reference_prompt": "...", "voice": {{"style": "warm/energetic/gruff/...", "exaggeration": 0.5}}}}]}}"""


class CastingAgent(Agent):
    name = "casting"

    def run(self, ctx: Context) -> Context:
        p = ctx.project
        if not p.characters:
            ctx.emit("warn", "no characters to cast")
            return ctx

        # count appearances to choose lora vs multi-reference
        appearances: dict[str, int] = {c.id: 0 for c in p.characters}
        for sh in p.all_shots():
            speaking = {d.character for d in sh.dialogue}
            for cid in speaking:
                if cid in appearances:
                    appearances[cid] += 1

        llm_by_name: dict[str, dict] = {}
        if ctx.llm is not None:
            chars = "\n".join(f"- {c.name}: {c.description}" for c in p.characters)
            try:
                res = ctx.llm.json(TEMPLATE.format(chars=chars), system=SYSTEM)
                for c in res.get("characters", []):
                    llm_by_name[c.get("name", "").strip().lower()] = c
            except Exception as e:  # noqa: BLE001 - casting must not abort the run
                ctx.emit("warn", f"casting LLM step failed, using heuristics: {e}")

        plan = {}
        for c in p.characters:
            info = llm_by_name.get(c.name.strip().lower(), {})
            ref_prompt = info.get("reference_prompt") or f"{c.description} character reference sheet, {p.style}"
            voice = info.get("voice") or {}
            c.voice_profile = {
                "style": voice.get("style", "neutral"),
                "exaggeration": float(voice.get("exaggeration", 0.5)),
            }
            strategy = "lora" if appearances.get(c.id, 0) >= LORA_SHOT_THRESHOLD else "multi_reference"
            plan[c.id] = {
                "name": c.name,
                "reference_prompt": ref_prompt,
                "identity_strategy": strategy,
                "appearances": appearances.get(c.id, 0),
            }
        p.render_plan["casting"] = plan
        n_lora = sum(1 for v in plan.values() if v["identity_strategy"] == "lora")
        ctx.emit("info", f"Cast {len(p.characters)} characters ({n_lora} need a LoRA, rest multi-reference)")
        return ctx
