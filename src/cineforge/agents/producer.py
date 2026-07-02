"""Producer — resource planner. ViMax's Producer lineage.

Sets the per-shot quality/VRAM dial (resolution, frame count, best-of-N) from the
detected tier and turns the calibrated per-stage ETAs into a total time estimate.
Pure logic — no LLM.
"""

from __future__ import annotations

from ..hardware.benchmark import default_etas
from .base import Agent, Context

# Per-tier render settings. Deliberately conservative on the low tier (the honest
# "short, softer, lower-res" reality).
TIER_RENDER = {
    "low":  {"width": 512, "height": 384, "frames": 49,  "fps": 24, "best_of_n": 1},
    "mid":  {"width": 768, "height": 512, "frames": 97,  "fps": 24, "best_of_n": 2},
    "high": {"width": 1024, "height": 576, "frames": 121, "fps": 24, "best_of_n": 3},
}


class ProducerAgent(Agent):
    name = "producer"

    def run(self, ctx: Context) -> Context:
        p = ctx.project
        tier = ctx.tier
        render = dict(TIER_RENDER.get(tier, TIER_RENDER["mid"]))
        # Apply user quality overrides from the GUI (resolution, steps, model, etc.).
        override = p.render_plan.get("quality") or {}
        for k in ("width", "height", "frames", "fps", "best_of_n", "steps", "cfg", "checkpoint"):
            if override.get(k) is not None:
                render[k] = override[k]
        p.render_plan["render"] = render
        p.render_plan["tier"] = tier
        p.render_plan["license_mode"] = ctx.license_mode

        # ETA: sum of per-stage seconds * counts, using calibration if present.
        etas = dict(default_etas(tier))
        etas.update(p.calibration)  # measured values win
        shots = p.all_shots()
        n_shots = max(1, len(shots))
        # best-of-N generation is a v0.3 quality lever the coordinator does not yet
        # execute, so the ETA reflects ONE pass per shot (render['best_of_n'] is kept
        # for when it is wired). This keeps the estimate honest, not 2-3x inflated.
        n = n_shots
        est = {
            "keyframe": etas.get("keyframe", 15) * n,
            "video": etas.get("video", 150) * n,
            "voice": etas.get("voice", 6) * sum(len(sh.dialogue) for sh in shots),
            "lipsync": etas.get("lipsync", 70) * sum(1 for sh in shots if sh.dialogue),
            "sfx": etas.get("sfx", 30) * n_shots,
            "music": etas.get("music", 90) * max(1, len(p.scenes)),
            "enhance": etas.get("enhance", 180) * n_shots,
        }
        total_min = round(sum(est.values()) / 60.0, 1)
        p.render_plan["eta_seconds_by_stage"] = est
        p.render_plan["eta_total_minutes"] = total_min
        if total_min > 180:
            ctx.emit("warn", f"Estimated ~{round(total_min / 60, 1)} h of GPU time for {n_shots} shots "
                             f"at tier {tier}. Long-form runs unattended with checkpoint/resume — "
                             f"budget accordingly.")
        ctx.emit(
            "info",
            f"Render plan: tier={tier} {render['width']}x{render['height']}; "
            f"est. ~{total_min} min for {n_shots} shots",
        )
        return ctx
