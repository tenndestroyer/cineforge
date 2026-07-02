"""Editor — assemble the timeline, transitions, and the mix plan for mastering."""

from __future__ import annotations

from .base import Agent, Context


class EditorAgent(Agent):
    name = "editor"

    def run(self, ctx: Context) -> Context:
        p = ctx.project
        timeline = []
        clock = 0.0
        for scene in p.scenes:
            for i, shot in enumerate(scene.shots):
                # simple, safe defaults: cut between shots, gentle fade at scene starts
                transition = "fade" if i == 0 else "cut"
                timeline.append({
                    "shot_id": shot.id,
                    "scene_id": scene.id,
                    "start_s": round(clock, 2),
                    "duration_s": shot.duration_s,
                    "transition_in": transition,
                })
                clock += shot.duration_s
        p.render_plan["timeline"] = timeline
        p.render_plan["total_duration_s"] = round(clock, 2)
        p.render_plan.setdefault("mix", {})
        p.render_plan["mix"].update({
            "buses": ["dialogue", "music", "sfx", "ambience"],
            "dialogue_priority": True,
        })
        ctx.emit("info", f"Assembled timeline: {len(timeline)} shots, {round(clock,1)}s total")
        return ctx
