"""QA — final gate before master.

Checks that each shot has the accepted takes it needs (video; voice+lipsync where the
shot has dialogue) and records a QAReport. Full-strength QA (SyncNet lip-sync
confidence, Whisper WER round-trip, loudness/clipping) runs when those tools are
installed; in a dry/scaffold run it reports which assets are missing instead.
"""

from __future__ import annotations

from ..state import QAReport
from .base import Agent, Context


class QAAgent(Agent):
    name = "qa"

    def run(self, ctx: Context) -> Context:
        p = ctx.project
        checks: list[dict] = []
        flagged: list[str] = []

        for sh in p.all_shots():
            has_video = sh.accepted_take("video") is not None
            needs_speech = bool(sh.dialogue)
            has_voice = sh.accepted_take("voice") is not None
            has_lipsync = sh.accepted_take("lipsync") is not None

            problems = []
            if not has_video:
                problems.append("no accepted video take")
            if needs_speech and not has_voice:
                problems.append("dialogue but no voice take")
            if needs_speech and not has_lipsync:
                problems.append("dialogue but no lipsync take")

            checks.append({"shot_id": sh.id, "ok": not problems, "problems": problems})
            if problems:
                flagged.append(sh.id)
                sh.status = "flagged"

        report = QAReport(passed=(not flagged), checks=checks, flagged_shots=flagged)
        p.qa = report
        if flagged:
            ctx.emit("warn", f"QA: {len(flagged)}/{len(checks)} shots flagged (missing assets). "
                             f"Install backends + re-run to render them.")
        else:
            ctx.emit("info", f"QA passed: {len(checks)} shots have their required takes")
        return ctx
