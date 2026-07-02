"""Consistency — the identity auditor ("secret sauce").

Scores a candidate take against a character's canonical reference using the local
VLM judge (and, when available, face/embedding similarity). Rejects drifted takes and
asks for regeneration. It is a reject/retry MITIGATION, never a guarantee — cross-shot
character consistency is unsolved industry-wide (see docs/QUALITY_CEILING.md).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..state import Character
from .base import Agent, Context

DEFAULT_THRESHOLD = 0.62


@dataclass
class Verdict:
    accept: bool
    score: float | None
    reason: str


class ConsistencyAgent(Agent):
    name = "consistency"

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold

    def run(self, ctx: Context) -> Context:
        # Not a standalone stage; it primes the threshold and is invoked per-take by
        # the generation stages / best_of_n. Recording it keeps the run auditable.
        ctx.data["consistency_threshold"] = self.threshold
        ctx.emit("info", f"Consistency auditor ready (threshold {self.threshold})")
        return ctx

    def audit(self, ctx: Context, take_path: str, character: Character | None) -> Verdict:
        """Score `take_path` against `character.canonical_ref`."""
        if character is None or not character.canonical_ref:
            return Verdict(True, None, "no canonical reference to compare against")
        if ctx.vlm is None:
            return Verdict(True, None, "no VLM judge available; accepting without audit")
        try:
            score = ctx.vlm.score(take_path, character.canonical_ref, criteria="same character identity")
        except Exception as e:  # noqa: BLE001 - audit failure must not crash the run
            return Verdict(True, None, f"audit skipped ({e})")
        ok = score >= self.threshold
        return Verdict(ok, score, "match" if ok else f"identity drift (score {score:.2f} < {self.threshold})")
