"""Reusable best-of-N candidate selection.

Generate N candidates (varied seed/params), score each with a subsystem scorer
(CLAP for audio, ArcFace/CLIP/VLM for image/video, SyncNet for lipsync), keep the
winner. A candidate that raises is skipped, not fatal.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..logging_setup import get_logger

_log = get_logger("cineforge.best_of_n")


@dataclass
class Selection:
    winner: Any
    score: float | None
    candidates: list[tuple[Any, float | None]]


def best_of_n(
    gen_fn: Callable[[int], Any],
    scorer: Callable[[Any], float] | None,
    n: int,
) -> Selection:
    n = max(1, int(n))
    candidates: list[tuple[Any, float | None]] = []
    best = None
    best_score: float | None = None
    for i in range(n):
        try:
            cand = gen_fn(i)
        except Exception as e:  # noqa: BLE001 - one bad candidate shouldn't sink the shot
            _log.warning("best_of_n candidate %d failed: %s", i, e)
            continue
        score = None
        if scorer is not None:
            try:
                score = float(scorer(cand))
            except Exception as e:  # noqa: BLE001
                _log.warning("best_of_n scorer failed on candidate %d: %s", i, e)
                score = None
        candidates.append((cand, score))
        if best is None or (score is not None and (best_score is None or score > best_score)):
            best, best_score = cand, score
    if best is None:
        raise RuntimeError(f"best_of_n produced no usable candidate out of {n}")
    return Selection(best, best_score, candidates)
