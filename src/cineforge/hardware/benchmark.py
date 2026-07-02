"""First-run self-calibration of per-stage timing.

Research across sources showed hardcoded ETAs are unreliable across GPUs/quants,
so Cineforge measures. In this scaffold `calibrate` returns conservative tier-based
defaults; the real implementation times a tiny generation per stage on first run and
persists the result to the project (Project.calibration). The GUI shows these as
honest ETAs rather than pretending renders are fast.
"""

from __future__ import annotations

# Rough seconds-per-unit fallbacks by tier (unit = 1 shot at that stage). Deliberately
# pessimistic so ETAs don't under-promise. Replaced by measured values after calibration.
_DEFAULTS: dict[str, dict[str, float]] = {
    "low": {"keyframe": 45, "video": 420, "voice": 12, "lipsync": 180, "sfx": 60, "music": 240, "enhance": 480},
    "mid": {"keyframe": 15, "video": 150, "voice": 6, "lipsync": 70, "sfx": 30, "music": 90, "enhance": 180},
    "high": {"keyframe": 8, "video": 70, "voice": 4, "lipsync": 40, "sfx": 18, "music": 55, "enhance": 90},
}


def default_etas(tier: str) -> dict[str, float]:
    return dict(_DEFAULTS.get(tier, _DEFAULTS["mid"]))


def calibrate(tier: str, stages: list[str] | None = None) -> dict[str, float]:
    """Return seconds-per-unit estimates for the given stages.

    Placeholder: returns tier defaults. TODO(v0.6): run a tiny real generation per
    stage and measure. Kept as a pure function so it's trivially testable.
    """
    etas = default_etas(tier)
    if stages is None:
        return etas
    return {s: etas.get(s, 60.0) for s in stages}
