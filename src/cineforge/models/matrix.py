"""Resolve (subsystem x VRAM-tier x license_mode) -> a concrete model choice.

The mapping lives in data/model_matrix.json (an updateable data file, NOT code) so
the fast model churn of 2026 never forces a code release — you edit JSON and pin a
new revision. See docs/MODELS.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import ConfigError

TIERS = ("low", "mid", "high")


@dataclass
class ModelChoice:
    subsystem: str
    tier: str
    model_id: str
    license_id: str = ""
    variant: str = ""
    quant: str = ""
    repo: str = ""
    workflow: str = ""            # ComfyUI workflow template name
    min_vram_gb: float = 0.0
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class ModelMatrix:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.version = data.get("version", 1)

    @classmethod
    def load(cls, path: Path) -> ModelMatrix:
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"model matrix not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ConfigError(f"model matrix is not valid JSON: {e}") from e
        return cls(data)

    def subsystems(self) -> list[str]:
        return sorted(self.data.get("subsystems", {}).keys())

    def resolve(self, subsystem: str, tier: str, license_mode: str = "safe") -> ModelChoice:
        if tier not in TIERS:
            raise ConfigError(f"unknown tier {tier!r}")
        subs = self.data.get("subsystems", {})
        if subsystem not in subs:
            raise ConfigError(f"no matrix entry for subsystem {subsystem!r}")
        tiers = subs[subsystem].get("tiers", {})
        if tier not in tiers:
            # Graceful degrade toward the CLOSEST tier: prefer the nearest LOWER tier
            # (safer for VRAM), and only fall UPWARD if no lower tier is defined.
            idx = TIERS.index(tier)
            order = list(reversed(TIERS[:idx])) + list(TIERS[idx + 1:])
            fallback = next((t for t in order if t in tiers), None)
            if fallback is None:
                raise ConfigError(f"subsystem {subsystem!r} has no tiers")
            tier = fallback
        entry = tiers[tier]
        # 'research' mode may unlock a gated 'best' pick; 'safe' always uses 'safe'.
        chosen = entry.get("best") if (license_mode == "research" and "best" in entry) else entry.get("safe")
        if chosen is None:
            chosen = entry.get("safe") or entry.get("best")
        if chosen is None:
            raise ConfigError(f"{subsystem}/{tier} has neither 'safe' nor 'best' entry")
        return ModelChoice(
            subsystem=subsystem,
            tier=tier,
            model_id=chosen["model_id"],
            license_id=chosen.get("license_id", chosen["model_id"]),
            variant=chosen.get("variant", ""),
            quant=chosen.get("quant", ""),
            repo=chosen.get("repo", ""),
            workflow=chosen.get("workflow", ""),
            min_vram_gb=float(chosen.get("min_vram_gb", 0)),
            notes=chosen.get("notes", ""),
            extra=chosen.get("extra", {}),
        )

    def resolve_all(self, tier: str, license_mode: str = "safe") -> dict[str, ModelChoice]:
        return {s: self.resolve(s, tier, license_mode) for s in self.subsystems()}
