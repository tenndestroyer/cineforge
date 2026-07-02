"""LicenseGate — enforce data/licenses.json at model-selection time.

Safe mode (default) allows ONLY models flagged `safe` (Apache/MIT). Gated models
(revenue caps, territory exclusions, non-commercial, watermarks) are blocked in Safe
mode and require explicit per-project consent in Research mode. Territory-excluded
models are always blocked for that territory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ConfigError

OK = "ok"
REQUIRES_ACK = "requires_ack"
BLOCKED = "blocked"


@dataclass
class Verdict:
    status: str            # OK | REQUIRES_ACK | BLOCKED
    reason: str = ""
    terms: dict[str, Any] | None = None

    @property
    def allowed(self) -> bool:
        return self.status == OK


class LicenseGate:
    def __init__(self, data: dict[str, Any]) -> None:
        self.models: dict[str, dict] = data.get("models", {})

    @classmethod
    def load(cls, path: Path) -> LicenseGate:
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"licenses file not found: {path}")
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def info(self, model_id: str) -> dict[str, Any]:
        return self.models.get(model_id, {})

    def check(
        self,
        model_id: str,
        license_mode: str = "safe",
        acks: list[str] | None = None,
        territory: str | None = None,
    ) -> Verdict:
        acks = acks or []
        entry = self.models.get(model_id)
        # Unknown model -> treat as permissive (nothing to enforce), but say so.
        if entry is None:
            return Verdict(OK, reason=f"no license record for {model_id!r}; treated as permissive")

        if entry.get("safe"):
            return Verdict(OK, reason=entry.get("license", "permissive"))

        # Territory exclusion is absolute (e.g. HunyuanVideo excludes EU/UK/KR).
        excluded = [t.upper() for t in entry.get("excluded_territories", [])]
        if territory and territory.upper() in excluded:
            return Verdict(
                BLOCKED,
                reason=f"{model_id} license excludes territory {territory}",
                terms=entry,
            )

        if license_mode == "safe":
            return Verdict(
                BLOCKED,
                reason=f"{model_id} is a gated/non-permissive model; enable Research mode to use it",
                terms=entry,
            )

        # research mode: allowed only after explicit consent
        if model_id in acks:
            return Verdict(OK, reason="user acknowledged terms", terms=entry)
        return Verdict(REQUIRES_ACK, reason=_ack_reason(entry), terms=entry)


def _ack_reason(entry: dict[str, Any]) -> str:
    bits = []
    if entry.get("non_commercial"):
        bits.append("NON-COMMERCIAL use only")
    if entry.get("arr_cap"):
        bits.append(f"commercial use capped at {entry['arr_cap']} ARR")
    if entry.get("mau_cap"):
        bits.append(f"commercial use capped at {entry['mau_cap']} MAU")
    if entry.get("excluded_territories"):
        bits.append("excluded in " + ", ".join(entry["excluded_territories"]))
    if entry.get("watermark"):
        bits.append(f"embeds a {entry['watermark']} watermark")
    lic = entry.get("license", "custom license")
    return f"{lic}: " + "; ".join(bits) if bits else lic
