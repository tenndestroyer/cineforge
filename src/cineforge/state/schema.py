"""Versioned validation + migration for project.json."""

from __future__ import annotations

from typing import Any

from ..errors import CheckpointError
from .project import SCHEMA_VERSION

REQUIRED_KEYS = ("name", "schema_version")


def validate(raw: dict[str, Any]) -> None:
    if not isinstance(raw, dict):
        raise CheckpointError("project state must be a JSON object")
    for key in REQUIRED_KEYS:
        if key not in raw:
            raise CheckpointError(f"project state missing required key: {key!r}")
    ver = raw.get("schema_version")
    if not isinstance(ver, int) or ver < 1:
        raise CheckpointError(f"invalid schema_version: {ver!r}")
    if ver > SCHEMA_VERSION:
        raise CheckpointError(
            f"project schema_version {ver} is newer than this Cineforge ({SCHEMA_VERSION}); upgrade Cineforge"
        )


def migrate(raw: dict[str, Any]) -> dict[str, Any]:
    """Bring an older project dict up to the current schema. No-op for v1."""
    # Future: while ver < SCHEMA_VERSION: apply migration step; ver += 1; then set to ver.
    raw["schema_version"] = SCHEMA_VERSION
    return raw
