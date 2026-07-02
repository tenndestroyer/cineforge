"""Atomic load/save of project.json."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from ..errors import CheckpointError
from .project import Project
from .schema import migrate, validate

PROJECT_FILE = "project.json"


def project_json_path(project_dir: Path) -> Path:
    return Path(project_dir) / PROJECT_FILE


def save(project: Project, project_dir: Path) -> Path:
    """Write project.json atomically (tmp file + os.replace) so a crash mid-write
    never corrupts the last-good checkpoint."""
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    target = project_json_path(project_dir)
    data = json.dumps(project.to_dict(), indent=2, ensure_ascii=False)

    fd, tmp = tempfile.mkstemp(dir=str(project_dir), prefix=".project.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except OSError as e:  # pragma: no cover - filesystem failure
        raise CheckpointError(f"could not write {target}: {e}") from e
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return target


def load(project_dir: Path) -> Project:
    path = project_json_path(project_dir)
    if not path.is_file():
        raise CheckpointError(f"no project.json in {project_dir}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CheckpointError(f"corrupt project.json: {e}") from e
    validate(raw)
    raw = migrate(raw)
    return Project.from_dict(raw)
