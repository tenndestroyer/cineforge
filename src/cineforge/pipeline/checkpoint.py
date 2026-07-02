"""Checkpoint/resume built on the atomic project.json save.

`project.stage` records the last COMPLETED stage. After every stage the coordinator
calls `save`, so a crash/OOM/interrupt resumes at the next stage without redoing
finished work (rendered takes already live in the project + on disk).
"""

from __future__ import annotations

from pathlib import Path

from ..state import Project, store
from .stages import STAGE_NAMES, stage_index


def save(project: Project, project_dir: Path, stage: str) -> None:
    project.stage = stage
    store.save(project, project_dir)


def load_latest_stage(project_dir: Path) -> str:
    """Return the last completed stage recorded in the saved project (or 'ingest')."""
    project = store.load(project_dir)
    return project.stage or "ingest"


def is_done(project: Project, stage: str) -> bool:
    return stage_index(project.stage) >= stage_index(stage)


def next_stage(project: Project) -> str | None:
    idx = stage_index(project.stage)
    if idx < 0:
        return STAGE_NAMES[0]
    if idx + 1 >= len(STAGE_NAMES):
        return None
    return STAGE_NAMES[idx + 1]
