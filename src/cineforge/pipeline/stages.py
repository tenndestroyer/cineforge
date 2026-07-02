"""The ordered pipeline: idea -> final master.

Each Stage is either a planning `agent` stage, a `generate` stage driven by a backend
subsystem, or a structural stage (`setup` / `assemble`). Data-driven so stages can be
reordered or inserted without touching the coordinator.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    name: str
    kind: str            # 'setup' | 'agent' | 'generate' | 'assemble'
    agent: str = ""      # for kind == 'agent'
    subsystem: str = ""  # for kind == 'generate'
    checkpointed: bool = True


STAGES: list[Stage] = [
    Stage("ingest", "setup"),
    Stage("screenwriting", "agent", agent="screenwriter"),
    Stage("direction", "agent", agent="director"),
    Stage("casting", "agent", agent="casting"),
    Stage("storyboard", "agent", agent="storyboard"),
    Stage("producing", "agent", agent="producer"),
    Stage("consistency_setup", "agent", agent="consistency"),
    Stage("keyframes", "generate", subsystem="image"),
    Stage("video", "generate", subsystem="video"),
    Stage("voice", "generate", subsystem="voice"),
    Stage("lipsync", "generate", subsystem="lipsync"),
    Stage("sound_design", "agent", agent="sound_designer"),
    Stage("sfx", "generate", subsystem="sfx"),
    Stage("music", "generate", subsystem="music"),
    Stage("enhance", "generate", subsystem="enhance"),
    Stage("edit", "agent", agent="editor"),
    Stage("qa", "agent", agent="qa"),
    Stage("master", "assemble"),
]

STAGE_NAMES: list[str] = [s.name for s in STAGES]


def stage_index(name: str) -> int:
    try:
        return STAGE_NAMES.index(name)
    except ValueError:
        return -1


def stage_by_name(name: str) -> Stage | None:
    return next((s for s in STAGES if s.name == name), None)
