"""The agentic director's brain: nine specialized agents over a shared Context.

`AGENTS` maps an agent name to its class; `build_agents()` instantiates them.
"""

from __future__ import annotations

from .base import Agent, Context
from .casting import CastingAgent
from .consistency import ConsistencyAgent
from .director import DirectorAgent
from .editor import EditorAgent
from .producer import ProducerAgent
from .qa import QAAgent
from .screenwriter import ScreenwriterAgent
from .sound_designer import SoundDesignerAgent
from .storyboard import StoryboardAgent

AGENTS: dict[str, type[Agent]] = {
    "screenwriter": ScreenwriterAgent,
    "director": DirectorAgent,
    "casting": CastingAgent,
    "storyboard": StoryboardAgent,
    "producer": ProducerAgent,
    "consistency": ConsistencyAgent,
    "sound_designer": SoundDesignerAgent,
    "editor": EditorAgent,
    "qa": QAAgent,
}


def build_agents() -> dict[str, Agent]:
    return {name: cls() for name, cls in AGENTS.items()}


__all__ = ["AGENTS", "Agent", "Context", "build_agents"]
