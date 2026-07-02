"""Agent ABC + the shared pipeline Context.

Every agent is a pure-ish function over the Context: `run(ctx) -> ctx`. It reads and
mutates `ctx.project` and may use the local LLM/VLM handles. This uniform contract is
what makes the pipeline orderable, testable, and resumable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..errors import NotInstalledError

if TYPE_CHECKING:
    from ..config import Config
    from ..logging_setup import EventLog
    from ..state import Project


@dataclass
class Context:
    """Everything an agent or a generation stage needs. Built by the coordinator."""

    project: Project
    config: Config
    events: EventLog
    llm: Any = None                 # OllamaLLM (or a fake in tests)
    vlm: Any = None                 # OllamaVLM (or a fake in tests)
    tier: str = "mid"
    license_mode: str = "safe"
    stage: str = ""                 # set by the coordinator before each stage
    matrix: Any = None              # ModelMatrix
    gate: Any = None                # LicenseGate
    plan: Any = None                # BackendPlan
    data: dict[str, Any] = field(default_factory=dict)

    def emit(self, kind: str, message: str, pct: float | None = None, **extra) -> None:
        self.events.emit(kind, self.stage or "-", message, pct, **extra)

    def require_llm(self):
        if self.llm is None:
            raise NotInstalledError(
                "This stage needs a local LLM. Install Ollama and pull a model "
                "(e.g. `ollama pull qwen2.5:7b`), then run `cineforge doctor`."
            )
        return self.llm


class Agent(ABC):
    name: str = ""

    @abstractmethod
    def run(self, ctx: Context) -> Context:
        """Read + mutate ctx.project, return ctx."""
        raise NotImplementedError
