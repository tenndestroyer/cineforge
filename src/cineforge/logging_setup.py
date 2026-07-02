"""Structured logging + a tiny event emitter shared by the pipeline and GUI.

The GUI polls `EventLog` to render live progress; the coordinator and agents push
events into it. Kept stdlib-only and importable everywhere.
"""

from __future__ import annotations

import logging
import sys
from collections import deque
from dataclasses import asdict, dataclass, field
from threading import Lock

_CONFIGURED = False


def get_logger(name: str = "cineforge") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        # Windows consoles default to cp1252 and mangle em-dashes/arrows in log lines.
        try:
            sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("cineforge")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name)


@dataclass
class Event:
    seq: int
    kind: str          # 'stage' | 'progress' | 'info' | 'warn' | 'error'
    stage: str
    message: str
    pct: float | None = None
    extra: dict = field(default_factory=dict)


class EventLog:
    """Bounded, thread-safe, in-memory event ring buffer for the GUI to poll."""

    def __init__(self, maxlen: int = 2000) -> None:
        self._events: deque[Event] = deque(maxlen=maxlen)
        self._seq = 0
        self._lock = Lock()
        self._log = get_logger("cineforge.events")

    def emit(self, kind: str, stage: str, message: str, pct: float | None = None, **extra) -> Event:
        with self._lock:
            self._seq += 1
            ev = Event(self._seq, kind, stage, message, pct, dict(extra))
            self._events.append(ev)
        level = {"error": self._log.error, "warn": self._log.warning}.get(kind, self._log.info)
        level("[%s] %s", stage, message)
        return ev

    def since(self, seq: int) -> list[dict]:
        with self._lock:
            return [asdict(e) for e in self._events if e.seq > seq]

    def latest_seq(self) -> int:
        with self._lock:
            return self._seq
