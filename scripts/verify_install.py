#!/usr/bin/env python
"""Standalone doctor — same checks as `cineforge doctor`, runnable before install."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cineforge.config import Config  # noqa: E402
from cineforge.scripts_verify import run_doctor  # noqa: E402


def main() -> int:
    cfg = Config.load()
    report = run_doctor(cfg)
    ok = all(c["ok"] for c in report)
    for c in report:
        print(f"[{'OK ' if c['ok'] else 'XX '}] {c['name']}: {c['detail']}")
    print("\nAll green." if ok else "\nSome checks failed — see docs/TROUBLESHOOTING.md")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
