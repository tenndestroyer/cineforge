"""A lightweight RAG-style asset index (no heavy vector-DB dependency).

Holds accepted character/scene reference frames and trained LoRAs so agents retrieve
the correct canonical reference per shot instead of drifting from an unindexed frame.
Embeddings are optional; when absent, retrieval falls back to tag matching. A real
CLIP/DINO vector store slots in behind the same interface later.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path

from ..logging_setup import get_logger


@dataclass
class Asset:
    path: str
    kind: str                       # 'character_ref' | 'scene_ref' | 'lora'
    tags: list[str] = field(default_factory=list)
    character: str | None = None    # character id, if applicable
    embedding: list[float] | None = None


class AssetStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.index_path = self.root / "assets.json"
        self.assets: list[Asset] = []
        self._log = get_logger("cineforge.asset_store")
        self._load()

    def _load(self) -> None:
        if not self.index_path.is_file():
            return
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._log.warning("assets.json unreadable/corrupt; starting a fresh index")
            self.assets = []
            return
        known = {f.name for f in dataclass_fields(Asset)}
        out: list[Asset] = []
        for a in raw.get("assets", []):
            if not isinstance(a, dict):
                continue
            try:
                out.append(Asset(**{k: v for k, v in a.items() if k in known}))
            except TypeError:
                self._log.warning("skipping malformed asset entry: %r", a)
        self.assets = out

    def save(self) -> None:
        """Atomic write (tmp file + os.replace), mirroring state/store.py, so a crash
        mid-write can't corrupt the index."""
        self.root.mkdir(parents=True, exist_ok=True)
        data = json.dumps({"assets": [asdict(a) for a in self.assets]}, indent=2)
        fd, tmp = tempfile.mkstemp(dir=str(self.root), prefix=".assets.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.index_path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def add(self, asset: Asset) -> None:
        self.assets.append(asset)
        self.save()

    def query_character(self, character_id: str) -> Asset | None:
        """Best reference for a character: prefer a canonical/hero-tagged frame."""
        matches = [a for a in self.assets if a.character == character_id and a.kind == "character_ref"]
        if not matches:
            return None
        for a in matches:
            if "canonical" in a.tags or "hero" in a.tags:
                return a
        return matches[0]

    def lora_for(self, character_id: str) -> Asset | None:
        return next((a for a in self.assets if a.character == character_id and a.kind == "lora"), None)
