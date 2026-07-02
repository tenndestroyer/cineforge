"""Central configuration — the single source of paths, flags, and secrets.

Everything else consumes a `Config`. It is intentionally dependency-free (stdlib
only) so it can be imported from anywhere without circular-import risk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ConfigError

LICENSE_MODES = ("safe", "research")
TIERS = ("low", "mid", "high")


def _repo_root() -> Path:
    # src/cineforge/config.py -> repo root is three parents up.
    return Path(__file__).resolve().parents[2]


def _load_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file. Silent if the file is absent."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip().strip('"').strip("'")
    return out


@dataclass
class Config:
    """Resolved runtime configuration.

    Prefer `Config.load()` over constructing directly.
    """

    repo_root: Path
    home: Path                      # CineforgeData: weights, comfy, projects, scratch
    hf_token: str = ""
    license_mode: str = "safe"      # 'safe' = Apache/MIT only; 'research' = allow gated
    tier_override: str | None = None  # force 'low'|'mid'|'high' instead of auto-detect
    comfy_url: str = "http://127.0.0.1:8188"
    ollama_url: str = "http://127.0.0.1:11434"
    flags: dict[str, str] = field(default_factory=dict)

    # ---- derived paths (all under `home`) ----
    @property
    def models_dir(self) -> Path:
        return self.home / "models_store"

    @property
    def comfy_dir(self) -> Path:
        return self.home / "ComfyUI"

    @property
    def projects_dir(self) -> Path:
        return self.home / "projects"

    @property
    def scratch_dir(self) -> Path:
        return self.home / "scratch"

    @property
    def data_dir(self) -> Path:
        """Repo `data/` — model_matrix.json, licenses.json, presets, luts."""
        return self.repo_root / "data"

    def project_dir(self, name: str) -> Path:
        return self.projects_dir / _slug(name)

    def ensure_dirs(self) -> None:
        for p in (self.home, self.models_dir, self.projects_dir, self.scratch_dir):
            p.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        if self.license_mode not in LICENSE_MODES:
            raise ConfigError(f"license_mode must be one of {LICENSE_MODES}, got {self.license_mode!r}")
        if self.tier_override is not None and self.tier_override not in TIERS:
            raise ConfigError(f"tier_override must be one of {TIERS} or None, got {self.tier_override!r}")

    @classmethod
    def load(cls, repo_root: Path | None = None) -> Config:
        root = Path(repo_root) if repo_root else _repo_root()
        env = _load_env_file(root / "keys.env")

        home_env = os.environ.get("CINEFORGE_HOME") or env.get("CINEFORGE_HOME")
        home = Path(home_env) if home_env else (root / "CineforgeData")

        cfg = cls(
            repo_root=root,
            home=home,
            hf_token=os.environ.get("HF_TOKEN") or env.get("HF_TOKEN", ""),
            license_mode=(os.environ.get("CINEFORGE_LICENSE_MODE") or "safe").lower(),
            tier_override=((os.environ.get("CINEFORGE_TIER") or "").strip().lower() or None),
            comfy_url=os.environ.get("CINEFORGE_COMFY_URL", "http://127.0.0.1:8188"),
            ollama_url=os.environ.get("CINEFORGE_OLLAMA_URL", "http://127.0.0.1:11434"),
        )
        cfg.validate()
        return cfg


def _slug(name: str) -> str:
    keep = "-_"
    s = "".join(c if (c.isalnum() or c in keep) else "-" for c in name.strip().lower())
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-") or "project"
