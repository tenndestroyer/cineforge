"""The serializable project state model: Project -> Scenes -> Shots -> Takes.

Agents are pure functions over a `Context` that read and mutate the `Project`.
Everything here is a plain dataclass of JSON-safe primitives so it round-trips
through `project.json` (see state/store.py) for checkpoint/resume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1


def _new_id(prefix: str, n: int) -> str:
    return f"{prefix}{n:03d}"


@dataclass
class Dialogue:
    character: str = ""       # Character.id
    line: str = ""
    emotion: str = "neutral"  # drives voice exaggeration + lipsync guidance


@dataclass
class Take:
    """One generated candidate for a shot's asset (keyframe/video/voice/lipsync)."""

    id: str = ""
    shot_id: str = ""
    kind: str = ""            # 'keyframe' | 'video' | 'voice' | 'lipsync' | 'foley' | 'music'
    path: str = ""            # relative to the project dir
    score: float | None = None
    accepted: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Character:
    id: str = ""
    name: str = ""
    description: str = ""
    reference_images: list[str] = field(default_factory=list)
    canonical_ref: str | None = None      # the locked hero keyframe used for I2V
    lora_path: str | None = None          # trained per-character identity LoRA
    voice_ref: str | None = None          # clip used for voice cloning
    voice_profile: dict[str, Any] = field(default_factory=dict)  # exaggeration, model, etc.


@dataclass
class Shot:
    id: str = ""
    scene_id: str = ""
    index: int = 0
    description: str = ""
    keyframe_prompt: str = ""
    pose: str = ""            # ControlNet/OpenPose brief
    camera: str = ""          # framing / movement note
    duration_s: float = 4.0
    dialogue: list[Dialogue] = field(default_factory=list)
    takes: list[Take] = field(default_factory=list)
    status: str = "pending"   # pending | rendered | approved | flagged

    def accepted_take(self, kind: str) -> Take | None:
        for t in self.takes:
            if t.kind == kind and t.accepted:
                return t
        return None


@dataclass
class Scene:
    id: str = ""
    index: int = 0
    heading: str = ""
    description: str = ""
    shots: list[Shot] = field(default_factory=list)


@dataclass
class AudioCue:
    id: str = ""
    kind: str = ""            # 'foley' | 'ambience' | 'music'
    start_s: float = 0.0
    duration_s: float = 0.0
    prompt: str = ""
    path: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class QAReport:
    passed: bool = False
    checks: list[dict[str, Any]] = field(default_factory=list)
    flagged_shots: list[str] = field(default_factory=list)


@dataclass
class Project:
    name: str = ""
    idea: str = ""
    style: str = "stylized 3D cartoon"
    logline: str = ""
    script: str = ""
    scenes: list[Scene] = field(default_factory=list)
    characters: list[Character] = field(default_factory=list)
    audio_cues: list[AudioCue] = field(default_factory=list)
    render_plan: dict[str, Any] = field(default_factory=dict)
    qa: QAReport | None = None
    license_acks: list[str] = field(default_factory=list)   # model ids the user consented to
    calibration: dict[str, float] = field(default_factory=dict)  # stage -> seconds/unit
    tier: str | None = None
    stage: str = ""          # last COMPLETED stage name (for resume); "" = never run
    created: str = ""        # ISO date, stamped by the caller (scripts can't use Date.now)
    schema_version: int = SCHEMA_VERSION

    # ---- convenience ----
    def all_shots(self) -> list[Shot]:
        return [shot for scene in self.scenes for shot in scene.shots]

    def character(self, cid: str) -> Character | None:
        return next((c for c in self.characters if c.id == cid), None)

    # ---- (de)serialization ----
    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Project:
        p = Project()
        for f in ("name", "idea", "style", "logline", "script", "render_plan",
                  "license_acks", "calibration", "tier", "stage", "created", "schema_version"):
            if f in d and d[f] is not None:
                setattr(p, f, d[f])
        p.characters = [_char(c) for c in d.get("characters", [])]
        p.scenes = [_scene(s) for s in d.get("scenes", [])]
        p.audio_cues = [_cue(c) for c in d.get("audio_cues", [])]
        if d.get("qa"):
            p.qa = QAReport(**{k: d["qa"].get(k, QAReport().__dict__[k]) for k in QAReport().__dict__})
        return p


# --- reconstruction helpers (lenient: unknown keys ignored, missing keys defaulted) ---
def _pick(cls, d: dict[str, Any]):
    valid = cls().__dict__
    return {k: d.get(k, valid[k]) for k in valid}


def _char(d: dict[str, Any]) -> Character:
    return Character(**_pick(Character, d))


def _cue(d: dict[str, Any]) -> AudioCue:
    return AudioCue(**_pick(AudioCue, d))


def _take(d: dict[str, Any]) -> Take:
    return Take(**_pick(Take, d))


def _dlg(d: dict[str, Any]) -> Dialogue:
    return Dialogue(**_pick(Dialogue, d))


def _shot(d: dict[str, Any]) -> Shot:
    base = _pick(Shot, d)
    base["dialogue"] = [_dlg(x) for x in d.get("dialogue", [])]
    base["takes"] = [_take(x) for x in d.get("takes", [])]
    return Shot(**base)


def _scene(d: dict[str, Any]) -> Scene:
    base = _pick(Scene, d)
    base["shots"] = [_shot(x) for x in d.get("shots", [])]
    return Scene(**base)


def _to_dict(obj: Any) -> Any:
    from dataclasses import fields, is_dataclass

    if is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj
