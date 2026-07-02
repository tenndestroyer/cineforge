"""The ONE common Backend interface + typed per-subsystem request/result dataclasses.

This is the load-bearing abstraction of Cineforge: every model — Wan 2.2, LTX-2.3,
Chatterbox, ACE-Step, EchoMimicV3, SeedVR2, ... — is a `Backend` subclass that
implements `capabilities`, `estimate_vram`, and `generate`. Selecting/swapping a
model never touches the coordinator, agents, or GUI; it's one adapter file + one
row in data/model_matrix.json.

Most heavy inference runs inside a local ComfyUI graph, so real adapters subclass
`ComfyBackend`, which handles readiness checks and submission. Adapters stay thin:
build the workflow, apply the per-model sampler preset, submit, normalize outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import NotInstalledError

if TYPE_CHECKING:
    from ..config import Config
    from ..models.matrix import ModelChoice

# ---- subsystems ----
SUBSYSTEMS = ("video", "image", "voice", "music", "lipsync", "sfx", "enhance", "llm")

# ---- capability flags (advertised by capabilities()) ----
CAP_NATIVE_AUDIO = "native_audio"        # video model emits synced audio in one pass
CAP_MULTI_REFERENCE = "multi_reference"  # image model accepts several identity refs
CAP_IC_LORA = "ic_lora"                  # image-conditioning LoRA (LTX-2.3)
CAP_IMG2VID = "img2vid"                  # can drive video from a locked keyframe
CAP_CLONING = "cloning"                  # voice model can clone from a reference clip
CAP_DURATION_CONTROL = "duration_control"
CAP_INSTRUMENTAL = "instrumental"
CAP_VOCALS = "vocals"
CAP_VIDEO_CONDITIONED = "video_conditioned"  # V2A foley synced to video
CAP_STYLIZED_FACE = "stylized_face"      # lipsync works on non-photoreal faces
CAP_INTERPOLATE = "interpolate"
CAP_UPSCALE = "upscale"


# =========================================================================
# Request / Result dataclasses (one request type per subsystem)
# =========================================================================
@dataclass
class Result:
    """A generated asset on disk. `extra_paths` carries side outputs, e.g. the
    audio stem from a native-audio video model."""

    path: str
    kind: str
    meta: dict[str, Any] = field(default_factory=dict)
    extra_paths: list[str] = field(default_factory=list)
    score: float | None = None


@dataclass
class VideoRequest:
    prompt: str = ""
    keyframe: str | None = None       # locked hero still -> I2V (primary consistency path)
    frames: int = 97
    fps: int = 24
    width: int = 768
    height: int = 512
    duration_s: float = 4.0
    style_lora: str | None = None
    ic_lora: str | None = None        # LTX-2.3 image-conditioning LoRA
    native_audio: bool = False
    seed: int | None = None
    steps: int | None = None          # None => use preset
    cfg: float | None = None


@dataclass
class ImageRequest:
    prompt: str = ""
    refs: list[str] = field(default_factory=list)   # identity/style reference images
    instruction: str | None = None                  # for edit-style models (Qwen-Image-Edit)
    style_lora: str | None = None
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    steps: int | None = None
    draft: bool = False               # fast preview pass vs high-fidelity final


@dataclass
class SpeechRequest:
    text: str = ""
    ref_clip: str | None = None       # voice-clone reference
    voice_id: str | None = None       # for no-clone narration models (Kokoro)
    emotion: str = "neutral"
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    language: str = "en"
    duration_s: float | None = None   # for duration-controllable models (dub sync)


@dataclass
class MusicRequest:
    style: str = ""
    lyrics: str | None = None         # None => instrumental
    duration_s: float = 30.0
    bpm: int | None = None
    instrumental: bool = False
    seed: int | None = None


@dataclass
class LipsyncRequest:
    face_source: str = ""             # image or short video of the character face
    audio: str = ""                   # clean pre-mix dialogue
    guidance: float = 2.0             # audio_guidance_scale
    mask: str | None = None           # manual mouth region for non-human faces


@dataclass
class FoleyRequest:
    video: str = ""                   # V2A conditions foley on the picture
    prompt: str = ""
    cfg: float = 4.5
    seconds: float | None = None


@dataclass
class AmbienceRequest:
    prompt: str = ""
    seconds: float = 10.0


@dataclass
class EnhanceRequest:
    clip: str = ""
    target_width: int | None = None
    target_height: int | None = None
    interpolate_factor: int = 1       # 1 = no interpolation
    restore: bool = True              # restore-before-upscale


# =========================================================================
# Backend ABC
# =========================================================================
class Backend(ABC):
    """Common interface for every model adapter."""

    subsystem: str = ""       # one of SUBSYSTEMS
    model_id: str = ""        # registry key; set by @register(...)
    license_id: str = ""      # key into data/licenses.json

    def __init__(self, cfg: Config, choice: ModelChoice | None = None) -> None:
        self.cfg = cfg
        self.choice = choice
        self._loaded = False

    # --- capability advertisement (override to declare) ---
    def capabilities(self) -> set[str]:
        return set()

    def supports(self, cap: str) -> bool:
        return cap in self.capabilities()

    # --- VRAM budgeting (used by the Producer to pick tier/quant) ---
    @abstractmethod
    def estimate_vram(self, request: Any) -> float:
        """Estimated peak VRAM in GB for this request."""

    # --- explicit lifecycle so the coordinator controls resident VRAM ---
    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @abstractmethod
    def generate(self, request: Any) -> Result:
        """Produce the asset. Raise NotInstalledError if weights/nodes are missing,
        VRAMError on OOM, BackendError on other failures."""

    # --- niceties ---
    @property
    def name(self) -> str:
        return self.model_id or type(self).__name__

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{type(self).__name__} model_id={self.model_id!r} subsystem={self.subsystem!r}>"


class ComfyBackend(Backend):
    """Base for adapters whose inference runs in the local ComfyUI.

    Subclasses implement `build_workflow(request) -> dict` (an API-format graph) and
    `parse_outputs(outputs, request) -> Result`. `generate` wires readiness checks,
    submission, and progress. In this scaffold, `build_workflow` is a documented
    placeholder for most models — the readiness check makes the "not yet wired /
    not installed" state explicit and honest rather than silently fake.
    """

    #: ComfyUI custom nodes this backend needs installed (checked by doctor).
    required_nodes: tuple[str, ...] = ()
    #: weight files (relative to models_store) this backend needs.
    required_weights: tuple[str, ...] = ()
    #: fallback VRAM estimate (GB) when no ModelChoice.min_vram_gb is set.
    default_vram: float = 12.0

    def estimate_vram(self, request: Any) -> float:
        if self.choice and self.choice.min_vram_gb:
            return float(self.choice.min_vram_gb)
        return self.default_vram

    def _client(self):
        from ..models.comfy_client import ComfyClient

        return ComfyClient(self.cfg.comfy_url)

    def _require_ready(self) -> None:
        client = self._client()
        if not client.is_reachable():
            raise NotInstalledError(
                f"ComfyUI is not reachable at {self.cfg.comfy_url}. Run `cineforge doctor` "
                f"or (re)run setup to install it. ({self.model_id})"
            )
        missing = [w for w in self.required_weights if not (self.cfg.models_dir / w).exists()]
        if missing:
            raise NotInstalledError(
                f"{self.model_id}: missing weights {missing}. Re-run setup with a valid HF_TOKEN "
                f"if the repo is gated (see keys.env)."
            )

    def build_workflow(self, request: Any) -> dict:  # pragma: no cover - per-model
        raise NotInstalledError(
            f"{self.model_id}: ComfyUI workflow not wired in this build. This adapter "
            f"declares its interface, VRAM estimate, and required nodes/weights; wiring the "
            f"graph is tracked in the roadmap. See docs/ARCHITECTURE.md."
        )

    def parse_outputs(self, outputs: dict, request: Any) -> Result:  # pragma: no cover
        raise NotImplementedError

    def generate(self, request: Any) -> Result:
        self._require_ready()
        client = self._client()
        workflow = self.build_workflow(request)
        prompt_id = client.submit(workflow)
        outputs = client.wait(prompt_id)
        return self.parse_outputs(outputs, request)


def out_path(cfg: Config, *parts: str) -> Path:
    """Scratch output path helper for adapters."""
    p = cfg.scratch_dir.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
