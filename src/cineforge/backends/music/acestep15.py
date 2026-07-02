"""ACE-Step music — WORKING via ComfyUI's native ACE-Step nodes (verified on GPU).

CheckpointLoaderSimple(all-in-one) -> TextEncodeAceStepAudio(tags/lyrics) ->
EmptyAceStepLatentAudio(seconds) -> KSampler -> VAEDecodeAudio -> SaveAudioMP3.
Generates a music/score track (instrumental by default; add lyrics for vocals).
"""

from __future__ import annotations

from typing import Any

from ...errors import BackendError
from ...models.registry import register
from ..base import CAP_DURATION_CONTROL, CAP_INSTRUMENTAL, CAP_VOCALS, ComfyBackend, Result

DEFAULT_CKPT = "ace_step_v1_3.5b.safetensors"


@register("acestep15", subsystem="music", license_id="acestep15")
class AceStep15(ComfyBackend):
    subsystem = "music"
    default_vram = 8.0

    def capabilities(self) -> set[str]:
        return {CAP_VOCALS, CAP_INSTRUMENTAL, CAP_DURATION_CONTROL}

    def _ckpt(self) -> str:
        if self.choice and self.choice.extra:
            return self.choice.extra.get("checkpoint", DEFAULT_CKPT)
        return DEFAULT_CKPT

    def build_workflow(self, request: Any) -> dict:
        tags = (getattr(request, "style", "") or "instrumental cinematic orchestral score, cheerful").strip()
        lyrics = getattr(request, "lyrics", None) or ""
        seconds = float(getattr(request, "duration_s", None) or 30.0)
        seed = int(request.seed) if getattr(request, "seed", None) is not None else 42
        return {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": self._ckpt()}},
            "2": {"class_type": "TextEncodeAceStepAudio", "inputs": {
                "clip": ["1", 1], "tags": tags, "lyrics": lyrics, "lyrics_strength": 1.0}},
            "3": {"class_type": "TextEncodeAceStepAudio", "inputs": {
                "clip": ["1", 1], "tags": "", "lyrics": "", "lyrics_strength": 1.0}},
            "4": {"class_type": "EmptyAceStepLatentAudio", "inputs": {"seconds": seconds, "batch_size": 1}},
            "5": {"class_type": "KSampler", "inputs": {
                "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0],
                "seed": seed, "steps": 50, "cfg": 5.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
            "6": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
            "7": {"class_type": "SaveAudioMP3", "inputs": {
                "audio": ["6", 0], "filename_prefix": "cineforge_music", "quality": "320k"}},
        }

    def parse_outputs(self, outputs: dict, request: Any) -> Result:
        for nid, o in outputs.items():
            if not isinstance(o, dict):
                continue
            for key in ("audio", "audios"):
                items = o.get(key)
                if items:
                    f = items[0]
                    fn = f.get("filename") if isinstance(f, dict) else None
                    if fn:
                        sub = (f.get("subfolder", "") if isinstance(f, dict) else "") or ""
                        return Result(path=str(self.cfg.comfy_dir / "output" / sub / fn),
                                      kind="music", meta={"node": nid, "workflow": "acestep"})
        raise BackendError("ComfyUI returned no audio output")
