"""SD-checkpoint image backend — the first PROVEN render path.

Renders keyframes via ComfyUI's canonical graph
(CheckpointLoaderSimple -> CLIPTextEncode -> KSampler -> VAEDecode -> SaveImage),
verified working on the user's RTX 5070 Ti. Uses an all-in-one .safetensors checkpoint
(SD1.5/SDXL). This is the reliable first render; the higher-quality Z-Image path (native
ComfyUI diffusion-model loaders) is the next upgrade — see zimage.py.
"""

from __future__ import annotations

from typing import Any

from ...errors import BackendError
from ...models.registry import register
from ..base import ComfyBackend, Result

DEFAULT_CKPT = "v1-5-pruned-emaonly-fp16.safetensors"
NEG_PROMPT = "blurry, low quality, deformed, disfigured, text, watermark, extra limbs"


@register("sd_checkpoint", subsystem="image", license_id="sd15")
class SDCheckpoint(ComfyBackend):
    subsystem = "image"
    default_vram = 4.0

    def capabilities(self) -> set[str]:
        return set()

    def _ckpt_name(self) -> str:
        if self.choice and self.choice.extra:
            return self.choice.extra.get("checkpoint", DEFAULT_CKPT)
        return DEFAULT_CKPT

    def build_workflow(self, request: Any) -> dict:
        ckpt = getattr(request, "checkpoint", None) or self._ckpt_name()
        w = int(getattr(request, "width", 512) or 512)
        h = int(getattr(request, "height", 512) or 512)
        steps = int(getattr(request, "steps", None) or 20)
        cfg = float(getattr(request, "cfg", None) or 7.0)
        neg = getattr(request, "negative", None) or NEG_PROMPT
        seed = int(request.seed) if getattr(request, "seed", None) is not None else 42
        prompt = (getattr(request, "prompt", "") or "a scene").strip()
        return {
            "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["4", 1]}},
            "3": {"class_type": "KSampler", "inputs": {
                "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler",
                "scheduler": "normal", "denoise": 1.0,
                "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "cineforge", "images": ["8", 0]}},
        }

    def parse_outputs(self, outputs: dict, request: Any) -> Result:
        for nid, o in outputs.items():
            if isinstance(o, dict) and o.get("images"):
                img = o["images"][0]
                sub = img.get("subfolder", "") or ""
                path = self.cfg.comfy_dir / "output" / sub / img["filename"]
                return Result(path=str(path), kind="keyframe",
                              meta={"node": nid, "workflow": "sd_checkpoint", "model": self._ckpt_name()})
        raise BackendError("ComfyUI returned no image output")
