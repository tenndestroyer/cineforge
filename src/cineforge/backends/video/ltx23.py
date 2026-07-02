"""LTX-2.3 — OPT-IN max-quality video (Lightricks Open Weights, revenue-gated).

The only open model that emits synced audio + dialogue + lipsync in ONE diffusion
pass, up to native 4K/50fps, with IC-LoRA image-conditioning for character/style
consistency and a latent upscaler. License is revenue-gated ($10M ARR), NOT Apache —
blocked in Safe mode. Also requires the gated google/gemma-3-12b-it text encoder.
"""

from __future__ import annotations

from ...models.registry import register
from ..base import CAP_IC_LORA, CAP_IMG2VID, CAP_NATIVE_AUDIO, ComfyBackend


@register("ltx23", subsystem="video", license_id="ltx23")
class LTX23(ComfyBackend):
    subsystem = "video"
    default_vram = 24.0
    required_nodes = ("ComfyUI-LTXVideo", "ComfyUI-GGUF")
    required_weights = ("video/ltx23/", "text_encoders/gemma-3-12b-it/")

    def capabilities(self) -> set[str]:
        return {CAP_IMG2VID, CAP_NATIVE_AUDIO, CAP_IC_LORA}
