"""SVD (Stable Video Diffusion) — the first PROVEN image->video path.

Animates a locked keyframe into a short clip via ComfyUI's canonical SVD graph
(ImageOnlyCheckpointLoader -> SVD_img2vid_Conditioning -> VideoLinearCFGGuidance ->
KSampler -> VAEDecode -> SaveWEBM). Reliable single-checkpoint path.

Honest ceiling: SVD adds MOTION/camera to the still — it makes the keyframe move; it
does not choreograph new action. Wan i2v is the quality/motion upgrade. Kept
VRAM-conservative (640x384, 14 frames) so it fits a 12 GB laptop GPU.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ...errors import BackendError, NotInstalledError
from ...models.registry import register
from ..base import CAP_IMG2VID, ComfyBackend, Result

DEFAULT_CKPT = "svd_xt.safetensors"


@register("svd", subsystem="video", license_id="svd")
class SVD(ComfyBackend):
    subsystem = "video"
    default_vram = 10.0
    required_nodes = ()

    def capabilities(self) -> set[str]:
        return {CAP_IMG2VID}

    def _ckpt(self) -> str:
        if self.choice and self.choice.extra:
            return self.choice.extra.get("checkpoint", DEFAULT_CKPT)
        return DEFAULT_CKPT

    def build_workflow(self, request: Any) -> dict:
        kf = getattr(request, "keyframe", None)
        if not kf or not Path(kf).is_file():
            raise NotInstalledError("SVD needs a keyframe image — render keyframes first.")
        # SVD's LoadImage reads from ComfyUI/input, so stage the keyframe there.
        inp = self.cfg.comfy_dir / "input"
        inp.mkdir(parents=True, exist_ok=True)
        name = Path(kf).name
        try:
            shutil.copyfile(kf, inp / name)
        except OSError as e:  # pragma: no cover
            raise BackendError(f"could not stage keyframe for SVD: {e}") from e

        frames = min(int(getattr(request, "frames", None) or 14), 25)
        fps = int(getattr(request, "fps", None) or 8)
        seed = int(request.seed) if getattr(request, "seed", None) is not None else 42
        w, h = 640, 384  # conservative for 12 GB VRAM (SVD is memory-heavy)
        return {
            "1": {"class_type": "ImageOnlyCheckpointLoader", "inputs": {"ckpt_name": self._ckpt()}},
            "2": {"class_type": "LoadImage", "inputs": {"image": name}},
            "3": {"class_type": "SVD_img2vid_Conditioning", "inputs": {
                "clip_vision": ["1", 1], "init_image": ["2", 0], "vae": ["1", 2],
                "width": w, "height": h, "video_frames": frames, "motion_bucket_id": 127,
                "fps": fps, "augmentation_level": 0.0}},
            "4": {"class_type": "VideoLinearCFGGuidance", "inputs": {"model": ["1", 0], "min_cfg": 1.0}},
            "5": {"class_type": "KSampler", "inputs": {
                "model": ["4", 0], "positive": ["3", 0], "negative": ["3", 1], "latent_image": ["3", 2],
                "seed": seed, "steps": 20, "cfg": 2.5, "sampler_name": "euler",
                "scheduler": "karras", "denoise": 1.0}},
            "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
            "7": {"class_type": "SaveWEBM", "inputs": {
                "images": ["6", 0], "filename_prefix": "cineforge_vid", "codec": "vp9",
                "fps": fps, "crf": 32}},
        }

    def parse_outputs(self, outputs: dict, request: Any) -> Result:
        for nid, o in outputs.items():
            if not isinstance(o, dict):
                continue
            for key in ("images", "gifs", "videos", "animated"):
                items = o.get(key)
                if items:
                    f = items[0]
                    fn = f.get("filename") if isinstance(f, dict) else None
                    if fn:
                        sub = (f.get("subfolder", "") if isinstance(f, dict) else "") or ""
                        return Result(path=str(self.cfg.comfy_dir / "output" / sub / fn),
                                      kind="video", meta={"node": nid, "workflow": "svd"})
        raise BackendError("ComfyUI returned no video output")
