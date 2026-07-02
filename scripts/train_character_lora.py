#!/usr/bin/env python
"""Train a per-character identity LoRA locally (wrapper).

Character consistency's strongest local lever is a dedicated LoRA per recurring
character, trained on a curated reference set (the Casting agent flags who needs one).
This wraps an external trainer (Kohya_ss / ai-toolkit / diffusion-pipe depending on the
base model). It is a long GPU job, intentionally separate from the render pipeline.

STATUS: scaffold. It validates inputs and prints the exact command to run for the
chosen base model; wiring each trainer is tracked in the roadmap (v0.3).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a character identity LoRA")
    ap.add_argument("--character", required=True, help="character name/id")
    ap.add_argument("--images", required=True, help="folder of 15-30 reference images")
    ap.add_argument("--base", default="wan22", help="base model the LoRA targets (wan22|ltx23|zimage|flux2)")
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()

    img_dir = Path(args.images)
    if not img_dir.is_dir():
        print(f"images folder not found: {img_dir}", file=sys.stderr)
        return 1
    n = len([p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])
    if n < 8:
        print(f"WARNING: only {n} images; 15-30 curated, consistent shots give far better identity locking.")

    print(f"Character : {args.character}")
    print(f"Base model: {args.base}")
    print(f"Images    : {n} in {img_dir}")
    print(f"Steps     : {args.steps}")
    print()
    print("TODO(v0.3): dispatch to the trainer for this base model, e.g.:")
    if args.base in ("zimage", "flux2"):
        print("  ai-toolkit / kohya_ss FLUX-style LoRA training")
    else:
        print("  diffusion-pipe / musubi-tuner video-LoRA training (Wan/LTX)")
    print("The trained .safetensors goes to CineforgeData/models_store/loras/<character>.safetensors")
    print("and is registered on the Character via the asset store.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
