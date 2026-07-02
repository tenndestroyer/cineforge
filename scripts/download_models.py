#!/usr/bin/env python
"""Tier-aware, size-verified model weight fetcher.

Resolves the model matrix for the detected (or forced) tier + license mode and pulls
each chosen model's repo from Hugging Face. Gated repos need HF_TOKEN in keys.env.
Use --dry to preview the plan without downloading (default prints the plan and asks
for --confirm before pulling tens of GB).
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cineforge.config import Config  # noqa: E402
from cineforge.errors import DownloadIntegrityError  # noqa: E402
from cineforge.hardware import classify_vram, detect_gpus, primary_gpu  # noqa: E402
from cineforge.models.downloader import verify_size  # noqa: E402
from cineforge.models.matrix import ModelMatrix  # noqa: E402

_WEIGHT_EXTS = {".safetensors", ".gguf", ".bin", ".pth", ".ckpt", ".onnx"}


def _verify_weights(dest) -> list[str]:
    """Byte-size/HTML verify each downloaded WEIGHT file so a gated pull that wrote a
    ~136-byte error page is rejected (the 'size-verified' the docstrings promise)."""
    problems: list[str] = []
    for f in dest.rglob("*"):
        if f.is_file() and f.suffix.lower() in _WEIGHT_EXTS:
            try:
                verify_size(f, None)
            except DownloadIntegrityError as e:
                problems.append(str(e))
    return problems


def _allow_patterns(choice) -> list[str] | None:
    """Restrict a repo download to only the files we need. GGUF repos ship MANY quant
    variants (all of which snapshot_download would grab) — we fetch only the tier's
    quant. A model_matrix `extra.allow_patterns` overrides. Non-GGUF single-variant
    repos download whole (returns None)."""
    override = (choice.extra or {}).get("allow_patterns")
    if override:
        return list(override)
    q = (choice.quant or "").lower()
    if "gguf" in q or re.fullmatch(r"q\d(_.*)?", q or ""):
        m = re.search(r"q(\d)", q)
        tag = f"q{m.group(1)}" if m else "q4"
        return [f"*{tag}*.gguf", f"*{tag.upper()}*.gguf",
                "*.json", "*.txt", "*.yaml", "*.model", "config*", "tokenizer*"]
    return None


def _has_weight_file(dest) -> bool:
    return any(f.is_file() and f.suffix.lower() in _WEIGHT_EXTS for f in dest.rglob("*"))


def _purge_repo_cache(repo: str) -> None:
    """Delete the HuggingFace hub cache copy for a repo. snapshot_download(local_dir=...)
    can leave a full duplicate in the cache on Windows, doubling disk use."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        d = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    except Exception:  # noqa: BLE001 - best-effort cleanup
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Download Cineforge model weights")
    ap.add_argument("--tier", default="auto", choices=["auto", "low", "mid", "high"])
    ap.add_argument("--license-mode", default="safe", choices=["safe", "research"])
    ap.add_argument("--dry", action="store_true", help="print the plan only")
    ap.add_argument("--confirm", action="store_true", help="actually download")
    ap.add_argument("--only", default="", help="download only this subsystem (e.g. image, video)")
    args = ap.parse_args()

    cfg = Config.load()
    tier = args.tier
    if tier == "auto":
        tier = cfg.tier_override or classify_vram(primary_gpu(detect_gpus()).vram_gb)
    matrix = ModelMatrix.load(cfg.data_dir / "model_matrix.json")

    plan = []
    for sub in matrix.subsystems():
        if args.only and sub != args.only:
            continue
        c = matrix.resolve(sub, tier, args.license_mode)
        if c.repo and not c.repo.startswith("ollama:"):
            plan.append((sub, c))

    print(f"Tier: {tier}   License mode: {args.license_mode}")
    print(f"Models dir: {cfg.models_dir}\n")
    for sub, c in plan:
        print(f"  {sub:<10} {c.model_id:<16} {c.repo}  [{c.variant} {c.quant}]")
    print()

    if args.dry or not args.confirm:
        print("Dry run. Re-run with --confirm to download (this can be tens of GB).")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface-hub not installed; run setup first.", file=sys.stderr)
        return 1

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    total = len(plan)
    for i, (sub, c) in enumerate(plan, 1):
        dest = cfg.models_dir / sub / c.model_id
        print(f"\n[{i}/{total}] {sub}: {c.repo}\n    -> {dest}", flush=True)
        patterns = _allow_patterns(c)
        if patterns:
            print(f"    (only: {patterns})", flush=True)
        try:
            snapshot_download(repo_id=c.repo, local_dir=str(dest), token=cfg.hf_token or None,
                              allow_patterns=patterns)
        except Exception as e:  # noqa: BLE001
            gated_hint = " (gated? set HF_TOKEN in keys.env)" if not cfg.hf_token else ""
            print(f"    FAILED: {e}{gated_hint}", file=sys.stderr)
            failures.append(c.repo)
            continue
        _purge_repo_cache(c.repo)  # drop the cache duplicate to save disk
        problems = _verify_weights(dest)
        if problems:
            for pr in problems:
                print(f"    INTEGRITY FAILURE: {pr}", file=sys.stderr)
            failures.append(c.repo)
        elif patterns and not _has_weight_file(dest):
            print("    WARNING: allow_patterns matched no weight file — patterns may be wrong "
                  "for this repo (widen them in model_matrix extra.allow_patterns)", file=sys.stderr)
            failures.append(c.repo)
        else:
            print(f"    [{i}/{total}] verified OK.", flush=True)
    if failures:
        print(f"\n{len(failures)} download(s) failed: {failures}", file=sys.stderr)
        return 1
    print("\nAll downloads complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
