# Install

## TL;DR

**Windows:** double-click `Run.bat`. **Linux/macOS:** `./run.sh`. The first run installs
everything; subsequent runs just launch the GUI.

## What the installer does

1. Provisions a **pinned standalone CPython 3.12** (`python_embeded/` on Windows,
   `.venv/` on Linux). torch has **no wheels for 3.13/3.14**, so we never use a newer system Python.
2. Installs the **right torch** for your GPU:
   - NVIDIA Ada/Ampere/Hopper → `cu128` stable.
   - NVIDIA **Blackwell / RTX 50-series (sm_120)** → `cu128` **nightly** (stable lacks sm_120 kernels).
   - AMD → **ROCm** wheels, verified; falls back to `torch-directml` if ROCm can't import.
   - No GPU → CPU torch (video generation is impractical; everything else works, slowly).
3. Installs the `cineforge` package + deps.
4. Clones **ComfyUI** + pinned custom nodes (GGUF, LTXVideo, WanVideoWrapper, Frame-Interpolation, …).
5. Installs **Ollama** models (`qwen2.5:7b` for the agents, `qwen2.5vl:7b` for the visual judge).
6. Downloads **Safe-mode weights** for your detected tier (size-verified).
7. Writes `.installed` **only after `import torch` succeeds**.

## Prerequisites

- **git** and **ffmpeg** on PATH (ffmpeg is used for the final master).
- **Ollama** ([ollama.com](https://ollama.com)) — needed for the planning agents.
- ~**120 GB** free disk for a full multi-tier model install.
- A Hugging Face token in `keys.env` **only** if you enable gated/opt-in models
  (Flux.2, LTX-2.3). The Apache/MIT defaults need no token. See `keys.env.example`.

## Tiers (auto-detected, override with `CINEFORGE_TIER`)

| Tier | VRAM | Expect |
|---|---|---|
| low  | 8–15 GB | short, softer, lower-res clips; minutes per clip |
| mid  | 16–31 GB | RTX 4090-class; the sweet spot |
| high | 32 GB+ | native 4K / concurrent hero models |

## Verify

```
cineforge doctor     # torch+CUDA, ComfyUI, Ollama, ffmpeg, weights, registry
cineforge models     # the model matrix resolved for YOUR gpu
```

## Environment variables

| Var | Effect |
|---|---|
| `HF_TOKEN` | token for gated downloads (also read from `keys.env`) |
| `CINEFORGE_HOME` | move the data dir (weights/comfy/projects) off the repo |
| `CINEFORGE_TIER` | force `low`/`mid`/`high` |
| `CINEFORGE_LICENSE_MODE` | `safe` (default) or `research` |
| `CINEFORGE_SKIP_WEIGHTS` | install code only; download weights later |
