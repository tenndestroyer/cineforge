# Cineforge

**A fully-local, fully-offline, open-source AI animated-video studio.**

Cineforge takes an idea, a script, or a novel and drives it through an agentic
production pipeline — screenwriter → director → casting → storyboard → keyframes →
video → voice → lip-sync → sound design → music → enhance → edit → QA → master —
using the best **open-weight local models of 2026**, auto-tuned to your GPU. No
cloud APIs at render time. Nothing leaves your machine.

It borrows the **agentic "director's brain"** of [HKUDS/ViMax](https://github.com/HKUDS/ViMax)
(Director / Screenwriter / Producer / Consistency agents, RAG script engine,
best-frame selection) and replaces ViMax's *cloud* generation backends
(Google Veo / Gemini / Nanobanana) with local ones.

---

## ⚠️ Read this first: the honest quality ceiling

Cineforge produces a **polished, character-consistent, stylized AI animated short** —
**not** rigged Pixar/DreamWorks/Moonbug-grade animation, and it does not auto-produce
broadcast content. This is a limitation of *local open-source AI in 2026*, not of
Cineforge's engineering.

The single biggest reason: the "Pixar-ish" demos you see from tools like Veo come
from **frontier cloud models**. The moment you require *fully local + offline*, you
are on open weights (Wan 2.2, LTX-2.3, HunyuanVideo, Z-Image, Flux) whose realistic
ceiling is a *strong indie stylized 3D look with visible character drift over long
sequences*. Cineforge pushes that ceiling as far as it goes (keyframe-locking,
per-character LoRA, a reject/retry consistency auditor, best-of-N, a real enhance +
master chain) and is **modular** so next quarter's stronger open models drop in as a
single adapter file. See [`docs/QUALITY_CEILING.md`](docs/QUALITY_CEILING.md) for a
per-subsystem breakdown.

---

## What you get

- **Auto hardware tuning** — detects NVIDIA / AMD / Intel / CPU, classifies a VRAM
  tier (low 8–12 GB · mid 16–24 GB · high 32 GB+), and picks the best model + quant
  per tier. Special-cases Blackwell (RTX 50-series, `sm_120`) onto a nightly torch
  branch so it doesn't silently fall back to CPU.
- **One common `Backend` interface** — every model (video/image/voice/music/lipsync/
  sfx/enhance/llm) is a swappable adapter. Add a new SOTA model = one adapter file +
  one row in `data/model_matrix.json`.
- **Agentic pipeline** — 9 specialized agents (screenwriter, director, casting,
  storyboard, producer, consistency, sound-designer, editor, QA) coordinate the run,
  with **checkpoint/resume** so a crash or OOM doesn't restart from zero.
- **License-aware** — a **Safe mode** (Apache/MIT models only) is the default. Gated
  models (LTX-2.3's revenue cap, HunyuanVideo's EU/UK/KR exclusion, Flux2 non-commercial,
  IndexTTS-2's bilibili terms) are opt-in behind an explicit consent dialog. See
  [`docs/LICENSES.md`](docs/LICENSES.md).
- **Offline GUI** — a stdlib `http.server` on `127.0.0.1` serving one self-contained
  HTML page (no CDN, no build step, no telemetry).
- **Bulletproof installer** — `Run.bat` (Windows) / `run.sh` (Linux) provisions a
  pinned CPython 3.12, the right torch wheel, ComfyUI, and size-verified model weights.

## Quick start (Windows)

```bat
:: 1) put your Hugging Face token in keys.env (only needed for gated/opt-in models)
copy keys.env.example keys.env    &&  notepad keys.env

:: 2) one click — installs everything the first time, launches the GUI thereafter
Run.bat
```

Linux:

```bash
cp keys.env.example keys.env   # optional, only for gated models
./run.sh
```

The first run downloads ~30–100 GB of weights depending on your tier (this is the
long part). After that, Cineforge is fully offline.

### CLI

```bash
cineforge doctor      # verify torch+CUDA, ComfyUI, weight integrity
cineforge models      # show the model matrix resolved for YOUR gpu
cineforge new "a brave little robot who learns to garden"   # scaffold a project
cineforge run  <project>      # run the pipeline
cineforge resume <project>    # resume after a crash/interrupt
cineforge gui                 # launch the local GUI
```

## Requirements

- Windows 10/11 or Linux. macOS is best-effort (MPS; most video models are unsupported).
- An NVIDIA GPU with ≥8 GB VRAM is strongly recommended (16–24 GB for the good stuff).
  AMD (ROCm) is supported with a DirectML fallback; Intel/CPU works but is slow.
- ~120 GB free disk for a full multi-tier model install.

## Documentation

| Doc | What |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | How the pieces fit together |
| [`docs/INSTALL.md`](docs/INSTALL.md) | Full install + hardware notes |
| [`docs/QUALITY_CEILING.md`](docs/QUALITY_CEILING.md) | The honest, per-subsystem ceiling |
| [`docs/LICENSES.md`](docs/LICENSES.md) | Per-model license matrix + gates |
| [`docs/MODELS.md`](docs/MODELS.md) | How to update the model matrix |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | When things break |

## Status

**v0.1 — scaffold.** The architecture, orchestration, agents, GUI, installer, and tests
are real: the package compiles, `ruff` is clean, and **45 unit tests pass** (mocked
backends, no GPU). The code has been through two adversarial review passes; the confirmed
findings — including long-form checkpoint/resume, per-shot skip on re-run, chunked
long-form scripting, and a real ffmpeg master-assembly — are fixed. Wiring each backend
end-to-end against installed ComfyUI nodes + weights is the v0.2→v0.5 work tracked in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#roadmap). This repo is honest about being a
foundation you extend, not a finished render farm — see
[`docs/QUALITY_CEILING.md`](docs/QUALITY_CEILING.md#long-form-1520-minutes--render-time--the-honest-numbers)
for what a 15–20 minute render actually costs.

## Ethics

Voice/likeness cloning carries impersonation risk regardless of a model's license.
The GUI requires you to attest you have the rights to any cloned voice or likeness.
Do not use Cineforge to deceive, impersonate, or harm.

## License

Cineforge's own code is **Apache-2.0** (see [`LICENSE`](LICENSE)). The AI models it
downloads carry their **own separate licenses** — read [`docs/LICENSES.md`](docs/LICENSES.md).
