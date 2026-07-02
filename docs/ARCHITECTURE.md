# Cineforge architecture

Cineforge = **ViMax's agentic director's brain** (screenwriter/director/producer/
consistency agents, RAG script engine, best-frame selection) wired to the **best 2026
open-weight local models**, auto-tuned to your GPU. Nothing calls the cloud at render
time.

## The three load-bearing ideas

1. **One `Backend` interface** (`backends/base.py`). Every model is a swappable adapter
   with `capabilities()`, `estimate_vram()`, `load/unload`, `generate(request)`.
   Adding a new SOTA model = one adapter file + one row in `data/model_matrix.json`.
   Most inference runs inside a local ComfyUI graph, so adapters are thin.

2. **Data-driven model selection.** `data/model_matrix.json` maps
   `(subsystem × VRAM-tier × license-mode) → model+quant+workflow`.
   `data/licenses.json` gates each model. Neither is code — model churn (fast in 2026)
   is a JSON edit, not a release.

3. **Agents over a shared `Context`.** Each agent is `run(ctx) -> ctx`, mutating the
   `Project` state model. The coordinator runs the ordered stages and checkpoints after
   each, so a crash/OOM/interrupt resumes cleanly.

## Directory

```
src/cineforge/
  config.py            central paths/flags/secrets
  errors.py            typed exceptions (drive retry/skip/abort)
  hardware/            detect GPUs, classify VRAM tier, pick torch runtime (Blackwell/ROCm aware)
  models/              matrix (resolve), licenses (gate), downloader (size-verified), registry, comfy_client
  backends/            base ABC + adapters: video/ image/ voice/ music/ lipsync/ sfx/ enhance/ llm/
  agents/              screenwriter director casting storyboard producer consistency sound_designer editor qa
  pipeline/            coordinator (run loop), stages, checkpoint, best_of_n, asset_store
  state/               Project/Scene/Shot/Take/Character dataclasses + versioned store
  gui/                 stdlib http.server + one offline HTML page
  cli.py               new | run | resume | gui | doctor | models
data/                  model_matrix.json, licenses.json, presets/, luts/
scripts/               download_models.py, verify_install.py, train_character_lora.py
```

## Pipeline (idea → master)

`ingest → screenwriting → direction → casting → storyboard → producing →
keyframes → video → voice → lipsync → sound_design → sfx → music → enhance →
edit → qa → master`

- **Planning stages** are agents (LLM-driven via Ollama).
- **Generation stages** resolve a model, enforce its license, and render via the
  registered backend. On a fresh install (no ComfyUI/weights) they run in **dry mode**:
  they report exactly what they *would* render with which model, then continue — so the
  whole orchestration is inspectable before you download 100 GB.
- **Consistency** is embedded, not a standalone stage: keyframe/video takes are audited
  by the VLM judge (`agents/consistency.py`) in a reject/retry loop.

## Consistency strategy (the "secret sauce", done locally)

Cross-shot character identity is unsolved industry-wide. Cineforge *mitigates* it:
1. Generate one canonical hero keyframe per character (image model + multi-ref / LoRA).
2. Drive every shot's video by **image-to-video off that locked keyframe** (not fresh T2V).
3. Train a **per-character LoRA** for recurring characters (Casting flags who needs one).
4. **Audit** each take with ArcFace/CLIP + the local VLM judge; reject drift, regenerate.
5. **best-of-N** per shot, scored, keep the winner.

## Roadmap

- **v0.1 Scaffold** *(this repo)* — interfaces, hardware/tier/runtime select, size-verified
  downloader, config/state/checkpoint, agents, stdlib GUI, installer, CI (mocked backends).
- **v0.2 Vertical slice** — Wan 2.2 + Z-Image + Chatterbox + ACE-Step wired end-to-end (mid tier); idea → single scene.
- **v0.3 Consistency** — local LoRA training, asset-store RAG, VLM auditor, best-of-N everywhere.
- **v0.4 Audio + polish** — EchoMimicV3 lipsync, HunyuanVideo-Foley + Stable Audio, SeedVR2 + RIFE, ffmpeg two-pass master, QA gates.
- **v0.5 Opt-in premium** — LTX-2.3 (native audio), FLUX.2 multi-ref, InfiniteTalk, IndexTTS-2/Higgs, all behind LicenseGate.
- **v0.6 Tier hardening** — verified 8-12 GB paths (GGUF/offload/BlockSwap), AMD ROCm, honest first-run ETAs, DirectML fallback.
- **v0.7 UX + docs** — full review/approve flow, consent dialogs.
- **v1.0 Model-swap maturity** — monthly SOTA re-verification, one-adapter-file onboarding.
