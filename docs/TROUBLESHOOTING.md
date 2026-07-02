# Troubleshooting

Run `cineforge doctor` first — it pinpoints most of these.

### The window opens and closes instantly (Windows)
`Run.bat` must stay a **flat goto-label** script. Big parenthesized `if (...)` blocks with
escaped parens/special chars make cmd.exe abort with `. was unexpected at this time`. If you
edit it, keep the goto structure.

### "torch has no matching distribution" / torch won't install
You're on Python 3.13/3.14. torch ships **no wheels** for those. Cineforge pins CPython 3.12
in `python_embeded/` for exactly this reason — use `Run.bat`/`run.sh`, not a system 3.13/3.14.

### NVIDIA RTX 50-series (Blackwell): CUDA "not available" or a kernel error
Stable torch still lacks **sm_120** kernels. The installer detects Blackwell and installs a
**nightly** `cu128` build. If you installed torch manually, reinstall from
`download.pytorch.org/whl/nightly/cu128`. This is the fastest-moving fact in the stack — recheck monthly.

### AMD: renders are absurdly slow or fail on start
GGUF is broken/unusably slow on ROCm (especially Windows). Cineforge defaults AMD to
**fp8/safetensors**. If ROCm torch won't import, it falls back to `torch-directml` (degraded,
no real VRAM management). Prefer a ROCm-supported card (RX 7000/9000 / W7000).

### A model "installed" but fails to load — file is ~136 bytes
Classic **gated-download false positive**: a token-less pull of a gated repo (Flux,
`google/gemma-3-12b-it`) silently writes an HTML error page. Cineforge's downloader verifies
**byte size**, so this shouldn't slip through — but if you fetched manually, delete the tiny
file, put a valid `HF_TOKEN` in `keys.env`, and re-run `scripts/download_models.py`.

### "Ollama not reachable" — the pipeline stops at screenwriting
The planning agents need a local LLM. Install [Ollama](https://ollama.com) and
`ollama pull qwen2.5:7b qwen2.5vl:7b`, then re-run. (Generation stages don't need Ollama.)

### "ComfyUI not running" in doctor
Expected unless a render is active — ComfyUI is launched on demand. It's only a problem if a
render fails to start.

### A generation stage says "not installed — N item(s) pending"
That's **dry mode**: the orchestration ran, but the weights/ComfyUI nodes for that model
aren't installed yet. Run `setup` (or `scripts/download_models.py --confirm`) and re-run;
checkpoint/resume picks up where it left off (`cineforge resume <name>`).

### A gated model is "blocked" / "needs consent"
Safe mode blocks non-permissive models. Switch to Research mode (`CINEFORGE_LICENSE_MODE=research`
or the GUI toggle) and acknowledge the model's terms. See `docs/LICENSES.md`.

### Out of VRAM mid-render
Force a lower tier (`CINEFORGE_TIER=low`) or let the Producer's best-of-N drop to 1. The 8–12 GB
tier is genuinely tight — see `docs/QUALITY_CEILING.md`.
