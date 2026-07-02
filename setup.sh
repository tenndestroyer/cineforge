#!/usr/bin/env bash
# Cineforge setup (Linux). Mirrors setup.ps1: CUDA/ROCm/CPU torch branches (no
# DirectML on Linux), ComfyUI + nodes, Ollama, size-verified weights. Idempotent.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"
DATA="$ROOT/CineforgeData"
COMFY="$DATA/ComfyUI"

info() { printf '\033[36m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[setup]\033[0m %s\n' "$*"; }
die()  { printf '\033[31m[setup] ERROR:\033[0m %s\n' "$*"; exit 1; }

# ---- 1. python 3.12 venv (torch has no 3.13/3.14 wheels) ----
PYBIN="$(command -v python3.12 || true)"
[ -n "$PYBIN" ] || PYBIN="$(command -v python3 || true)"
[ -n "$PYBIN" ] || die "python3 not found (need 3.10-3.12)"
PYVER="$("$PYBIN" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
case "$PYVER" in 3.10|3.11|3.12) : ;; *) warn "python $PYVER may lack torch wheels; 3.12 recommended" ;; esac
[ -d "$VENV" ] || { info "creating venv ($PYVER)"; "$PYBIN" -m venv "$VENV"; }
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip setuptools wheel >/dev/null

# ---- 2. torch per backend ----
if command -v nvidia-smi >/dev/null 2>&1; then
  NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true)"
  info "NVIDIA GPU: $NAME"
  if echo "$NAME" | grep -Eq 'RTX 50|5090|5080|5070|5060|Blackwell'; then
    warn "Blackwell -> nightly torch (stable lacks sm_120 kernels)"
    "$PY" -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
  else
    "$PY" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  fi
elif command -v rocminfo >/dev/null 2>&1; then
  warn "AMD/ROCm detected -> ROCm torch"
  "$PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2 || warn "ROCm wheel failed"
else
  warn "no GPU -> CPU torch (video generation impractical)"
  "$PY" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# ---- 3. cineforge + deps ----
info "installing Cineforge (editable)"
"$PY" -m pip install -e "$ROOT"

# ---- 4. ComfyUI + nodes (+ their requirements) ----
if command -v git >/dev/null 2>&1; then
  [ -d "$COMFY" ] || { info "cloning ComfyUI"; git clone --depth 1 https://github.com/comfyanonymous/ComfyUI "$COMFY"; }
  [ -f "$COMFY/requirements.txt" ] && { info "installing ComfyUI requirements"; "$PY" -m pip install -r "$COMFY/requirements.txt"; }
  mkdir -p "$COMFY/custom_nodes"
  for r in city96/ComfyUI-GGUF kijai/ComfyUI-WanVideoWrapper Lightricks/ComfyUI-LTXVideo \
           Fannovel16/ComfyUI-Frame-Interpolation diodiogod/TTS-Audio-Suite ace-step/ACE-Step-ComfyUI \
           smthemex/ComfyUI_EchoMimic smthemex/ComfyUI_StableAudio_Open numz/ComfyUI-SeedVR2_VideoUpscaler; do
    d="$COMFY/custom_nodes/$(basename "$r")"
    [ -d "$d" ] || git clone --depth 1 "https://github.com/$r" "$d"
    [ -f "$d/requirements.txt" ] && "$PY" -m pip install -r "$d/requirements.txt" || true
  done
else
  warn "git not found; skipping ComfyUI clone"
fi

# ---- 5. Ollama ----
if command -v ollama >/dev/null 2>&1; then
  info "pulling Ollama models"; ollama pull qwen2.5:7b; ollama pull qwen2.5vl:7b
else
  warn "Ollama not installed. See https://ollama.com then: ollama pull qwen2.5:7b qwen2.5vl:7b"
fi

# ---- 6. weights (size-verified) ----
if [ -z "${CINEFORGE_SKIP_WEIGHTS:-}" ]; then
  info "downloading weights for the wired render stages (image + video)"
  "$PY" "$ROOT/scripts/download_models.py" --tier auto --license-mode safe --confirm
else
  warn "CINEFORGE_SKIP_WEIGHTS set -> skipping weights"
fi

# ---- 7. gate on torch import ----
"$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" || die "torch import failed; not writing .installed"
date -u +%FT%TZ > "$ROOT/.installed"
info "done. launch with ./run.sh"
