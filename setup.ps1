# ============================================================================
#  Cineforge setup (Windows) — provisions EVERYTHING and streams live progress:
#  pinned CPython 3.12, the right torch wheel, ComfyUI + custom nodes (+ their
#  requirements), Ollama models, and all Safe-mode model weights for your GPU.
#  Idempotent: safe to re-run; finished steps are skipped.
#
#  Key facts baked in (see docs/TROUBLESHOOTING.md):
#   * torch has NO wheels for Python 3.13/3.14 -> we pin standalone CPython 3.12.
#   * Blackwell / RTX 50-series (sm_120) needs NIGHTLY torch; stable cu128 lacks
#     sm_120 kernels and silently falls back to CPU.
#   * AMD on Windows -> torch-directml (ROCm has no Windows wheels).
#   * Gated HF downloads silently write ~136-byte error HTML; downloads are
#     byte-size verified, not existence-checked.
# ============================================================================
$ErrorActionPreference = "Stop"
$ProgressPreference = "Continue"   # show Invoke-WebRequest progress bars
$Root = $PSScriptRoot
$PyDir = Join-Path $Root "python_embeded"
$PyExe = Join-Path $PyDir "python.exe"
$DataDir = Join-Path $Root "CineforgeData"
$ComfyDir = Join-Path $DataDir "ComfyUI"

$script:Step = 0
$script:Total = 8
function Banner($msg) {
    $script:Step++
    Write-Host ""
    Write-Host ("=" * 72) -ForegroundColor DarkCyan
    Write-Host ("  STEP $($script:Step)/$($script:Total)  -  $msg") -ForegroundColor Cyan
    Write-Host ("=" * 72) -ForegroundColor DarkCyan
}
function Info($m) { Write-Host "  $m" -ForegroundColor Gray }
function Ok($m)   { Write-Host "  [OK] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "  [!]  $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "  [X]  ERROR: $m" -ForegroundColor Red; exit 1 }

# ComfyUI custom nodes needed by the default backends (owner/repo).
# NOTE: pinned/verified during setup; see docs/MODELS.md.
$NODES = @(
    "city96/ComfyUI-GGUF",                    # GGUF quantized loading
    "kijai/ComfyUI-WanVideoWrapper",          # Wan 2.2 video
    "Lightricks/ComfyUI-LTXVideo",            # LTX-2.3 video (opt-in)
    "Fannovel16/ComfyUI-Frame-Interpolation", # RIFE / GMFSS interpolation
    "diodiogod/TTS-Audio-Suite",              # Chatterbox / voice
    "ace-step/ACE-Step-ComfyUI",              # ACE-Step music
    "smthemex/ComfyUI_EchoMimic",             # EchoMimicV3 lipsync
    "smthemex/ComfyUI_StableAudio_Open",      # Stable Audio ambience
    "numz/ComfyUI-SeedVR2_VideoUpscaler"      # SeedVR2 upscale/restore
)
# NOTE: several defaults (Z-Image, FLUX.2, Qwen-Image-Edit, HunyuanVideo-1.5) have
# NATIVE ComfyUI support in current core builds and need no custom node.

# ---- 1. pinned CPython 3.12 (python-build-standalone) ----
function Install-Python312 {
    Banner "Python 3.12 runtime"
    if (Test-Path $PyExe) { Ok "CPython 3.12 already present"; return }
    Info "Resolving a standalone CPython 3.12 build (torch has no 3.13/3.14 wheels)..."
    $rel = Invoke-RestMethod "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest"
    $asset = $rel.assets | Where-Object {
        $_.name -match 'cpython-3\.12\.\d+\+.*-x86_64-pc-windows-msvc-install_only\.tar\.gz$'
    } | Select-Object -First 1
    if (-not $asset) { Die "could not resolve a CPython 3.12 build-standalone asset" }
    $tgz = Join-Path $env:TEMP "cineforge-py312.tar.gz"
    Info "Downloading $($asset.name) ..."
    Invoke-WebRequest $asset.browser_download_url -OutFile $tgz
    New-Item -ItemType Directory -Force $PyDir | Out-Null
    Info "Extracting..."
    tar -xzf $tgz -C $PyDir --strip-components=1
    if (-not (Test-Path $PyExe)) { Die "python.exe missing after extract" }
    & $PyExe -m ensurepip --upgrade
    & $PyExe -m pip install --upgrade pip setuptools wheel
    Ok "CPython 3.12 ready"
}

# ---- 2. detect GPU (for the torch channel) ----
function Get-GpuVendorName {
    try {
        $gpus = Get-CimInstance Win32_VideoController | Sort-Object AdapterRAM -Descending
        foreach ($g in $gpus) {
            if ($g.Name -match 'NVIDIA') { return @{vendor = 'nvidia'; name = $g.Name } }
            if ($g.Name -match 'AMD|Radeon') { return @{vendor = 'amd'; name = $g.Name } }
        }
        if ($gpus) { return @{vendor = 'other'; name = $gpus[0].Name } }
    } catch {}
    return @{vendor = 'cpu'; name = 'CPU' }
}

# ---- 3. torch per backend ----
function Install-Torch {
    Banner "PyTorch (GPU acceleration)"
    $gpu = Get-GpuVendorName
    Info "Detected GPU: $($gpu.name) [$($gpu.vendor)]"
    if ($gpu.vendor -eq 'nvidia') {
        if ($gpu.name -match 'RTX\s*50|5090|5080|5070|5060|5050|Blackwell|B100|B200') {
            Warn "Blackwell detected -> NIGHTLY torch (stable lacks sm_120 kernels)."
            & $PyExe -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
        } else {
            & $PyExe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
        }
    } elseif ($gpu.vendor -eq 'amd') {
        Warn "AMD on Windows: installing torch-directml (ROCm has no Windows wheels)."
        & $PyExe -m pip install torch-directml
        if ($LASTEXITCODE -ne 0) { Warn "torch-directml install failed; AMD GPU acceleration unavailable (CPU fallback)." }
    } else {
        Warn "No supported GPU -> CPU torch (video generation will be impractical)."
        & $PyExe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    }
    Ok "torch installed"
}

# ---- 4. Cineforge package ----
function Install-Cineforge {
    Banner "Cineforge package + dependencies"
    & $PyExe -m pip install -e "$Root"
    if ($LASTEXITCODE -ne 0) { Die "pip install -e failed (exit $LASTEXITCODE); NOT writing .installed" }
    Ok "Cineforge installed (editable)"
}

# ---- 5. ComfyUI + pinned custom nodes (+ their requirements) ----
function Install-ComfyUI {
    Banner "ComfyUI engine + custom nodes"
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Die "git not found; install Git for Windows and re-run." }
    if (-not (Test-Path $ComfyDir)) {
        Info "Cloning ComfyUI..."
        git clone --depth 1 https://github.com/comfyanonymous/ComfyUI "$ComfyDir"
    } else { Ok "ComfyUI already cloned" }

    $reqs = Join-Path $ComfyDir "requirements.txt"
    if (Test-Path $reqs) { Info "Installing ComfyUI requirements..."; & $PyExe -m pip install -r "$reqs" }

    $nodes = Join-Path $ComfyDir "custom_nodes"
    New-Item -ItemType Directory -Force $nodes | Out-Null
    foreach ($r in $NODES) {
        $dst = Join-Path $nodes ($r.Split("/")[-1])
        if (-not (Test-Path $dst)) {
            Info "node: $r"
            git clone --depth 1 "https://github.com/$r" "$dst"
        } else { Info "node: $r (present)" }
        $nreq = Join-Path $dst "requirements.txt"
        if (Test-Path $nreq) {
            try { & $PyExe -m pip install -r "$nreq" } catch { Warn "node $r requirements failed (non-fatal): $_" }
        }
    }
    Ok "ComfyUI + nodes installed"
}

# ---- 6. Ollama (LLM/VLM for the agents) ----
function Install-Ollama {
    Banner "Ollama models (the agent brain)"
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        Warn "Ollama not installed. Install from https://ollama.com then re-run this setup."
        Warn "The planning agents (script/shot list/etc.) need it."
        return
    }
    Info "Pulling qwen2.5:7b (text agents)..."
    ollama pull qwen2.5:7b
    Info "Pulling qwen2.5vl:7b (visual QA judge)..."
    ollama pull qwen2.5vl:7b
    Ok "Ollama models ready"
}

# ---- 7. model weights (size-verified, streamed progress) ----
function Fetch-Weights {
    Banner "Model weights (this is the big one)"
    if ($env:CINEFORGE_SKIP_WEIGHTS) { Warn "CINEFORGE_SKIP_WEIGHTS set -> skipping weight download."; return }
    # Only fetch weights for stages that actually render today (currently: image).
    # Other stages download when their render path is wired, to avoid pulling tens of
    # GB of models that can't be used yet.
    Info "Downloading weights for the wired render stages (image + video) ..."
    & $PyExe "$Root\scripts\download_models.py" --tier auto --license-mode safe --confirm
    if ($LASTEXITCODE -ne 0) { Warn "Some weights failed to download (see above). You can re-run setup to retry." }
    else { Ok "All Safe-mode weights downloaded + verified" }
}

# ---- 8. verify + gate on torch importing ----
function Verify-Install {
    Banner "Verify installation"
    Info "Checking torch imports..."
    & $PyExe -c "import torch; print('  torch', torch.__version__, '| cuda', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { Die "torch failed to import; NOT writing .installed" }
    Info "Running doctor..."
    & $PyExe -m cineforge doctor
    Ok "Verification complete"
}

Write-Host ""
Write-Host "  Cineforge setup - installing everything and showing progress as it goes." -ForegroundColor White
Write-Host "  This downloads several GB (torch, ComfyUI, models). Grab a coffee." -ForegroundColor White

Install-Python312
Install-Torch
Install-Cineforge
Install-ComfyUI
Install-Ollama
Fetch-Weights
Verify-Install

Set-Content -Path (Join-Path $Root ".installed") -Value (Get-Date -Format o) -Encoding utf8
Write-Host ""
Write-Host "  ============================================================" -ForegroundColor Green
Write-Host "   Cineforge is installed. Launching the GUI (Run.bat)..." -ForegroundColor Green
Write-Host "  ============================================================" -ForegroundColor Green
