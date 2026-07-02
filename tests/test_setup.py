"""Tests for the onboarding/install flow, AMD routing, and the ComfyUI launcher."""

from __future__ import annotations

import sys

from cineforge.hardware.detect import GpuInfo


def test_install_status_structure(cfg):
    from cineforge.scripts_verify import install_status
    st = install_status(cfg)
    assert isinstance(st["ready"], bool)
    keys = {c["key"] for c in st["components"]}
    assert {"torch", "comfyui", "ollama", "ffmpeg", "weights"} <= keys
    assert "gpu" in st and "has_hf_token" in st


def test_save_keys_writes_env(cfg, tmp_path):
    from cineforge.gui.api import GuiApi
    api = GuiApi(cfg)
    api.cfg.repo_root = tmp_path  # redirect keys.env to a temp dir
    r = api.save_keys("hf_abc123")
    assert r["ok"]
    assert (tmp_path / "keys.env").read_text(encoding="utf-8").strip().endswith("hf_abc123")
    assert api.cfg.hf_token == "hf_abc123"


def test_comfy_server_not_installed(cfg):
    from cineforge.models import comfy_server
    assert comfy_server.installed(cfg) is False
    # Not installed -> ensure_running must return False (never tries to launch)
    assert comfy_server.ensure_running(cfg) is False


def test_amd_windows_routes_to_directml(monkeypatch):
    import cineforge.hardware.backend_select as bs
    monkeypatch.setattr(sys, "platform", "win32")
    gpu = GpuInfo(vendor="amd", name="AMD Radeon RX 7800 XT", vram_gb=16, backend_hint="directml")
    plan = bs.select_backend([gpu])
    assert plan.runtime == "directml"
    assert any("DirectML" in w for w in plan.warnings)


def test_amd_linux_routes_to_rocm(monkeypatch):
    import cineforge.hardware.backend_select as bs
    monkeypatch.setattr(sys, "platform", "linux")
    gpu = GpuInfo(vendor="amd", name="AMD Radeon RX 7900 XTX", vram_gb=24, backend_hint="rocm")
    plan = bs.select_backend([gpu])
    assert plan.runtime == "rocm"


def test_vendor_from_name():
    from cineforge.hardware.detect import _vendor_from_name
    assert _vendor_from_name("NVIDIA GeForce RTX 5070 Ti")[0] == "nvidia"
    assert _vendor_from_name("AMD Radeon RX 7900 XTX")[0] == "amd"
    assert _vendor_from_name("Intel Arc A770")[0] == "intel"
