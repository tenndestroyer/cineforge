"""Regression tests for the bugs found in the adversarial review passes."""

from __future__ import annotations

from cineforge.agents import Context, build_agents
from cineforge.hardware import detect_gpus, select_backend
from cineforge.models.matrix import ModelMatrix
from cineforge.models.registry import BackendRegistry
from cineforge.pipeline import Coordinator, checkpoint
from cineforge.state import Project, Take
from cineforge.state.project import SCHEMA_VERSION
from cineforge.state.schema import migrate


# ---- matrix tier degrade ----
def test_matrix_degrade_prefers_nearest_lower_tier():
    only_mid_high = {"version": 1, "subsystems": {"video": {"tiers": {
        "mid": {"safe": {"model_id": "wan22", "min_vram_gb": 16}},
        "high": {"safe": {"model_id": "wan22", "min_vram_gb": 32}}}}}}
    # 'low' requested, absent -> nearest available is 'mid', NOT the VRAM-hungry 'high'
    assert ModelMatrix(only_mid_high).resolve("video", "low", "safe").tier == "mid"

    only_low_mid = {"version": 1, "subsystems": {"video": {"tiers": {
        "low": {"safe": {"model_id": "wan22", "min_vram_gb": 8}},
        "mid": {"safe": {"model_id": "wan22", "min_vram_gb": 16}}}}}}
    # 'high' requested, absent -> nearest lower is 'mid'
    assert ModelMatrix(only_low_mid).resolve("video", "high", "safe").tier == "mid"


# ---- config normalization ----
def test_config_tier_override_normalized(monkeypatch):
    from cineforge.config import Config
    monkeypatch.setenv("CINEFORGE_TIER", "  LOW ")
    assert Config.load().tier_override == "low"


# ---- schema migrate ----
def test_schema_migrate_sets_current_version():
    assert migrate({"name": "x", "schema_version": 1})["schema_version"] == SCHEMA_VERSION


# ---- coordinator resume semantics ----
def _coord_with_fake(cfg, fake_llm) -> Coordinator:
    coord = Coordinator(cfg)

    def build(project):
        return Context(project=project, config=cfg, events=coord.events, llm=fake_llm, vlm=None,
                       tier="low", license_mode="safe", matrix=coord.matrix, gate=coord.gate,
                       plan=select_backend(detect_gpus()))

    coord.build_context = build
    return coord


def test_run_stops_at_first_uninstalled_generate_stage(cfg, fake_llm):
    coord = _coord_with_fake(cfg, fake_llm)
    p = Project(name="rs", idea="a robot gardens")
    coord.run(p)
    # Planning stages complete; 'keyframes' can't render (no ComfyUI) -> run stops there,
    # leaving the pointer at the last COMPLETED stage so resume re-attempts keyframes.
    assert p.stage == "consistency_setup"
    assert checkpoint.next_stage(p) == "keyframes"


def test_run_generate_returns_false_when_not_installed(cfg, fake_llm):
    coord = _coord_with_fake(cfg, fake_llm)
    p = Project(name="ni", idea="x")
    ctx = coord.build_context(p)
    build_agents()["screenwriter"].run(ctx)
    build_agents()["producer"].run(ctx)
    assert coord._run_generate(ctx, "image") is False  # ComfyUI unreachable -> not complete


def test_items_for_skips_already_rendered_shots(cfg, fake_llm):
    coord = _coord_with_fake(cfg, fake_llm)
    p = Project(name="sk", idea="x")
    ctx = coord.build_context(p)
    build_agents()["screenwriter"].run(ctx)
    shots = p.all_shots()
    assert len(shots) == 2
    backend = BackendRegistry.get("wan22", cfg)
    assert len(coord._items_for(ctx, "image", backend)) == 2
    shots[0].takes.append(Take(id="t1", shot_id=shots[0].id, kind="keyframe", path="k.png", accepted=True))
    assert len(coord._items_for(ctx, "image", backend)) == 1  # the done shot is skipped


# ---- asset store robustness ----
def test_asset_store_survives_corrupt_index(tmp_path):
    from cineforge.pipeline.asset_store import Asset, AssetStore
    (tmp_path / "assets.json").write_text("{ not valid json", encoding="utf-8")
    store = AssetStore(tmp_path)   # must not raise
    assert store.assets == []
    store.add(Asset(path="a.png", kind="character_ref", character="c001", tags=["canonical"]))
    assert AssetStore(tmp_path).query_character("c001") is not None


# ---- gui new_project overwrite guard ----
def test_gui_new_project_no_silent_overwrite(cfg):
    from cineforge.gui.api import GuiApi
    api = GuiApi(cfg)
    assert api.new_project("a robot gardens", name="dup").get("name") == "dup"
    again = api.new_project("a robot gardens", name="dup")
    assert again.get("ok") is False and "exists" in again.get("error", "")
    assert api.new_project("a robot gardens", name="dup", overwrite=True).get("ok") is True


# ---- blackwell datacenter detection ----
def test_datacenter_blackwell_routes_to_nightly():
    from cineforge.hardware.detect import GpuInfo
    b200 = GpuInfo(vendor="nvidia", name="NVIDIA B200", vram_gb=180, compute_cap="10.0", backend_hint="cuda")
    assert select_backend([b200]).torch_channel == "cu128-nightly"
