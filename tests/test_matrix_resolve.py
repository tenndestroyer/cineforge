from cineforge.models.matrix import ModelMatrix
from cineforge.models.registry import BackendRegistry


def _matrix(cfg):
    return ModelMatrix.load(cfg.data_dir / "model_matrix.json")


def test_every_subsystem_resolves_all_tiers(cfg):
    m = _matrix(cfg)
    for sub in m.subsystems():
        for tier in ("low", "mid", "high"):
            for mode in ("safe", "research"):
                c = m.resolve(sub, tier, mode)
                assert c.model_id, f"{sub}/{tier}/{mode} has no model"
                assert c.subsystem == sub


def test_safe_vs_research_video_mid(cfg):
    m = _matrix(cfg)
    assert m.resolve("video", "mid", "safe").model_id == "wan22"
    assert m.resolve("video", "mid", "research").model_id == "ltx23"


def test_every_matrix_model_has_adapter(cfg):
    m = _matrix(cfg)
    for sub in m.subsystems():
        for tier in ("low", "mid", "high"):
            for mode in ("safe", "research"):
                c = m.resolve(sub, tier, mode)
                assert BackendRegistry.has(c.model_id), f"no adapter for {c.model_id}"
