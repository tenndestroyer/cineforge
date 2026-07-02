from cineforge.backends.base import Backend
from cineforge.models.registry import BackendRegistry


def test_registry_populated():
    allb = BackendRegistry.all()
    assert len(allb) >= 20
    assert "wan22" in allb and "chatterbox" in allb and "acestep15" in allb


def test_for_subsystem():
    vids = {c.model_id for c in BackendRegistry.for_subsystem("video")}
    assert {"wan22", "ltx23", "hunyuanvideo15"} <= vids


def test_get_returns_backend(cfg):
    b = BackendRegistry.get("wan22", cfg)
    assert isinstance(b, Backend)
    assert b.subsystem == "video"
    assert b.model_id == "wan22"


def test_every_backend_declares_subsystem():
    from cineforge.backends.base import SUBSYSTEMS

    for mid, cls in BackendRegistry.all().items():
        assert cls.subsystem in SUBSYSTEMS, f"{mid} has bad subsystem {cls.subsystem!r}"


def test_estimate_vram_is_numeric(cfg):
    for mid in ("wan22", "zimage", "chatterbox", "acestep15", "echomimic3", "seedvr2"):
        b = BackendRegistry.get(mid, cfg)
        assert isinstance(b.estimate_vram(None), float)
