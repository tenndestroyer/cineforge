from cineforge.pipeline import checkpoint
from cineforge.state import Project, store


def test_checkpoint_roundtrip_and_resume(cfg):
    p = Project(name="demo", idea="a robot gardens")
    pdir = cfg.project_dir("demo")

    checkpoint.save(p, pdir, "direction")
    reloaded = store.load(pdir)
    assert reloaded.stage == "direction"

    assert checkpoint.is_done(reloaded, "screenwriting")
    assert checkpoint.is_done(reloaded, "direction")
    assert not checkpoint.is_done(reloaded, "video")
    assert checkpoint.next_stage(reloaded) == "casting"


def test_next_stage_none_at_end(cfg):
    p = Project(name="d2", idea="x", stage="master")
    assert checkpoint.next_stage(p) is None
