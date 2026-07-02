import pytest

from cineforge.errors import CheckpointError
from cineforge.state import Character, Dialogue, Project, Scene, Shot, Take, store
from cineforge.state.schema import migrate, validate


def _sample() -> Project:
    p = Project(name="p", idea="idea", logline="a log")
    p.characters = [Character(id="c001", name="Bolt", description="robot")]
    shot = Shot(id="sh001", scene_id="s001", index=1, description="rolls in",
                dialogue=[Dialogue(character="c001", line="hi", emotion="curious")],
                takes=[Take(id="t001", shot_id="sh001", kind="video", path="v.mp4", accepted=True)])
    p.scenes = [Scene(id="s001", index=1, heading="EXT", shots=[shot])]
    return p


def test_roundtrip_preserves_nested(cfg):
    p = _sample()
    pdir = cfg.project_dir("p")
    store.save(p, pdir)
    r = store.load(pdir)
    assert r.name == "p" and r.logline == "a log"
    assert r.characters[0].name == "Bolt"
    sh = r.scenes[0].shots[0]
    assert sh.dialogue[0].line == "hi"
    assert sh.takes[0].kind == "video" and sh.takes[0].accepted is True
    assert sh.accepted_take("video") is not None


def test_validate_rejects_missing_keys():
    with pytest.raises(CheckpointError):
        validate({"schema_version": 1})  # no name
    with pytest.raises(CheckpointError):
        validate({"name": "x"})           # no schema_version


def test_migrate_sets_version():
    raw = {"name": "x", "schema_version": 1}
    assert migrate(raw)["schema_version"] >= 1
