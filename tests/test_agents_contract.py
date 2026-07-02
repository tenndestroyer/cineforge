import pytest

from cineforge.agents import Context, build_agents
from cineforge.errors import NotInstalledError
from cineforge.logging_setup import EventLog
from cineforge.state import Project


def _ctx(cfg, llm=None, vlm=None) -> Context:
    return Context(project=Project(name="t", idea="a robot gardens", style="3D cartoon"),
                   config=cfg, events=EventLog(), llm=llm, vlm=vlm, tier="mid", license_mode="safe")


def test_every_agent_has_name_and_run():
    for name, agent in build_agents().items():
        assert agent.name == name
        assert callable(agent.run)


def test_full_planning_pipeline_runs(cfg, fake_llm, fake_vlm):
    ctx = _ctx(cfg, fake_llm, fake_vlm)
    agents = build_agents()
    order = ["screenwriter", "director", "casting", "storyboard", "producer",
             "consistency", "sound_designer", "editor", "qa"]
    for name in order:
        ctx.stage = name
        out = agents[name].run(ctx)
        assert out is ctx, f"{name}.run must return the Context"

    p = ctx.project
    assert len(p.scenes) == 1
    assert len(p.all_shots()) == 2
    assert len(p.characters) >= 2
    assert all(sh.camera for sh in p.all_shots()), "director should fill camera"
    assert all(sh.keyframe_prompt for sh in p.all_shots()), "storyboard should fill prompts"
    assert p.render_plan.get("render"), "producer should set render settings"
    assert p.render_plan.get("timeline"), "editor should build a timeline"
    assert any(c.kind == "music" for c in p.audio_cues), "sound designer should add a music cue"
    assert p.qa is not None, "qa should produce a report"


def test_screenwriter_requires_llm(cfg):
    ctx = _ctx(cfg, llm=None)
    ctx.stage = "screenwriting"
    with pytest.raises(NotInstalledError):
        build_agents()["screenwriter"].run(ctx)


def test_director_degrades_without_llm(cfg, fake_llm):
    # build a script first, then run director with no LLM -> defaults, no crash
    ctx = _ctx(cfg, fake_llm)
    build_agents()["screenwriter"].run(ctx)
    ctx.llm = None
    build_agents()["director"].run(ctx)
    assert all(sh.camera for sh in ctx.project.all_shots())
