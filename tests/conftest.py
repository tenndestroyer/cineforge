"""Shared fixtures: a tmp-homed Config, a fake LLM/VLM, and a fake backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from cineforge.config import Config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(tmp_path) -> Config:
    """Real repo data/ (so matrix + licenses load) but an isolated tmp home.

    comfy_url/ollama_url point at dead ports so tests are deterministic even when a
    real ComfyUI/Ollama happens to be running on the dev machine (otherwise a
    generation stage would actually render against it)."""
    c = Config(repo_root=REPO_ROOT, home=tmp_path / "CineforgeData")
    c.comfy_url = "http://127.0.0.1:59571"
    c.ollama_url = "http://127.0.0.1:59572"
    c.ensure_dirs()
    return c


class FakeLLM:
    """Deterministic stand-in for OllamaLLM so agents run without Ollama."""

    available_ = True

    def available(self) -> bool:
        return self.available_

    def chat(self, messages, temperature=0.7) -> str:
        return "ok"

    def complete(self, prompt, system=None, temperature=0.7) -> str:
        return "ok"

    def json(self, prompt: str, system: str | None = None, temperature: float = 0.3) -> dict:
        # Branch on distinctive markers unique to each agent's prompt template.
        if "scene OUTLINE" in prompt:  # screenwriter outline pass
            return {
                "logline": "A little robot learns to garden.",
                "characters": [
                    {"name": "Bolt", "description": "a curious round robot"},
                    {"name": "Snappy", "description": "a grumpy snail"},
                ],
                "scenes": [{"heading": "EXT. GARDEN - DAY",
                            "description": "Bolt discovers a wilting plant.", "target_shots": 2}],
            }
        if "shots for THIS scene" in prompt:  # screenwriter per-scene pass
            return {"shots": [
                {"description": "Bolt rolls up to a droopy flower", "duration_s": 4,
                 "dialogue": [{"character": "Bolt", "line": "Oh no, you look sad!", "emotion": "curious"}]},
                {"description": "Snappy peeks out grumpily", "duration_s": 3,
                 "dialogue": [{"character": "Snappy", "line": "Keep it down.", "emotion": "grumpy"}]},
            ]}
        if "keyframe_prompt" in prompt:  # storyboard
            return {"shots": [{"id": "sh001", "keyframe_prompt": "Bolt by a flower, 3D cartoon", "pose": "leaning in"},
                              {"id": "sh002", "keyframe_prompt": "Snappy peeking, 3D cartoon", "pose": "peeking"}]}
        if "camera framing" in prompt:  # director
            return {"shots": [{"id": "sh001", "camera": "wide", "movement": "push-in", "duration_s": 4},
                              {"id": "sh002", "camera": "close-up", "movement": "static", "duration_s": 3}]}
        if "reference_prompt" in prompt:  # casting
            return {"characters": [{"name": "Bolt", "reference_prompt": "round robot sheet",
                                    "voice": {"style": "bright", "exaggeration": 0.6}}]}
        if "foley" in prompt:  # sound designer
            return {"shots": [{"id": "sh001", "foley": "servo whir", "ambience": "birdsong"}],
                    "music": {"style": "gentle ukulele", "lyrics": None}}
        return {}


class FakeVLM:
    def score(self, image, reference, criteria="") -> float:
        return 0.9

    def caption(self, image) -> str:
        return "a frame"


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_vlm() -> FakeVLM:
    return FakeVLM()
