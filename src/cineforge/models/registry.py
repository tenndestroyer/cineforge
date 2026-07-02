"""BackendRegistry — maps a model id to its Backend adapter class.

Adapters self-register at import time via the @register decorator. `_autoload()`
imports the cineforge.backends package once so callers never have to.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from ..errors import BackendError

if TYPE_CHECKING:
    from ..backends.base import Backend
    from ..config import Config
    from .matrix import ModelChoice

_REGISTRY: dict[str, type] = {}
_AUTOLOADED = False


def register(model_id: str, *, subsystem: str | None = None, license_id: str | None = None):
    """Class decorator: register a Backend subclass under `model_id`."""

    def deco(cls):
        cls.model_id = model_id
        if subsystem is not None:
            cls.subsystem = subsystem
        if license_id is not None:
            cls.license_id = license_id
        if not cls.subsystem:
            raise BackendError(f"{cls.__name__} registered as {model_id!r} without a subsystem")
        _REGISTRY[model_id] = cls
        return cls

    return deco


def _autoload() -> None:
    global _AUTOLOADED
    if not _AUTOLOADED:
        _AUTOLOADED = True
        importlib.import_module("cineforge.backends")


class BackendRegistry:
    @staticmethod
    def all() -> dict[str, type]:
        _autoload()
        return dict(_REGISTRY)

    @staticmethod
    def has(model_id: str) -> bool:
        _autoload()
        return model_id in _REGISTRY

    @staticmethod
    def cls_for(model_id: str) -> type:
        _autoload()
        if model_id not in _REGISTRY:
            raise BackendError(f"no backend registered for model id {model_id!r}")
        return _REGISTRY[model_id]

    @staticmethod
    def for_subsystem(subsystem: str) -> list[type]:
        _autoload()
        return [c for c in _REGISTRY.values() if getattr(c, "subsystem", "") == subsystem]

    @staticmethod
    def get(model_id: str, cfg: Config, choice: ModelChoice | None = None) -> Backend:
        cls = BackendRegistry.cls_for(model_id)
        return cls(cfg, choice)


# Convenience for tests / internal use: register a class without the decorator.
def register_class(cls, model_id: str | None = None) -> None:
    mid = model_id or getattr(cls, "model_id", "")
    if not mid:
        raise BackendError("register_class needs a model_id")
    cls.model_id = mid
    _REGISTRY[mid] = cls
