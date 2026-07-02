"""Model backends.

Importing this package self-registers every adapter into the BackendRegistry
(each adapter module calls @register(...) at import time). `registry._autoload()`
imports this package lazily so callers never have to remember to.
"""

# Import subpackages so their adapters register. Keep this list in sync when
# adding a new subsystem folder.
from . import enhance, image, lipsync, llm, music, sfx, video, voice  # noqa: F401

__all__ = ["video", "image", "voice", "music", "lipsync", "sfx", "enhance", "llm"]
