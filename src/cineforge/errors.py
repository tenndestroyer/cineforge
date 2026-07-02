"""Typed exception hierarchy.

The coordinator inspects these to decide retry / skip / abort:
  - VRAMError            -> drop to a lower quant/res, or a smaller model, and retry
  - DownloadIntegrityError -> re-fetch (often a missing HF_TOKEN)
  - LicenseError         -> abort the shot; needs user consent / model swap
  - NotInstalledError    -> abort; run `cineforge doctor` / re-run setup
  - BackendError         -> generic; retry-once then flag the shot
  - CheckpointError      -> abort the run; state is untrustworthy
"""

from __future__ import annotations


class CineforgeError(Exception):
    """Base class for every Cineforge error."""


class ConfigError(CineforgeError):
    """Configuration is missing or invalid."""


class BackendError(CineforgeError):
    """A model backend failed to produce a valid result."""


class VRAMError(BackendError):
    """Out of GPU memory (or estimated to be). Retry at a lower tier/quant."""


class LicenseError(CineforgeError):
    """A model may not be used under the current license_mode / territory / consent."""


class DownloadIntegrityError(CineforgeError):
    """A downloaded file failed byte-size / checksum verification.

    The load-bearing case: a token-less pull of a *gated* repo silently writes a
    ~136-byte HTML error page that passes a naive `os.path.exists` check. We treat
    an undersized file as a hard failure, not a success.
    """


class NotInstalledError(CineforgeError):
    """A required weight, ComfyUI node, or runtime is not installed."""


class CheckpointError(CineforgeError):
    """Project checkpoint could not be saved or is corrupt."""
