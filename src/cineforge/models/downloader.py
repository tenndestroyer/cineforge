"""Gated-safe model downloader.

THE load-bearing gotcha (learned painfully on a prior project): a token-less pull
of a *gated* Hugging Face repo does not error — it silently writes a ~136-byte HTML
error page to the destination filename. A naive `os.path.exists` check then reports
the model as "installed" (false positive), and the failure only surfaces later as a
cryptic load error.

So every download is verified by BYTE SIZE (and optional sha256), and any file that
sniffs as HTML or is implausibly small for a weight is rejected as
DownloadIntegrityError.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..errors import DownloadIntegrityError, NotInstalledError

# A real model weight is never this small. Below this, with no expected size, we
# assume it's an error page / truncated download.
MIN_PLAUSIBLE_WEIGHT_BYTES = 4096
_HTML_MARKERS = (b"<html", b"<!doctype", b"<?xml", b"<head", b"error", b"not found")


def _sniff_html(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            head = fh.read(256).lstrip().lower()
    except OSError:
        return False
    return head.startswith(b"<") or any(m in head for m in _HTML_MARKERS)


def verify_size(path: Path, expected_size: int | None, tolerance: float = 0.98) -> None:
    """Raise DownloadIntegrityError if `path` looks like an error page or is too small.

    - If it sniffs as HTML/XML -> reject (the classic gated-download error page).
    - If `expected_size` is known -> require >= expected_size * tolerance.
    - If unknown -> require at least MIN_PLAUSIBLE_WEIGHT_BYTES.
    """
    path = Path(path)
    if not path.is_file():
        raise DownloadIntegrityError(f"expected file was not written: {path}")
    size = path.stat().st_size

    if _sniff_html(path):
        raise DownloadIntegrityError(
            f"{path.name} ({size} bytes) looks like an HTML/error page, not a weight. "
            f"This usually means the repo is GATED and HF_TOKEN is missing or lacks access."
        )

    if expected_size:
        if size < expected_size * tolerance:
            raise DownloadIntegrityError(
                f"{path.name} is {size} bytes but expected ~{expected_size} "
                f"(<{tolerance:.0%}); download is truncated or an error page."
            )
    elif size < MIN_PLAUSIBLE_WEIGHT_BYTES:
        raise DownloadIntegrityError(
            f"{path.name} is only {size} bytes — implausibly small for a model weight "
            f"(likely a gated-download error page)."
        )


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    actual = sha256_of(path)
    if actual.lower() != expected.lower():
        raise DownloadIntegrityError(f"{path.name} sha256 mismatch: got {actual}, expected {expected}")


def fetch(
    repo: str,
    filename: str,
    *,
    expected_size: int | None = None,
    revision: str | None = None,
    token: str | None = None,
    dest_dir: Path | None = None,
    sha256: str | None = None,
) -> Path:
    """Download one file from Hugging Face, then verify it.

    Pins `revision` when given (reproducible installs). Raises NotInstalledError if
    huggingface-hub isn't available, DownloadIntegrityError on a bad file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover - env-dependent
        raise NotInstalledError("huggingface-hub is not installed; re-run setup") from e

    local = hf_hub_download(
        repo_id=repo,
        filename=filename,
        revision=revision,
        token=token or None,
        local_dir=str(dest_dir) if dest_dir else None,
    )
    path = Path(local)
    verify_size(path, expected_size)
    if sha256:
        verify_sha256(path, sha256)
    return path
