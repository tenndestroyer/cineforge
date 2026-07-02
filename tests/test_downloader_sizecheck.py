import pytest

from cineforge.errors import DownloadIntegrityError
from cineforge.models.downloader import verify_size


def test_html_error_page_rejected(tmp_path):
    """The load-bearing gotcha: a ~136-byte gated-download error page must fail."""
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"<!DOCTYPE html><html><head><title>Not Found</title></head><body>gated</body></html>")
    assert f.stat().st_size < 200
    with pytest.raises(DownloadIntegrityError):
        verify_size(f, expected_size=None)


def test_truncated_download_rejected(tmp_path):
    f = tmp_path / "w.bin"
    f.write_bytes(b"\x00" * 1000)
    with pytest.raises(DownloadIntegrityError):
        verify_size(f, expected_size=10_000_000)


def test_tiny_binary_rejected_without_expected(tmp_path):
    f = tmp_path / "w.bin"
    f.write_bytes(b"\x00" * 100)
    with pytest.raises(DownloadIntegrityError):
        verify_size(f, expected_size=None)


def test_plausible_weight_accepted(tmp_path):
    f = tmp_path / "w.bin"
    f.write_bytes(b"\x00" * 50_000)
    verify_size(f, expected_size=None)          # no raise
    verify_size(f, expected_size=50_000)        # exact
    verify_size(f, expected_size=51_000)        # within tolerance
