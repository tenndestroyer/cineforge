from cineforge.hardware.vram_tier import classify_vram


def test_tier_boundaries():
    assert classify_vram(4) == "low"
    assert classify_vram(8) == "low"
    assert classify_vram(12) == "low"
    assert classify_vram(16) == "mid"
    assert classify_vram(24) == "mid"
    assert classify_vram(32) == "high"
    assert classify_vram(48) == "high"
