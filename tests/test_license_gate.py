from cineforge.models.licenses import BLOCKED, OK, REQUIRES_ACK, LicenseGate


def _gate(cfg):
    return LicenseGate.load(cfg.data_dir / "licenses.json")


def test_apache_model_always_ok(cfg):
    g = _gate(cfg)
    assert g.check("wan22", "safe").status == OK
    assert g.check("wan22", "research").status == OK


def test_gated_blocked_in_safe_mode(cfg):
    g = _gate(cfg)
    assert g.check("ltx23", "safe").status == BLOCKED


def test_gated_requires_ack_then_ok_in_research(cfg):
    g = _gate(cfg)
    assert g.check("ltx23", "research", acks=[]).status == REQUIRES_ACK
    assert g.check("ltx23", "research", acks=["ltx23"]).status == OK


def test_territory_exclusion_absolute(cfg):
    g = _gate(cfg)
    v = g.check("hunyuanvideo15", "research", acks=["hunyuanvideo15"], territory="EU")
    assert v.status == BLOCKED


def test_unknown_model_permissive(cfg):
    g = _gate(cfg)
    assert g.check("some_future_model", "safe").status == OK
