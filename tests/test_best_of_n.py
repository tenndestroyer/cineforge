import pytest

from cineforge.pipeline.best_of_n import best_of_n


def test_picks_highest_score():
    sel = best_of_n(gen_fn=lambda i: i, scorer=lambda c: float(c), n=3)
    assert sel.winner == 2
    assert sel.score == 2.0
    assert len(sel.candidates) == 3


def test_skips_failing_candidate():
    def gen(i):
        if i == 1:
            raise RuntimeError("boom")
        return i
    sel = best_of_n(gen_fn=gen, scorer=lambda c: float(c), n=3)
    assert sel.winner == 2
    assert len(sel.candidates) == 2  # the failing one is dropped


def test_no_scorer_returns_first():
    sel = best_of_n(gen_fn=lambda i: f"c{i}", scorer=None, n=3)
    assert sel.winner == "c0"


def test_all_fail_raises():
    with pytest.raises(RuntimeError):
        best_of_n(gen_fn=lambda i: (_ for _ in ()).throw(ValueError()), scorer=None, n=2)
