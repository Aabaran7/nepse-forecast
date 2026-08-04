"""Decision-rule tests (plan §6.3, §6.4).

The rule layer is where a backtest is easiest to flatter by accident: an
off-by-one in the position series, a rule that silently never trades, or one
that degenerates into buy-and-hold while still being reported as a model. Each
of those produces a plausible Sharpe and no error.

Run: .venv/bin/pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.eval.strategy import Rule  # noqa: E402


def test_zero_deadband_reproduces_the_naive_rule():
    """delta=0, min_hold=1 must be exactly `p > 0.5`, or §6.2 and §6.3 are not
    comparable and the whole experiment measures nothing."""
    p = np.array([0.6, 0.4, 0.7, 0.3, 0.55])
    got = Rule(delta=0.0, min_hold=1).positions(p)
    assert got.tolist() == [1, 0, 1, 0, 1]


def test_deadband_holds_position_through_the_middle():
    """The point of hysteresis: a probability wobbling around 0.5 must not
    generate a trade on every crossing."""
    p = np.array([0.7, 0.52, 0.48, 0.51, 0.30])
    got = Rule(delta=0.10, min_hold=1).positions(p)
    #        enter   hold  hold  hold  exit
    assert got.tolist() == [1, 1, 1, 1, 0]


def test_wider_deadband_never_increases_turnover():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.3, 0.7, 500)
    trades = []
    for d in (0.0, 0.05, 0.10, 0.15):
        pos = Rule(delta=d, min_hold=1).positions(p)
        trades.append(int(np.abs(np.diff(pos)).sum()))
    assert trades == sorted(trades, reverse=True)


def test_min_hold_freezes_the_position():
    p = np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
    got = Rule(delta=0.0, min_hold=3).positions(p)
    # enters at 0, then cannot change until index 3
    assert got[0] == 1
    assert got[1] == got[2] == 1
    assert got[3] == 0


def test_longer_min_hold_never_increases_turnover():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.2, 0.8, 800)
    trades = [int(np.abs(np.diff(Rule(min_hold=m).positions(p))).sum())
              for m in (1, 5, 21, 63)]
    assert trades == sorted(trades, reverse=True)


# --- the asymmetric family (§6.4) -------------------------------------------

def test_asymmetric_rule_starts_and_stays_long_by_default():
    """The structural point: on a drifting asset a weak signal should justify
    LEAVING the market, not entering it."""
    p = np.full(20, 0.5)
    got = Rule(asymmetric=True, exit_delta=0.10).positions(p)
    assert got.tolist() == [1] * 20


def test_asymmetric_rule_exits_only_on_a_confident_bearish_signal():
    p = np.array([0.5, 0.45, 0.35, 0.5])
    got = Rule(asymmetric=True, exit_delta=0.10, delta=0.0).positions(p)
    assert got[0] == 1          # default long
    assert got[1] == 1          # 0.45 is not below 0.40
    assert got[2] == 0          # 0.35 is
    assert got[3] == 1          # recovers


def test_asymmetric_holds_more_than_symmetric_on_the_same_signal():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.35, 0.65, 600)
    sym = Rule(delta=0.05, min_hold=1, asymmetric=False).positions(p)
    asym = Rule(delta=0.05, min_hold=1, asymmetric=True, exit_delta=0.10).positions(p)
    assert asym.mean() > sym.mean()


# --- degenerate cases the guards exist to catch -----------------------------

def test_a_rule_that_never_trades_is_visible_as_such():
    """§6.3's guard 3 exists because a rule can pass a Sharpe bar by simply
    becoming buy-and-hold. That has to be detectable, not hidden."""
    p = np.full(300, 0.5)
    pos = Rule(asymmetric=True, exit_delta=0.4).positions(p)
    assert int(np.abs(np.diff(pos)).sum()) == 0
    assert pos.mean() == 1.0        # i.e. buy-and-hold, and obviously so


def test_positions_are_binary_and_length_preserving():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 250)
    for r in (Rule(), Rule(delta=0.1, min_hold=21),
              Rule(asymmetric=True, exit_delta=0.05)):
        pos = r.positions(p)
        assert len(pos) == len(p)
        assert set(np.unique(pos)) <= {0, 1}


def test_empty_input_does_not_crash():
    assert len(Rule().positions(np.array([]))) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
