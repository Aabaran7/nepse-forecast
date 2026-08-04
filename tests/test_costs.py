"""Cost model and ledger tests (plan §4).

§4's rule is that every constant is a date-indexed lookup and none of them may
be guessed. Both halves need enforcing: the lookup has to pick the right era,
and a missing constant has to raise rather than quietly default -- §6.1 is
explicit that a backtest must not run against placeholder costs, and a silent
default produces a Sharpe that looks perfectly reasonable.

Run: .venv/bin/pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.eval import costs, portfolio  # noqa: E402

PARAMS = Path("configs/market_params.yaml")


@pytest.fixture(scope="module")
def P() -> costs.Params:
    if not PARAMS.exists():
        pytest.skip("market_params.yaml absent")
    return costs.Params()


# --- era lookup -------------------------------------------------------------

def test_no_overlapping_eras_in_the_real_config(P):
    """The regression. settlement_cycle's T+3 entry was left open-ended, so it
    matched every date and T+2 was unreachable -- the backtest applied a 3-day
    settlement lag to 2026 and nothing errored, because era lookup is
    first-match and the first match was always there."""
    assert P.overlapping_eras() == []


def test_overlapping_eras_are_rejected_at_construction():
    bad = {"settlement_cycle": [
        {"effective_from": None, "effective_to": None, "days": 3},
        {"effective_from": "2018-01-01", "effective_to": None, "days": 2},
    ]}
    with pytest.raises(ValueError, match="overlapping"):
        costs.Params(bad)


def test_settlement_switches_from_t3_to_t2(P):
    assert P.settlement_days(pd.Timestamp("2016-06-01")) == 3
    assert P.settlement_days(pd.Timestamp("2020-06-01")) == 2


def test_circuits_widen_on_the_2026_amendment(P):
    """Both limits changed on 2026-04-20 (§4). Fill logic reading a scalar
    would block fills that were legal and allow ones that were not."""
    before, after = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-05-01")
    assert P.scrip_circuit(before) == 0.10
    assert P.scrip_circuit(after) == 0.15
    assert P.index_circuit(before) == 0.06
    assert P.index_circuit(after) == 0.08


def test_commission_tiers_are_by_trade_value(P):
    d = pd.Timestamp("2025-01-01")
    assert P.commission_rate(10_000, d) == pytest.approx(0.0036)
    assert P.commission_rate(100_000, d) == pytest.approx(0.0033)
    assert P.commission_rate(50_000_000, d) == pytest.approx(0.0024)


def test_commission_fell_in_may_2024(P):
    lo = P.commission_rate(1_000_000, pd.Timestamp("2024-06-01"))
    hi = P.commission_rate(1_000_000, pd.Timestamp("2024-01-01"))
    assert lo < hi


def test_cgt_rose_in_july_2026_and_depends_on_holding_period(P):
    before, after = pd.Timestamp("2026-06-01"), pd.Timestamp("2026-08-01")
    assert P.cgt_rate(30, before) == pytest.approx(0.075)
    assert P.cgt_rate(400, before) == pytest.approx(0.05)
    assert P.cgt_rate(30, after) == pytest.approx(0.10)
    assert P.cgt_rate(400, after) == pytest.approx(0.075)
    assert P.cgt_rate(30, before, entity="institutional") == pytest.approx(0.10)


def test_a_null_constant_raises_rather_than_defaulting():
    """§6.1: no backtesting against placeholders. The dangerous failure is the
    quiet one -- a zero-cost default returns a beautiful Sharpe."""
    p = costs.Params({"dp_charge_npr": [
        {"effective_from": None, "effective_to": None, "amount": None}]})
    with pytest.raises(costs.MissingConstant):
        p.dp_charge(pd.Timestamp("2020-01-01"))


def test_a_todo_constant_also_raises():
    p = costs.Params({"dp_charge_npr": [
        {"effective_from": None, "effective_to": None, "amount": "TODO"}]})
    with pytest.raises(costs.MissingConstant):
        p.dp_charge(pd.Timestamp("2020-01-01"))


def test_a_date_no_era_covers_raises(P):
    p = costs.Params({"x": [{"effective_from": "2020-01-01",
                             "effective_to": "2020-12-31", "v": 1}]})
    with pytest.raises(costs.MissingConstant):
        p.era("x", pd.Timestamp("2019-01-01"))


# --- the flat charge --------------------------------------------------------

def test_dp_charge_does_not_scale_so_small_accounts_pay_more(P):
    """§4's core claim, and the reason results must state the capital base."""
    d = pd.Timestamp("2025-01-01")
    small = costs.CostModel(P, capital=50_000).round_trip_bps(d)
    large = costs.CostModel(P, capital=10_000_000).round_trip_bps(d)
    assert small > large


def test_n_scrips_multiplies_the_flat_charge_only(P):
    """The index is not one instrument; a basket pays DP per name."""
    d = pd.Timestamp("2025-01-01")
    one = costs.CostModel(P, capital=1_000_000, n_scrips=1)
    twenty = costs.CostModel(P, capital=1_000_000, n_scrips=20)
    diff = twenty.trade_cost(1e6, d, "buy") - one.trade_cost(1e6, d, "buy")
    assert diff == pytest.approx(19 * P.dp_charge(d))


def test_dp_charged_on_sell_only_is_cheaper_than_both(P):
    d = pd.Timestamp("2025-01-01")
    both = costs.CostModel(P, capital=1e6, dp_charged_on="both")
    sell = costs.CostModel(P, capital=1e6, dp_charged_on="sell")
    assert sell.trade_cost(1e6, d, "buy") < both.trade_cost(1e6, d, "buy")
    assert sell.trade_cost(1e6, d, "sell") == both.trade_cost(1e6, d, "sell")


def test_cgt_is_charged_on_gains_not_turnover(P):
    cm = costs.CostModel(P, capital=1e6)
    d = pd.Timestamp("2025-01-01")
    assert cm.capital_gains_tax(-5000, 30, d) == 0.0
    assert cm.capital_gains_tax(10_000, 30, d) == pytest.approx(750.0)


def test_zero_friction_means_zero_cost(P):
    cm = costs.CostModel(P, capital=1e6, friction_multiplier=0.0)
    assert cm.trade_cost(1e6, pd.Timestamp("2025-01-01"), "buy") == 0.0


# --- fills ------------------------------------------------------------------

def test_limit_up_blocks_buying_but_not_selling(P):
    row = pd.Series({"date": pd.Timestamp("2020-01-01"), "day_return": 0.0601})
    assert costs.fill_blocked(row, +1, P)
    assert not costs.fill_blocked(row, -1, P)


def test_limit_down_blocks_selling_but_not_buying(P):
    row = pd.Series({"date": pd.Timestamp("2020-01-01"), "day_return": -0.0601})
    assert costs.fill_blocked(row, -1, P)
    assert not costs.fill_blocked(row, +1, P)


def test_the_same_move_is_not_a_limit_after_the_2026_widening(P):
    """6% locked the market before 2026-04-20 and is an ordinary day after."""
    r = {"day_return": 0.0601}
    assert costs.fill_blocked(pd.Series({**r, "date": pd.Timestamp("2026-04-01")}), +1, P)
    assert not costs.fill_blocked(pd.Series({**r, "date": pd.Timestamp("2026-05-01")}), +1, P)


# --- ledger -----------------------------------------------------------------

def prices(n=60, start="2020-01-01", drift=0.001) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"date": dates,
                         "close": 1000 * np.exp(np.arange(n) * drift)})


def test_settlement_prevents_selling_before_the_cycle_completes(P):
    f = prices(20)
    sig = np.zeros(20, dtype=int)
    sig[5] = 1                                  # buy, then immediately want out
    cm = costs.CostModel(P, capital=1_000_000)
    r = portfolio.run_backtest(f, sig, cm)
    assert r.settlement_blocked > 0
    sells = [t for t in r.trades if t.side == "sell"]
    buys = [t for t in r.trades if t.side == "buy"]
    gap = (sells[0].date - buys[0].date).days
    assert gap >= 2


def test_costs_reduce_equity_relative_to_gross(P):
    f = prices(120)
    sig = np.tile([1, 0], 60)
    cm = costs.CostModel(P, capital=1_000_000)
    r = portfolio.run_backtest(f, sig, cm)
    assert r.equity.iloc[-1] < r.gross_equity.iloc[-1]


def test_equity_never_goes_negative_and_a_ruined_account_stops(P):
    """The regression: without a solvency check the ledger bought a negative
    number of units, printed a -100.1% drawdown, and made MORE trades at 2x
    friction than at 1x."""
    f = prices(300)
    sig = np.tile([1, 0], 150)
    cm = costs.CostModel(P, capital=2_000, friction_multiplier=2.0)
    r = portfolio.run_backtest(f, sig, cm)
    assert (r.equity >= 0).all()
    assert r.stats["max_drawdown"] >= -1.0


def test_buy_and_hold_trades_exactly_twice(P):
    f = prices(200)
    cm = costs.CostModel(P, capital=1_000_000)
    r = portfolio.buy_and_hold(f, cm)
    assert r.stats["n_trades"] == 2
    assert r.stats["time_in_market"] > 0.9


def test_zero_friction_backtest_matches_gross(P):
    f = prices(120)
    sig = np.tile([1, 0], 60)
    cm = costs.CostModel(P, capital=1_000_000, friction_multiplier=0.0)
    r = portfolio.run_backtest(f, sig, cm)
    assert r.equity.iloc[-1] == pytest.approx(r.gross_equity.iloc[-1], rel=1e-9)


def test_signal_length_must_match_the_frame(P):
    cm = costs.CostModel(P, capital=1e6)
    with pytest.raises(ValueError):
        portfolio.run_backtest(prices(50), np.ones(10, dtype=int), cm)


def test_top5_share_is_undefined_when_the_strategy_lost_money(P):
    f = prices(200, drift=-0.002)
    sig = np.tile([1, 0], 100)
    cm = costs.CostModel(P, capital=100_000)
    r = portfolio.run_backtest(f, sig, cm)
    assert r.stats["total_return"] < 0
    assert np.isnan(r.stats["pct_pnl_top5_days"])


def test_sessions_per_year_is_measured_not_assumed_252(P):
    f = prices(460, start="2020-01-01")
    sig = np.ones(len(f), dtype=int)
    cm = costs.CostModel(P, capital=1e6)
    r = portfolio.run_backtest(f, sig, cm)
    assert 200 < r.stats["sessions_per_year"] < 275


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
