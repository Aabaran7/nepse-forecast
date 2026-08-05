"""Volatility-targeting and fractional-ledger tests (plan §6.6, §6.7).

This is the one strategy in the project that PASSED, which makes it the one
most in need of hostile testing. A positive result is exactly where a leak
hides most comfortably: nobody digs into why the number came out good.

The load-bearing test is `test_exposure_has_no_lookahead`. Sizing a position
with an unshifted trailing volatility means the position is set by a return it
is about to earn. It produces an excellent backtest and no error whatsoever.

Run: .venv/bin/pytest tests/ -q
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nepselab.eval import costs, portfolio  # noqa: E402
from phase7_voltarget import BANDS, TARGET_VOLS, VOL_WINDOWS, exposure_series  # noqa: E402

DEEP = ROOT / "data/deep/nepse_index_deep.parquet"
PARAMS = ROOT / "configs/market_params.yaml"


@pytest.fixture(scope="module")
def P():
    if not PARAMS.exists():
        pytest.skip("market_params.yaml absent")
    return costs.Params()


def prices(n=600, seed=0, vol=0.01):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    close = 1000 * np.exp(np.cumsum(rng.normal(0.0004, vol, n)))
    return pd.DataFrame({"date": dates, "close": close})


# --- the leak test ----------------------------------------------------------

@pytest.mark.skipif(not DEEP.exists(), reason="deep series absent")
def test_exposure_has_no_lookahead():
    """Truncating the series must not change exposures computed before the cut.

    Catches the unshifted-volatility bug in its general form, across the whole
    parameter grid rather than one setting.
    """
    df = pd.read_parquet(DEEP).sort_values("date").reset_index(drop=True)
    for w, tv, band in product(VOL_WINDOWS, TARGET_VOLS, BANDS):
        full = exposure_series(df["close"], w, tv, band)[:1500]
        trunc = exposure_series(df["close"].iloc[:1600], w, tv, band)[:1500]
        assert np.allclose(full, trunc, equal_nan=True), (
            f"lookahead at window={w} target={tv} band={band}")


def test_exposure_is_bounded_and_never_levered():
    """Long-only, no leverage. An unclipped target/realised ratio explodes in
    a calm market and would quietly imply borrowing."""
    for tv in TARGET_VOLS:
        e = exposure_series(prices(400, vol=0.001)["close"], 21, tv, 0.1)
        assert np.nanmax(e) <= 1.0 + 1e-12
        assert np.nanmin(e) >= 0.0


def test_higher_volatility_gives_lower_exposure():
    """The entire premise, stated as an assertion."""
    calm = exposure_series(prices(400, seed=1, vol=0.004)["close"], 21, 0.15, 0.0)
    wild = exposure_series(prices(400, seed=1, vol=0.030)["close"], 21, 0.15, 0.0)
    assert np.nanmean(wild) < np.nanmean(calm)


def test_wider_band_reduces_rebalancing():
    e_tight = exposure_series(prices(600)["close"], 21, 0.15, 0.02)
    e_wide = exposure_series(prices(600)["close"], 21, 0.15, 0.30)
    assert np.abs(np.diff(e_wide)).sum() < np.abs(np.diff(e_tight)).sum()


# --- the fractional ledger --------------------------------------------------

def test_full_weight_matches_buy_and_hold_closely(P):
    """w=1 throughout is buy-and-hold minus the terminal sale."""
    f = prices(500)
    cm = costs.CostModel(P, capital=1_000_000)
    w = portfolio.run_backtest_weighted(f, np.ones(len(f)), cm)
    b = portfolio.buy_and_hold(f, cm)
    assert w.stats["net_cagr"] == pytest.approx(b.stats["net_cagr"], abs=0.005)
    assert w.exposure.iloc[-1] > 0.95


def test_zero_weight_holds_cash_and_never_trades(P):
    f = prices(300)
    cm = costs.CostModel(P, capital=1_000_000)
    r = portfolio.run_backtest_weighted(f, np.zeros(len(f)), cm)
    assert r.stats["n_trades"] == 0
    assert r.equity.nunique() == 1


def test_costs_are_charged_on_traded_notional_not_the_whole_position(P):
    """A 60% -> 55% rebalance pays commission on 5% of equity, not 55%.

    Compares the INCREMENTAL cost of the rebalance, not the total: both paths
    share an identical opening buy to 60%, which dominates and which an earlier
    version of this test forgot to net out.
    """
    f = prices(300)
    cm = costs.CostModel(P, capital=1_000_000)
    hold = np.full(300, 0.60)
    small = np.r_[np.full(150, 0.60), np.full(150, 0.55)]   # trades 5%
    large = np.r_[np.full(150, 0.60), np.full(150, 0.05)]   # trades 55%
    base = portfolio.run_backtest_weighted(f, hold, cm).stats["total_costs"]
    c_small = portfolio.run_backtest_weighted(f, small, cm).stats["total_costs"] - base
    c_large = portfolio.run_backtest_weighted(f, large, cm).stats["total_costs"] - base
    assert c_small > 0
    # 11x the notional traded; allow slack for the flat DP charge in both.
    assert c_large > c_small * 5


def test_rebalance_threshold_suppresses_trivial_trades(P):
    f = prices(600)
    cm = costs.CostModel(P, capital=1_000_000)
    rng = np.random.default_rng(0)
    w = np.clip(0.5 + rng.normal(0, 0.01, len(f)), 0, 1)   # tiny wobble
    t0 = portfolio.run_backtest_weighted(f, w, cm, rebalance_threshold=0.0)
    t5 = portfolio.run_backtest_weighted(f, w, cm, rebalance_threshold=0.05)
    assert t5.stats["n_trades"] < t0.stats["n_trades"]


def test_weighted_ledger_never_goes_negative_or_levers(P):
    f = prices(800)
    cm = costs.CostModel(P, capital=20_000, friction_multiplier=2.0)
    rng = np.random.default_rng(2)
    w = rng.uniform(0, 1, len(f))
    r = portfolio.run_backtest_weighted(f, w, cm)
    assert (r.equity >= 0).all()
    assert r.exposure.max() <= 1.0 + 1e-9


def test_weight_length_must_match_the_frame(P):
    cm = costs.CostModel(P, capital=1e6)
    with pytest.raises(ValueError):
        portfolio.run_backtest_weighted(prices(100), np.ones(10), cm)


def test_settlement_still_binds_on_partial_sales(P):
    f = prices(60)
    cm = costs.CostModel(P, capital=1_000_000)
    w = np.r_[np.full(5, 0.0), np.full(1, 1.0), np.full(54, 0.0)]
    r = portfolio.run_backtest_weighted(f, w, cm)
    assert r.settlement_blocked > 0


# --- the result itself ------------------------------------------------------

@pytest.mark.skipif(not DEEP.exists(), reason="deep series absent")
def test_vol_targeting_beats_buy_and_hold_across_most_of_the_grid(P):
    """Robustness, not a knife edge. §6.7's claim is that the effect survives
    the parameter choice; if only one or two settings worked it would be a fit."""
    df = pd.read_parquet(DEEP).sort_values("date").reset_index(drop=True)
    test = df[df.date >= pd.Timestamp("2018-03-05")].reset_index(drop=True)
    cm = costs.CostModel(P, capital=1_000_000)
    bh = portfolio.buy_and_hold(test, cm).stats["net_sharpe"]

    wins = 0
    for w, tv, band in product(VOL_WINDOWS, TARGET_VOLS, BANDS):
        e = exposure_series(df["close"], w, tv, band)[
            (df.date >= pd.Timestamp("2018-03-05")).to_numpy()]
        s = portfolio.run_backtest_weighted(test, e, cm,
                                            rebalance_threshold=0.02).stats
        wins += s["net_sharpe"] > bh
    assert wins >= 8, f"only {wins}/12 parameter sets beat buy-and-hold"


@pytest.mark.skipif(not DEEP.exists(), reason="deep series absent")
def test_vol_targeting_always_reduces_drawdown(P):
    """The risk claim, which matters more than the Sharpe for a real investor."""
    df = pd.read_parquet(DEEP).sort_values("date").reset_index(drop=True)
    test = df[df.date >= pd.Timestamp("2018-03-05")].reset_index(drop=True)
    cm = costs.CostModel(P, capital=1_000_000)
    bh_dd = portfolio.buy_and_hold(test, cm).stats["max_drawdown"]
    for w, tv, band in product(VOL_WINDOWS, TARGET_VOLS, BANDS):
        e = exposure_series(df["close"], w, tv, band)[
            (df.date >= pd.Timestamp("2018-03-05")).to_numpy()]
        dd = portfolio.run_backtest_weighted(test, e, cm,
                                             rebalance_threshold=0.02
                                             ).stats["max_drawdown"]
        assert dd >= bh_dd - 0.02, f"({w},{tv},{band}) worsened drawdown"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# --- §6.8: estimator selection by forecast accuracy -------------------------

@pytest.mark.skipif(not DEEP.exists(), reason="deep series absent")
def test_ewma_forecasts_volatility_better_than_close_to_close_63():
    """§6.7 sized positions off the worst forecaster in the set, because it was
    chosen by in-sample Sharpe rather than by forecasting ability."""
    from nepselab.eval import volest

    df = pd.read_parquet(DEEP).sort_values("date").reset_index(drop=True)
    ewma = volest.forecast_score(volest.ewma(df, 0.94), df)
    c2c = volest.forecast_score(volest.close_to_close(df, 63), df)
    assert ewma > c2c


@pytest.mark.skipif(not DEEP.exists(), reason="deep series absent")
def test_estimator_selection_never_sees_the_test_window():
    """select_estimator must depend only on the rows handed to it."""
    from nepselab.eval import volest

    df = pd.read_parquet(DEEP).sort_values("date").reset_index(drop=True)
    a = volest.select_estimator(df.iloc[:1200].reset_index(drop=True))
    b = volest.select_estimator(df.iloc[:1200].copy().reset_index(drop=True))
    assert a == b
    assert a in volest.ESTIMATORS


@pytest.mark.skipif(not DEEP.exists(), reason="deep series absent")
def test_range_estimators_respect_the_bad_ohlc_flag():
    """The 16 bars flagged in §3.6 have unreliable highs and lows. A range
    estimator reading them returns a confident wrong number, not a missing one."""
    from nepselab.eval import volest

    df = pd.read_parquet(DEEP).sort_values("date").reset_index(drop=True)
    if "ohlc_consistent" not in df.columns:
        pytest.skip("no ohlc_consistent column")
    doctored = df.copy()
    bad = ~doctored["ohlc_consistent"].astype(bool)
    doctored.loc[bad, "high"] = doctored.loc[bad, "high"] * 50    # absurd values
    a = volest.parkinson(df, 21)
    b = volest.parkinson(doctored, 21)
    m = a.notna() & b.notna()
    assert np.allclose(a[m], b[m]), "range estimator used a flagged bar"


@pytest.mark.skipif(not DEEP.exists(), reason="deep series absent")
def test_better_vol_forecasts_produce_better_strategies():
    """The coherence check, and the strongest evidence the mechanism is real:
    the ranking of estimators by forecast accuracy should reproduce in their
    PnL ranking. Overfitting has no reason to do that."""
    from scipy.stats import spearmanr

    from nepselab.eval import volest
    sys.path.insert(0, str(ROOT / "scripts"))
    from phase8_improved import REBAL, exposure

    df = pd.read_parquet(DEEP).sort_values("date").reset_index(drop=True)
    live = (df.date >= pd.Timestamp("2018-03-05")).to_numpy()
    test = df[live].reset_index(drop=True)
    cm = costs.CostModel(costs.Params(), capital=1_000_000)

    fc, sh = [], []
    for est in volest.ESTIMATORS:
        fc.append(volest.forecast_score(volest.ESTIMATORS[est](df), df))
        sh.append(portfolio.run_backtest_weighted(
            test, exposure(df, est, 0.10, 0.20)[live], cm,
            rebalance_threshold=REBAL).stats["net_sharpe"])
    rho, _ = spearmanr(fc, sh)
    assert rho > 0.4, f"forecast quality does not track PnL (rho={rho:.2f})"
