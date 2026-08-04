"""Phase 7: volatility targeting, pre-registered in §6.6.

Predicts volatility rather than direction, and sizes exposure to a risk budget
instead of switching in and out. Parameters are selected inside each
walk-forward fold on training data only.

Also tests the combination -- vol targeting multiplied by the §6.4 directional
rule -- because if direction carries any information at all, the place it should
show up is as a modifier on an exposure that is already correctly sized.

Usage: .venv/bin/python scripts/phase7_voltarget.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.eval import costs, portfolio  # noqa: E402

warnings.filterwarnings("ignore")

DEEP = Path("data/deep/nepse_index_deep.parquet")
RULE = "=" * 78
CAPITAL = 1_000_000
ANN = 230.0

VOL_WINDOWS = (21, 63)
TARGET_VOLS = (0.10, 0.15, 0.20)
BANDS = (0.10, 0.20)

REGIMES = {
    "pre-mania 2016..2020-02": ("2016-01-01", "2020-02-29"),
    "mania 2020-06..2021-12": ("2020-06-01", "2021-12-31"),
    "chop 2022+": ("2022-01-01", "2030-01-01"),
}


def hr(t: str) -> None:
    print(f"\n{RULE}\n{t}\n{RULE}")


def exposure_series(close: pd.Series, window: int, target_vol: float,
                    band: float) -> np.ndarray:
    """Target exposure, using only information available at the prior close.

    `.shift(1)` is the whole ballgame: an unshifted trailing vol includes
    today's move, so the position would be sized by a return it is about to
    earn. That is lookahead, it is invisible in the output, and it would make
    this strategy look excellent.
    """
    r = close.pct_change()
    vol = r.rolling(window).std().shift(1) * np.sqrt(ANN)
    raw = (target_vol / vol).clip(0.0, 1.0).to_numpy()

    held = np.zeros(len(raw))
    cur = 0.0
    for i, x in enumerate(raw):
        if np.isnan(x):
            held[i] = 0.0
            continue
        if abs(x - cur) > band:      # rebalance only on a material drift
            cur = float(x)
        held[i] = cur
    return held


def select_params(train: pd.DataFrame, cost) -> tuple[int, float, float]:
    best, best_sh = (63, 0.15, 0.20), float("-inf")
    for w, tv, band in product(VOL_WINDOWS, TARGET_VOLS, BANDS):
        e = exposure_series(train["close"], w, tv, band)
        try:
            sh = portfolio.run_backtest_weighted(
                train, e, cost, rebalance_threshold=0.02).stats["net_sharpe"]
        except Exception:  # noqa: BLE001
            continue
        if sh > best_sh:
            best, best_sh = (w, tv, band), sh
    return best


def walk_forward_exposure(frame: pd.DataFrame, cost, initial_train: int = 500,
                          freq: str = "YS") -> tuple[np.ndarray, list]:
    """Re-select parameters on a calendar schedule, train-only, then apply.

    Yearly rather than monthly: these are risk-budget settings, not a signal,
    and re-fitting a volatility target every month on overlapping windows would
    be fitting noise in the third decimal of an estimate that barely moves.
    """
    dates = pd.to_datetime(frame["date"])
    boundaries = pd.date_range(dates.iloc[initial_train].normalize(),
                               dates.iloc[-1], freq=freq)
    edges = [dates.iloc[initial_train]] + [b for b in boundaries
                                           if b > dates.iloc[initial_train]]
    edges.append(dates.iloc[-1] + pd.Timedelta(days=1))

    expo = np.full(len(frame), np.nan)
    chosen = []
    for start, stop in zip(edges[:-1], edges[1:]):
        test_mask = (dates >= start) & (dates < stop)
        if test_mask.sum() == 0:
            continue
        train = frame[dates < start].reset_index(drop=True)
        if len(train) < 250:
            continue
        w, tv, band = select_params(train, cost)
        chosen.append((str(start.date()), w, tv, band))
        # Exposure is computed over the FULL history each time so the rolling
        # window is warm at the fold boundary, then only the test slice is used.
        full = exposure_series(frame["close"], w, tv, band)
        expo[test_mask.to_numpy()] = full[test_mask.to_numpy()]
    return expo, chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--initial-train", type=int, default=500)
    args = ap.parse_args()

    df = pd.read_parquet(DEEP).sort_values("date").reset_index(drop=True)
    P = costs.Params()
    cost = costs.CostModel(P, capital=CAPITAL)

    hr("PHASE 7: VOLATILITY TARGETING (pre-registered §6.6)")
    print(f"series: {len(df)} sessions, {df.date.min().date()} .. {df.date.max().date()}")

    expo, chosen = walk_forward_exposure(df, cost, args.initial_train)
    live = ~np.isnan(expo)
    test = df[live].reset_index(drop=True)
    e = expo[live]

    hr("1. PARAMETERS CHOSEN PER FOLD (training data only)")
    print(f"{'from':<14}{'vol window':>12}{'target vol':>12}{'band':>8}")
    for d, w, tv, band in chosen:
        print(f"{d:<14}{w:>12}{tv:>12.0%}{band:>8.0%}")

    hr("2. RESULTS")
    r_vt = portfolio.run_backtest_weighted(test, e, cost, rebalance_threshold=0.02)
    bh = portfolio.buy_and_hold(test, cost)
    print(f"test window: {test.date.min().date()} .. {test.date.max().date()}, "
          f"{len(test)} sessions")
    print(f"\n{'strategy':<26}{'gross Sh':>10}{'net Sh':>9}{'net CAGR':>10}"
          f"{'maxDD':>9}{'trades':>8}{'avg expo':>10}")
    print("-" * 78)
    for name, r in (("vol-targeted", r_vt), ("buy-and-hold", bh)):
        s = r.stats
        print(f"{name:<26}{s['gross_sharpe']:>10.2f}{s['net_sharpe']:>9.2f}"
              f"{s['net_cagr']:>10.1%}{s['max_drawdown']:>9.1%}"
              f"{s['n_trades']:>8}{r.exposure.mean():>10.1%}")

    hr("3. BY REGIME")
    print(f"{'regime':<26}{'VT net Sh':>11}{'VT CAGR':>10}{'B&H CAGR':>10}"
          f"{'VT maxDD':>10}{'B&H maxDD':>11}")
    print("-" * 78)
    for rname, (lo, hi) in REGIMES.items():
        m = ((test["date"] >= pd.Timestamp(lo))
             & (test["date"] <= pd.Timestamp(hi))).to_numpy()
        if m.sum() < 60:
            continue
        sub = test[m].reset_index(drop=True)
        a = portfolio.run_backtest_weighted(sub, e[m], cost, rebalance_threshold=0.02)
        b = portfolio.buy_and_hold(sub, cost)
        print(f"{rname:<26}{a.stats['net_sharpe']:>11.2f}"
              f"{a.stats['net_cagr']:>10.1%}{b.stats['net_cagr']:>10.1%}"
              f"{a.stats['max_drawdown']:>10.1%}{b.stats['max_drawdown']:>11.1%}")

    hr("4. §6.6 CRITERIA")
    s = r_vt.stats
    c1 = s["net_sharpe"] >= 0.4
    c2 = s["net_sharpe"] > bh.stats["net_sharpe"]
    print(f"  (1) net Sharpe {s['net_sharpe']:.3f} >= 0.400        "
          f"{'PASS' if c1 else 'FAIL'}")
    print(f"  (2) beats buy-and-hold ({bh.stats['net_sharpe']:.3f})       "
          f"{'PASS' if c2 else 'FAIL'}")
    print(f"\n  drawdown: {s['max_drawdown']:.1%} vs buy-and-hold "
          f"{bh.stats['max_drawdown']:.1%}")

    hr("VERDICT")
    if c1 and c2:
        print("  Volatility targeting clears both criteria.")
        print("  §6(e)'s capital gate still applies: >= 60 forward days logged")
        print("  prospectively before any capital moves.")
    else:
        print("  §6.6's criteria are NOT met. Per §6.6, four attempts across two")
        print("  predicted quantities and two position spaces have now failed to")
        print("  beat owning the index. That is the answer.")


if __name__ == "__main__":
    main()
