"""Phase 8: better volatility estimation, and does direction add anything? (§6.8)

Two changes to §6.7's strategy, neither of them a parameter scan:

  1. **The volatility estimator is chosen by forecast accuracy**, on training
     data, not by the Sharpe of the strategy built on it. §6.7 used
     close-to-close-63, which is the worst forecaster of the eight available
     (correlation 0.275 vs 0.452 for EWMA). It was picked on the wrong criterion.

  2. **A directional tilt is tested on top.** Four attempts showed a 55.5%
     directional signal cannot be traded on its own. The remaining question is
     whether it adds anything once exposure is already correctly sized -- a
     weak signal can be useful as a modifier while being useless as a switch.
     Reported whether it helps or not.

Same bar as always: net Sharpe >= 0.4, must beat buy-and-hold, must not depend
on the mania regime.

Usage: .venv/bin/python scripts/phase8_improved.py
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
from nepselab.eval import costs, labels, portfolio, volest  # noqa: E402
from nepselab.features import base as fbase  # noqa: E402
from nepselab.features import price as fprice  # noqa: E402
from nepselab.models.classifiers import Logistic  # noqa: E402

warnings.filterwarnings("ignore")

DEEP = Path("data/deep/nepse_index_deep.parquet")
RULE = "=" * 78
CAPITAL = 1_000_000
TARGET_VOLS = (0.10, 0.15, 0.20)
BANDS = (0.10, 0.20)
REBAL = 0.02

REGIMES = {
    "pre-mania 2016..2020-02": ("2016-01-01", "2020-02-29"),
    "mania 2020-06..2021-12": ("2020-06-01", "2021-12-31"),
    "chop 2022+": ("2022-01-01", "2030-01-01"),
}


def hr(t: str) -> None:
    print(f"\n{RULE}\n{t}\n{RULE}")


def banded(raw: np.ndarray, band: float) -> np.ndarray:
    held = np.zeros(len(raw))
    cur = 0.0
    for i, x in enumerate(raw):
        if np.isnan(x):
            held[i] = 0.0
            continue
        if abs(x - cur) > band:
            cur = float(x)
        held[i] = cur
    return held


def exposure(df: pd.DataFrame, est_name: str, target_vol: float,
             band: float) -> np.ndarray:
    """Risk-budgeted exposure. `.shift(1)` keeps the sizing strictly lagged."""
    vol = volest.ESTIMATORS[est_name](df).shift(1)
    raw = (target_vol / vol).clip(0.0, 1.0).to_numpy()
    return banded(raw, band)


def walk_forward(df: pd.DataFrame, cost, initial_train=500, freq="YS",
                 tilt: np.ndarray | None = None):
    dates = pd.to_datetime(df["date"])
    bnds = pd.date_range(dates.iloc[initial_train].normalize(), dates.iloc[-1],
                         freq=freq)
    edges = [dates.iloc[initial_train]] + [b for b in bnds
                                           if b > dates.iloc[initial_train]]
    edges.append(dates.iloc[-1] + pd.Timedelta(days=1))

    expo = np.full(len(df), np.nan)
    chosen = []
    for start, stop in zip(edges[:-1], edges[1:]):
        mask = ((dates >= start) & (dates < stop)).to_numpy()
        if mask.sum() == 0:
            continue
        train = df[(dates < start).to_numpy()].reset_index(drop=True)
        if len(train) < 300:
            continue

        # (1) estimator by FORECAST accuracy -- not by Sharpe.
        est = volest.select_estimator(train)
        # (2) risk budget by in-sample net Sharpe, as in §6.3/§6.6.
        best, best_sh = (0.15, 0.20), float("-inf")
        for tv, band in product(TARGET_VOLS, BANDS):
            e = exposure(train, est, tv, band)
            try:
                sh = portfolio.run_backtest_weighted(
                    train, e, cost, rebalance_threshold=REBAL).stats["net_sharpe"]
            except Exception:  # noqa: BLE001
                continue
            if sh > best_sh:
                best, best_sh = (tv, band), sh
        tv, band = best
        chosen.append((str(start.date()), est, tv, band))
        full = exposure(df, est, tv, band)
        if tilt is not None:
            full = full * tilt
        expo[mask] = full[mask]
    return expo, chosen


def summarise(name, r, bh=None):
    s = r.stats
    extra = ""
    if bh is not None:
        extra = f"{'YES' if s['net_sharpe'] > bh else 'no':>7}"
    print(f"{name:<28}{s['gross_sharpe']:>10.2f}{s['net_sharpe']:>9.2f}"
          f"{s['net_cagr']:>10.1%}{s['max_drawdown']:>9.1%}"
          f"{s['n_trades']:>8}{r.exposure.mean():>9.1%}{extra}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--initial-train", type=int, default=500)
    args = ap.parse_args()

    df = pd.read_parquet(DEEP).sort_values("date").reset_index(drop=True)
    P = costs.Params()
    cost = costs.CostModel(P, capital=CAPITAL)

    hr("PHASE 8: BETTER VOL ESTIMATION + DIRECTIONAL TILT (§6.8)")

    hr("1. WHICH ESTIMATOR FORECASTS VOLATILITY BEST? (full sample, diagnostic)")
    print(f"{'estimator':<20}{'corr with next-21d realised vol':>34}")
    scored = sorted(((n, volest.forecast_score(f(df), df))
                     for n, f in volest.ESTIMATORS.items()),
                    key=lambda t: -t[1])
    for n, sc in scored:
        mark = "   <- §6.7 used this" if n == "close2close-63" else ""
        print(f"{n:<20}{sc:>34.3f}{mark}")

    hr("2. WALK-FORWARD, ESTIMATOR CHOSEN BY FORECAST ACCURACY")
    expo, chosen = walk_forward(df, cost, args.initial_train)
    print(f"{'from':<14}{'estimator':<18}{'target vol':>12}{'band':>8}")
    for d, est, tv, band in chosen:
        print(f"{d:<14}{est:<18}{tv:>12.0%}{band:>8.0%}")

    live = ~np.isnan(expo)
    test = df[live].reset_index(drop=True)

    # Directional tilt: scale exposure by the model's probability, mapped so
    # p=0.5 leaves exposure unchanged. Bounded [0.5, 1.5] so a weak signal can
    # modulate the risk budget but never override it.
    a = fbase.assemble(df, [fprice.PriceFeatures(), fprice.TurnoverFeatures()],
                       start=pd.Timestamp("2017-01-01"))
    lab = labels.make_labels(a.frame, horizon=5)
    usable = lab[lab["usable"]].reset_index(drop=True)
    feats = a.feature_names
    probs = pd.Series(np.nan, index=df.index)
    idx = {d: i for i, d in enumerate(df["date"])}
    split = int(len(usable) * 0.25)
    model = Logistic().fit(usable[feats].iloc[:split],
                           usable["label"].to_numpy().astype(int)[:split])
    p = model.predict_proba(usable[feats].iloc[split:])
    for d, pv in zip(usable["date"].iloc[split:], p):
        if d in idx:
            probs.iloc[idx[d]] = pv
    tilt = (0.5 + probs).clip(0.5, 1.5).shift(1).fillna(1.0).to_numpy()

    hr("3. RESULTS")
    print(f"{'strategy':<28}{'gross Sh':>10}{'net Sh':>9}{'net CAGR':>10}"
          f"{'maxDD':>9}{'trades':>8}{'avg exp':>9}{'>B&H':>7}")
    print("-" * 78)
    bh = portfolio.buy_and_hold(test, cost)
    bh_sh = bh.stats["net_sharpe"]

    r_new = portfolio.run_backtest_weighted(test, expo[live], cost,
                                            rebalance_threshold=REBAL)
    e_tilt = np.clip(expo[live] * tilt[live], 0, 1)
    r_tilt = portfolio.run_backtest_weighted(test, e_tilt, cost,
                                             rebalance_threshold=REBAL)
    # §6.7 baseline: close2close-63, target 10%, band 10%
    r_old = portfolio.run_backtest_weighted(
        test, exposure(df, "close2close-63", 0.10, 0.10)[live], cost,
        rebalance_threshold=REBAL)

    summarise("§6.7 vol-target (c2c-63)", r_old, bh_sh)
    summarise("§6.8 vol-target (forecast)", r_new, bh_sh)
    summarise("§6.8 + directional tilt", r_tilt, bh_sh)
    summarise("buy-and-hold", bh)

    hr("4. BY REGIME (best §6.8 variant)")
    best_r, best_e, best_name = max(
        ((r_new, expo[live], "forecast-selected"), (r_tilt, e_tilt, "+ tilt")),
        key=lambda t: t[0].stats["net_sharpe"])[:3]
    print(f"variant: {best_name}")
    print(f"{'regime':<26}{'net Sh':>9}{'CAGR':>9}{'B&H CAGR':>10}"
          f"{'maxDD':>9}{'B&H maxDD':>11}")
    print("-" * 78)
    for rname, (lo, hi) in REGIMES.items():
        m = ((test["date"] >= pd.Timestamp(lo))
             & (test["date"] <= pd.Timestamp(hi))).to_numpy()
        if m.sum() < 60:
            continue
        sub = test[m].reset_index(drop=True)
        x = portfolio.run_backtest_weighted(sub, best_e[m], cost,
                                            rebalance_threshold=REBAL)
        b = portfolio.buy_and_hold(sub, cost)
        print(f"{rname:<26}{x.stats['net_sharpe']:>9.2f}"
              f"{x.stats['net_cagr']:>9.1%}{b.stats['net_cagr']:>10.1%}"
              f"{x.stats['max_drawdown']:>9.1%}{b.stats['max_drawdown']:>11.1%}")

    hr("VERDICT")
    s = best_r.stats
    print(f"  best §6.8 variant: {best_name}")
    print(f"    net Sharpe {s['net_sharpe']:.3f} "
          f"({'PASS' if s['net_sharpe'] >= 0.4 else 'FAIL'} vs 0.4)")
    print(f"    vs buy-and-hold {bh_sh:.3f} "
          f"({'PASS' if s['net_sharpe'] > bh_sh else 'FAIL'})")
    print(f"    vs §6.7 {r_old.stats['net_sharpe']:.3f} "
          f"({'improvement' if s['net_sharpe'] > r_old.stats['net_sharpe'] else 'NO improvement'})")
    d = r_tilt.stats["net_sharpe"] - r_new.stats["net_sharpe"]
    print(f"\n  directional tilt contribution: {d:+.3f} Sharpe")
    print(f"    {'It helps.' if d > 0.02 else 'It does not help -- consistent with §6.2-6.4.'}")


if __name__ == "__main__":
    main()
