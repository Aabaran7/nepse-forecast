"""Phase 2b: the cost model, applied to the signal Phase 2a found (plan §4, §6).

§3.7 recorded a naive momentum rule beating the majority class by +3.32pp at
h=1, and immediately flagged that there were no costs yet. §4 predicts what
happens next: the DP charge is FLAT, so it does not shrink with the trade, and a
signal that switches position every few days pays it every time.

This runs that prediction. It reports gross and net side by side at several
capital bases and friction multipliers, because §4 requires 0x/1x/2x and
because the flat charge means the answer depends on the declared capital base.

Usage: .venv/bin/python scripts/phase2b_costs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.eval import baselines, costs, labels, portfolio  # noqa: E402

DEEP = Path("data/deep/nepse_index_deep.parquet")
RULE = "=" * 78
CAPITALS = [50_000, 200_000, 1_000_000, 10_000_000]


def hr(t: str) -> None:
    print(f"\n{RULE}\n{t}\n{RULE}")


def main() -> None:
    df = pd.read_parquet(DEEP).sort_values("date").reset_index(drop=True)
    df["prev_return"] = df["close"].pct_change()
    lab = labels.make_labels(df, horizon=1)
    frame = lab[lab["usable"] & lab["prev_return"].notna()].reset_index(drop=True)

    P = costs.Params()

    hr("PHASE 2B: COSTS AND FILLS")
    print(f"frame: {len(frame)} sessions, {frame.date.min().date()} .. "
          f"{frame.date.max().date()}")

    hr("1. WHAT A ROUND TRIP COSTS, BY DATE AND CAPITAL BASE")
    print(f"{'capital (NPR)':>15}" + "".join(f"{d:>14}" for d in
          ("2016-06", "2024-06", "2026-05")))
    for cap in CAPITALS:
        cm = costs.CostModel(P, capital=cap)
        row = "".join(f"{cm.round_trip_bps(pd.Timestamp(d)):>12.1f}bp"
                      for d in ("2016-06-01", "2024-06-01", "2026-05-01"))
        print(f"{cap:>15,}{row}")
    print("\n  The flat DP charge is why the small base costs more. It does not")
    print("  scale, so it is a larger share of a smaller trade -- §4's reason for")
    print("  insisting the capital base is declared wherever a Sharpe appears.")

    # Walk the momentum signal forward exactly as Phase 2a did.
    preds, _ = baselines.run_walk_forward(
        frame, baselines.Momentum(), horizon=1,
        feature_cols=["prev_return"], initial_train=500)
    test = frame[frame["date"].isin(preds["date"])].reset_index(drop=True)
    signal = preds.sort_values("date")["y_pred"].to_numpy()

    hr("2. MOMENTUM SIGNAL, GROSS vs NET")
    print(f"test window: {test.date.min().date()} .. {test.date.max().date()}, "
          f"{len(test)} sessions")
    print(f"\n{'capital':>12}{'friction':>10}{'gross Sh':>10}{'net Sh':>9}"
          f"{'net CAGR':>10}{'maxDD':>9}{'trades':>8}{'costs/cap':>11}")
    print("-" * 78)
    results = {}
    for cap in CAPITALS:
        for mult, mlabel in ((0.0, "0x"), (1.0, "1x"), (2.0, "2x")):
            cm = costs.CostModel(P, capital=cap, friction_multiplier=mult)
            r = portfolio.run_backtest(test, signal, cm)
            s = r.stats
            results[(cap, mult)] = r
            print(f"{cap:>12,}{mlabel:>10}{s['gross_sharpe']:>10.2f}"
                  f"{s['net_sharpe']:>9.2f}{s['net_cagr']:>9.1%}"
                  f"{s['max_drawdown']:>9.1%}{s['n_trades']:>8}"
                  f"{s['total_costs'] / cap:>10.1%}")

    hr("3. BUY AND HOLD, SAME COSTS (§1's third mandatory benchmark)")
    print(f"{'capital':>12}{'net Sharpe':>12}{'net CAGR':>10}{'maxDD':>9}"
          f"{'trades':>8}")
    print("-" * 78)
    for cap in CAPITALS:
        cm = costs.CostModel(P, capital=cap)
        r = portfolio.buy_and_hold(test, cm)
        s = r.stats
        print(f"{cap:>12,}{s['net_sharpe']:>12.2f}{s['net_cagr']:>10.1%}"
              f"{s['max_drawdown']:>9.1%}{s['n_trades']:>8}")

    hr("4. WHERE THE MONEY WENT (capital = 1,000,000, 1x friction)")
    r = results[(1_000_000, 1.0)]
    s = r.stats
    print(f"  round trips              {s['n_round_trips']}")
    print(f"  turnover                 {s['turnover_x_capital']:.1f}x capital")
    print(f"  total costs              NPR {s['total_costs']:,.0f} "
          f"({s['total_costs'] / r.capital:.1%} of capital)")
    print(f"  time in market           {s['time_in_market']:.1%}")
    print(f"  fills blocked by circuit {r.blocked_fills}")
    print(f"  exits blocked by settlement {r.settlement_blocked}")
    print(f"  gross CAGR               {s['gross_cagr']:>7.2%}")
    print(f"  net CAGR                 {s['net_cagr']:>7.2%}")
    print(f"  cost drag                {s['gross_cagr'] - s['net_cagr']:>7.2%} /yr")
    print(f"  %% of PnL from top 5 days {s['pct_pnl_top5_days']:>7.1%}")
    print(f"  sessions/year (measured) {s['sessions_per_year']:.0f}")

    hr("5. §6 THRESHOLD: NET SHARPE >= 0.4 AFTER REALISTIC COSTS")
    cm = costs.CostModel(P, capital=1_000_000)
    r1 = portfolio.run_backtest(test, signal, cm)
    net = r1.stats["net_sharpe"]
    print(f"  momentum, 1,000,000 NPR, 1x friction: net Sharpe {net:.3f}")
    print(f"  §6 threshold: 0.400")
    print(f"  VERDICT: {'PASSES' if net >= 0.4 else 'FAILS -- no capital, per §6'}")
    print("\n  Caveats that must travel with this number:")
    print("   - market_params is SECONDARY-sourced, not circular-verified.")
    print("   - n_scrips=1 assumes the index is tradeable as ONE instrument.")
    print("     It is not; a real basket multiplies the flat DP charge.")
    print("     This is therefore a LOWER BOUND on cost, i.e. optimistic.")
    print("   - dp_charged_on='both' is the conservative reading of a point")
    print("     the sources genuinely disagree on.")


if __name__ == "__main__":
    main()
