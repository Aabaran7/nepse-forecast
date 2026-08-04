"""Phase 6: the pre-registered low-turnover rule (plan §6.3).

Runs exactly what §6.3 committed to before seeing any of it, and applies the
same §6(b) threshold plus the two guards §6.3 added because the obvious way to
pass is to degenerate into buy-and-hold.

Reports as found. §6.3 declared in advance that there is no third attempt.

Usage: .venv/bin/python scripts/phase6_lowturnover.py [--horizon 1]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.eval import costs, labels, portfolio, strategy  # noqa: E402
from nepselab.features import base as fbase  # noqa: E402
from nepselab.features import price as fprice  # noqa: E402
from nepselab.models.classifiers import GBM, Logistic  # noqa: E402

warnings.filterwarnings("ignore")

DEEP = Path("data/deep/nepse_index_deep.parquet")
RULE = "=" * 78
CAPITAL = 1_000_000

# §6.3's pre-registered guards.
SHARPE_BAR = 0.4
MAX_TIME_IN_MARKET = 0.90
MIN_ROUND_TRIPS = 6


def hr(t: str) -> None:
    print(f"\n{RULE}\n{t}\n{RULE}")


def build(horizon: int):
    df = pd.read_parquet(DEEP).sort_values("date").reset_index(drop=True)
    a = fbase.assemble(df, [fprice.PriceFeatures(), fprice.TurnoverFeatures()],
                       start=pd.Timestamp("2017-01-01"))
    lab = labels.make_labels(a.frame, horizon=horizon)
    return lab[lab["usable"]].reset_index(drop=True), a.feature_names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--initial-train", type=int, default=400)
    args = ap.parse_args()

    hr(f"PHASE 6: LOW-TURNOVER DECISION RULE (h={args.horizon})")
    print("Pre-registered in §6.3 before this ran. Criterion unchanged: §6(b).")

    frame, feats = build(args.horizon)
    P = costs.Params()
    cost = costs.CostModel(P, capital=CAPITAL)
    print(f"frame: {len(frame)} sessions, {frame.date.min().date()}"
          f" .. {frame.date.max().date()}, {len(feats)} features")

    results = {}
    for label, factory in (("logistic", lambda: Logistic()),
                           ("gbm", lambda: GBM(seed=0))):
        preds = strategy.run_walk_forward_strategy(
            frame, factory, horizon=args.horizon, feature_cols=feats,
            cost=cost, initial_train=args.initial_train)
        test = frame[frame["date"].isin(preds["date"])].reset_index(drop=True)

        # Baseline: the SAME model with §6.2's rule (act on p>0.5 every day).
        naive_pos = (preds["prob_up"].to_numpy() > 0.5).astype(int)
        r_naive = portfolio.run_backtest(test, naive_pos, cost)
        r_rule = portfolio.run_backtest(test, preds["position"].to_numpy(), cost)
        results[label] = (preds, r_naive, r_rule, test)

    bh = portfolio.buy_and_hold(results["logistic"][3], cost)

    hr("1. RULES CHOSEN PER FOLD (on training data only)")
    for label, (preds, _, _, _) in results.items():
        counts = preds.groupby(["delta", "min_hold"]).size().sort_values(ascending=False)
        st = preds.attrs["stitched_rule"]
        print(f"\n{label}: stitched rule delta={st.delta}, min_hold={st.min_hold}")
        print("  most-selected (delta, min_hold) across folds:")
        for (d, m), n in counts.head(4).items():
            print(f"    delta={d:<5} min_hold={m:<3} on {n:>5} test sessions")

    hr("2. NAIVE RULE vs LOW-TURNOVER RULE")
    print(f"{'model / rule':<26}{'net Sh':>9}{'net CAGR':>10}{'maxDD':>9}"
          f"{'trades':>8}{'in mkt':>9}{'costs/cap':>11}")
    print("-" * 78)
    for label, (_, r_naive, r_rule, _) in results.items():
        for tag, r in (("naive p>0.5", r_naive), ("deadband+hold", r_rule)):
            s = r.stats
            print(f"{label + ' / ' + tag:<26}{s['net_sharpe']:>9.2f}"
                  f"{s['net_cagr']:>10.1%}{s['max_drawdown']:>9.1%}"
                  f"{s['n_trades']:>8}{s['time_in_market']:>9.1%}"
                  f"{s['total_costs'] / CAPITAL:>10.1%}")
    s = bh.stats
    print(f"{'buy-and-hold':<26}{s['net_sharpe']:>9.2f}{s['net_cagr']:>10.1%}"
          f"{s['max_drawdown']:>9.1%}{s['n_trades']:>8}"
          f"{s['time_in_market']:>9.1%}{s['total_costs'] / CAPITAL:>10.1%}")

    hr("3. §6.3 PRE-REGISTERED CRITERIA")
    bh_sharpe = bh.stats["net_sharpe"]
    any_pass = False
    for label, (_, _, r_rule, _) in results.items():
        s = r_rule.stats
        c1 = s["net_sharpe"] >= SHARPE_BAR
        c2 = s["net_sharpe"] > bh_sharpe
        c3 = s["time_in_market"] < MAX_TIME_IN_MARKET
        c4 = s["n_round_trips"] >= MIN_ROUND_TRIPS
        print(f"\n  {label}:")
        print(f"    (1) net Sharpe {s['net_sharpe']:.3f} >= {SHARPE_BAR}          "
              f"{'PASS' if c1 else 'FAIL'}")
        print(f"    (2) beats buy-and-hold ({bh_sharpe:.3f})            "
              f"{'PASS' if c2 else 'FAIL'}")
        print(f"    (3) time in market {s['time_in_market']:.1%} < "
              f"{MAX_TIME_IN_MARKET:.0%}         {'PASS' if c3 else 'FAIL'}")
        print(f"    (4) round trips {s['n_round_trips']} >= {MIN_ROUND_TRIPS}"
              f"                     {'PASS' if c4 else 'FAIL'}")
        if all((c1, c2, c3, c4)):
            any_pass = True
            print(f"    ALL FOUR PASS")

    hr("VERDICT")
    if any_pass:
        print("  §6.3's criteria are met by at least one model.")
        print("  This does NOT reopen §6.2 -- that null stands as recorded. What")
        print("  it shows is that the decision rule, not the classifier, was the")
        print("  binding constraint, which is what §6.3 predicted from arithmetic.")
        print("  §6's capital gate (>= 60 forward days) still applies before any")
        print("  capital moves, and §6(c)'s regime check must be re-read here.")
    else:
        print("  §6.3's criteria are NOT met. Per §6.3, this is written up as a")
        print("  second null and there is no third attempt -- a rule needing a")
        print("  third redesign to clear a fixed bar is fitting the bar.")


if __name__ == "__main__":
    main()
