"""Phase 2a: walk-forward harness, baselines and metrics on the deep series.

§1 calls the harness the primary objective -- "models are interchangeable; the
harness is what makes any result believable" -- so this runs before any model
exists, on baselines only. That ordering is deliberate. A coin flip that scores
56% here is a harness bug; discovered now it costs an afternoon, discovered in
Phase 4 it would have been reported as an edge.

What this establishes, in advance of any modelling:
  - the walk-forward folds are leak-free (asserted, not assumed);
  - the majority-class bar §6 measures against, per regime, fitted on train;
  - the width of the confidence interval any Phase 4 result will carry.

NOT here: cost model, fill logic, net PnL, Sharpe. Those are Phase 2b and are
blocked on §4 -- every cost constant in market_params.yaml is still null, and
§6.1 is explicit that a backtest must not run against placeholder costs.

Usage: .venv/bin/python scripts/phase2_walkforward.py [--horizon 1] [--h5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.eval import baselines, labels, metrics  # noqa: E402

DEEP = Path("data/deep/nepse_index_deep.parquet")
RULE = "=" * 78

# §2/§3.6. The mania window is the one where the majority class flips to UP.
REGIMES = {
    "pre-mania 2016..2020-02": ("2016-01-01", "2020-02-29"),
    "mania 2020-06..2021-12": ("2020-06-01", "2021-12-31"),
    "chop 2022+": ("2022-01-01", "2030-01-01"),
}


def hr(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def build_frame(horizon: int) -> pd.DataFrame:
    df = pd.read_parquet(DEEP).sort_values("date").reset_index(drop=True)

    # The only feature Phase 2a needs: yesterday's return, strictly lagged.
    # Phase 3 replaces this with the real modules; it exists here so the
    # momentum baseline has something to be wrong with.
    df["prev_return"] = df["close"].pct_change()

    lab = labels.make_labels(df, horizon=horizon)
    print(f"labelled {len(lab)} rows at h={horizon}: "
          f"{lab.attrs['dropped_flat']} flat, "
          f"{lab.attrs['dropped_long_span']} spanning a closure, "
          f"{int((~lab['usable']).sum())} unusable in total")

    frame = lab[lab["usable"] & lab["prev_return"].notna()].reset_index(drop=True)
    print(f"usable for modelling: {len(frame)} rows, "
          f"{frame.date.min().date()} .. {frame.date.max().date()}")
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--initial-train", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    if not DEEP.exists():
        print(f"{DEEP} missing -- run scripts/phase1c_deep_history.py")
        sys.exit(1)

    hr(f"PHASE 2A: WALK-FORWARD HARNESS (h={args.horizon})")
    frame = build_frame(args.horizon)

    models = [baselines.MajorityClass(), baselines.AlwaysUp(),
              baselines.Coin(seed=0), baselines.Momentum()]

    hr("1. FOLDS")
    _, splits = baselines.run_walk_forward(
        frame, baselines.MajorityClass(), horizon=args.horizon,
        feature_cols=["prev_return"], initial_train=args.initial_train)
    print(f"{len(splits)} monthly folds, embargo {args.horizon} row(s) per fold")
    print(f"  first: {splits[0]}")
    print(f"  last:  {splits[-1]}")
    total_test = sum(len(s.test) for s in splits)
    print(f"  pooled test rows: {total_test}")
    print("  leakage assertion: PASSED (checked inside run_walk_forward)")

    # The baseline is the majority-class predictor WALKED FORWARD -- refit at
    # every retrain, exactly as a live competitor would be. Every edge below is
    # measured against these predictions, on identical rows.
    base_preds, _ = baselines.run_walk_forward(
        frame, baselines.MajorityClass(), horizon=args.horizon,
        feature_cols=["prev_return"], initial_train=args.initial_train)
    y_base = base_preds["y_pred"].to_numpy()

    hr("2. POOLED RESULTS")
    print(f"{'model':<16}{'n':>7}{'acc':>9}{'F1(up)':>9}{'majority':>10}"
          f"{'edge pp':>9}{'edge 95% CI':>20}")
    print("-" * 78)
    results = {}
    for m in models:
        preds, _ = baselines.run_walk_forward(
            frame, m, horizon=args.horizon, feature_cols=["prev_return"],
            initial_train=args.initial_train)
        s = metrics.score(preds["y_true"].to_numpy(), preds["y_pred"].to_numpy(),
                          y_base, horizon=args.horizon, n_boot=args.n_boot)
        results[m.name] = (preds, s)
        elo, ehi = s.extra["edge_ci"]
        print(f"{m.name:<16}{s.n:>7}{s.accuracy:>9.4f}{s.f1_up:>9.4f}"
              f"{s.baseline_majority:>10.4f}{s.edge_vs_majority * 100:>+9.2f}"
              f"   [{elo * 100:+.2f}, {ehi * 100:+.2f}]pp")

    hr("3. HARNESS SELF-CHECK")
    coin_s = results["coin"][1]
    ok = coin_s.ci[0] <= 0.5 <= coin_s.ci[1]
    print(f"  coin accuracy {coin_s.accuracy:.4f}, CI covers 0.50: "
          f"{'YES' if ok else 'NO -- HARNESS BUG'}")
    maj_s = results["majority-class"][1]
    print(f"  majority-class edge vs itself: {maj_s.edge_vs_majority * 100:+.2f}pp "
          f"(must be ~0)")
    print("  A coin that beats 50%, or a majority baseline that beats itself,")
    print("  means the harness is scoring something other than what it claims.")

    hr("4. BY REGIME (the majority class flips -- §3.6)")
    for name in ("majority-class", "momentum"):
        preds, _ = results[name]
        tbl = metrics.by_regime(preds["date"], preds["y_true"].to_numpy(),
                                preds["y_pred"].to_numpy(), y_base,
                                REGIMES, horizon=args.horizon,
                                n_boot=args.n_boot)
        print(f"\n{name}:")
        print(tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    hr("5. BLOCK-LENGTH SENSITIVITY (how fragile is that edge?)")
    mom = results["momentum"][0]
    yt, yp = mom["y_true"].to_numpy(), mom["y_pred"].to_numpy()
    rec = metrics.recommended_block(len(yt), args.horizon)
    print(f"  momentum edge vs majority, by bootstrap block length "
          f"(recommended: {rec})")
    for b in sorted({1, args.horizon, rec, 2 * rec, 25}):
        lo, hi = metrics.paired_edge_ci(yt, yp, y_base, block=b,
                                        n_boot=args.n_boot)
        star = " <-- used" if b == rec else ""
        print(f"    block={b:<4} CI [{lo * 100:+6.2f}, {hi * 100:+6.2f}]pp"
              f"   excludes 0: {'yes' if not (lo <= 0 <= hi) else 'NO':<3}{star}")
    print("  Dependence outlives the label overlap -- trends and regimes do not")
    print("  reset every h sessions -- so block=h is anti-conservative. Any")
    print("  Phase 4 claim of significance must state the block length it used.")

    hr("6. WHAT A PHASE 4 RESULT WILL BE ABLE TO SAY")
    width = (coin_s.ci[1] - coin_s.ci[0]) * 100
    print(f"  Pooled test n = {total_test} at h={args.horizon}")
    print(f"  95% CI width on accuracy: {width:.1f}pp "
          f"(±{width / 2:.1f}pp around a point estimate)")
    print(f"  §6 bar: majority ({maj_s.baseline_majority:.4f}) + 2pp = "
          f"{maj_s.baseline_majority + 0.02:.4f}")
    if width / 2 > 2.0:
        print(f"  The interval is wider than the 2pp threshold. A point estimate")
        print(f"  clearing the bar will NOT be distinguishable from one that")
        print(f"  doesn't -- report the interval next to every number.")
    else:
        print(f"  The interval is narrower than the 2pp threshold; a result here")
        print(f"  is interpretable.")


if __name__ == "__main__":
    main()
