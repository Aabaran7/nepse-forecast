"""Phase 4: models, ablation table, and the §6 thresholds applied (plan §6).

This is the run the whole project exists to make, and §6's numbers were frozen
on 2026-08-02 precisely so that this script could not be tuned against them.
It reports whatever it finds.

What it does:
  1. Walks logistic regression and GBM forward through the SAME harness the
     baselines used -- same folds, same embargo, same paired scoring.
  2. An ablation over feature sets: price-only / +turnover / +reddit / all.
     §5 requires this table; the alt-data gate in §6 reads it.
  3. The regime table, restored by §3.5.
  4. Applies §6's thresholds and prints the verdict.

Ablation honesty note: the feature sets do not all cover the same window. Reddit
starts 2020-06, turnover 2017-01, price 2016-01. Comparing a price-only model on
2,400 sessions with an all-features model on 1,300 measures the window as much
as the features. So the ablation runs TWICE: once per-set on its own maximal
window, and once with every set restricted to the common window. The second is
the one the §6 alt-data gate reads.

Usage: .venv/bin/python scripts/phase4_models.py [--horizon 1] [--quick]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.eval import baselines, costs, labels, metrics, portfolio  # noqa: E402
from nepselab.features import base as fbase  # noqa: E402
from nepselab.features import price as fprice  # noqa: E402
from nepselab.features import reddit as freddit  # noqa: E402
from nepselab.models.classifiers import GBM, Logistic  # noqa: E402

warnings.filterwarnings("ignore")

DEEP = Path("data/deep/nepse_index_deep.parquet")
RULE = "=" * 78
CAPITAL = 1_000_000

REGIMES = {
    "pre-mania 2016..2020-02": ("2016-01-01", "2020-02-29"),
    "mania 2020-06..2021-12": ("2020-06-01", "2021-12-31"),
    "chop 2022+": ("2022-01-01", "2030-01-01"),
}

FEATURE_SETS = {
    "price-only": (["price"], "2016-01-01"),
    "+turnover": (["price", "turnover"], "2017-01-01"),
    "+reddit": (["price", "reddit"], "2020-06-01"),
    "all": (["price", "turnover", "reddit"], "2020-06-01"),
}


def hr(t: str) -> None:
    print(f"\n{RULE}\n{t}\n{RULE}")


def make_modules(names: list[str]) -> list:
    reg = {"price": fprice.PriceFeatures, "turnover": fprice.TurnoverFeatures,
           "reddit": freddit.RedditAttention}
    return [reg[n]() for n in names]


def build(horizon: int, module_names: list[str], start: str,
          restrict_to: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(DEEP).sort_values("date").reset_index(drop=True)
    a = fbase.assemble(df, make_modules(module_names), start=pd.Timestamp(start))
    frame = a.frame
    if restrict_to is not None:
        frame = frame[frame["date"] >= pd.Timestamp(restrict_to)].reset_index(drop=True)
    lab = labels.make_labels(frame, horizon=horizon)
    usable = lab[lab["usable"]].reset_index(drop=True)
    return usable, a.feature_names


def evaluate(frame: pd.DataFrame, features: list[str], model, horizon: int,
             initial_train: int, n_boot: int) -> tuple[metrics.Score, pd.DataFrame]:
    base_p, _ = baselines.run_walk_forward(
        frame, baselines.MajorityClass(), horizon=horizon,
        feature_cols=features, initial_train=initial_train)
    preds, _ = baselines.run_walk_forward(
        frame, model, horizon=horizon, feature_cols=features,
        initial_train=initial_train)
    s = metrics.score(preds["y_true"].to_numpy(), preds["y_pred"].to_numpy(),
                      base_p["y_pred"].to_numpy(), horizon=horizon, n_boot=n_boot)
    preds = preds.assign(y_base=base_p["y_pred"].to_numpy())
    return s, preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--initial-train", type=int, default=400)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))

    hr(f"PHASE 4: MODELS AND §6 THRESHOLDS (h={args.horizon})")
    print("§6's numbers were frozen 2026-08-02, before any of this ran.")

    # ---- 1. ablation, each set on its own maximal window --------------------
    hr("1. ABLATION -- each feature set on its OWN maximal window")
    print("Windows differ, so this table conflates features with sample. Read it")
    print("for coverage, not for the alt-data gate.\n")
    print(f"{'feature set':<14}{'window':<24}{'n':>6}{'feats':>7}"
          f"{'LR edge':>10}{'GBM edge':>11}")
    print("-" * 78)
    own_window = {}
    for name, (mods, start) in FEATURE_SETS.items():
        frame, feats = build(args.horizon, mods, start)
        lr_s, _ = evaluate(frame, feats, Logistic(), args.horizon,
                           args.initial_train, args.n_boot)
        gbm_edges = []
        for sd in seeds:
            g_s, _ = evaluate(frame, feats, GBM(seed=sd), args.horizon,
                              args.initial_train, args.n_boot)
            gbm_edges.append(g_s.edge_vs_majority * 100)
        own_window[name] = (lr_s, np.mean(gbm_edges), np.std(gbm_edges))
        win = f"{frame.date.min().date()}..{frame.date.max().date()}"
        print(f"{name:<14}{win:<24}{len(frame):>6}{len(feats):>7}"
              f"{lr_s.edge_vs_majority * 100:>+9.2f}"
              f"{np.mean(gbm_edges):>+9.2f}±{np.std(gbm_edges):.2f}")

    # ---- 2. ablation on the common window ----------------------------------
    common_start = max(pd.Timestamp(s) for _, s in FEATURE_SETS.values())
    hr(f"2. ABLATION -- all sets restricted to the COMMON window "
       f"(from {common_start.date()})")
    print("This is the comparison §6's alt-data gate reads: same rows, same")
    print("folds, only the feature set changes.\n")
    print(f"{'feature set':<14}{'n':>6}{'feats':>7}{'LR acc':>9}{'LR edge':>10}"
          f"{'edge 95% CI':>20}{'GBM edge':>12}")
    print("-" * 78)
    common = {}
    for name, (mods, start) in FEATURE_SETS.items():
        frame, feats = build(args.horizon, mods, start,
                             restrict_to=str(common_start.date()))
        lr_s, lr_p = evaluate(frame, feats, Logistic(), args.horizon,
                              args.initial_train, args.n_boot)
        gbm_edges, gbm_preds, gbm_score = [], None, None
        for sd in seeds:
            g_s, g_p = evaluate(frame, feats, GBM(seed=sd), args.horizon,
                                args.initial_train, args.n_boot)
            gbm_edges.append(g_s.edge_vs_majority * 100)
            if sd == 0:
                gbm_preds, gbm_score = g_p, g_s
        elo, ehi = lr_s.extra["edge_ci"]
        common[name] = {"lr": lr_s, "lr_preds": lr_p, "frame": frame,
                        "features": feats, "gbm_mean": float(np.mean(gbm_edges)),
                        "gbm_sd": float(np.std(gbm_edges)),
                        "gbm_preds": gbm_preds, "gbm_score": gbm_score}
        print(f"{name:<14}{len(frame):>6}{len(feats):>7}{lr_s.accuracy:>9.4f}"
              f"{lr_s.edge_vs_majority * 100:>+10.2f}"
              f"   [{elo * 100:+.2f},{ehi * 100:+.2f}]pp"
              f"{np.mean(gbm_edges):>+8.2f}±{np.std(gbm_edges):.2f}")

    base_edge = common["price-only"]["lr"].edge_vs_majority * 100
    print(f"\n  Marginal contribution over price-only (LR):")
    for name in ("+turnover", "+reddit", "all"):
        print(f"    {name:<12}{common[name]['lr'].edge_vs_majority * 100 - base_edge:>+7.2f}pp")

    # ---- 3. regime table ---------------------------------------------------
    hr("3. BY REGIME (§2's split, restored by §3.5)")
    # Deliberately NOT the common window. Restricting to 2020-06 and then
    # spending 400 sessions on the initial training window leaves the mania
    # regime entirely inside the training data, so §6(c) -- the bull-market
    # artifact check -- would have no rows to test on. Price-only reaches back
    # to 2017 and is the only set that can answer the question it was
    # reinstated to answer.
    wide_frame, wide_feats = build(args.horizon, ["price"], "2016-01-01")
    wide_s, wide_p = evaluate(wide_frame, wide_feats, Logistic(), args.horizon,
                              args.initial_train, args.n_boot)
    tbl = metrics.by_regime(wide_p["date"], wide_p["y_true"].to_numpy(),
                            wide_p["y_pred"].to_numpy(),
                            wide_p["y_base"].to_numpy(),
                            REGIMES, horizon=args.horizon, n_boot=args.n_boot)
    print(f"logistic, price-only, {wide_frame.date.min().date()}.."
          f"{wide_frame.date.max().date()} (widest window with mania coverage):")
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}")
          if len(tbl) else "  (no regime rows)")

    # ---- 4. costs ----------------------------------------------------------
    hr("4. NET-OF-COST PnL (capital = NPR 1,000,000, 1x friction)")
    P = costs.Params()
    print(f"{'model':<22}{'gross Sh':>10}{'net Sh':>9}{'net CAGR':>10}"
          f"{'maxDD':>9}{'trades':>8}")
    print("-" * 78)
    net_sharpes = {}
    for label, key, pk in (("LR (all features)", "all", "lr_preds"),
                           ("GBM (all features)", "all", "gbm_preds"),
                           ("LR (price-only)", "price-only", "lr_preds")):
        d = common[key]
        preds = d[pk]
        test = d["frame"][d["frame"]["date"].isin(preds["date"])].reset_index(drop=True)
        sig = preds.sort_values("date")["y_pred"].to_numpy()
        cm = costs.CostModel(P, capital=CAPITAL)
        r = portfolio.run_backtest(test, sig, cm)
        net_sharpes[label] = r.stats["net_sharpe"]
        print(f"{label:<22}{r.stats['gross_sharpe']:>10.2f}"
              f"{r.stats['net_sharpe']:>9.2f}{r.stats['net_cagr']:>9.1%}"
              f"{r.stats['max_drawdown']:>9.1%}{r.stats['n_trades']:>8}")

    d = common["all"]
    test = d["frame"][d["frame"]["date"].isin(d["lr_preds"]["date"])].reset_index(drop=True)
    cm = costs.CostModel(P, capital=CAPITAL)
    bh = portfolio.buy_and_hold(test, cm)
    print(f"{'buy-and-hold':<22}{bh.stats['gross_sharpe']:>10.2f}"
          f"{bh.stats['net_sharpe']:>9.2f}{bh.stats['net_cagr']:>9.1%}"
          f"{bh.stats['max_drawdown']:>9.1%}{bh.stats['n_trades']:>8}")

    # ---- 5. the thresholds -------------------------------------------------
    hr("5. §6 ABANDONMENT THRESHOLDS -- FROZEN 2026-08-02, APPLIED NOW")
    # Consider BOTH model families, and quote the CI belonging to whichever
    # actually won -- an earlier version printed the best edge next to a
    # different feature set's interval.
    candidates = []
    for nm, v in common.items():
        candidates.append((f"LR / {nm}", v["lr"]))
        if v.get("gbm_score") is not None:
            candidates.append((f"GBM(seed0) / {nm}", v["gbm_score"]))
    best_name, best_score = max(candidates, key=lambda kv: kv[1].edge_vs_majority)
    best_edge = best_score.edge_vs_majority * 100
    elo, ehi = best_score.extra["edge_ci"]

    print(f"\n  (a) PRIMARY: h={args.horizon} accuracy > majority + 2pp")
    print(f"      best of {len(candidates)} model x feature-set combinations:")
    print(f"        {best_name}  edge {best_edge:+.2f}pp")
    print(f"        95% CI [{elo * 100:+.2f}, {ehi * 100:+.2f}]pp "
          f"(block={best_score.extra['block']})")
    a_pass = best_edge > 2.0
    print(f"      point estimate clears +2pp: {'YES' if a_pass else 'NO'}")
    print(f"      CI excludes +2pp:           "
          f"{'YES' if elo * 100 > 2.0 else 'NO'}")
    print(f"      CI excludes ZERO:           "
          f"{'YES' if elo > 0 else 'NO'}")
    print(f"      NOTE: picking the best of {len(candidates)} combinations is")
    print(f"      itself a multiple-comparisons problem. The CI above is not")
    print(f"      corrected for it, so the true interval is WIDER than shown.")

    print(f"\n  (b) Net Sharpe >= 0.4 after realistic costs at NPR {CAPITAL:,}")
    for k, v in net_sharpes.items():
        print(f"      {k:<22}{v:>7.3f}  {'PASS' if v >= 0.4 else 'FAIL'}")

    print(f"\n  (c) Edge present only in the mania regime -> abandon")
    mania_only = False
    if len(tbl):
        for _, r_ in tbl.iterrows():
            print(f"      {r_['regime']:<26}{r_['edge_pp']:>+7.2f}pp  "
                  f"[{r_['edge_lo_pp']:+.2f},{r_['edge_hi_pp']:+.2f}]")
        mania = tbl[tbl["regime"].str.startswith("mania")]
        others = tbl[~tbl["regime"].str.startswith("mania")]
        if len(mania) and len(others):
            mania_only = (float(mania["edge_pp"].iloc[0]) > 2.0
                          and (others["edge_pp"] <= 2.0).all())
        print(f"      edge confined to the mania regime: "
              f"{'YES -- §6(c) TRIGGERS' if mania_only else 'no'}")
    else:
        print("      (regime table empty on this window)")

    print(f"\n  (d) Alt-data gate: Reddit must add >= 1pp over price-only")
    reddit_gain = (common["+reddit"]["lr"].edge_vs_majority
                   - common["price-only"]["lr"].edge_vs_majority) * 100
    print(f"      +reddit vs price-only (common window): {reddit_gain:+.2f}pp")
    print(f"      gate: {'KEEP reddit' if reddit_gain >= 1.0 else 'DROP reddit module'}")

    print(f"\n  (e) Capital gate: >= 60 forward trading days logged")
    fwd = Path("predictions")
    n_fwd = len(list(fwd.glob("*.csv"))) if fwd.exists() else 0
    print(f"      forward log entries: {n_fwd} -- gate NOT met (Phase 5 just built)")

    hr("VERDICT")
    sharpe_pass = any(v >= 0.4 for k, v in net_sharpes.items()
                      if not k.startswith("buy"))
    if not sharpe_pass:
        print("  §6(b) FAILS: no model clears net Sharpe 0.4 after costs.")
        print("  Per §6 this means NO CAPITAL, regardless of accuracy.")
    if not a_pass:
        print("  §6(a) FAILS: no feature set clears majority + 2pp.")
    if a_pass and elo * 100 <= 2.0:
        print("  §6(a) point estimate clears the bar but its interval does not.")
        print("  Per §2 and §3.7 that is NOT evidence of an edge.")
    if mania_only:
        print("  §6(c) TRIGGERS: the edge lives in the mania regime and is absent")
        print("  (or negative) elsewhere. §6 calls that a bull-market artifact and")
        print("  abandons on it. This criterion was RETIRED on 2026-08-02 for want")
        print("  of data and reinstated by §3.5 -- it earned its place back.")
    print("\n  These numbers are reported as found. §6 and §9 forbid re-tuning")
    print("  against them; a second run with adjusted thresholds is not a")
    print("  better result, it is an invalidated one.")


if __name__ == "__main__":
    main()
