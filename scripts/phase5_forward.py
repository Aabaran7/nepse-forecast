"""Phase 5: the daily forward prediction job and its scorer (plan §7).

Runs after archive_pull. Builds features from everything known as of the latest
session, predicts t+1 and t+5, and appends to the immutable log.

§7 is emphatic about what this is FOR: leak detection and pipeline validation.
~40 trading days cannot distinguish 55% accuracy from 50%, so the forward log
is not evidence of an edge and must never be reported as if it were. What it
catches is a feature that silently used future information, a data source that
changed shape, a fill rule that never fires on live data -- the failures that a
backtest cannot see because the backtest and the bug share the same code path.

The log is append-only and refuses to overwrite (§7). If the model changes, bump
model_version; do not rewrite history.

Usage:
    .venv/bin/python scripts/phase5_forward.py            # predict + log
    .venv/bin/python scripts/phase5_forward.py --score    # score the log
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.eval import labels  # noqa: E402
from nepselab.features import base as fbase  # noqa: E402
from nepselab.features import price as fprice  # noqa: E402
from nepselab.forward import log as flog  # noqa: E402
from nepselab.models.classifiers import Logistic  # noqa: E402

warnings.filterwarnings("ignore")

DEEP = Path("data/deep/nepse_index_deep.parquet")
RULE = "=" * 74
MODEL_VERSION = "lr-price-turnover-v1"
HORIZONS = (1, 5)


def hr(t: str) -> None:
    print(f"\n{RULE}\n{t}\n{RULE}")


def load_features() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(DEEP).sort_values("date").reset_index(drop=True)
    # price + turnover: the set that scored best in Phase 4, and the one whose
    # inputs the daily job can actually obtain. Reddit was dropped by §6's
    # alt-data gate (it subtracted 0.91pp), so the forward job does not need
    # the dump to be refreshed to keep running.
    mods = [fprice.PriceFeatures(), fprice.TurnoverFeatures()]
    a = fbase.assemble(df, mods, start=pd.Timestamp("2017-01-01"))
    return a.frame, a.feature_names


def predict_once(horizon: int, frame: pd.DataFrame, feats: list[str]) -> dict:
    """Train on everything with a formed label, predict the latest session.

    The embargo matters live too: the most recent `horizon` sessions cannot have
    labels yet, so they are excluded from training rather than filled in.
    """
    lab = labels.make_labels(frame, horizon=horizon)
    train = lab[lab["usable"]].reset_index(drop=True)
    model = Logistic().fit(train[feats], train["label"].to_numpy().astype(int))

    latest = frame.iloc[[-1]]
    pred = int(model.predict(latest[feats])[0])
    prob = float(model.predict_proba(latest[feats])[0])
    return {"horizon": horizon, "prediction": pred, "prob_up": round(prob, 6),
            "n_train": len(train), "close": float(latest["close"].iloc[0])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true", help="score the log instead")
    args = ap.parse_args()

    frame, feats = load_features()

    if args.score:
        hr("FORWARD LOG SCORING (§7)")
        actuals = pd.read_parquet(DEEP)[["date", "close"]]
        scored = flog.score_log(actuals)
        if scored.empty:
            print("  log is empty -- nothing to score yet")
            return
        resolved = scored[scored["correct"].notna()]
        print(f"  logged predictions : {len(scored)}")
        print(f"  resolved           : {len(resolved)}")
        print(f"  still open         : {len(scored) - len(resolved)}")
        if len(resolved):
            for h in sorted(resolved["horizon"].unique()):
                sub = resolved[resolved["horizon"] == h]
                print(f"    h={h}: {len(sub)} resolved, "
                      f"accuracy {sub['correct'].mean():.4f}")
        print("\n  §7: this is pipeline validation, NOT evidence of an edge.")
        print("  §6's capital gate needs >= 60 forward days AND forward accuracy")
        print("  within 5pp of the backtest before any capital is deployed.")
        return

    as_of = frame["date"].iloc[-1]
    hr(f"FORWARD PREDICTIONS as of {pd.Timestamp(as_of).date()}")
    rows = [predict_once(h, frame, feats) for h in HORIZONS]
    for r in rows:
        print(f"  h={r['horizon']}: {'UP' if r['prediction'] else 'DOWN'} "
              f"(p_up={r['prob_up']:.4f}, trained on {r['n_train']} sessions)")

    try:
        path = flog.append(as_of, rows, model_version=MODEL_VERSION)
        print(f"\n  appended -> {path}")
    except flog.PredictionExists as e:
        print(f"\n  NOT LOGGED: {e}")
        print("  This is the §7 guarantee working, not a failure.")


if __name__ == "__main__":
    main()
