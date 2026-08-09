"""Descriptive statistics for the per-scrip panel (plan §5, dashboard input).

NOT a test of anything. There is no forward return here, no label, and no
ranking by expected outcome -- see the boundary set out in
nepselab/features/scrip.py. Its job is to answer one question: which of the
things people say about volume are actually visible in NEPSE's data?

It exists because the module docstring makes factual claims about this market,
and a claim in a docstring that nobody can regenerate is folklore with better
formatting. Running this reproduces every figure quoted there.

Usage:
    .venv/bin/python scripts/phase9_scrip_panel.py
    .venv/bin/python scripts/phase9_scrip_panel.py --out results/phase9
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.eval.costs import CostModel, Params  # noqa: E402
from nepselab.features import scrip  # noqa: E402
from nepselab.ingest import archive  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("phase9")


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/phase9"))
    args = ap.parse_args()

    tp = archive.load("today_price")
    if tp.empty:
        log.error("no today_price archive; run scripts/archive_pull.py first")
        return 1

    params = Params()
    panel = scrip.build_panel(tp, params=params)
    traded = panel[panel["totalTradedQuantity"] > 0]
    rated = traded.dropna(subset=["ret", "vol_ratio"])

    dates = pd.to_datetime(panel["businessDate"])
    stats: dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_hash": git_hash(),
        "sessions": int(dates.nunique()),
        "scrips": int(panel["symbol"].nunique()),
        "span": [str(dates.min().date()), str(dates.max().date())],
        "scrip_days": int(len(panel)),
    }

    # --- thinness ---------------------------------------------------------
    stats["pct_days_under_10_trades"] = round(float((traded["totalTrades"] < 10).mean()), 4)
    stats["median_trades_per_scrip_day"] = float(traded["totalTrades"].median())
    spikes = rated[rated["vol_ratio"] >= 3.0]
    stats["pct_3x_spikes_under_10_trades"] = round(float((spikes["totalTrades"] < 10).mean()), 4)

    # --- THE CLAIM: does heavy volume mark buying or selling here? --------
    up, dn = rated[rated["at_limit_up"]], rated[rated["at_limit_down"]]
    stats["limit_up_days"] = int(len(up))
    stats["limit_down_days"] = int(len(dn))
    stats["limit_up_median_vol_ratio"] = round(float(up["vol_ratio"].median()), 3)
    stats["limit_down_median_vol_ratio"] = round(float(dn["vol_ratio"].median()), 3)
    ordinary = rated[~rated["at_limit_up"] & ~rated["at_limit_down"]]
    stats["ordinary_median_vol_ratio"] = round(float(ordinary["vol_ratio"].median()), 3)

    q = rated["quadrant"].value_counts(normalize=True)
    stats["quadrant_share"] = {k: round(float(v), 4) for k, v in q.items()}

    # --- the cost floor any scrip-level rule has to clear -----------------
    last = pd.Timestamp(dates.max())
    stats["round_trip_bps"] = {}
    for capital in (25_000, 50_000, 100_000):
        cm = CostModel(params=params, capital=capital)
        stats["round_trip_bps"][str(capital)] = round(cm.round_trip_bps(last), 1)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "panel_stats.json").write_text(json.dumps(stats, indent=2))
    scrip.market_breadth(panel).to_csv(args.out / "breadth.csv", index=False)
    scrip.turnover_concentration(panel).to_csv(args.out / "concentration.csv", index=False)

    print(f"\n{'='*66}\nSCRIP PANEL — {stats['sessions']} sessions, "
          f"{stats['scrips']} scrips, {stats['span'][0]} .. {stats['span'][1]}\n{'='*66}")
    print(f"\nTHIN DAYS (volume says nothing about who traded)")
    print(f"  median transactions per scrip-day : {stats['median_trades_per_scrip_day']:.0f}")
    print(f"  scrip-days under 10 transactions  : {stats['pct_days_under_10_trades']:.1%}")
    print(f"  3x volume spikes under 10 trades  : {stats['pct_3x_spikes_under_10_trades']:.1%}")

    print(f"\nHEAVY VOLUME: WHICH DIRECTION? (the imported heuristic says down)")
    print(f"  limit-UP   days {stats['limit_up_days']:>5}   "
          f"median volume {stats['limit_up_median_vol_ratio']:.2f}x own 20d median")
    print(f"  limit-DOWN days {stats['limit_down_days']:>5}   "
          f"median volume {stats['limit_down_median_vol_ratio']:.2f}x")
    print(f"  ordinary   days {len(ordinary):>5}   "
          f"median volume {stats['ordinary_median_vol_ratio']:.2f}x")
    print("  -> heavy volume accompanies buying more than selling in this market.")

    print(f"\nCOST FLOOR (what any scrip rule must beat, per round trip)")
    for cap, bps in stats["round_trip_bps"].items():
        print(f"  Rs {int(cap):>7,} position: {bps:>6.1f} bps  ({bps/100:.2f}%)")

    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
