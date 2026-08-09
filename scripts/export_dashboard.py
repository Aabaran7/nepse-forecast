"""Write the dashboard's JSON. The only bridge between the pipeline and the page.

The dashboard is a static site: no server, no database, no query at page load.
This script reduces the archive to the few hundred kilobytes the page actually
renders and drops it in `public/data/`, where Vite copies it verbatim into the
build. The page fetches it at runtime, so a daily data refresh does NOT require
a rebuild -- only a re-run of this script and a commit.

Two things it is careful about.

  IT NEVER INVENTS A FIELD. The design was drawn against mock data with columns
  this project does not have (a per-prediction `note`, an `id`, a `scrip`). Those
  are gone; the log carries model_version, prob_up and git_hash instead, and
  those are what ships. A dashboard column with nothing real behind it is the
  cheapest possible way to mislead someone, including yourself.

  IT DOES NOT DECIDE WHAT IS TRUE. Accuracy, baselines and the minimum sample
  size come from the same nepselab code the Streamlit app used, not from a
  reimplementation in TypeScript. If those rules change they change in one place.

Usage:
    .venv/bin/python scripts/export_dashboard.py
    .venv/bin/python scripts/export_dashboard.py --out public/data/dashboard.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.dashboard import data as dash  # noqa: E402
from nepselab.features import scrip as scrip_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("export_dashboard")

OUT = Path("public/data/dashboard.json")

# Same threshold the Streamlit page used: below this many resolved predictions
# an accuracy figure is noise wearing a percent sign, and one correct call
# renders a bar at 100%.
MIN_RESOLVED = 30

# How much history the page needs. The full archive is ~76k scrip-days; sending
# all of it to a browser to render a 60-day bar chart would be silly.
BREADTH_SESSIONS = 60
CONCENTRATION_SESSIONS = 252
STALE_DAYS = 5

FRESHNESS_LABELS = {
    "archive": "Exchange archive",
    "scrips": "Scrip prices",
    "deep": "Deep history",
    "news": "News",
}


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def clean(v):
    """JSON has no NaN. Pandas produces it constantly. Convert once, here.

    json.dumps writes bare `NaN`, which is valid JavaScript but NOT valid JSON --
    JSON.parse rejects it and the page dies with a parse error pointing at a
    character offset, which is a miserable way to discover a missing price.
    """
    if v is None:
        return None
    # NaT FIRST. It is not a float, so the isnan check misses it, and it is
    # Timestamp-ish enough that str() cheerfully returns the string "NaT" --
    # which then ships to the browser and renders as a publication date.
    if v is pd.NaT:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return None if pd.isna(v) else str(pd.Timestamp(v).date())
    if hasattr(v, "item"):          # numpy scalar
        v = v.item()
        return None if isinstance(v, float) and math.isnan(v) else v
    if v is pd.NA or (isinstance(v, float) and pd.isna(v)):
        return None
    return v


def build_freshness() -> list[dict]:
    today = pd.Timestamp.today().normalize()
    rows = []
    for key, ts in dash.freshness().items():
        if ts is None:
            rows.append({"label": FRESHNESS_LABELS.get(key, key), "date": None,
                         "daysAgo": None, "stale": True})
            continue
        days = int((today - pd.Timestamp(ts).normalize()).days)
        rows.append({
            "label": FRESHNESS_LABELS.get(key, key),
            "date": str(pd.Timestamp(ts).date()),
            "daysAgo": days,
            # `deep` is refreshed by a manual run, not the cron, so it is
            # expected to lag. Flagging it stale every single day would train
            # the reader to ignore the flag on the sources that matter.
            "stale": days > STALE_DAYS and key != "deep",
        })
    return rows


def build_forward_log(closes: pd.DataFrame) -> dict:
    scored = dash.forward_log()
    if scored.empty:
        return {"totalPredictions": 0, "resolved": 0, "open": 0,
                "exposureRows": 0, "minN": MIN_RESOLVED,
                "horizons": [], "predictions": []}

    directional = dash.directional_log(scored)
    resolved = directional[directional["correct"].notna()]
    exposure = scored[scored["kind"] == "exposure"]

    horizons = []
    for h in sorted(directional["horizon"].unique()):
        sub = resolved[resolved["horizon"] == h]
        fwd = closes["close"].pct_change(int(h)).shift(-int(h)).dropna()
        baseline = max((fwd > 0).mean(), (fwd <= 0).mean()) if len(fwd) else 0.5
        horizons.append({
            "horizon": f"t+{int(h)} day" + ("s" if int(h) != 1 else ""),
            "n": int(len(sub)),
            "modelPct": round(float(sub["correct"].mean()) * 100, 1) if len(sub) else None,
            "baselinePct": round(float(baseline) * 100, 1),
            "minN": MIN_RESOLVED,
        })

    preds = []
    for _, r in scored.sort_values(["as_of", "horizon"], ascending=[False, True]).iterrows():
        outcome = (None if pd.isna(r.get("correct"))
                   else ("correct" if int(r["correct"]) == 1 else "incorrect"))
        preds.append({
            "asOf": clean(r["as_of"]),
            "kind": r.get("kind"),
            "horizon": f"t+{int(r['horizon'])}",
            "direction": ("up" if int(r["prediction"]) == 1 else "down")
                         if r.get("kind") == "direction" else None,
            "probUp": clean(r.get("prob_up")),
            "exposure": clean(r.get("exposure")),
            "modelVersion": clean(r.get("model_version")),
            "gitHash": clean(r.get("git_hash")),
            "outcome": outcome,
            "fwdReturn": clean(r.get("fwd_return")),
        })

    return {
        "totalPredictions": int(len(directional)),
        "resolved": int(len(resolved)),
        "open": int(len(directional) - len(resolved)),
        "exposureRows": int(len(exposure)),
        "minN": MIN_RESOLVED,
        "horizons": horizons,
        "predictions": preds,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    closes = dash.index_closes()
    panel = dash.scrip_panel()
    news = dash.headlines()

    payload: dict = {
        "generatedUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gitHash": git_hash(),
        "freshness": build_freshness(),
        "forwardLog": build_forward_log(closes),
        "indexData": [], "breadthData": [], "concentrationData": [],
        "stocks": [], "session": None, "news": [],
    }

    if not closes.empty:
        payload["indexData"] = [
            {"date": str(d.date()), "value": round(float(c), 2)}
            for d, c in zip(closes["date"], closes["close"]) if not math.isnan(c)
        ]

    if not panel.empty:
        breadth = scrip_features.market_breadth(panel).tail(BREADTH_SESSIONS)
        payload["breadthData"] = [
            {"date": str(pd.Timestamp(r.businessDate).date()),
             # Falling is sent negative: the chart diverges from a zero centre
             # line, and doing the sign here keeps the component dumb.
             "rising": int(r.advancers), "falling": -int(r.decliners)}
            for r in breadth.itertuples()
        ]

        conc = scrip_features.turnover_concentration(panel).tail(CONCENTRATION_SESSIONS)
        payload["concentrationData"] = [
            {"date": str(pd.Timestamp(r.businessDate).date()),
             "pct": round(float(r.top10_share) * 100, 1)}
            for r in conc.itertuples() if not pd.isna(r.top10_share)
        ]

        latest = scrip_features.latest_session(panel)
        payload["session"] = str(pd.Timestamp(latest["businessDate"].iloc[0]).date())
        payload["stocks"] = [{
            "symbol": clean(r.get("symbol")),
            "close": clean(r.get("closePrice")),
            # Percent, to match how the table renders it.
            "change": None if pd.isna(r.get("ret")) else round(float(r["ret"]) * 100, 2),
            "volumeVsNormal": None if pd.isna(r.get("vol_ratio")) else round(float(r["vol_ratio"]), 2),
            "trades": clean(r.get("totalTrades")),
            # Shares per trade. Rounded because 13 decimal places of a share
            # count is noise, and it is repeated across ~350 rows of payload.
            "avgTrade": None if pd.isna(r.get("avg_trade_size"))
                        else round(float(r["avg_trade_size"]), 1),
            "thin": bool(r.get("is_thin")),
            "dayType": clean(r.get("quadrant")),
            "limitUp": bool(r.get("at_limit_up", False)),
            "limitDown": bool(r.get("at_limit_down", False)),
            "turnover": clean(r.get("totalTradedValue")),
        } for _, r in latest.iterrows()]

    if not news.empty:
        payload["news"] = [{
            "session": clean(r.get("session")),
            "source": clean(r.get("source")),
            "headline": clean(r.get("title")),
            "publishedAt": clean(r.get("published")),
            "url": clean(r.get("url")),
            "scored": False,     # flips when scripts/score_sentiment.py exists
        } for _, r in news.iterrows()]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False turns "I emitted invalid JSON" from a runtime mystery in
    # the browser into a loud failure here, where the data actually is.
    args.out.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":")))

    size_kb = args.out.stat().st_size / 1024
    log.info("wrote %s (%.0f KB)", args.out, size_kb)
    log.info("  index %d pts | stocks %d | news %d | predictions %d | resolved %d",
             len(payload["indexData"]), len(payload["stocks"]), len(payload["news"]),
             payload["forwardLog"]["totalPredictions"], payload["forwardLog"]["resolved"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
