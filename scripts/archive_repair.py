"""Re-pull a session and replace a broken archived capture. Use rarely.

The archive is append-only (§3.4) and that is almost always right. This is the
exception it cannot handle on its own: when OUR capture of a session was broken,
append-only enshrines the broken value forever and rejects every corrected one.

The case this was written for, 2026-08-04: a pull ran mid-session, NEPSE
returned `closingIndex` 0 for all 17 indices, 109 of ~350 scrips and 1.2% of a
normal day's turnover, and all of it was archived as final. Every pull since has
re-fetched the real closes and dutifully discarded them, logging 17 conflicts a
run. `scripts/archive_pull.py` now refuses to archive an in-progress session, so
this should stay rare -- but the damage already done needs undoing.

Always dry-run first. The default IS a dry run.

Usage:
    .venv/bin/python scripts/archive_repair.py --date 2026-08-04
    .venv/bin/python scripts/archive_repair.py --date 2026-08-04 --apply \\
        --reason "captured mid-session; closingIndex 0 on all 17 indices"
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.ingest import archive  # noqa: E402
from nepselab.ingest.nepse_client import NepseClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("archive_repair")


def fetch(c: NepseClient, day: str) -> dict[str, pd.DataFrame]:
    """Everything NEPSE now reports for `day`, keyed by dataset."""
    out: dict[str, pd.DataFrame] = {}

    sectors = c.sector_indices()
    frames = []
    for _, row in sectors.iterrows():
        idx_id = int(row["id"])
        try:
            df = c.index_history("2000-01-01", date.today().isoformat(), index_id=idx_id)
        except Exception as exc:  # noqa: BLE001
            log.error("index %s failed: %s", idx_id, exc)
            continue
        if df.empty:
            continue
        df["exchangeIndexId"] = idx_id
        df["indexCode"] = row.get("indexCode")
        frames.append(df)
    if frames:
        idx = pd.concat(frames, ignore_index=True)
        out["indices"] = idx[pd.to_datetime(idx["businessDate"]).dt.normalize() == day]

    ms = c.market_summary_history(day, day)
    if not ms.empty:
        out["market_summary"] = ms[pd.to_datetime(ms["businessDate"]).dt.normalize() == day]

    try:
        tp = c.today_price(day)
        if not tp.empty:
            out["today_price"] = tp
    except Exception as exc:  # noqa: BLE001
        log.error("today_price %s failed: %s", day, exc)

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True, help="session to repair, YYYY-MM-DD")
    ap.add_argument("--apply", action="store_true",
                    help="actually write. Without this it is a dry run.")
    ap.add_argument("--reason", default="",
                    help="why this capture is being replaced; recorded in _repairs/")
    args = ap.parse_args()

    day = pd.Timestamp(args.date).normalize()

    log.info("fetching what NEPSE reports for %s now ...", day.date())
    fresh = fetch(NepseClient(), day.strftime("%Y-%m-%d"))
    if not fresh:
        log.error("NEPSE returned nothing for %s; refusing to remove anything",
                  day.date())
        return 1

    print(f"\n{'=' * 70}\nREPAIR PREVIEW — {day.date()}\n{'=' * 70}")
    for ds, new in fresh.items():
        have = archive.load(ds)
        if have.empty:
            continue
        hit = pd.to_datetime(have["businessDate"]).dt.normalize() == day
        print(f"\n{ds}: archived {int(hit.sum())} row(s) -> upstream now has {len(new)}")
        if ds == "indices" and int(hit.sum()):
            cmp = (have[hit][["exchangeIndexId", "closingIndex"]]
                   .merge(new[["exchangeIndexId", "closingIndex"]],
                          on="exchangeIndexId", suffixes=("_archived", "_upstream")))
            print(cmp.head(20).to_string(index=False))

    if not args.apply:
        print("\nDry run. Nothing written. Re-run with --apply and --reason.")
        return 0

    if not args.reason.strip():
        log.error("--apply needs --reason. This is the only operation that "
                  "removes archived rows; it does not happen unexplained.")
        return 1

    results = [archive.repair(ds, new, [day], args.reason) for ds, new in fresh.items()]
    archive.record(results)
    print(f"\n{'=' * 70}\nREPAIRED\n{'=' * 70}")
    for r in results:
        print(f"  {r.dataset:<18} {r.total:>7,} rows total")
    print("\nPrevious values are in data/archive/_repairs/. Re-run "
          "scripts/export_dashboard.py to refresh the page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
