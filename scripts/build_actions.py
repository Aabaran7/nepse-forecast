"""Rebuild the corporate action calendar, and refresh sector labels.

Corporate actions are DERIVED, not fetched: NEPSE restates a scrip's previous
close on the ex-date, so the ratio between `previousDayClosePrice` and the prior
session's actual close is the adjustment factor (nepselab/adjust/actions.py).
That makes this cheap enough to re-run daily and reproducible from the archive
alone -- there is no third-party source to go stale or disagree.

Sector labels come from a different NEPSE call than the securities snapshot and
are stored dated, because a company can be reclassified.

Usage:
    .venv/bin/python scripts/build_actions.py            # actions only, offline
    .venv/bin/python scripts/build_actions.py --sectors  # also refresh sectors
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.adjust import actions  # noqa: E402
from nepselab.ingest import archive  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("build_actions")

SECTOR_COLS = ["symbol", "companyName", "securityName", "sectorName",
               "instrumentType", "regulatoryBody", "status"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sectors", action="store_true",
                    help="also refresh sector labels from NEPSE (one call)")
    args = ap.parse_args()

    tp = archive.load("today_price")
    if tp.empty:
        log.error("no today_price archive; run scripts/archive_pull.py first")
        return 1

    acts = actions.detect(tp)
    log.info("%s", actions.summarise(acts))
    # Append-only like everything else: an ex-date, once observed, does not
    # change. A re-run adds only events from sessions archived since.
    res = [archive.merge("corporate_actions", acts)]

    if args.sectors:
        from nepselab.ingest.nepse_client import NepseClient
        info = NepseClient().company_info()
        if info.empty:
            log.error("company_info returned nothing; sectors not refreshed")
        else:
            keep = [c for c in SECTOR_COLS if c in info.columns]
            info = info[keep].copy()
            info["snapshot_date"] = pd.Timestamp(date.today())
            res.append(archive.merge("company_info", info))
            log.info("sectors: %d listing(s), %d sector(s)",
                     len(info), info["sectorName"].nunique())

    archive.record(res)
    print(f"\n{'=' * 66}\nARCHIVE\n{'=' * 66}")
    for r in res:
        print(f"  {r.dataset:<20} +{r.added:<7} {r.total:>7} total")

    if not acts.empty:
        print(f"\nLargest adjustments (factor < 1 means the share count grew):")
        print(acts.nsmallest(8, "factor")[
            ["ex_date", "symbol", "prior_close", "restated_close", "factor"]
        ].to_string(index=False))
        print("\n`kind` is 'unknown' by design: the factor cannot distinguish a "
              "bonus issue\nfrom a rights issue or a large special dividend. "
              "Naming it needs the company\nannouncement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
