"""Fold the Phase 0 raw pull into the archive. Run once, before archive_pull.py.

data/raw/ was pulled on 2026-07-25 and contains five sessions (2025-07-23 ..
2025-07-29) that NEPSE no longer serves -- the rolling window has since moved
past them. This is the only copy of those sessions in existence, so it gets
folded in before anything else touches the archive.

Usage: .venv/bin/python scripts/archive_seed.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.ingest import archive  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("seed")

RAW = Path("data/raw")


def main() -> None:
    results = []

    idx = pd.read_parquet(RAW / "index_nepse.parquet")
    if not idx.empty:
        idx["indexCode"] = "NEPSE"
        log.info("index_nepse: %d rows", len(idx))
    results.append(archive.merge("indices", idx))

    scrips = pd.read_parquet(RAW / "scrips.parquet")
    if not scrips.empty:
        # `security` is a nested dict (id, isin, listingDate, instrumentType...).
        # Parquet handles it, but it is dead weight in the archive and the id is
        # the only field we would ever join on -- so lift that and drop the rest.
        if "security" in scrips.columns:
            scrips["securityId"] = scrips["security"].map(
                lambda s: s.get("id") if isinstance(s, dict) else None
            )
            scrips = scrips.drop(columns=["security"])
        log.info("scrips: %d rows, %d symbols", len(scrips), scrips.symbol.nunique())
    results.append(archive.merge("ticker_history", scrips))

    secs = pd.read_parquet(RAW / "securities.parquet")
    if not secs.empty:
        secs = secs.rename(columns={"securitySymbol": "symbol"})
        # Dated to the Phase 0 pull, not to today -- this is what the listing
        # looked like on 2026-07-25, and the state machine needs that distinction.
        secs["snapshot_date"] = pd.Timestamp("2026-07-25")
        log.info("securities: %d rows", len(secs))
    results.append(archive.merge("securities", secs))

    archive.record(results)

    print(f"\n{'=' * 66}\nSEEDED FROM data/raw/\n{'=' * 66}")
    print(f"{'dataset':<18}{'added':>9}{'total':>10}{'conflicts':>11}")
    for r in results:
        print(f"{r.dataset:<18}{r.added:>9}{r.total:>10}{len(r.conflicts):>11}")


if __name__ == "__main__":
    main()
