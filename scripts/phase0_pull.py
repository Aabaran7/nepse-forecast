"""Phase 0: pull 1 year of NEPSE index + 5 liquid scrips, write parquet, report quality.

Usage: .venv/bin/python scripts/phase0_pull.py
"""

from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.ingest.nepse_client import NepseClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("phase0")

RAW = Path("data/raw")
SCRIPS = ["NABIL", "NICA", "NRIC", "HDL", "NTC"]


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    end = date.today()
    start = end - timedelta(days=365)
    s, e = start.isoformat(), end.isoformat()
    log.info("window %s -> %s", s, e)

    c = NepseClient()

    secs = c.securities()
    secs.to_parquet(RAW / "securities.parquet")
    log.info("securities: %d rows, cols=%s", len(secs), list(secs.columns)[:8])

    sym_col = next((x for x in ("symbol", "securitySymbol") if x in secs.columns), None)
    known = set(secs[sym_col]) if sym_col else set()
    for t in SCRIPS:
        if known and t not in known:
            log.warning("ticker %s NOT in securities list", t)

    idx = c.index_history(s, e)
    idx.to_parquet(RAW / "index_nepse.parquet")
    log.info("index: %d rows, cols=%s", len(idx), list(idx.columns))

    frames = []
    for t in SCRIPS:
        try:
            df = c.ticker_history(t, s, e)
        except Exception as exc:  # noqa: BLE001
            log.error("ticker %s failed: %s: %s", t, type(exc).__name__, exc)
            continue
        log.info("ticker %s: %d rows", t, len(df))
        if not df.empty:
            frames.append(df)

    if frames:
        scrips = pd.concat(frames, ignore_index=True)
        scrips.to_parquet(RAW / "scrips.parquet")
        log.info("scrips: %d rows total", len(scrips))
    else:
        log.error("no scrip data pulled")

    try:
        disc = c.disclosures()
        disc.to_parquet(RAW / "disclosures.parquet")
        log.info("disclosures: %d rows, cols=%s", len(disc), list(disc.columns)[:10])
    except Exception as exc:  # noqa: BLE001
        log.error("disclosures failed: %s: %s", type(exc).__name__, exc)


if __name__ == "__main__":
    main()
