"""Capture the order book. Runs DURING market hours; there is no history to fetch.

This is the only endpoint in the project with no past. `today_price` and the
index histories serve a rolling year, which is bad enough (§3.4). Supply and
demand serve NOW: NEPSE returns empty lists outside the session, so a snapshot
not taken is not merely aged out, it never existed. Measured 2026-08-09 (Sunday,
closed) and 2026-08-10 01:45 NPT (pre-open): both empty.

That makes it the strongest argument in the project for capturing before
analysing. Volume tells you what traded. The book tells you what people WANTED
to trade and at what price -- which is the question a thin market actually
raises, and the one "heavy volume means distribution" tries and fails to answer
from prints alone.

TWO RULES FOLLOW FROM UNRECOVERABILITY

  1. INTRADAY, NOT DAILY. A book is a time series inside the session; one
     snapshot at the close throws away the shape of the day. This runs several
     times while the market is open.

  2. NEVER LOSE A PAYLOAD TO A SCHEMA SURPRISE. The field names are unknown at
     the time of writing -- the endpoint was empty every time it could be
     probed. So the raw JSON is written to disk BEFORE any parsing is attempted,
     and a parse failure downgrades to "raw kept, table not updated" instead of
     an exception that discards the response. Being unable to read it today is
     recoverable; not having it is not.

Usage:
    .venv/bin/python scripts/snapshot_orderbook.py
    .venv/bin/python scripts/snapshot_orderbook.py --force   # ignore market hours
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.ingest import archive  # noqa: E402
from nepselab.ingest.nepse_client import NepseClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("orderbook")

BOOK_DIR = Path("data/orderbook")
RAW_DIR = BOOK_DIR / "_raw"

KTM = timezone(timedelta(hours=5, minutes=45))
# configs/market_params.yaml, effective 2026-04-20. Widened by 15 minutes each
# side so a snapshot straddling the bell is kept rather than refused.
OPEN_NPT = time(10, 45)
CLOSE_NPT = time(15, 15)

# Candidate names for the symbol column, tried in order. The endpoint has never
# returned a populated row here, so this is a guess -- and `store` treats a miss
# as "keep the raw file, skip the table", never as an error.
SYMBOL_FIELDS = ("symbol", "securitySymbol", "securityName", "securityId",
                 "scrip", "stockSymbol")


def market_hours(now: datetime | None = None) -> bool:
    now = (now or datetime.now(timezone.utc)).astimezone(KTM)
    if now.weekday() > 4:          # Mon-Fri since 2026-04-10
        return False
    return OPEN_NPT <= now.time() <= CLOSE_NPT


def save_raw(payload: dict, stamp: str) -> Path:
    """Write the response to disk before anything can go wrong with parsing."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{stamp.replace(':', '')}.json"
    path.write_text(json.dumps(payload))
    return path


def to_frame(payload: dict, stamp: str) -> pd.DataFrame:
    rows = []
    for side in ("supplyList", "demandList"):
        for row in payload.get(side) or []:
            if isinstance(row, dict):
                r = dict(row)
                r["side"] = "supply" if side == "supplyList" else "demand"
                r["captured_utc"] = stamp
                rows.append(r)
    return pd.DataFrame(rows)


def store(df: pd.DataFrame, raw_path: Path) -> bool:
    """Fold into the archive. Returns False if the schema was unrecognisable."""
    sym = next((c for c in SYMBOL_FIELDS if c in df.columns), None)
    if sym is None:
        log.warning(
            "no symbol-like column in %s -- raw payload kept at %s and NOT "
            "discarded. Add the real name to SYMBOL_FIELDS and re-run "
            "scripts/rebuild_orderbook.py to fold it in.",
            sorted(df.columns), raw_path)
        return False

    if sym != "symbol":
        df = df.rename(columns={sym: "symbol"})
    res = archive.merge("orderbook", df, root=BOOK_DIR)
    archive.record([res], root=BOOK_DIR)
    log.info("%s", res)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="capture even outside market hours (to learn the schema)")
    args = ap.parse_args()

    if not args.force and not market_hours():
        now = datetime.now(timezone.utc).astimezone(KTM)
        log.info("market closed (%s NPT); the book is empty outside the session, "
                 "so nothing to capture", now.strftime("%a %H:%M"))
        return 0

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = NepseClient().supply_demand()
    raw_path = save_raw(payload, stamp)

    df = to_frame(payload, stamp)
    if df.empty:
        log.info("book empty at %s -- market likely closed or pre-open. Raw kept "
                 "at %s so the empty response is itself on record.", stamp, raw_path)
        return 0

    log.info("captured %d row(s): %s", len(df),
             df["side"].value_counts().to_dict())
    store(df, raw_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
