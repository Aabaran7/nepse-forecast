"""Daily archival pull. Run this every trading day; missed days are lost forever.

NEPSE serves a rolling ~1 year and nothing older (scripts/phase1_probe_depth.py).
The only way this project ever gets a sample long enough to test anything is to
capture the window before it rolls. This job is therefore not a convenience --
it is the data collection strategy, and it is also the Phase 5 forward-run
ingest, which is now the same job.

Pulls, then folds into the append-only store in data/archive/:
  - all 17 indices (NEPSE, Sensitive, Float, and the 14 sector indices)
  - market summary history (turnover, traded shares, transactions, tradedScrips)
  - per-session today_price snapshots -- the breadth source, ~340 rows/session
  - a dated securities snapshot, for the scrip state machine (plan §3.2)

Idempotent: re-running the same day adds nothing. Safe to run more than once.

Usage:
    .venv/bin/python scripts/archive_pull.py              # incremental (default)
    .venv/bin/python scripts/archive_pull.py --backfill   # sweep the whole window
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.ingest import archive  # noqa: E402
from nepselab.ingest.nepse_client import NepseClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("archive_pull")


def pull_indices(c: NepseClient) -> pd.DataFrame:
    sectors = c.sector_indices()
    frames = []
    for _, row in sectors.iterrows():
        idx_id = int(row["id"])
        try:
            df = c.index_history("2000-01-01", date.today().isoformat(), index_id=idx_id)
        except Exception as exc:  # noqa: BLE001
            log.error("index %s (%s) failed: %s", idx_id, row.get("indexCode"), exc)
            continue
        if df.empty:
            continue
        df["exchangeIndexId"] = idx_id
        df["indexCode"] = row.get("indexCode")
        frames.append(df)
        log.info("  index %-12s id=%-3s %d sessions", row.get("indexCode"), idx_id, len(df))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def sessions_to_pull(c: NepseClient, backfill: bool, fresh_first: int = 2) -> list[str]:
    """Trading dates we want today_price for, ordered by EXPIRY RISK.

    The index history doubles as the trading calendar -- it lists exactly the
    sessions that happened. Incremental mode pulls only sessions not already
    archived; backfill re-sweeps the whole window to close gaps from missed runs.

    Ordering is the part that matters and it used to be wrong. This queue was
    newest-first, which is the correct order for exactly one purpose (get
    today's session) and exactly backwards for the other (close a backlog before
    the window rolls). With 153 sessions queued and NEPSE throttling every run
    down to ~15 fetches, newest-first spends weeks on sessions that are in no
    danger while the oldest ones -- the only ones actually expiring -- sit at the
    back of the queue and fall off.

    So: the newest `fresh_first` sessions go first, because a long backfill must
    never delay today's data, and everything after them is oldest-first, because
    that is the order they expire in.
    """
    idx = c.index_history("2000-01-01", date.today().isoformat())
    if idx.empty:
        return []
    all_sessions = sorted(pd.to_datetime(idx["businessDate"]).dt.normalize().unique())

    want = all_sessions
    if not backfill:
        have_df = archive.load("today_price")
        if not have_df.empty:
            have = set(pd.to_datetime(have_df["businessDate"]).dt.normalize())
            want = [d for d in want if d not in have]

    if fresh_first > 0:
        head, tail = want[-fresh_first:], want[:-fresh_first]
    else:
        head, tail = [], want      # pure oldest-first
    ordered = list(reversed(head)) + tail
    return [pd.Timestamp(d).date().isoformat() for d in ordered]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true",
                    help="re-sweep every session in the window, not just new ones")
    ap.add_argument("--interval", type=float, default=1.6,
                    help="min seconds between calls (0.7 got throttled at ~75 calls)")
    ap.add_argument("--chunk", type=int, default=10, help="commit every N sessions")
    ap.add_argument("--max-consecutive", type=int, default=4,
                    help="consecutive failures before cooling down")
    ap.add_argument("--cooldown", type=int, default=120, help="cool-down seconds")
    ap.add_argument("--give-up", type=int, default=40,
                    help="total failures before ending the run (rest stay queued)")
    args = ap.parse_args()

    c = NepseClient(min_interval=args.interval)
    results = []

    log.info("pulling all indices ...")
    results.append(archive.merge("indices", pull_indices(c)))

    log.info("pulling market summary history ...")
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=400)).isoformat()
    results.append(archive.merge("market_summary", c.market_summary_history(start, end)))

    log.info("pulling securities snapshot ...")
    secs = c.securities()
    if not secs.empty:
        secs["snapshot_date"] = pd.Timestamp(date.today())
    results.append(archive.merge("securities", secs))

    todo = sessions_to_pull(c, args.backfill)
    log.info("today_price: %d session(s) to pull%s",
             len(todo), " (backfill)" if args.backfill else "")

    # NEPSE throttles a sustained sweep: a 2026-08-02 backfill ran ~75 sessions
    # clean, then returned HTTPError for every subsequent date. Those dates were
    # NOT empty -- retried in isolation minutes later they all returned ~340 rows.
    # So a run of failures means "back off", never "no data", and the sweep must
    # not grind through hundreds of doomed 4-deep retries to discover that.
    tp_result = archive.merge("today_price", pd.DataFrame())  # current state
    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    consecutive = 0

    added_total = 0

    def flush() -> None:
        """Commit what we have. Called often: a killed sweep must keep its work."""
        nonlocal frames, tp_result, added_total
        if frames:
            r = archive.merge("today_price", pd.concat(frames, ignore_index=True))
            # merge() reports rows added by THIS call; the summary wants the run.
            added_total += r.added
            tp_result = archive.MergeResult("today_price", added_total, r.total, r.conflicts)
            log.info("  committed -> %d rows archived", r.total)
            frames = []

    for i, d in enumerate(todo, 1):
        try:
            df = c.today_price(d)
            consecutive = 0
        except Exception as exc:  # noqa: BLE001
            failed.append(d)
            consecutive += 1
            log.warning("  today_price %s failed (%s), %d consecutive",
                        d, type(exc).__name__, consecutive)
            if consecutive >= args.max_consecutive:
                # Commit before sleeping, so a kill during the cool-down is free.
                flush()
                log.warning("throttled after %d consecutive failures; cooling down %ds",
                            consecutive, args.cooldown)
                time.sleep(args.cooldown)
                consecutive = 0
                if len(failed) > args.give_up:
                    log.error("giving up this run at %d failures. Remaining sessions "
                              "stay queued -- just run again.", len(failed))
                    break
            continue
        if not df.empty:
            frames.append(df)
        if i % args.chunk == 0:
            flush()
            log.info("  ... %d/%d sessions", i, len(todo))

    flush()
    results.append(tp_result)

    archive.record(results)

    print(f"\n{'=' * 66}\nARCHIVE STATE\n{'=' * 66}")
    print(f"{'dataset':<18}{'added':>9}{'total':>10}{'conflicts':>11}")
    for r in results:
        print(f"{r.dataset:<18}{r.added:>9}{r.total:>10}{len(r.conflicts):>11}")
    if failed:
        print(f"\n{len(failed)} session(s) not retrieved this run -- almost certainly "
              f"throttling,\nnot missing data. They stay queued; run again to pick them up.")
    idx = archive.load("indices")
    if not idx.empty:
        nepse = idx[idx.exchangeIndexId == 58]
        bd = pd.to_datetime(nepse["businessDate"])
        print(f"\nNEPSE index archived: {len(nepse)} sessions, "
              f"{bd.min().date()} .. {bd.max().date()}")
        print("Run this every trading day. A session missed is a session gone.")


if __name__ == "__main__":
    main()
