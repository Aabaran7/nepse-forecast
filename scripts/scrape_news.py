"""Daily headline scrape. Runs after archive_pull.py, never before it.

Ordering matters and it is not arbitrary. `archive_pull.py` is collecting data
that NEPSE destroys on a rolling window (§3.4); this job collects data that the
news sites keep in their own archives indefinitely. If a run has time for only
one of them, it must be the pull. So this script is designed to be *skippable*:
it exits 0 on a partial scrape and never touches data/archive/.

Session attribution happens here rather than at read time, so the attribution is
written down once, next to the headline, with the calendar that produced it --
instead of being recomputed by every consumer and drifting between them.

Usage:
    .venv/bin/python scripts/scrape_news.py                      # all sources
    .venv/bin/python scripts/scrape_news.py --sources sharesansar --pages 3
    .venv/bin/python scripts/scrape_news.py --dry-run            # parse, store nothing
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.ingest import archive, news  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("scrape_news")

NEWS_ROOT = Path(news.NEWS_DIR)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", nargs="*", default=None, choices=list(news.SOURCES),
                    help="default: every source")
    ap.add_argument("--pages", type=int, default=1,
                    help="listing pages per source; >1 only for backfill")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-robots", action="store_true",
                    help="skip robots.txt checks (don't)")
    args = ap.parse_args()

    fetcher = news.Fetcher(respect_robots=not args.no_robots)
    df, errors = news.scrape(args.sources, pages=args.pages, fetcher=fetcher)

    requested = args.sources or list(news.SOURCES)
    if len(errors) == len(requested):
        log.error("every source failed: %s", errors)
        return 1
    if errors:
        log.warning("%d/%d sources failed, continuing: %s",
                    len(errors), len(requested), errors)

    # Attribution is computed for the log line only -- see below. Most headlines
    # in a normal after-close run legitimately show no session yet.
    sessions = news.trading_sessions()
    if not sessions:
        log.warning("no archived index history; run scripts/archive_pull.py first.")
    preview = news.attribute(df, sessions)
    pending = int(preview["session"].isna().sum()) if "session" in preview else len(df)
    log.info("scraped %d headlines from %d source(s); %d await a session that "
             "has not traded yet", len(df), len(requested) - len(errors), pending)

    if args.dry_run:
        cols = [c for c in ("session", "source", "title") if c in preview.columns]
        print(preview[cols].head(20).to_string(index=False))
        return 0

    # Store the raw observation only. `session` is derived from the trading
    # calendar, which grows -- a headline scraped after Thursday's close belongs
    # to a session that does not exist yet, and would be frozen as null by an
    # append-only write. Consumers call news.attribute() at read time instead.
    res = archive.merge("headlines", df, root=NEWS_ROOT, ignore=["first_seen_utc"])
    archive.record([res], root=NEWS_ROOT)
    log.info("%s", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
