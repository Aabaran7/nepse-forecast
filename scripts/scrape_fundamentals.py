"""Company fundamentals from MeroLagani. Weekly is plenty; daily is waste.

NEPSE serves no fundamentals, so these come from MeroLagani's company pages --
one fetch per company, ~280 of them, a few minutes at a polite rate.

WHY NOT DAILY
Reported figures change when a company files, which is quarterly. Re-fetching
280 pages every day would produce 279 identical rows out of 280 and hammer a
site that is doing us a favour by being scrapeable. The daily job does not call
this; run it weekly, or after results season.

WHAT IT SCRAPES
Active equities only. Debentures, mutual funds and preference shares have no
meaningful EPS or book value, and delisted companies have no meaningful
anything -- 211 of 645 listings carry status D.

POINT-IN-TIME
Each row records `snapshot_date` (when we looked) alongside the reporting period
MeroLagani labels the value with (`eps_fy`, `eps_quarter`). A row therefore
says "as of this date, the latest reported EPS was X for period P". The page
shows only current values, so past point-in-time state cannot be recovered --
this store starts today and accumulates. Same shape as the price archive: the
first run is the beginning of the record, not a view of the past.

Usage:
    .venv/bin/python scripts/scrape_fundamentals.py --limit 5   # try it
    .venv/bin/python scripts/scrape_fundamentals.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.ingest import archive, fundamentals, news  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("fundamentals")

FUND_DIR = Path("data/fundamentals")


def targets(limit: int | None, only: list[str] | None) -> list[str]:
    if only:
        return [s.upper() for s in only]

    ci = archive.load("company_info")
    if ci.empty:
        raise SystemExit("no company_info; run scripts/build_actions.py --sectors")

    eq = ci[(ci["instrumentType"].astype(str).str.strip() == "Equity")]
    if "status" in eq.columns:
        # 'D' is delisted. Scraping those spends requests on companies that no
        # longer trade and pollutes every sector median with stale ratios.
        eq = eq[eq["status"].astype(str).str.upper() == "A"]

    syms = sorted(eq["symbol"].dropna().astype(str).unique())

    # Prefer names that actually traded recently: a listing with no prints is
    # not somewhere a ratio can be acted on anyway.
    tp = archive.load("today_price")
    if not tp.empty:
        tp["businessDate"] = pd.to_datetime(tp["businessDate"])
        recent = set(tp[tp["businessDate"] >= tp["businessDate"].max()
                        - pd.Timedelta(days=30)]["symbol"].astype(str))
        syms = [s for s in syms if s in recent] + [s for s in syms if s not in recent]

    return syms[:limit] if limit else syms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--interval", type=float, default=1.5,
                    help="seconds between fetches; be polite")
    args = ap.parse_args()

    syms = targets(args.limit, args.symbols)
    log.info("fetching %d company page(s)", len(syms))

    fetcher = news.Fetcher(min_interval=args.interval)
    snapshot = pd.Timestamp(date.today())
    records, divs, failed = [], [], []

    for i, sym in enumerate(syms, 1):
        try:
            html = fetcher.get(fundamentals.url_for(sym))
        except Exception as exc:  # noqa: BLE001
            failed.append(sym)
            log.warning("  %s failed: %s", sym, type(exc).__name__)
            continue

        rec = fundamentals.parse(html, sym)
        # A page that yields only the symbol parsed to nothing -- a delisting
        # placeholder, or a layout change. Storing it would write a row of nulls
        # over a company that has real numbers on file.
        if len(rec) <= 2:
            failed.append(sym)
            log.warning("  %s: page had no recognisable fields", sym)
            continue
        rec["snapshot_date"] = snapshot
        records.append(rec)

        d = fundamentals.parse_dividend_history(html, sym)
        if not d.empty:
            divs.append(d)

        if i % 25 == 0:
            log.info("  ... %d/%d", i, len(syms))

    if not records:
        log.error("nothing scraped")
        return 1

    res = [archive.merge("fundamentals", pd.DataFrame(records), root=FUND_DIR)]
    if divs:
        res.append(archive.merge("dividend_history", pd.concat(divs, ignore_index=True),
                                 root=FUND_DIR))
    archive.record(res, root=FUND_DIR)

    df = pd.DataFrame(records)
    print(f"\n{'=' * 70}\nFUNDAMENTALS — {len(df)} companies, {snapshot.date()}\n{'=' * 70}")
    for r in res:
        print(f"  {r.dataset:<20} +{r.added:<6} {r.total:>6} total")
    if failed:
        print(f"\n{len(failed)} failed: {', '.join(failed[:12])}"
              + (" ..." if len(failed) > 12 else ""))

    med = fundamentals.sector_medians(df)
    if not med.empty:
        print("\nSector medians — a P/E means nothing until compared to these:")
        print(med.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
