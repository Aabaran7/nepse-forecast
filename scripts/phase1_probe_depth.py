"""Phase 1 probe: how far back does NEPSE actually serve history?

The plan's power calculation (§2) assumes ~1500 usable daily observations from
2020-06. Phase 0 only pulled one year, so that assumption is untested. This
walks yearly windows backwards and reports what each endpoint really returns,
before we build anything that depends on the answer.

Read-only. Writes a summary table to results/phase1_depth_probe.csv.

Usage: .venv/bin/python scripts/phase1_probe_depth.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.ingest.nepse_client import NEPSE_INDEX_ID, NepseClient  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("probe")

OUT = Path("results")
YEARS = list(range(2012, 2027))


def probe_window(fn, label: str) -> dict:
    """Call fn(), reducing any failure to a row rather than aborting the sweep."""
    try:
        df = fn()
    except Exception as exc:  # noqa: BLE001 - probing, every failure is a datum
        return {"window": label, "rows": 0, "first": None, "last": None,
                "status": f"{type(exc).__name__}: {str(exc)[:80]}"}
    if df is None or len(df) == 0:
        return {"window": label, "rows": 0, "first": None, "last": None, "status": "empty"}
    datecol = next((c for c in ("businessDate", "date") if c in df.columns), None)
    first = last = None
    if datecol:
        first, last = df[datecol].min(), df[datecol].max()
    return {"window": label, "rows": len(df),
            "first": None if first is None else str(pd.Timestamp(first).date()),
            "last": None if last is None else str(pd.Timestamp(last).date()),
            "status": "ok"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c = NepseClient()
    rows = []

    print(f"\n{'=' * 74}\nINDEX HISTORY DEPTH (id={NEPSE_INDEX_ID}), one window per year\n{'=' * 74}")
    print(f"{'window':<24}{'rows':>7}  {'first':<12}{'last':<12}status")
    for y in YEARS:
        s, e = f"{y}-01-01", f"{y}-12-31"
        r = probe_window(lambda s=s, e=e: c.index_history(s, e), f"{y}")
        rows.append({"endpoint": "index_history", **r})
        print(f"{r['window']:<24}{r['rows']:>7}  {str(r['first']):<12}{str(r['last']):<12}{r['status']}")

    print(f"\n{'=' * 74}\nMARKET SUMMARY HISTORY\n{'=' * 74}")
    path = c._api.endpoints["market_summary_history_api"]["api"]
    for y in (2015, 2018, 2020, 2023, 2026):
        s, e = f"{y}-01-01", f"{y}-12-31"

        def call(s=s, e=e):
            raw = c._paged(path, {"startDate": s, "endDate": e}, f"mkt_summary({s})")
            return pd.DataFrame(raw)

        r = probe_window(call, f"{y}")
        rows.append({"endpoint": "market_summary_history", **r})
        print(f"{r['window']:<24}{r['rows']:>7}  {str(r['first']):<12}{str(r['last']):<12}{r['status']}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "phase1_depth_probe.csv", index=False)

    idx = out[(out.endpoint == "index_history") & (out.rows > 0)]
    print(f"\n{'=' * 74}\nVERDICT\n{'=' * 74}")
    if idx.empty:
        print("No index history returned for ANY year. Endpoint shape or auth changed.")
        return

    # Do NOT sum rows across windows. If the server ignores the date params it
    # returns the same rolling window every time, and summing counts one year of
    # data once per probe -- which is exactly the mistake this check exists to
    # catch. Distinct (first, last) spans are the only honest measure of depth.
    spans = idx[["first", "last"]].drop_duplicates()
    if len(spans) == 1:
        first, last = spans.iloc[0]["first"], spans.iloc[0]["last"]
        print("DATE PARAMETERS ARE IGNORED. Every probed year returned the same")
        print(f"rolling window: {first} .. {last} ({int(idx.rows.iloc[0])} sessions).")
        print("The API serves ~1 year of history and no more, silently -- it does")
        print("not 400 on an out-of-range request, it just hands back the window.")
        print("\nConsequences:")
        print("  - The plan's ~1500-observation sample is NOT obtainable from this API.")
        print("  - The 2020-06..2021-12 regime split has no data and cannot be run.")
        print("  - The window ROLLS: sessions older than the span are already")
        print("    unrecoverable from NEPSE. Archiving daily from now on is the only")
        print("    way this project ever accumulates a longer sample.")
    else:
        print(f"distinct spans returned : {len(spans)}  (date params appear to work)")
        print(f"earliest date seen      : {idx['first'].min()}")
        print(f"latest date seen        : {idx['last'].max()}")
    print(f"\nwrote {OUT / 'phase1_depth_probe.csv'}")


if __name__ == "__main__":
    main()
