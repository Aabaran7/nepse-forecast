"""Phase 0 quality report: sanity-check the pulled year against the plan's §3 claims.

Checks the trading calendar, OHLC consistency, return outliers (corporate-action
candidates), zero-volume days, and scrip-level circuit-limit hits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
pd.set_option("display.width", 120)

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def hr(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def main() -> None:
    idx = pd.read_parquet(RAW / "index_nepse.parquet")
    scrips = pd.read_parquet(RAW / "scrips.parquet")

    idx["businessDate"] = pd.to_datetime(idx["businessDate"])
    scrips["businessDate"] = pd.to_datetime(scrips["businessDate"])

    hr("1. INDEX COVERAGE & TRADING CALENDAR")
    print(f"rows={len(idx)}  {idx.businessDate.min().date()} -> {idx.businessDate.max().date()}")
    dow = idx.businessDate.dt.dayofweek.value_counts().sort_index()
    print("sessions by weekday:")
    for d, n in dow.items():
        print(f"  {WEEKDAYS[d]:<4} {n:>4}")
    print(f"duplicate dates: {idx.businessDate.duplicated().sum()}")
    gaps = idx.businessDate.diff().dt.days.dropna()
    print(f"gap>4 calendar days (unscheduled closures): {(gaps > 4).sum()}")
    if (gaps > 4).any():
        big = idx.loc[gaps[gaps > 4].index, "businessDate"]
        print("  longest gaps end on:", ", ".join(str(d.date()) for d in big.head(5)))

    hr("2. INDEX OHLC CONSISTENCY")
    bad_hi = (idx.highIndex < idx[["openIndex", "closingIndex", "lowIndex"]].max(axis=1)).sum()
    bad_lo = (idx.lowIndex > idx[["openIndex", "closingIndex", "highIndex"]].min(axis=1)).sum()
    print(f"high < max(o,c,l): {bad_hi}    low > min(o,c,h): {bad_lo}")
    print(f"non-positive closes: {(idx.closingIndex <= 0).sum()}")
    print(f"zero/na turnover days: {int((idx.turnoverValue.fillna(0) <= 0).sum())}")

    idx = idx.sort_values("businessDate")
    idx["ret"] = idx.closingIndex.pct_change()
    print(f"\nindex daily return: sd={idx.ret.std():.4%}  min={idx.ret.min():.2%}  max={idx.ret.max():.2%}")
    print(f"|return| > 4%: {(idx.ret.abs() > 0.04).sum()} days")
    print(f"up days: {(idx.ret > 0).sum()} / {idx.ret.notna().sum()} "
          f"({(idx.ret > 0).sum() / idx.ret.notna().sum():.1%}) <- majority-class baseline")

    hr("3. SCRIP COVERAGE")
    per = scrips.groupby("symbol").agg(
        rows=("businessDate", "size"),
        start=("businessDate", "min"),
        end=("businessDate", "max"),
    )
    per["missing_vs_index"] = len(idx) - per["rows"]
    print(per.to_string())

    hr("4. SCRIP RETURN OUTLIERS (corporate-action candidates)")
    cols = {c.lower(): c for c in scrips.columns}
    close_col = next((cols[k] for k in ("closeprice", "close", "lasttradedprice") if k in cols), None)
    if close_col is None:
        print("no close column found; columns =", list(scrips.columns))
        return
    print(f"using close column: {close_col}")
    s = scrips.sort_values(["symbol", "businessDate"]).copy()
    s["ret"] = s.groupby("symbol")[close_col].pct_change()
    out = s[s.ret.abs() > 0.10]
    print(f"\n|return| > 10% (beyond the daily circuit): {len(out)} rows")
    if len(out):
        print(out[["businessDate", "symbol", close_col, "ret"]].to_string(index=False))
        print("\n^ these should each reconcile to a bonus/rights/split, or the")
        print("  adjustment layer is wrong. Unexplained ones are the red flag.")

    hr("5. CIRCUIT-LIMIT PROXIMITY (phantom-alpha check)")
    near = s[s.ret.abs() > 0.09].groupby("symbol").size()
    print("days with |return| > 9% (at/near the ±10% scrip limit):")
    print(near.to_string() if len(near) else "  none in this window")
    flat = s[(s.ret.abs() > 0.09)]
    print(f"\ntotal near-limit scrip-days: {len(flat)} of {len(s)} ({len(flat)/len(s):.2%})")
    print("If a signal concentrates on these days, fills are not achievable.")

    hr("6. ZERO-VOLUME / SUSPENSION DAYS")
    vol_col = next((cols[k] for k in ("totaltradedquantity", "totalvolume", "volume") if k in cols), None)
    if vol_col:
        z = s[s[vol_col].fillna(0) <= 0]
        print(f"using volume column: {vol_col}")
        print(f"zero-volume scrip-days: {len(z)}")
        if len(z):
            print(z.groupby("symbol").size().to_string())
    else:
        print("no volume column found; columns =", list(scrips.columns))


if __name__ == "__main__":
    main()
