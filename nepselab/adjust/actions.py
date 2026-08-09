"""Corporate actions, reconstructed from data already in the archive.

Plan §3.2 descoped corporate actions at the Phase 0 gate on the grounds that
sourcing them was poor and that they do not move an index-level target. Both
were true. Neither is true any more:

  - Sourcing is no longer needed. NEPSE restates a scrip's previous close on the
    ex-date, so the ratio between its `previousDayClosePrice` and the actual
    prior session's close IS the adjustment factor. It costs nothing to compute
    and it is the exchange's own number, not a third party's.
  - The target changed. §3.2's reasoning was about the INDEX, which the exchange
    already keeps continuous. The scrip panel is per-company, and a bonus issue
    is the largest single thing that happens to a share price.

How large: NABBC on 2025-08-25 shows -40.9% on a raw close-to-close series and
+9.99% -- limit up -- on the exchange's own basis. 171 events over one year,
touching 160 of 454 traded symbols. Any per-scrip price series built by shifting
closes is wrong on those days, in the most alarming possible direction.

WHAT THIS DOES AND DOES NOT TELL YOU
It recovers the DATE and the SIZE of an adjustment. It does not recover the
KIND: a 2:1 bonus, a 100% rights issue at par, and a large special dividend can
imply similar factors, and this cannot tell them apart. Naming the kind needs
the company announcement (see the notices endpoint), so `kind` stays "unknown"
here rather than being guessed. Recording the ratio honestly is worth more than
labelling it confidently.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

# Below this the difference is rounding, tick size, or a stale quote rather than
# a corporate action. Bonus issues in Nepal are rarely under a few percent, and
# the smallest real event in the archive is ~4%.
MIN_FACTOR_MOVE = 0.02


def detect(today_price: pd.DataFrame) -> pd.DataFrame:
    """Find every session where NEPSE restated a scrip's previous close.

    Returns one row per (symbol, ex-date) with the implied factor. `factor` is
    what a pre-event price must be MULTIPLIED by to be comparable with prices
    after it -- so 0.5 means the share count roughly doubled.
    """
    need = {"businessDate", "symbol", "closePrice", "previousDayClosePrice"}
    missing = need - set(today_price.columns)
    if missing:
        raise ValueError(f"corporate action detection needs {sorted(missing)}")

    df = today_price[list(need)].copy()
    df["businessDate"] = pd.to_datetime(df["businessDate"])
    df = df.sort_values(["symbol", "businessDate"])

    # Yesterday's ACTUAL close, from our own archive of that session.
    df["prior_close"] = df.groupby("symbol", observed=True)["closePrice"].shift(1)
    df = df.dropna(subset=["prior_close"])
    df = df[(df["prior_close"] > 0) & (df["previousDayClosePrice"] > 0)]

    df["factor"] = df["previousDayClosePrice"] / df["prior_close"]
    hits = df[(df["factor"] - 1.0).abs() > MIN_FACTOR_MOVE].copy()
    if hits.empty:
        return pd.DataFrame(columns=["symbol", "ex_date", "factor", "prior_close",
                                     "restated_close", "implied_pct", "kind"])

    hits = hits.rename(columns={"businessDate": "ex_date",
                                "previousDayClosePrice": "restated_close"})
    hits["implied_pct"] = (hits["factor"] - 1.0).round(6)
    # Deliberately not guessed -- see the module docstring.
    hits["kind"] = "unknown"
    return (hits[["symbol", "ex_date", "factor", "prior_close", "restated_close",
                  "implied_pct", "kind"]]
            .sort_values(["ex_date", "symbol"])
            .reset_index(drop=True))


def adjust_series(prices: pd.DataFrame, actions: pd.DataFrame,
                  price_col: str = "closePrice") -> pd.DataFrame:
    """Add `adj_close`: a price series comparable across corporate actions.

    Standard back-adjustment. Prices BEFORE an ex-date are scaled by the factor,
    so the most recent price is left untouched and equals what you would pay
    today. That direction matters: adjusting forwards instead would silently
    restate today's quoted price, and someone would eventually compare it with a
    broker screen and conclude the archive was broken.
    """
    out = prices.copy()
    out["businessDate"] = pd.to_datetime(out["businessDate"])
    out["adj_close"] = out[price_col].astype(float)
    if actions.empty:
        return out

    for sym, evs in actions.groupby("symbol", observed=True):
        mask = out["symbol"] == sym
        if not mask.any():
            continue
        # Apply newest-first so each event scales everything before it, and the
        # factors compound the way successive events actually did.
        for ex_date, factor in evs.sort_values("ex_date", ascending=False)[
                ["ex_date", "factor"]].itertuples(index=False):
            before = mask & (out["businessDate"] < pd.Timestamp(ex_date))
            out.loc[before, "adj_close"] *= float(factor)
    return out


def summarise(actions: pd.DataFrame) -> str:
    if actions.empty:
        return "no corporate actions detected"
    biggest = actions.nsmallest(1, "factor").iloc[0]
    return (f"{len(actions)} event(s) across {actions['symbol'].nunique()} symbol(s), "
            f"{actions['ex_date'].min().date()}..{actions['ex_date'].max().date()}; "
            f"largest {biggest['symbol']} x{biggest['factor']:.3f} "
            f"on {pd.Timestamp(biggest['ex_date']).date()}")
