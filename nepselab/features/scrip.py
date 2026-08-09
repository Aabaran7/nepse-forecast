"""Per-scrip daily facts, for the dashboard. DESCRIPTIVE ONLY.

Read this before adding anything: **nothing here looks forward, and nothing here
is a signal.** Every column describes the session that just closed, from data
available at its close. There is no forward return, no label, no ranking by
"what to buy". That is a deliberate boundary, not an oversight.

The reason is §6. Three decision rules have already been tested and abandoned,
and the plan's conclusion is that buy-and-hold is the tradeable answer. A screen
that sorts scrips by an untested pattern would quietly reverse that finding by
implication -- and it would do so on a panel that has never been through the
§6.1 power analysis. So this module renders facts, and any predictive claim goes
through a pre-registered test first.

This is NOT a FeatureModule (features/base.py). Those return one row per
session for the index-level model, and the assemble() contract checks exactly
that. This is a PANEL: one row per (session, scrip). Forcing it into that
protocol would either flatten away the scrip dimension or break assemble()'s
row-count invariant, so it stays separate.

--- what the numbers mean, and where they mislead --------------------------

VOLUME IS NOT DIRECTION. The popular reading -- heavy volume plus a falling
price means large holders are distributing -- does not transfer to NEPSE
unmodified, and the archive says so. Measured over 228 sessions and 475 scrips
(2025-08 .. 2026-08): scrips closing at their limit UP traded ~3.8x their own
20-day median volume, and there were 364 such days. Scrips closing at their
limit DOWN traded ~2.3x, and there were only 70. Heavy volume in this market
accompanies buying far more often than selling.

(Counting scrip-days that have a 20-session volume baseline, so a newly listed
scrip cannot register a spike on its third day of trading.)

The mechanism is structural. A daily circuit truncates the move, there is no
short selling, and a scrip that nobody wants simply locks down on a thin book.
The domestic shape of forced selling is therefore LOW volume at the limit, not
high volume -- close to the opposite of the imported heuristic.

THIN SCRIPS ARE NOISE, NOT INFORMATION. 15% of scrip-days close on fewer than 10
transactions, and 19% of 3x volume spikes are backed by fewer than 10 trades. On
those days "volume" is one or two participants, and no aggregate can tell you
whether they were accumulating or exiting. `is_thin` exists so the dashboard can
mark them rather than rank them.

Verified by scripts/phase9_scrip_panel.py; figures restated in plan §5.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Below this many transactions, a day's move is one or two participants and the
# volume figure carries no information about who they were. Not tuned -- it is
# the round number nearest the 15th percentile of the panel, chosen before
# looking at any outcome, and it is a DISPLAY threshold, not a model parameter.
THIN_TRADES = 10

# A "spike" for display purposes. Same status: a labelling cut, not a signal.
SPIKE_RATIO = 2.0

VOL_WINDOW = 20   # sessions in the volume baseline
# Median, not mean: one block trade otherwise redefines "normal" for a month.
MIN_VOL_PERIODS = 10


def _require(df: pd.DataFrame, cols: tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"scrip panel needs column(s) {missing}. Expected an archive "
            f"`today_price` frame (nepselab.ingest.archive.load('today_price')).")


def build_panel(today_price: pd.DataFrame, params=None) -> pd.DataFrame:
    """One row per (session, scrip), with the day's facts. No forward data.

    `params` is a costs.Params for the date-aware circuit limit. Passing None
    skips the circuit columns rather than assuming a limit -- the limit changed
    from 10% to 15% on 2026-04-20, and a hardcoded value is wrong on one side of
    that date and silently mislabels every limit day there.
    """
    _require(today_price, ("businessDate", "symbol", "closePrice",
                           "previousDayClosePrice", "totalTradedQuantity",
                           "totalTrades"))

    df = today_price.copy()
    df["businessDate"] = pd.to_datetime(df["businessDate"])
    df = df.sort_values(["symbol", "businessDate"]).reset_index(drop=True)

    prev = df["previousDayClosePrice"].replace(0, np.nan)
    df["ret"] = df["closePrice"] / prev - 1.0

    # Baseline is the scrip's OWN recent median, so a big illiquid name and a
    # small liquid one are comparable. min_periods keeps a newly listed scrip
    # from getting a "spike" on its third day of trading.
    g = df.groupby("symbol", group_keys=False, observed=True)
    df["vol_median_20"] = g["totalTradedQuantity"].transform(
        lambda s: s.rolling(VOL_WINDOW, min_periods=MIN_VOL_PERIODS).median())
    base = df["vol_median_20"].replace(0, np.nan)
    df["vol_ratio"] = df["totalTradedQuantity"] / base

    df["is_thin"] = df["totalTrades"] < THIN_TRADES
    df["avg_trade_size"] = (df["totalTradedQuantity"]
                            / df["totalTrades"].replace(0, np.nan))

    df["quadrant"] = _quadrant(df["ret"], df["vol_ratio"])

    if "fiftyTwoWeekHigh" in df and "fiftyTwoWeekLow" in df:
        span = df["fiftyTwoWeekHigh"] - df["fiftyTwoWeekLow"]
        df["pct_of_52w_range"] = np.where(
            span > 0, (df["closePrice"] - df["fiftyTwoWeekLow"]) / span, np.nan)

    if params is not None:
        df = _add_circuit(df, params)

    return df


def _quadrant(ret: pd.Series, vol_ratio: pd.Series) -> pd.Series:
    """Label the day by price direction and volume, as description only.

    The names are deliberately neutral -- "heavy_down", not "distribution". The
    module docstring explains why the loaded reading is wrong here; using the
    loaded word in the data would smuggle it back in through the column values,
    where it would end up on the dashboard as a verdict.
    """
    heavy = vol_ratio >= SPIKE_RATIO
    up = ret > 0
    flat = ret.isna() | (ret == 0)

    out = pd.Series(
        np.select(
            [flat,
             heavy & up, heavy & ~up,
             ~heavy & up, ~heavy & ~up],
            ["flat", "heavy_up", "heavy_down", "quiet_up", "quiet_down"],
            default="flat"),
        index=ret.index, dtype="object")
    return out


def _add_circuit(df: pd.DataFrame, params) -> pd.DataFrame:
    """Flag limit-up / limit-down using the limit in force ON THAT DATE."""
    limits = {}
    for d in df["businessDate"].unique():
        try:
            limits[d] = params.scrip_circuit(pd.Timestamp(d))
        except Exception as exc:  # noqa: BLE001 - MissingConstant and friends
            log.warning("no circuit limit for %s (%s); leaving unflagged",
                        pd.Timestamp(d).date(), exc)
            limits[d] = np.nan

    lim = df["businessDate"].map(limits).astype(float)
    # Tolerance because the exchange rounds to the tick, so a limit close lands
    # a few basis points either side of the nominal percentage. costs.py makes
    # the same allowance for the index.
    tol = 0.005
    df["circuit_limit"] = lim
    df["at_limit_up"] = df["ret"] >= (lim - tol)
    df["at_limit_down"] = df["ret"] <= -(lim - tol)
    return df


# --- session-level rollups -------------------------------------------------

def market_breadth(panel: pd.DataFrame) -> pd.DataFrame:
    """Advancers / decliners / unchanged per session, plus limit counts.

    Counts TRADED scrips only. A scrip with no transactions did not decline; it
    did not participate, and folding it into the denominator makes breadth drift
    with the listing count instead of with the market.
    """
    t = panel[panel["totalTradedQuantity"] > 0]
    g = t.groupby("businessDate", observed=True)
    out = pd.DataFrame({
        "traded": g.size(),
        "advancers": g["ret"].apply(lambda s: int((s > 0).sum())),
        "decliners": g["ret"].apply(lambda s: int((s < 0).sum())),
        "unchanged": g["ret"].apply(lambda s: int((s == 0).sum())),
        "thin": g["is_thin"].sum(),
    })
    if "at_limit_up" in t.columns:
        out["limit_up"] = g["at_limit_up"].sum()
        out["limit_down"] = g["at_limit_down"].sum()
    out["advance_decline_ratio"] = out["advancers"] / out["decliners"].replace(0, np.nan)
    return out.reset_index()


def turnover_concentration(panel: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Share of each session's traded VALUE taken by its largest `top_n` scrips.

    Worth showing because a rising index carried by ten names is a different
    market from the same index carried by two hundred, and the headline number
    cannot tell them apart.
    """
    if "totalTradedValue" not in panel.columns:
        raise ValueError("turnover_concentration needs `totalTradedValue`")
    rows = []
    for date, grp in panel.groupby("businessDate", observed=True):
        total = grp["totalTradedValue"].sum()
        top = grp.nlargest(top_n, "totalTradedValue")
        rows.append({
            "businessDate": date,
            "total_turnover": total,
            f"top{top_n}_share": (top["totalTradedValue"].sum() / total
                                  if total > 0 else np.nan),
            f"top{top_n}_symbols": ", ".join(top["symbol"].astype(str)),
        })
    return pd.DataFrame(rows)


def latest_session(panel: pd.DataFrame) -> pd.DataFrame:
    """The most recent session's rows, sorted by turnover. The screener's input."""
    if panel.empty:
        return panel
    last = panel["businessDate"].max()
    out = panel[panel["businessDate"] == last]
    sort_col = "totalTradedValue" if "totalTradedValue" in out.columns \
        else "totalTradedQuantity"
    return out.sort_values(sort_col, ascending=False).reset_index(drop=True)
