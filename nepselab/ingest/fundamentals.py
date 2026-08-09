"""Company fundamentals from MeroLagani. Point-in-time by construction.

NEPSE serves no fundamentals at all -- market cap and nothing else. MeroLagani's
company page carries the ratios: EPS, P/E, book value, PBV, dividend, shares
outstanding, and a ten-year dividend history.

THE THING THAT MAKES FUNDAMENTALS DANGEROUS
A ratio without its reporting date is a look-ahead leak wearing a respectable
name. NABIL's page says EPS 28.36 today; using that figure for a backtest dated
last March means trading on earnings nobody had yet. MeroLagani labels the
period -- "28.36 (FY:082-083, Q:4)" -- so `fiscal_period` is parsed out and
stored beside every value, and `snapshot_date` records when WE saw it. A row
therefore says "as of this date, the latest reported EPS was X, for period P",
which is the only form of this data that can be used honestly.

WHAT CANNOT BE RECOVERED
The page shows current values only. There is no way to ask what it said last
year, so a point-in-time history cannot be reconstructed -- it can only be
accumulated from now on. Same property as the order book and the price archive:
the first scrape is the start of the record, not a snapshot of the past.

WHAT IS NOT HERE
No balance sheet, no cash flow, no debt-to-equity, no non-performing loans, no
capital adequacy. Those live in quarterly report PDFs. Worth knowing that D/E is
the wrong question for most of this market anyway: 70% of NEPSE listings are
banks, development banks, finance and microfinance companies, where leverage is
the business model rather than a risk signal.
"""

from __future__ import annotations

import logging
import re

import pandas as pd
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

BASE = "https://merolagani.com/CompanyDetail.aspx?symbol="

# Page label -> our column. Anything unlisted is ignored rather than guessed at.
FIELDS = {
    "sector": "sector",
    "shares outstanding": "shares_outstanding",
    "market price": "market_price",
    "52 weeks high - low": "week52_high_low",
    "180 day average": "avg_180d",
    "120 day average": "avg_120d",
    "1 year yield": "yield_1y",
    "eps": "eps",
    "p/e ratio": "pe_ratio",
    "book value": "book_value",
    "pbv": "pbv",
    "% dividend": "dividend_pct",
    "market capitalization": "market_cap",
}

NUMERIC = {"shares_outstanding", "market_price", "avg_180d", "avg_120d",
           "yield_1y", "eps", "pe_ratio", "book_value", "pbv", "dividend_pct",
           "market_cap"}

# "28.36 (FY:082-083, Q:4)" -> value, fiscal year, quarter
_PERIOD = re.compile(r"\(\s*FY\s*:\s*([\d\-]+)\s*(?:,\s*Q\s*:\s*(\d+))?\s*\)", re.I)


def url_for(symbol: str) -> str:
    return BASE + str(symbol).upper()


def _number(text: str) -> float | None:
    """First number in a cell. Handles '1,234.56', '5.43%', '-'."""
    if text is None:
        return None
    m = re.search(r"-?[\d,]+\.?\d*", str(text).replace(" ", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def parse(html: str, symbol: str) -> dict:
    """One company's page -> a flat record. Missing fields are None, not zero.

    Zero would be a claim. A bank with no reported EPS this quarter has not
    earned nothing; we simply do not have it, and downstream ranking must be
    able to tell those apart.
    """
    soup = BeautifulSoup(html, "lxml")
    out: dict = {"symbol": str(symbol).upper()}

    for tr in soup.select("table tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if len(cells) < 2:
            continue
        label = cells[0].strip().lower().rstrip(":")
        col = FIELDS.get(label)
        if col is None or col in out:
            continue

        raw = cells[1].strip()
        out[col] = _number(raw) if col in NUMERIC else raw

        # The reporting period travels with the value it qualifies. Without it
        # the number cannot be used for anything dated.
        m = _PERIOD.search(raw)
        if m:
            out[f"{col}_fy"] = m.group(1)
            if m.group(2):
                out[f"{col}_quarter"] = int(m.group(2))

    if "week52_high_low" in out:
        parts = re.findall(r"[\d,]+\.?\d*", out.pop("week52_high_low"))
        if len(parts) == 2:
            out["week52_high"] = float(parts[0].replace(",", ""))
            out["week52_low"] = float(parts[1].replace(",", ""))

    return out


def parse_dividend_history(html: str, symbol: str) -> pd.DataFrame:
    """The ten-year dividend table. Rows look like: '1. | 12.50% | (FY: 081-082)'.

    Kept separate from the flat record because it is the one genuinely
    historical thing on the page -- everything else is a snapshot of today.
    """
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for tr in soup.select("table tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if len(cells) < 3 or not re.fullmatch(r"\d+\.?", cells[0].strip()):
            continue
        pct = _number(cells[1])
        fy = re.search(r"([\d]{3}-[\d]{3})", cells[2])
        if pct is None or not fy:
            continue
        rows.append({"symbol": str(symbol).upper(), "fiscal_year": fy.group(1),
                     "dividend_pct": pct})
    if not rows:
        return pd.DataFrame(columns=["symbol", "fiscal_year", "dividend_pct"])
    # The page lists bonus and cash rows separately under the same fiscal year;
    # keep the larger, which is the one a screen would quote.
    return (pd.DataFrame(rows)
            .sort_values("dividend_pct", ascending=False)
            .drop_duplicates(["symbol", "fiscal_year"], keep="first")
            .sort_values("fiscal_year", ascending=False)
            .reset_index(drop=True))


def sector_medians(df: pd.DataFrame, cols: tuple[str, ...] = ("pe_ratio", "pbv")
                   ) -> pd.DataFrame:
    """Median ratio per sector. The number that makes a ratio mean anything.

    A P/E of 19.6 is not high or low on its own. Against the commercial banking
    median it becomes a statement -- and in this market that statement is often
    the opposite of the naive one: banks trade near 18 while hydropower trades
    near 37, so a "low" 19.6 is slightly EXPENSIVE for a bank.

    NEGATIVE P/E IS EXCLUDED, and that is not a cosmetic choice. A loss-making
    company has no meaningful price-to-earnings ratio; the number it produces is
    negative and sorts as if it were the cheapest thing on the exchange. Left in,
    the raw medians read Hotels -123.7 and Others -16.1, which invites exactly
    the wrong conclusion. They are counted in `n_loss` instead, which is the
    genuinely useful fact about those sectors.

    Median rather than mean throughout: one company at 400x drags an average
    somewhere no company actually trades.
    """
    if df.empty or "sector" not in df.columns:
        return pd.DataFrame()
    have = [c for c in cols if c in df.columns]
    if not have:
        return pd.DataFrame()

    work = df.copy()
    loss = pd.Series(False, index=work.index)
    if "pe_ratio" in work.columns:
        loss = work["pe_ratio"] <= 0
        work.loc[loss, "pe_ratio"] = pd.NA

    g = work.groupby("sector", observed=True)
    out = g[have].median().round(3)
    out["n"] = g.size()
    if loss.any():
        out["n_loss"] = work.assign(_l=loss).groupby("sector", observed=True)["_l"].sum()
    return out.reset_index().sort_values("n", ascending=False)
