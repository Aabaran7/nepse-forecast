"""Third-party deep NEPSE index history, and the reconciliation that gates it.

NEPSE's own API serves a rolling ~1 year (plan §3.4), which left the project with
225 sessions and no way to test anything (§2 power table). §8.1 option D asks
whether a usable daily index series exists *somewhere else*. It does -- from more
than one place -- and that raises the question this module exists to answer:
which of them, over which dates, is trustworthy enough to model on?

Kept deliberately separate from `nepselab.ingest.archive`. The archive holds
exchange-sourced rows and is the only irreplaceable asset here; a scraped
third-party series has a different provenance class and must not be able to
contaminate it. Deep history lands in `data/deep/` instead.

Two independent sources:

  MEROLAGANI  a TradingView-style UDF chart endpoint. Full OHLC, 1997-07-20 on.
  GITHUB      a git-LFS CSV dump. Same nominal span, different scrape.

Neither can be verified before 2025-07 against anything authoritative, because
that is exactly the data NEPSE no longer serves. What *can* be done is:

  1. score each source against the 231 archived exchange sessions, and
  2. score the two sources against each other across the whole span.

(2) is the load-bearing test. Where two independent scrapes disagree about the
sign of a daily return, at least one is wrong, and we have no way to tell which.
Since the sign of the return IS this project's target, that disagreement rate is
a direct floor on how small an edge the data could ever support.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

DEEP = Path("data/deep")

_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

MEROLAGANI_URL = "https://merolagani.com/handlers/TechnicalChartHandler.ashx"
GITHUB_URL = ("https://media.githubusercontent.com/media/menaceXnadin/"
              "nepse-historical-market-data-csv/HEAD/data/index/unadjusted/"
              "NEPSE_unadjusted.csv")

# Columns every source is normalised to. `turnover` is market-wide turnover --
# see the note in fetch_github() about what these feeds actually put in `volume`.
COLUMNS = ["date", "open", "high", "low", "close", "turnover"]


# (connect, read). Split deliberately: githubusercontent publishes AAAA records,
# and on a host whose IPv6 route blackholes them the connect sits in SYN-SENT
# rather than failing, so a single scalar timeout reads as "the script hung".
# A short connect budget forces the fallback to IPv4 quickly; the read budget
# stays generous because the LFS blob is a few hundred KB over a slow hop.
_TIMEOUT = (15, 120)


def _get(url: str, **kw) -> requests.Response:
    r = requests.get(url, headers={"User-Agent": _UA, **kw.pop("headers", {})},
                     timeout=_TIMEOUT, **kw)
    r.raise_for_status()
    return r


def fetch_merolagani(start: str = "1990-01-01", end: str | None = None) -> pd.DataFrame:
    """MeroLagani's chart handler. Returns parallel arrays, UDF style.

    Timestamps carry a time-of-day that CHANGED mid-series -- 05:45 UTC for the
    older bars, 20:45 UTC for the newer ones. Normalising them in Kathmandu local
    time therefore pushes every recent bar onto the following day and silently
    misaligns the whole modern half of the series against the archive. They are
    UTC-dated; normalise in UTC and every one of the 231 archived sessions lines
    up exactly. This is the one trap in this endpoint.

    `end` defaults to today and should stay near it. A far-future bound (year
    2100) does not clamp -- it redirects to an ASP.NET error page with HTTP 200,
    so the failure arrives as a JSON parse error rather than anything readable.
    """
    end_ts = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC")
    params = {
        "type": "get_advanced_chart",
        "symbol": "NEPSE",
        "resolution": "1D",
        "rangeStartDate": int(pd.Timestamp(start, tz="UTC").timestamp()),
        "rangeEndDate": int(end_ts.timestamp()),
        "isAdjust": 1,
        "currencyCode": "NPR",
    }
    r = _get(MEROLAGANI_URL, params=params,
             headers={"Referer": "https://merolagani.com/"})
    try:
        d = json.loads(r.text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"merolagani did not return JSON (landed on {r.url}); "
            f"body starts {r.text[:80]!r}") from exc
    if d.get("s") not in (None, "ok"):
        raise RuntimeError(f"merolagani returned status {d.get('s')!r}")

    df = pd.DataFrame({
        "date": pd.to_datetime(d["t"], unit="s", utc=True).normalize().tz_localize(None),
        "open": d["o"], "high": d["h"], "low": d["l"], "close": d["c"],
        "turnover": d["v"],
    })
    return _tidy(df, "merolagani")


def fetch_github() -> pd.DataFrame:
    """Community CSV dump, served through git-LFS.

    `raw.githubusercontent.com` hands back the 131-byte LFS pointer rather than
    the file; `media.githubusercontent.com/media/...` serves the real content.

    Its `volume` column is not index turnover -- it is MARKET-WIDE turnover, and
    it reconciles to `market_summary.totalTurnover` almost exactly (213/224
    sessions within 0.1%) while disagreeing with `indices.turnoverValue` by a
    median 1.7%. Both deep sources carry the same quantity. Naming it `turnover`
    here rather than `volume` is the whole point: read as index turnover it looks
    like a badly broken feed, and it is nothing of the kind.
    """
    csv = _get(GITHUB_URL).content
    df = pd.read_csv(io.BytesIO(csv), parse_dates=["time"])
    df = df.rename(columns={"time": "date", "volume": "turnover"})
    return _tidy(df[COLUMNS], "github")


def _tidy(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Sort, de-duplicate, and tag. Both feeds carry a handful of repeated dates."""
    df = df[COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    dupes = int(df["date"].duplicated().sum())
    if dupes:
        log.info("%s: dropping %d duplicate date(s)", source, dupes)
    df = (df.drop_duplicates(subset="date", keep="first")
            .sort_values("date").reset_index(drop=True))
    df["source"] = source
    return df


def archive_index() -> pd.DataFrame:
    """The NEPSE index as the exchange gave it to us. The only ground truth."""
    from nepselab.ingest import archive

    idx = archive.load("indices")
    if idx.empty:
        return pd.DataFrame(columns=COLUMNS)
    ne = idx[idx["exchangeIndexId"] == 58].copy()
    ne["date"] = pd.to_datetime(ne["businessDate"]).dt.normalize()
    return (ne.rename(columns={"openIndex": "open", "highIndex": "high",
                               "lowIndex": "low", "closingIndex": "close",
                               "turnoverValue": "index_turnover"})
              [["date", "open", "high", "low", "close", "index_turnover"]]
              .sort_values("date").reset_index(drop=True))


@dataclass
class Agreement:
    """How two daily series line up. `sign_disagree` is the one that matters."""
    n_common: int
    only_a: list[pd.Timestamp]
    only_b: list[pd.Timestamp]
    exact: dict[str, int]
    max_abs: dict[str, float]
    sign_disagree: int

    @property
    def sign_disagree_pct(self) -> float:
        return 100.0 * self.sign_disagree / max(self.n_common - 1, 1)


def compare(a: pd.DataFrame, b: pd.DataFrame,
            cols: tuple[str, ...] = ("open", "high", "low", "close")) -> Agreement:
    m = a.merge(b, on="date", how="outer", suffixes=("_a", "_b"), indicator=True)
    m = m.sort_values("date")
    both = m[m["_merge"] == "both"]

    exact, max_abs = {}, {}
    for c in cols:
        if f"{c}_a" not in both or f"{c}_b" not in both:
            continue
        d = (both[f"{c}_a"] - both[f"{c}_b"]).abs()
        exact[c] = int((d < 1e-6).sum())
        max_abs[c] = float(d.max()) if len(d) else 0.0

    # Compare the sign of the close-to-close return, computed within the shared
    # dates only. A source can be right about every level and still disagree
    # about direction if it is missing a session in between.
    sa = (both["close_a"].diff() > 0)
    sb = (both["close_b"].diff() > 0)
    sign_disagree = int((sa != sb).iloc[1:].sum())

    return Agreement(
        n_common=len(both),
        only_a=sorted(m.loc[m["_merge"] == "left_only", "date"].tolist()),
        only_b=sorted(m.loc[m["_merge"] == "right_only", "date"].tolist()),
        exact=exact, max_abs=max_abs, sign_disagree=sign_disagree,
    )


def yearly_disagreement(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Per-year source disagreement -- this is what picks the usable start date."""
    m = (a[["date", "close"]].merge(b[["date", "close"]], on="date",
                                    how="outer", suffixes=("_a", "_b"),
                                    indicator=True)
           .sort_values("date").reset_index(drop=True))
    m["year"] = m["date"].dt.year
    both = m[m["_merge"] == "both"].copy()
    both["rel"] = (both["close_a"] - both["close_b"]).abs() / both["close_b"].abs()
    both["sign_disagree"] = ((both["close_a"].diff() > 0)
                             != (both["close_b"].diff() > 0))
    both.iloc[0, both.columns.get_loc("sign_disagree")] = False

    g = both.groupby("year").agg(
        n=("close_a", "size"),
        close_mismatch=("rel", lambda s: int((s > 1e-6).sum())),
        rel_gt_10bp=("rel", lambda s: int((s > 1e-3).sum())),
        max_rel=("rel", "max"),
        sign_disagree=("sign_disagree", "sum"),
    )
    cal = (m.groupby(["year", "_merge"], observed=True).size()
             .unstack(fill_value=0)
             .rename(columns={"left_only": "only_a", "right_only": "only_b"}))
    for c in ("only_a", "only_b"):
        g[c] = cal[c] if c in cal else 0
    g["sign_disagree_pct"] = 100 * g["sign_disagree"] / g["n"]
    return g


def usable_start(yearly: pd.DataFrame, max_pct: float = 0.5) -> int | None:
    """Earliest year from which EVERY later year stays under `max_pct` disagreement.

    Deliberately not "years that individually pass": a clean 2007 sitting inside
    a decade of dirty ones buys nothing, since the series has to be contiguous to
    be modelled on. The rule is a suffix condition.
    """
    years = sorted(yearly.index)
    ok = None
    for y in reversed(years):
        if yearly.loc[y, "sign_disagree_pct"] <= max_pct:
            ok = y
        else:
            break
    return ok


def turnover_scale_breaks(df: pd.DataFrame, factor: float = 20.0,
                          min_sessions: int = 10) -> pd.Series:
    """Month boundaries where median turnover jumps by more than `factor`.

    The close series is continuous across 2016->2017, but turnover is NOT: the
    2016 values sit around 1e6 and the 2017 ones around 5e8, a ~400x step with
    no matching move in the index. It is a change of units (2016 also carries
    literal zeros), not a change in market activity.

    This matters more than it looks. Every turnover feature in §5 is a ratio or
    z-score against a trailing window, so a feature computed across the break
    reads a 400x surge that never happened -- in a year otherwise perfectly good
    for price features.

    Returns ALL breaks rather than one, because they are not all the same kind
    of thing and the caller has to tell them apart: the 2017 step is spurious,
    while the 2020-05 step is NEPSE genuinely reopening into a boom after a
    two-month COVID closure. Only the units break should move a feature's start
    date. Months with fewer than `min_sessions` sessions are dropped first --
    those same closures otherwise manufacture ratios out of two-day months.
    """
    s = df.set_index("date")["turnover"].replace(0, pd.NA).dropna()
    monthly = s.resample("MS").agg(["median", "size"])
    monthly = monthly[monthly["size"] >= min_sessions]["median"]
    if len(monthly) < 3:
        return pd.Series(dtype=float)
    ratio = monthly / monthly.shift(1)
    return ratio[(ratio > factor) | (ratio < 1 / factor)].dropna()


def save(df: pd.DataFrame, name: str, root: Path = DEEP) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path
