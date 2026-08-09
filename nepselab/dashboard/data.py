"""Loading layer for the dashboard. Reads local files today, Supabase later.

Every read the UI does goes through here, so the day the serving layer moves to
Supabase, `app.py` does not change -- only the bodies of these functions do.

One piece of real logic lives here rather than in the UI: `index_closes()`
stitches the two index sources together. They do not overlap in span or
provenance (plan §3.5), and picking either alone is wrong in a way that shows up
as a silently broken dashboard:

  data/deep/  MeroLagani, 2016 -> present, but only refreshed by a manual
              phase1c run, so its tail is stale.
  data/archive/  the exchange itself, refreshed every day by the cron, but it
              only reaches back one rolling year.

Use deep alone and the forward log can never resolve, because the target session
is missing from the actuals. Use archive alone and every chart loses nine years
of history. So: archive wins wherever the two overlap, since it is the exchange's
own number, and deep supplies everything before that.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nepselab.ingest import archive

DEEP = Path("data/deep/nepse_index_deep.parquet")
NEWS = Path("data/news")
NEPSE_INDEX_ID = 58


def _latest(s: pd.Series | None) -> pd.Timestamp | None:
    """Newest value in `s`, as a tz-NAIVE timestamp.

    The stores disagree about timezones: exchange dates are naive business
    dates, while news `first_seen_utc` is an ISO string with a +00:00 offset.
    Mixing them makes `today - ts` raise on tz-aware vs tz-naive, so the
    boundary is here -- one place, rather than at every call site.
    """
    if s is None or len(s) == 0:
        return None
    ts = pd.to_datetime(s, errors="coerce", utc=True).max()
    if pd.isna(ts):
        return None
    return ts.tz_convert(None) if ts.tz is not None else ts


def freshness() -> dict[str, pd.Timestamp | None]:
    """Latest date in each store. The first thing to check when a number looks odd."""
    idx = archive.load("indices")
    tp = archive.load("today_price")
    heads = headlines()
    return {
        "archive": _latest(idx["businessDate"] if not idx.empty else None),
        "scrips": _latest(tp["businessDate"] if not tp.empty else None),
        "deep": _latest(pd.read_parquet(DEEP)["date"] if DEEP.exists() else None),
        "news": _latest(heads["first_seen_utc"] if not heads.empty else None),
    }


def index_closes() -> pd.DataFrame:
    """NEPSE index closes, `date` + `close`, from 2016 to the last session.

    Archive takes precedence on overlapping dates: it is the exchange's own
    figure, where deep is a third-party scrape kept only because it reaches
    further back (§3.5).
    """
    frames = []

    if DEEP.exists():
        d = pd.read_parquet(DEEP)[["date", "close"]].copy()
        d["date"] = pd.to_datetime(d["date"])
        d["src"] = "deep"
        frames.append(d)

    idx = archive.load("indices")
    if not idx.empty:
        a = idx[idx["exchangeIndexId"] == NEPSE_INDEX_ID].copy()
        a = a[["businessDate", "closingIndex"]].rename(
            columns={"businessDate": "date", "closingIndex": "close"})
        a["date"] = pd.to_datetime(a["date"])
        a["src"] = "archive"
        frames.append(a)

    if not frames:
        return pd.DataFrame(columns=["date", "close"])

    both = pd.concat(frames, ignore_index=True)
    # "archive" sorts before "deep", so keep="first" after sorting on src is
    # what makes the exchange win a disagreement.
    both = (both.sort_values(["date", "src"])
                .drop_duplicates("date", keep="first")
                .sort_values("date")
                .reset_index(drop=True))
    return both[["date", "close"]]


def scrip_panel() -> pd.DataFrame:
    """The per-scrip panel, with circuit flags when market_params allows."""
    from nepselab.features import scrip

    tp = archive.load("today_price")
    if tp.empty:
        return pd.DataFrame()
    try:
        from nepselab.eval.costs import Params
        params = Params()
    except Exception:  # noqa: BLE001 - a malformed params file must not blank the page
        params = None
    return scrip.build_panel(tp, params=params)


def headlines() -> pd.DataFrame:
    """Scraped headlines with their trading session attached.

    Attribution is applied HERE, at read time, for the reason set out in
    nepselab.ingest.news.attribute: a headline scraped after the close belongs
    to a session that has not happened yet, so the answer changes as the
    calendar grows and must never be frozen into storage.
    """
    from nepselab.ingest import news

    path = NEWS / "headlines.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)

    idx = archive.load("indices")
    sessions = news.trading_sessions(idx) if not idx.empty else []
    if sessions:
        df = news.attribute(df, sessions)

    # Market relevance is derived here for the same reason `session` is: it
    # depends on the listed-securities snapshot, which grows as companies list.
    # Storing it would freeze today's answer against tomorrow's company list,
    # and a headline about a firm that lists next month would stay marked
    # irrelevant forever.
    from nepselab.ingest import relevance
    df = relevance.tag(df, archive.load("securities"))

    return df.sort_values("first_seen_utc", ascending=False).reset_index(drop=True)


def sentiment(version: str | None = None) -> pd.DataFrame:
    """Headline sentiment scores, newest scorer version only unless asked.

    The store keeps every version ever run (§7's habit applied to scoring: a
    prompt change is a new version, not an overwrite). Showing all of them at
    once would double-count headlines scored twice, so the default is the latest.
    """
    path = NEWS / "sentiment.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty or "scorer_version" not in df.columns:
        return df
    if version is None:
        version = sorted(df["scorer_version"].unique())[-1]
    return df[df["scorer_version"] == version].reset_index(drop=True)


def stock_sentiment(sent: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per listed company mentioned in the news.

    This is the cheap half of stock-level sentiment: ~350 rows regardless of how
    many headlines exist, so it stays a fixed cost on the page while the headline
    history grows without bound behind it.
    """
    sent = sentiment() if sent is None else sent
    if sent.empty or "symbol" not in sent.columns:
        return pd.DataFrame()
    linked = sent[sent["symbol"].notna()]
    if linked.empty:
        return pd.DataFrame()

    g = linked.groupby("symbol", observed=True)
    out = pd.DataFrame({
        "mentions": g.size(),
        "score": g["score"].mean().round(3),
        "bullish": g["sentiment"].apply(lambda s: int((s == "bullish").sum())),
        "bearish": g["sentiment"].apply(lambda s: int((s == "bearish").sum())),
        "neutral": g["sentiment"].apply(lambda s: int((s == "neutral").sum())),
        "last_scored": g["scored_utc"].max(),
    }).reset_index()
    return out.sort_values("mentions", ascending=False).reset_index(drop=True)


def forward_log() -> pd.DataFrame:
    """The immutable prediction log, joined to outcomes as they arrive (§7).

    Adds a `kind` column, which the raw log does not have and the dashboard
    cannot do without. §6.6's volatility-target rows are logged at horizon 0
    because they are not forecasts of anything -- they carry today's exposure.
    score_log() has no way to know that: it computes close[i+0]/close[i] - 1,
    gets exactly 0, reads "not up", and scores every one of them WRONG.

    Left alone, the dashboard would show a headline accuracy dragged down by
    rows that never made a directional claim -- understating the model for the
    most embarrassing possible reason. So exposure rows are labelled here and
    excluded from accuracy wherever it is computed.
    """
    from nepselab.forward import log as flog

    actuals = index_closes()
    scored = flog.load_all() if actuals.empty else flog.score_log(actuals)
    if scored.empty:
        return scored

    scored = scored.copy()
    scored["kind"] = ["exposure" if int(h) == 0 else "direction"
                      for h in scored["horizon"]]
    for c in ("actual", "fwd_return", "correct"):
        if c in scored.columns:
            scored.loc[scored["kind"] == "exposure", c] = pd.NA
    return scored


def directional_log(scored: pd.DataFrame | None = None) -> pd.DataFrame:
    """Only the rows that actually predicted a direction. The accuracy input."""
    scored = forward_log() if scored is None else scored
    if scored.empty or "kind" not in scored.columns:
        return scored
    return scored[scored["kind"] == "direction"].reset_index(drop=True)
