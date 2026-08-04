"""r/NepalStock attention series (plan §5, module 2).

The EDA is done and §5 says not to redo it. What it concluded, and what this
module implements: Reddit is usable as an **index-level attention series only**.
Ticker-level cross-section is dead (only 15 tickers clear 500 mentions ever).

Two hazards, both of which produce a working-looking feature that is wrong.

**The subreddit grew, so raw counts trend.** §5 is blunt: a raw daily comment
count encodes "what year is it", and any model will happily use that as a proxy
for the market's own trend. Every feature this module emits is therefore either
a share of the sub's own activity or a z-score against a trailing window --
never a level.

**Reddit's clock is not NEPSE's clock.** `created_utc` is UTC; NEPSE closes at
15:00 Kathmandu (UTC+5:45), i.e. 09:15 UTC. Attention for session t must
include only what was posted before that session's close, or the feature knows
how the day went. Weekend and holiday activity is carried forward to the next
session rather than dropped -- a closure does not stop people posting, and
throwing it away discards the highest-attention moments of a panic.

Parsing 131k JSON lines takes ~10s, so the daily aggregate is cached to parquet
and only rebuilt when the source files are newer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

DUMP_DIR = Path.home() / "Desktop"
POSTS = DUMP_DIR / "r_nepalstock_posts.jsonl"
COMMENTS = DUMP_DIR / "r_nepalstock_comments.jsonl"
CACHE = Path("data/deep/reddit_daily.parquet")

# NEPSE's close in UTC. 15:00 Asia/Kathmandu = 09:15 UTC.
CLOSE_UTC_HOUR = 9
CLOSE_UTC_MINUTE = 15


def _read_jsonl(path: Path, fields: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append({k: d.get(k) for k in fields})
    return pd.DataFrame(rows)


def build_daily(posts_path: Path = POSTS, comments_path: Path = COMMENTS
                ) -> pd.DataFrame:
    """Per-UTC-day counts, plus a second copy cut at NEPSE's close.

    Emits both `*_day` (whole UTC day) and `*_precls` (only what existed before
    that day's 09:15 UTC close). The feature layer uses the pre-close variant;
    the full-day one is kept because the ratio between them is a useful check
    that the cut is doing something.
    """
    posts = _read_jsonl(posts_path, ("created_utc", "author", "id", "num_comments"))
    comments = _read_jsonl(comments_path, ("created_utc", "author", "id", "score"))

    out = []
    for name, df in (("posts", posts), ("comments", comments)):
        df = df.dropna(subset=["created_utc"]).copy()
        ts = pd.to_datetime(df["created_utc"], unit="s", utc=True)
        df["date"] = ts.dt.normalize().dt.tz_localize(None)
        before_close = ((ts.dt.hour < CLOSE_UTC_HOUR)
                        | ((ts.dt.hour == CLOSE_UTC_HOUR)
                           & (ts.dt.minute < CLOSE_UTC_MINUTE)))
        df["pre_close"] = before_close

        g = df.groupby("date")
        agg = pd.DataFrame({
            f"n_{name}_day": g.size(),
            f"n_{name}_authors_day": g["author"].nunique(),
        })
        gp = df[df["pre_close"]].groupby("date")
        agg[f"n_{name}_precls"] = gp.size()
        agg[f"n_{name}_authors_precls"] = gp["author"].nunique()
        out.append(agg)

    daily = pd.concat(out, axis=1).fillna(0.0)
    daily.index.name = "date"
    return daily.reset_index().sort_values("date").reset_index(drop=True)


def load_daily(rebuild: bool = False) -> pd.DataFrame:
    """Cached daily aggregate. Rebuilds when the dump is newer than the cache."""
    if CACHE.exists() and not rebuild:
        stale = (POSTS.exists() and POSTS.stat().st_mtime > CACHE.stat().st_mtime)
        if not stale:
            return pd.read_parquet(CACHE)
    if not POSTS.exists() or not COMMENTS.exists():
        raise FileNotFoundError(
            f"Reddit dump not found at {POSTS} / {COMMENTS}")
    log.info("parsing reddit dump (~131k lines) ...")
    daily = build_daily()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(CACHE, index=False)
    return daily


def align_to_sessions(daily: pd.DataFrame, sessions: pd.Series) -> pd.DataFrame:
    """Map calendar-day activity onto trading sessions.

    Activity on a non-trading day is carried forward to the NEXT session, not
    discarded: a closure does not stop people posting, and the loudest days on
    a stock subreddit are often the ones the market is shut. Each session
    therefore sums everything since the previous session.
    """
    sessions = pd.Series(pd.to_datetime(sessions)).sort_values().reset_index(drop=True)
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])

    # searchsorted with side="left" maps each calendar day to the first session
    # on or after it -- i.e. the session at which that activity is first usable.
    pos = sessions.searchsorted(d["date"], side="left")
    d = d[pos < len(sessions)].copy()
    d["session"] = sessions.iloc[pos[pos < len(sessions)]].to_numpy()

    value_cols = [c for c in d.columns if c.startswith("n_")]
    agg = d.groupby("session")[value_cols].sum()
    agg.index.name = "date"
    return agg.reindex(sessions.to_numpy(), fill_value=0.0).reset_index()
