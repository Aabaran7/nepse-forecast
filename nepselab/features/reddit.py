"""Reddit attention features (plan §5, module 2).

§5's EDA is complete and its conclusion is implemented rather than revisited:
index-level attention only, weekly solid, daily thinner but workable.

**The one hard rule, from §5 and §9's risk table:** attention must be a share of
the subreddit's own activity or a z-score against a trailing window -- never a
raw count. r/NepalStock grew across the sample, so a raw count is a proxy for
the calendar, and a model handed the calendar will use it to predict a market
that also trended. §9 rates this "High if unhandled". Nothing here emits a
level.

Sentiment is deliberately NOT here. §5 requires an LLM scoring pass with a
constrained schema and an on-disk cache, and requires validating the chosen
model against ~200 hand-labelled code-switched comments first. That is real
work with a real cost and it is not done, so this module ships attention only
and the ablation measures attention only. VADER/TextBlob are explicitly ruled
out by §5 -- they score romanized Nepali as neutral noise -- so there is no
shortcut available here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..ingest import reddit as reddit_ingest


class RedditAttention:
    """Attention normalised against its own trailing history.

    `available_from` is set well after the dump's first row: the subreddit was
    tiny in 2017-2019 and §5 puts the usable window at 2020-06. A trailing
    90-session z-score also needs that much history before it means anything.
    """

    name = "reddit"
    available_from = pd.Timestamp("2020-06-01")

    def __init__(self, z_window: int = 90, use_pre_close: bool = True):
        self.z_window = z_window
        self.use_pre_close = use_pre_close
        self._daily: pd.DataFrame | None = None

    def _counts(self, sessions: pd.DataFrame) -> pd.DataFrame:
        if self._daily is None:
            self._daily = reddit_ingest.load_daily()
        return reddit_ingest.align_to_sessions(self._daily, sessions["date"])

    def build(self, sessions: pd.DataFrame) -> pd.DataFrame:
        s = sessions.sort_values("date").reset_index(drop=True)
        counts = self._counts(s)
        sfx = "precls" if self.use_pre_close else "day"

        comments = counts[f"n_comments_{sfx}"].astype(float)
        posts = counts[f"n_posts_{sfx}"].astype(float)
        authors = counts[f"n_comments_authors_{sfx}"].astype(float)

        out = pd.DataFrame({"date": s["date"]})
        w = self.z_window

        def z(x: pd.Series) -> pd.Series:
            m = x.rolling(w, min_periods=w // 2).mean()
            sd = x.rolling(w, min_periods=w // 2).std()
            return (x - m) / sd.replace(0, np.nan)

        # Attention, z-scored against its own trailing window. This is the
        # detrending §5 demands: the level is discarded, only the surprise
        # relative to the recent norm survives.
        out["reddit_comments_z"] = z(comments)
        out["reddit_posts_z"] = z(posts)
        out["reddit_authors_z"] = z(authors)

        # Log ratio to the trailing mean -- same idea, less sensitive to the
        # variance estimate on quiet stretches.
        for name, x in (("comments", comments), ("posts", posts)):
            lx = np.log1p(x)
            out[f"reddit_{name}_ratio"] = lx - lx.rolling(w, min_periods=w // 2).mean()

        # Composition, which is scale-free by construction: a burst of comments
        # per post is an argument, a burst of posts is a news event.
        out["reddit_comments_per_post"] = (comments / posts.replace(0, np.nan))
        out["reddit_comments_per_author"] = (comments / authors.replace(0, np.nan))
        out["reddit_comments_per_post_z"] = z(out["reddit_comments_per_post"])

        # Short-window acceleration: attention today against attention this week.
        out["reddit_accel_5"] = (np.log1p(comments)
                                 - np.log1p(comments).rolling(5).mean())

        # The dump ends before the price series does. Those trailing sessions
        # get NaN rather than zero -- zero would read as "nobody posted", which
        # is a strong and false signal, where NaN correctly means "unknown" and
        # is dropped by assemble().
        last = self._daily["date"].max()
        out.loc[s["date"] > last, out.columns != "date"] = np.nan
        out.loc[s["date"] < self.available_from, out.columns != "date"] = np.nan
        return out
