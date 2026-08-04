"""Price and volume features (plan §5, module 1).

Everything here is built from the deep index series and every column is shifted
so that a feature at session t uses only closes up to and including t. The
target is the return from t to t+h, so t's own close is legitimately known --
but nothing beyond it is, and `rolling(...).mean()` on an unshifted column
quietly includes the future when the window is centred or the label is joined
carelessly.

Two constraints from §3.5/§3.6 that shape what is buildable:

**Turnover starts in 2017, not 2016.** There is a 420x units break at
2017-01-01 with no matching move in the index. Every turnover feature here is a
ratio to its own trailing window, which would read that break as a 420x surge.
Hence `TurnoverFeatures.available_from = 2017-01-01`, separate from price.

**16 bars are internally inconsistent.** high/low are unreliable on those days
(§3.6), so range-derived features read `ohlc_consistent` and emit NaN rather
than a plausible wrong number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PriceFeatures:
    """Returns, volatility, and distance from the running high."""

    name = "price"
    available_from = pd.Timestamp("2016-01-01")

    def __init__(self, lags: tuple[int, ...] = (1, 2, 3, 5, 10),
                 vol_windows: tuple[int, ...] = (5, 21, 63)):
        self.lags = lags
        self.vol_windows = vol_windows

    def build(self, sessions: pd.DataFrame) -> pd.DataFrame:
        s = sessions.sort_values("date").reset_index(drop=True)
        c = s["close"].astype(float)
        r = c.pct_change()

        out = pd.DataFrame({"date": s["date"]})

        # Lagged returns. lag=1 is the return INTO t, known at t's close.
        for k in self.lags:
            out[f"ret_lag{k}"] = r.shift(k - 1)

        # Cumulative momentum over several windows.
        for k in (5, 10, 21, 63):
            out[f"mom_{k}"] = c / c.shift(k) - 1.0

        # Realized volatility, and where today's vol sits within its own history.
        # min_periods everywhere: a rolling window whose min_periods defaults to
        # the full width returns NaN if ANY input in it is NaN, so one masked bar
        # silently deletes the next `w` rows. That is not hypothetical -- three
        # inconsistent bars in 2018 (§3.6) wiped 117 of 240 sessions through
        # intraday_range_z, and 2017-2019 lost ~60% of their rows before this
        # was caught. It cost the pre-mania regime, which §6(c) needs.
        for w in self.vol_windows:
            v = r.rolling(w, min_periods=max(2, w // 2)).std()
            out[f"vol_{w}"] = v
            out[f"vol_ratio_{w}"] = v / r.rolling(252, min_periods=60).std()

        # Distance from the trailing 52-week high/low. Uses closes, not the
        # high/low columns, so the 16 bad bars (§3.6) cannot corrupt it.
        roll_max = c.rolling(252, min_periods=60).max()
        roll_min = c.rolling(252, min_periods=60).min()
        out["dist_52w_high"] = c / roll_max - 1.0
        out["dist_52w_low"] = c / roll_min - 1.0

        # Where the close sits in the recent range: 0 = at the low, 1 = at the high.
        for w in (21, 63):
            lo = c.rolling(w, min_periods=w // 2).min()
            hi = c.rolling(w, min_periods=w // 2).max()
            out[f"pos_in_range_{w}"] = (c - lo) / (hi - lo).replace(0, np.nan)

        # Trend agreement, a plain moving-average cross expressed as a level.
        out["ma_ratio_5_21"] = (c.rolling(5, min_periods=3).mean()
                                / c.rolling(21, min_periods=10).mean() - 1.0)
        out["ma_ratio_21_63"] = (c.rolling(21, min_periods=10).mean()
                                 / c.rolling(63, min_periods=30).mean() - 1.0)

        # Run length: NEPSE has long directional streaks, which is exactly why
        # §2 insists the majority class is the baseline that matters.
        up = (r > 0).astype(float)
        grp = (up != up.shift()).cumsum()
        out["run_length"] = up.groupby(grp).cumcount() + 1
        out["run_is_up"] = up

        # Intraday range, but only where the bar is internally consistent.
        if "ohlc_consistent" in s.columns:
            rng = (s["high"] - s["low"]) / c
            out["intraday_range"] = rng.where(s["ohlc_consistent"].astype(bool))
            ir = out["intraday_range"]
            out["intraday_range_z"] = (
                (ir - ir.rolling(63, min_periods=30).mean())
                / ir.rolling(63, min_periods=30).std())
        return out


class TurnoverFeatures:
    """Market-wide turnover, as ratios to its own trailing window.

    Separate module from PriceFeatures purely because of the 2017 units break --
    same source file, different valid start date. Keeping them apart is what
    lets a config use price back to 2016 without silently dragging turnover
    across a 420x discontinuity.
    """

    name = "turnover"
    available_from = pd.Timestamp("2017-01-01")

    def build(self, sessions: pd.DataFrame) -> pd.DataFrame:
        s = sessions.sort_values("date").reset_index(drop=True)
        out = pd.DataFrame({"date": s["date"]})
        if "turnover" not in s.columns:
            return out

        t = s["turnover"].astype(float).replace(0, np.nan)
        # Log, because turnover is heavily right-skewed and a raw ratio is
        # dominated by a handful of frenzy days.
        lt = np.log(t)
        for w in (5, 21, 63):
            out[f"turnover_ratio_{w}"] = lt - lt.rolling(w, min_periods=max(2, w // 2)).mean()
        out["turnover_z_63"] = ((lt - lt.rolling(63, min_periods=30).mean())
                                / lt.rolling(63, min_periods=30).std())
        out["turnover_chg_1"] = lt.diff()

        # Turnover rising into a falling market means something different from
        # turnover rising into a rising one; give the model the interaction
        # rather than hoping a linear model finds it.
        r = s["close"].astype(float).pct_change()
        out["turnover_x_ret"] = out["turnover_z_63"] * np.sign(r)

        # Anything computed on the pre-2017 units is nonsense; mask it here so
        # a caller who ignores available_from still cannot use it.
        out.loc[s["date"] < self.available_from, out.columns != "date"] = np.nan
        return out
