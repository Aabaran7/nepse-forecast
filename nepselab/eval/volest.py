"""Volatility estimators (plan §6.8).

§6.7's strategy sized positions off a 63-session close-to-close standard
deviation, chosen because it maximised in-sample Sharpe. Measured directly
against what it is supposed to forecast -- the next 21 sessions' realised
volatility -- that is the WORST estimator available here (correlation 0.275,
against 0.452 for an EWMA). It was selected on the wrong criterion and got away
with it.

This module holds the alternatives, and §6.8 selects among them by **forecast
accuracy**, not by Sharpe. That matters beyond tidiness: choosing a volatility
model by the profitability of the strategy built on it is a search over
strategies wearing the costume of a statistical choice, and with enough
estimators it will always find one. Correlation with future realised volatility
is a criterion the backtest cannot see.

**Why correlation rather than R² or QLIKE.** Every estimator here targets the
same quantity on a different scale (Parkinson's constant assumes continuous
sampling, EWMA is a different weighting). A systematic scale bias is harmless
for position sizing because `target_vol` absorbs it -- doubling every forecast
just halves every exposure, which the target then rescales. What cannot be
absorbed is getting the *ordering* of calm and violent periods wrong, and that
is exactly what correlation measures.

All estimators respect `ohlc_consistent`: the 16 bars flagged in §3.6 have
unreliable highs and lows, and a range estimator reading them would produce a
confident wrong number rather than a missing one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 230.0


def _ohlc(df: pd.DataFrame):
    ok = (df["ohlc_consistent"].astype(bool) if "ohlc_consistent" in df.columns
          else pd.Series(True, index=df.index))
    return (df["open"].where(ok), df["high"].where(ok),
            df["low"].where(ok), df["close"])


def close_to_close(df: pd.DataFrame, window: int = 21) -> pd.Series:
    r = df["close"].pct_change()
    return r.rolling(window, min_periods=max(2, window // 2)).std() * np.sqrt(ANN)


def ewma(df: pd.DataFrame, lam: float = 0.94) -> pd.Series:
    """RiskMetrics-style exponentially weighted volatility.

    No window to choose, and it adapts immediately to a volatility shock rather
    than waiting for one to work through a rolling window. Best forecaster of
    the set on this series (correlation 0.452 with next-21-session realised vol).
    """
    r = df["close"].pct_change()
    return np.sqrt(r.pow(2).ewm(alpha=1 - lam).mean()) * np.sqrt(ANN)


def parkinson(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Parkinson (1980), from the high-low range.

    Roughly five times more efficient than close-to-close for the same sample,
    because a day's range carries far more information about its volatility than
    its two endpoints do. Blind to overnight gaps, which is its known weakness.
    """
    _, h, l, _ = _ohlc(df)
    x = (np.log(h / l) ** 2) / (4 * np.log(2))
    return np.sqrt(x.rolling(window, min_periods=max(2, window // 2)).mean()) * np.sqrt(ANN)


def garman_klass(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Garman-Klass (1980): range plus the open-to-close move."""
    o, h, l, c = _ohlc(df)
    x = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    return np.sqrt(x.clip(lower=0).rolling(
        window, min_periods=max(2, window // 2)).mean()) * np.sqrt(ANN)


ESTIMATORS = {
    "close2close-21": lambda d: close_to_close(d, 21),
    "close2close-63": lambda d: close_to_close(d, 63),
    "ewma-0.94": lambda d: ewma(d, 0.94),
    "ewma-0.97": lambda d: ewma(d, 0.97),
    "parkinson-21": lambda d: parkinson(d, 21),
    "parkinson-63": lambda d: parkinson(d, 63),
    "garman_klass-21": lambda d: garman_klass(d, 21),
    "garman_klass-63": lambda d: garman_klass(d, 63),
}


def forecast_score(est: pd.Series, df: pd.DataFrame, horizon: int = 21) -> float:
    """Correlation between the estimate and the NEXT `horizon` sessions' vol.

    Deliberately not a function of returns, costs or the strategy -- the whole
    point is a criterion the backtest cannot influence.
    """
    fut = df["close"].pct_change().shift(-horizon).rolling(horizon).std()
    m = est.notna() & fut.notna() & (est > 0) & (fut > 0)
    if m.sum() < 100:
        return float("-inf")
    return float(np.corrcoef(est[m], fut[m])[0, 1])


def select_estimator(train: pd.DataFrame, horizon: int = 21) -> str:
    """Best forecaster on the TRAINING window. Never sees a test row."""
    best, best_s = "ewma-0.94", float("-inf")
    for name, fn in ESTIMATORS.items():
        s = forecast_score(fn(train), train, horizon)
        if s > best_s:
            best, best_s = name, s
    return best
