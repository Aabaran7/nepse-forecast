"""Target construction: the sign of the h-step-ahead index return (plan §2).

Two things here are easy to get wrong and invisible afterwards.

**h is measured in SESSIONS, not calendar days**, and the two came apart badly
inside this sample. NEPSE closed for 51 days in 2020 and traded a six-day week
for four months in 2022 (§3.6), so "5 trading days ahead" is anywhere from 6 to
58 calendar days depending on where you stand. A label that silently spans the
COVID closure is not the same target as the one either side of it, and pooling
them is a modelling error dressed up as a data-cleaning shortcut.

**A flat close is not an up day.** NEPSE prints exact repeats. `> 0` and `>= 0`
differ by ~0.1% of days here, which sounds negligible until you remember §6's
whole threshold is 2pp. Flat days are dropped, not silently bucketed.
"""

from __future__ import annotations

import pandas as pd

# A gap this long means a closure, not a weekend. Chosen against the observed
# distribution (§3.6): 32 gaps exceed 4 days, but only three exceed 10, and
# those are the 2020 COVID closures plus one 11-day break.
MAX_LABEL_SPAN_DAYS = 15


def forward_return(df: pd.DataFrame, horizon: int, price_col: str = "close") -> pd.Series:
    """Close-to-close return `horizon` SESSIONS ahead, indexed like `df`."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    c = df[price_col]
    return c.shift(-horizon) / c - 1.0


def label_span_days(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Calendar days each label actually spans. Diagnostic for the check below."""
    d = df["date"]
    return (d.shift(-horizon) - d).dt.days


def make_labels(df: pd.DataFrame, horizon: int, price_col: str = "close",
                max_span_days: int = MAX_LABEL_SPAN_DAYS) -> pd.DataFrame:
    """Attach `fwd_return`, `label` (1 up / 0 down) and `usable` to a copy of df.

    `usable` is False where the label cannot be formed or cannot be trusted:
      - the horizon runs off the end of the series,
      - the forward return is exactly zero (flat close -- neither up nor down),
      - the label spans a market closure longer than `max_span_days`.

    Nothing is dropped here. The caller decides, and can see how much it lost --
    a filter applied silently inside label construction is a filter nobody
    audits.
    """
    out = df.sort_values("date").reset_index(drop=True).copy()
    out["fwd_return"] = forward_return(out, horizon, price_col)
    out["label_span_days"] = label_span_days(out, horizon)

    flat = out["fwd_return"] == 0
    too_long = out["label_span_days"] > max_span_days
    out["label"] = (out["fwd_return"] > 0).astype("float")
    out.loc[out["fwd_return"].isna(), "label"] = float("nan")

    out["usable"] = out["fwd_return"].notna() & ~flat & ~too_long
    out.attrs["horizon"] = horizon
    out.attrs["dropped_flat"] = int(flat.sum())
    out.attrs["dropped_long_span"] = int((too_long & out["fwd_return"].notna()).sum())
    return out


def weekly_blocks(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Non-overlapping block id for each row, for significance testing (§2).

    h=5 labels overlap: consecutive rows share four of their five days, so
    treating them as independent makes confidence intervals far too tight. §2
    requires non-overlapping blocks; this assigns every `horizon`-th row to a
    new block so a caller can take one row per block, or bootstrap over blocks.
    """
    return pd.Series(range(len(df)), index=df.index) // horizon
