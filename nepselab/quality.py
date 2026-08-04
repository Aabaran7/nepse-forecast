"""Sanity checks for daily index series (plan §3.3).

`scripts/phase0_quality.py` reports on the Phase 0 pull; this module holds the
checks themselves so the report script and the test suite run the *same* code.
§8's rule is that sanity tests pass before any experiment runs, which only means
something if the tests assert on the real files rather than on a paraphrase of
them.

Every function returns findings rather than raising. A check that throws stops
at the first problem; the point here is to see all of them at once, and to let
the caller decide which are fatal. Nothing in this module modifies data -- a
silently repaired bar is indistinguishable from a correct one downstream, and
the whole reason these checks exist is that the failures are already invisible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Relative to the close. The deep feed stores high/low to 2dp while open/close
# carry more digits, so a close can exceed its own high by <0.005 as pure
# representation noise. At 1e-4 (~0.2 points on a 2000-level index) that noise
# drops out and only genuine inconsistencies remain: on the 2016-2026 series
# it separates 83 nominal violations into 67 rounding artifacts and 16 real ones.
OHLC_TOL = 1e-4

# NEPSE's index circuit is +-6%, but observed extremes overshoot it slightly
# (max 6.0610% across 2016-2026, 6.0070% in the exchange-verified window). The
# overshoot is consistent across eras, so the check allows for it rather than
# reporting a dozen false positives every run.
INDEX_CAP = 0.06
INDEX_CAP_TOL = 0.001


@dataclass
class Finding:
    """One check's result. `rows` is empty when the check passed."""
    name: str
    passed: bool
    detail: str
    rows: pd.DataFrame = field(default_factory=pd.DataFrame)


def ohlc_violations(df: pd.DataFrame, tol: float = OHLC_TOL) -> pd.DataFrame:
    """Bars where high isn't the max or low isn't the min, beyond `tol`.

    Returns the offending rows with a `violation` column giving the magnitude
    relative to the close, so a caller can tell 0.005-point rounding from a
    close sitting three points above its own high.
    """
    need = {"open", "high", "low", "close"}
    if not need.issubset(df.columns):
        raise KeyError(f"expected {sorted(need)}, got {sorted(df.columns)}")

    hi_should_be = df[["open", "close", "low"]].max(axis=1)
    lo_should_be = df[["open", "close", "high"]].min(axis=1)
    over = (hi_should_be - df["high"]).clip(lower=0)
    under = (df["low"] - lo_should_be).clip(lower=0)
    violation = (over + under) / df["close"].abs()

    out = df.loc[violation > tol].copy()
    out["violation"] = violation[violation > tol]
    return out.sort_values("violation", ascending=False)


def close_outside_range(df: pd.DataFrame, tol: float = OHLC_TOL) -> pd.DataFrame:
    """Bars whose CLOSE sits outside [low, high] -- the subset that matters most.

    Separated from ohlc_violations because the close is this project's target.
    A bad open corrupts one feature; a bad close corrupts the label.
    """
    excess = (pd.concat([df["close"] - df["high"], df["low"] - df["close"]], axis=1)
                .max(axis=1)) / df["close"].abs()
    out = df.loc[excess > tol].copy()
    out["excess"] = excess[excess > tol]
    return out.sort_values("excess", ascending=False)


def weekday_violations(df: pd.DataFrame, eras: list[dict]) -> pd.DataFrame:
    """Sessions falling on a weekday the trading calendar does not allow.

    `eras` is market_params' `trading_week`: dicts with `effective_from`,
    `effective_to` (either may be None for open-ended) and `days`.

    This is the check that found the 2022 six-day era. It is worth keeping even
    once the calendar is believed correct, because a new era announces itself
    here and nowhere else -- the price data stays perfectly well-formed.
    """
    name_of = df["date"].dt.day_name()
    allowed = pd.Series(False, index=df.index)
    covered = pd.Series(False, index=df.index)

    for era in eras:
        lo = pd.Timestamp(era["effective_from"]) if era.get("effective_from") else None
        hi = pd.Timestamp(era["effective_to"]) if era.get("effective_to") else None
        in_era = pd.Series(True, index=df.index)
        if lo is not None:
            in_era &= df["date"] >= lo
        if hi is not None:
            in_era &= df["date"] <= hi
        covered |= in_era
        allowed |= in_era & name_of.isin(era["days"])

    out = df.loc[covered & ~allowed].copy()
    out["weekday"] = name_of[covered & ~allowed]
    return out


def weekday_profile(df: pd.DataFrame, freq: str = "Q") -> pd.DataFrame:
    """Session counts per weekday per period. Discovery aid, not a pass/fail.

    Reading this is how a human notices an era boundary that no assertion was
    written for yet.
    """
    p = df.assign(period=df["date"].dt.to_period(freq).dt.to_timestamp(),
                  weekday=df["date"].dt.day_name())
    return (p.pivot_table(index="period", columns="weekday", values="close",
                          aggfunc="size", fill_value=0))


def circuit_breaches(df: pd.DataFrame, cap: float = INDEX_CAP,
                     tol: float = INDEX_CAP_TOL) -> pd.DataFrame:
    """Close-to-close returns beyond the index circuit, allowing for overshoot.

    Passing is *positive* evidence, not merely an absence of errors: a
    third-party series that respects NEPSE's circuit across ten years is
    behaving like real NEPSE data. See §3.5.
    """
    d = df.sort_values("date").copy()
    d["ret"] = d["close"].pct_change()
    out = d.loc[d["ret"].abs() > cap + tol].copy()
    return out


def duplicate_dates(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["date"].duplicated(keep=False)].sort_values("date")


def calendar_gaps(df: pd.DataFrame, max_days: int = 4) -> pd.DataFrame:
    """Gaps longer than `max_days` calendar days. Closures, not missing data."""
    d = df.sort_values("date").copy()
    d["gap_days"] = d["date"].diff().dt.days
    return d.loc[d["gap_days"] > max_days, ["date", "gap_days"]]


def direction_baseline(df: pd.DataFrame) -> dict:
    """Majority-class baseline -- the bar §6 measures against, not 50%."""
    r = df.sort_values("date")["close"].pct_change().dropna()
    up = float((r > 0).mean())
    return {"n": int(len(r)), "up_share": up,
            "majority_class": "up" if up > 0.5 else "down",
            "majority_share": max(up, 1 - up)}


def run_all(df: pd.DataFrame, eras: list[dict] | None = None) -> list[Finding]:
    """Every check, as a list of findings. Empty `rows` means it passed."""
    findings: list[Finding] = []

    dup = duplicate_dates(df)
    findings.append(Finding("duplicate dates", dup.empty,
                            f"{len(dup)} duplicated date(s)", dup))

    nonpos = df[df["close"] <= 0]
    findings.append(Finding("positive closes", nonpos.empty,
                            f"{len(nonpos)} non-positive close(s)", nonpos))

    v = ohlc_violations(df)
    c = close_outside_range(df)

    # NEPSE's own published bars are internally inconsistent on 16 of 2,434
    # sessions (§3.5), and cross-checking both deep sources showed the defect is
    # upstream rather than a scraping artifact -- so "zero violations" is a bar
    # this data will never clear, and asserting it makes the §8 gate permanently
    # red and therefore ignored. What IS assertable is that every bad bar is
    # FLAGGED, so the feature layer can exclude it instead of rediscovering it.
    if "ohlc_consistent" in df.columns:
        flagged = set(df.index[~df["ohlc_consistent"]])
        agree = flagged == set(v.index)
        findings.append(Finding(
            "OHLC violations flagged", agree,
            f"{len(v)} inconsistent bar(s), {len(flagged)} flagged"
            + ("" if agree else " -- FLAG DISAGREES WITH THE DATA"),
            v if not agree else pd.DataFrame()))
        findings.append(Finding(
            "close within [low, high]", set(c.index) <= flagged,
            f"{len(c)} close(s) outside the day's range, all flagged"
            if set(c.index) <= flagged else f"{len(c)} unflagged",
            c if not set(c.index) <= flagged else pd.DataFrame()))
    else:
        findings.append(Finding("OHLC consistency", v.empty,
                                f"{len(v)} bar(s) beyond tol={OHLC_TOL:g} "
                                f"and no ohlc_consistent column to account "
                                f"for them", v))
        findings.append(Finding("close within [low, high]", c.empty,
                                f"{len(c)} close(s) outside the day's range", c))

    b = circuit_breaches(df)
    findings.append(Finding("index circuit", b.empty,
                            f"{len(b)} return(s) beyond "
                            f"±{(INDEX_CAP + INDEX_CAP_TOL):.1%}", b))

    if eras:
        w = weekday_violations(df, eras)
        findings.append(Finding("trading calendar", w.empty,
                                f"{len(w)} session(s) on a disallowed weekday", w))

    return findings
