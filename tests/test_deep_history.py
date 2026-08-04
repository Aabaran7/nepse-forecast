"""The deep-history gate, tested on its silent failure modes.

Everything here failed silently at least once while Phase 1c was being written,
which is the reason it is tested at all. None of these bugs raise; each one just
produces a plausible-looking series that is wrong, and the wrongness lands
directly on the label this project predicts.

Run: .venv/bin/pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.ingest import deep_history as dh  # noqa: E402


def series(dates: list[str], closes: list[float], turnover: float = 1e9) -> pd.DataFrame:
    d = pd.to_datetime(dates)
    return pd.DataFrame({
        "date": d, "open": closes, "high": closes, "low": closes,
        "close": closes, "turnover": turnover, "source": "test",
    })


# --- the timestamp trap -----------------------------------------------------
# MeroLagani's bars are 05:45 UTC early in the series and 20:45 UTC later. Read
# as Kathmandu local time (UTC+5:45) the 20:45 bars roll onto the NEXT day, so
# every modern bar shifts forward one session and silently misaligns against the
# archive. The series still looks perfectly well-formed afterwards.

def test_utc_normalisation_keeps_both_timestamp_conventions_on_the_same_day():
    early = pd.Timestamp("2016-03-04 05:45", tz="UTC")
    late = pd.Timestamp("2026-03-04 20:45", tz="UTC")
    for ts in (early, late):
        assert ts.normalize().tz_localize(None) == pd.Timestamp("2016-03-04") \
            or ts.normalize().tz_localize(None) == pd.Timestamp("2026-03-04")


def test_kathmandu_normalisation_would_shift_the_evening_bars():
    """Guards the bug, not the fix: if this ever stops being true the docstring
    in fetch_merolagani() is wrong and someone should re-derive the convention."""
    late = pd.Timestamp("2026-03-04 20:45", tz="UTC")
    shifted = late.tz_convert("Asia/Kathmandu").normalize().tz_localize(None)
    assert shifted == pd.Timestamp("2026-03-05")


# --- source agreement -------------------------------------------------------

def test_one_wrong_close_flips_two_days_of_direction():
    """Why sign disagreement is the metric and level agreement is not enough:
    a single bad close sits between two returns, so one wrong level corrupts
    two labels. 2/3 closes here match exactly and the series still disagrees
    about direction on every day it can."""
    a = series(["2020-01-01", "2020-01-02", "2020-01-03"], [100.0, 101.0, 99.0])
    b = series(["2020-01-01", "2020-01-02", "2020-01-03"], [100.0, 98.0, 99.0])
    agr = dh.compare(a, b)
    assert agr.n_common == 3
    assert agr.exact["close"] == 2          # only 2020-01-02 differs
    # a reads up-then-down, b reads down-then-up: both labels wrong
    assert agr.sign_disagree == 2


def test_compare_reports_dates_unique_to_each_source():
    a = series(["2020-01-01", "2020-01-02"], [100.0, 101.0])
    b = series(["2020-01-02", "2020-01-03"], [101.0, 102.0])
    agr = dh.compare(a, b)
    assert [d.date().isoformat() for d in agr.only_a] == ["2020-01-01"]
    assert [d.date().isoformat() for d in agr.only_b] == ["2020-01-03"]


def test_identical_series_disagree_nowhere():
    a = series(["2020-01-01", "2020-01-02", "2020-01-03"], [100.0, 101.0, 99.0])
    agr = dh.compare(a, a.copy())
    assert agr.sign_disagree == 0
    assert agr.exact["close"] == 3


# --- the usable-start rule --------------------------------------------------
# Deliberately a suffix condition: a clean year sitting inside dirty ones is
# useless because the series has to be contiguous to model on.

def _yearly(pcts: dict[int, float]) -> pd.DataFrame:
    return pd.DataFrame({"sign_disagree_pct": pd.Series(pcts)})


def test_usable_start_picks_the_first_year_of_a_clean_run():
    y = _yearly({2013: 4.0, 2014: 3.0, 2015: 2.0, 2016: 0.1, 2017: 0.0})
    assert dh.usable_start(y, max_pct=0.5) == 2016


def test_usable_start_ignores_a_clean_year_stranded_among_dirty_ones():
    y = _yearly({2013: 0.0, 2014: 3.0, 2015: 2.0, 2016: 0.1, 2017: 0.0})
    assert dh.usable_start(y, max_pct=0.5) == 2016     # not 2013


def test_usable_start_is_none_when_the_latest_year_is_dirty():
    y = _yearly({2015: 0.0, 2016: 0.0, 2017: 9.0})
    assert dh.usable_start(y, max_pct=0.5) is None


# --- turnover units ---------------------------------------------------------

def test_turnover_units_break_is_detected():
    dates = pd.bdate_range("2016-01-01", "2017-12-31")
    vals = np.where(dates < pd.Timestamp("2017-01-01"), 1.4e6, 5.5e8)
    df = pd.DataFrame({"date": dates, "turnover": vals})
    breaks = dh.turnover_scale_breaks(df)
    assert len(breaks) == 1
    assert breaks.index[0] == pd.Timestamp("2017-01-01")
    assert breaks.iloc[0] > 100


def test_steady_turnover_reports_no_break():
    dates = pd.bdate_range("2016-01-01", "2017-12-31")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"date": dates,
                       "turnover": 5e8 * rng.uniform(0.5, 2.0, len(dates))})
    assert len(dh.turnover_scale_breaks(df)) == 0


def test_short_months_do_not_manufacture_a_break():
    """NEPSE closed for two months in 2020. A two-session month has a wild
    median, and the naive version of this check flagged the COVID reopening as
    a units change because of it."""
    normal = pd.bdate_range("2020-01-01", "2020-02-28")
    stub = pd.DatetimeIndex(["2020-03-02", "2020-03-03"])
    after = pd.bdate_range("2020-04-01", "2020-06-30")
    dates = normal.append(stub).append(after)
    vals = np.concatenate([np.full(len(normal), 5e8),
                           np.full(len(stub), 1e6),      # thin, unrepresentative
                           np.full(len(after), 5e8)])
    df = pd.DataFrame({"date": dates, "turnover": vals})
    assert len(dh.turnover_scale_breaks(df)) == 0


def test_zero_turnover_rows_are_ignored_rather_than_dividing_by_zero():
    dates = pd.bdate_range("2016-01-01", "2016-06-30")
    vals = np.full(len(dates), 5e8)
    vals[:5] = 0.0
    df = pd.DataFrame({"date": dates, "turnover": vals})
    assert len(dh.turnover_scale_breaks(df)) == 0


# --- provenance -------------------------------------------------------------

def test_deep_series_is_not_written_into_the_archive(tmp_path: Path):
    """The archive is exchange-sourced and irreplaceable; a scrape must not be
    able to land in it. save() takes its own root and defaults outside it."""
    assert dh.DEEP != __import__("nepselab.ingest.archive",
                                 fromlist=["ARCHIVE"]).ARCHIVE
    df = series(["2020-01-01"], [100.0])
    out = dh.save(df, "probe", root=tmp_path / "deep")
    assert out.exists() and "archive" not in str(out)


def test_tidy_drops_duplicate_dates_and_sorts():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-01"]),
        "open": [3.0, 1.0, 1.0], "high": [3.0, 1.0, 1.0],
        "low": [3.0, 1.0, 1.0], "close": [3.0, 1.0, 1.5],
        "turnover": [1.0, 1.0, 1.0],
    })
    out = dh._tidy(df, "test")
    assert len(out) == 2
    assert out["date"].is_monotonic_increasing
    assert out.iloc[0]["close"] == 1.0     # keep="first"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
