"""Sanity-suite tests (plan §3.3, §8's "tests pass before any experiment runs").

Two halves, and both are needed:

  Synthetic cases pin the check LOGIC -- a checker that silently stops detecting
  anything is worse than no checker, and that failure is invisible when the only
  thing you ever run it on is data that currently passes.

  Real-file cases pin the DATA. §8's gate is meaningless if the tests assert on
  a paraphrase of the series rather than the series itself. These skip when the
  parquet is absent so a fresh clone still runs green.

Run: .venv/bin/pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab import quality  # noqa: E402

DEEP = Path("data/deep/nepse_index_deep.parquet")
PARAMS = Path("configs/market_params.yaml")


def bars(dates: list[str], o, h, l, c) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates),
                         "open": o, "high": h, "low": l, "close": c})


# --- OHLC -------------------------------------------------------------------

def test_clean_bars_have_no_violations():
    df = bars(["2020-01-01", "2020-01-02"], [10, 11], [12, 13], [9, 10], [11, 12])
    assert quality.ohlc_violations(df).empty


def test_close_above_high_is_caught():
    df = bars(["2020-01-01"], [10.0], [11.0], [9.0], [12.0])
    v = quality.ohlc_violations(df)
    assert len(v) == 1
    assert v.iloc[0]["violation"] == pytest.approx(1.0 / 12.0)


def test_low_above_open_is_caught():
    df = bars(["2020-01-01"], [10.0], [13.0], [11.0], [12.0])
    assert len(quality.ohlc_violations(df)) == 1


def test_sub_tolerance_rounding_is_not_a_violation():
    """high/low are stored to 2dp while close carries more digits, so a close
    can exceed its own high by a fraction of a cent. That is representation
    noise, and flagging it would bury the 16 real cases in 67 fake ones."""
    df = bars(["2020-01-01"], [1253.0], [1253.06], [1238.7], [1253.06067])
    assert quality.ohlc_violations(df).empty
    assert len(quality.ohlc_violations(df, tol=0)) == 1


def test_close_outside_range_is_a_strict_subset_of_violations():
    df = bars(["2020-01-01", "2020-01-02"],
              [10.0, 20.0], [11.0, 21.0], [9.0, 19.0], [12.0, 20.5])
    v, c = quality.ohlc_violations(df), quality.close_outside_range(df)
    assert set(c.index) <= set(v.index)
    assert len(c) == 1                      # only the first bar's close is out


def test_ohlc_violations_rejects_a_frame_missing_columns():
    with pytest.raises(KeyError):
        quality.ohlc_violations(pd.DataFrame({"date": [], "close": []}))


# --- calendar ---------------------------------------------------------------

ERAS = [
    {"effective_from": None, "effective_to": "2022-05-19",
     "days": ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]},
    {"effective_from": "2022-05-20", "effective_to": "2022-09-16",
     "days": ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]},
    {"effective_from": "2022-09-17", "effective_to": None,
     "days": ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]},
]


def test_friday_outside_the_six_day_era_is_flagged():
    df = pd.DataFrame({"date": pd.to_datetime(["2022-05-13"])})   # a Friday
    assert len(quality.weekday_violations(df, ERAS)) == 1


def test_friday_inside_the_six_day_era_is_allowed():
    """The 2022 block is exactly what this check exists to have caught."""
    df = pd.DataFrame({"date": pd.to_datetime(["2022-05-20", "2022-09-16"])})
    assert quality.weekday_violations(df, ERAS).empty


def test_era_boundaries_are_inclusive_on_both_ends():
    inside = pd.DataFrame({"date": pd.to_datetime(["2022-05-20"])})
    outside = pd.DataFrame({"date": pd.to_datetime(["2022-09-23"])})  # Friday, after
    assert quality.weekday_violations(inside, ERAS).empty
    assert len(quality.weekday_violations(outside, ERAS)) == 1


def test_saturday_is_never_allowed():
    df = pd.DataFrame({"date": pd.to_datetime(["2022-06-04"])})    # Saturday
    assert len(quality.weekday_violations(df, ERAS)) == 1


# --- circuit ----------------------------------------------------------------

def test_return_inside_the_circuit_passes():
    df = bars(["2020-01-01", "2020-01-02"], [100, 100], [100, 100], [100, 100],
              [100.0, 105.0])
    assert quality.circuit_breaches(df).empty


def test_return_beyond_the_circuit_is_caught():
    df = bars(["2020-01-01", "2020-01-02"], [100, 100], [100, 100], [100, 100],
              [100.0, 108.0])
    assert len(quality.circuit_breaches(df)) == 1


def test_small_overshoot_is_tolerated():
    """Observed extremes sit just past 6% (max 6.0610% over 2016-2026). Without
    tolerance the check cries wolf a dozen times on genuinely capped days."""
    df = bars(["2020-01-01", "2020-01-02"], [100, 100], [100, 100], [100, 100],
              [100.0, 106.05])
    assert quality.circuit_breaches(df).empty


# --- baselines --------------------------------------------------------------

def test_direction_baseline_reports_the_majority_not_fifty_percent():
    df = bars(["2020-01-0" + str(i) for i in range(1, 6)],
              [1] * 5, [1] * 5, [1] * 5, [100.0, 101.0, 102.0, 103.0, 102.0])
    b = quality.direction_baseline(df)
    assert b["n"] == 4
    assert b["majority_class"] == "up"
    assert b["majority_share"] == pytest.approx(0.75)


# --- the real series --------------------------------------------------------

@pytest.fixture(scope="module")
def deep() -> pd.DataFrame:
    if not DEEP.exists():
        pytest.skip(f"{DEEP} absent; run scripts/phase1c_deep_history.py")
    return pd.read_parquet(DEEP).sort_values("date").reset_index(drop=True)


@pytest.fixture(scope="module")
def eras() -> list[dict]:
    return yaml.safe_load(PARAMS.read_text())["trading_week"]


def test_deep_series_passes_every_check(deep, eras):
    failed = [f.name for f in quality.run_all(deep, eras) if not f.passed]
    assert not failed, f"failing checks: {failed}"


def test_deep_series_flags_exactly_the_inconsistent_bars(deep):
    """Policy is flag-never-repair, so the flag has to stay true to the data.
    A repair upstream would show up here as a disagreement, not as silence."""
    assert "ohlc_consistent" in deep.columns
    assert set(deep.index[~deep["ohlc_consistent"]]) == set(
        quality.ohlc_violations(deep).index)


def test_deep_series_respects_the_index_circuit(deep):
    """Corroboration, not just absence of error: a third-party feed obeying
    NEPSE's ±6% cap for a decade is behaving like real NEPSE data (§3.5)."""
    assert quality.circuit_breaches(deep).empty
    ret = deep["close"].pct_change()
    assert ret.abs().max() > 0.06          # it does reach the cap
    assert ret.abs().max() < 0.061         # and never meaningfully exceeds it


def test_deep_series_has_no_duplicate_dates(deep):
    assert quality.duplicate_dates(deep).empty


def test_deep_series_covers_both_regimes(deep):
    """§2's regime split is only restored if both halves are actually present."""
    mania = deep[(deep.date >= "2020-06-01") & (deep.date <= "2021-12-31")]
    chop = deep[deep.date >= "2022-01-01"]
    assert len(mania) > 300
    assert len(chop) > 900


def test_majority_class_flips_between_regimes(deep):
    """Recorded because it is a trap, not a curiosity: §6's bar is
    'majority + 2pp', and the majority is DOWN overall but UP in the mania
    window. A single pooled baseline would set the wrong bar in both."""
    full = quality.direction_baseline(deep)
    mania = quality.direction_baseline(
        deep[(deep.date >= "2020-06-01") & (deep.date <= "2021-12-31")])
    assert full["majority_class"] == "down"
    assert mania["majority_class"] == "up"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
