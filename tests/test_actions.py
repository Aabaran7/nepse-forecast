"""Corporate actions recovered from the archive, and the adjustment they imply.

The failure this prevents is not subtle, it is just invisible until someone
plots a single company. When a scrip issues bonus shares the quoted price falls
mechanically, and a price series built by shifting closes reads that as a crash.
NABBC on 2025-08-25: -40.9% on raw closes, +9.99% -- limit up -- on the
exchange's own basis. 171 such days over one year, on 160 of 454 symbols.

The detector is deliberately modest about what it knows. It recovers the DATE
and the SIZE of an adjustment from NEPSE's own restatement of the previous
close. It does NOT recover the kind: a 2:1 bonus, a par-priced rights issue and
a large special dividend can imply similar factors. So `kind` stays "unknown",
and these tests assert that it stays unknown rather than being guessed.

Run: .venv/bin/pytest tests/test_actions.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.adjust import actions  # noqa: E402


def frame(rows: list[tuple]) -> pd.DataFrame:
    """rows of (date, symbol, close, previousDayClose)."""
    return pd.DataFrame(
        [{"businessDate": pd.Timestamp(d), "symbol": s,
          "closePrice": c, "previousDayClosePrice": p} for d, s, c, p in rows])


class TestDetection:
    def test_a_bonus_issue_is_found(self):
        # Close 1336.65, then the exchange restates the prior close to 718.33
        # and the scrip closes 790.10 -- the NABBC event, in miniature.
        df = frame([
            ("2026-01-05", "AAA", 1336.65, 1300.00),
            ("2026-01-06", "AAA", 790.10, 718.33),
        ])
        out = actions.detect(df)
        assert len(out) == 1
        assert out["symbol"].iloc[0] == "AAA"
        assert out["ex_date"].iloc[0] == pd.Timestamp("2026-01-06")
        assert out["factor"].iloc[0] == pytest.approx(718.33 / 1336.65, rel=1e-6)

    def test_an_ordinary_session_is_not_an_event(self):
        df = frame([
            ("2026-01-05", "AAA", 100.0, 99.0),
            ("2026-01-06", "AAA", 103.0, 100.0),   # prior close matches exactly
        ])
        assert actions.detect(df).empty

    def test_rounding_noise_is_not_an_event(self):
        df = frame([
            ("2026-01-05", "AAA", 100.00, 99.0),
            ("2026-01-06", "AAA", 101.00, 100.05),  # 0.05% -- a tick, not a bonus
        ])
        assert actions.detect(df).empty

    def test_the_kind_is_not_guessed(self):
        df = frame([
            ("2026-01-05", "AAA", 200.0, 199.0),
            ("2026-01-06", "AAA", 105.0, 100.0),
        ])
        out = actions.detect(df)
        assert out["kind"].iloc[0] == "unknown", (
            "the factor cannot distinguish a bonus from a rights issue")

    def test_symbols_do_not_contaminate_each_other(self):
        df = frame([
            ("2026-01-05", "AAA", 100.0, 99.0),
            ("2026-01-05", "BBB", 500.0, 495.0),
            ("2026-01-06", "AAA", 102.0, 100.0),
            ("2026-01-06", "BBB", 260.0, 250.0),   # only BBB has an event
        ])
        out = actions.detect(df)
        assert list(out["symbol"]) == ["BBB"]

    def test_a_scrips_first_ever_session_cannot_be_an_event(self):
        # No prior close of our own to compare against; inventing one would make
        # every new listing look like a corporate action.
        df = frame([("2026-01-06", "NEW", 100.0, 50.0)])
        assert actions.detect(df).empty

    def test_missing_columns_are_reported(self):
        with pytest.raises(ValueError, match="previousDayClosePrice"):
            actions.detect(pd.DataFrame({"businessDate": [], "symbol": [],
                                         "closePrice": []}))


class TestAdjustment:
    def test_prices_before_the_event_are_scaled_and_today_is_untouched(self):
        prices = pd.DataFrame({
            "businessDate": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            "symbol": ["AAA", "AAA"],
            "closePrice": [1336.65, 790.10],
        })
        acts = pd.DataFrame({"symbol": ["AAA"],
                             "ex_date": [pd.Timestamp("2026-01-06")],
                             "factor": [718.33 / 1336.65]})
        out = actions.adjust_series(prices, acts).sort_values("businessDate")

        # The newest price must equal the quoted price, or it stops matching a
        # broker screen and someone concludes the archive is broken.
        assert out["adj_close"].iloc[-1] == pytest.approx(790.10)
        # And the fake crash becomes the real move.
        ret = out["adj_close"].iloc[-1] / out["adj_close"].iloc[0] - 1
        assert ret == pytest.approx(0.0999, abs=1e-3)

    def test_two_events_compound(self):
        prices = pd.DataFrame({
            "businessDate": pd.to_datetime(["2026-01-01", "2026-01-10", "2026-01-20"]),
            "symbol": ["AAA"] * 3,
            "closePrice": [400.0, 200.0, 100.0],
        })
        acts = pd.DataFrame({
            "symbol": ["AAA", "AAA"],
            "ex_date": pd.to_datetime(["2026-01-10", "2026-01-20"]),
            "factor": [0.5, 0.5],
        })
        out = actions.adjust_series(prices, acts).sort_values("businessDate")
        # Oldest price sits behind both halvings: 400 * 0.5 * 0.5 = 100.
        assert out["adj_close"].iloc[0] == pytest.approx(100.0)
        assert out["adj_close"].iloc[1] == pytest.approx(100.0)
        assert out["adj_close"].iloc[2] == pytest.approx(100.0)

    def test_an_unaffected_symbol_is_left_alone(self):
        prices = pd.DataFrame({
            "businessDate": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            "symbol": ["BBB", "BBB"], "closePrice": [50.0, 51.0],
        })
        acts = pd.DataFrame({"symbol": ["AAA"],
                             "ex_date": [pd.Timestamp("2026-01-06")],
                             "factor": [0.5]})
        out = actions.adjust_series(prices, acts)
        assert list(out["adj_close"]) == [50.0, 51.0]

    def test_no_actions_means_adj_close_equals_close(self):
        prices = pd.DataFrame({
            "businessDate": pd.to_datetime(["2026-01-05"]),
            "symbol": ["AAA"], "closePrice": [50.0]})
        out = actions.adjust_series(prices, pd.DataFrame())
        assert out["adj_close"].iloc[0] == 50.0


def test_against_the_real_archive_if_present():
    """The 171 events are a fact about the stored data, not a fixture."""
    from nepselab.ingest import archive
    tp = archive.load("today_price")
    if tp.empty:
        pytest.skip("no archive in this checkout")

    out = actions.detect(tp)
    assert not out.empty
    # Every detected event must be a genuine restatement, not a price move: a
    # scrip cannot legally move more than 15% in a session, so anything inside
    # that band would be indistinguishable from ordinary trading.
    assert (out["factor"] - 1.0).abs().min() > actions.MIN_FACTOR_MOVE
    assert (out["kind"] == "unknown").all()
    assert out["ex_date"].is_monotonic_increasing or len(out) == 1
