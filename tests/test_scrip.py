"""The per-scrip panel: no leaks, and no hardcoded circuit limit.

Two failures matter here and neither one raises.

  A LEAK. Every column must be computable at the close of its own session. A
  rolling window that centres instead of trailing, or a groupby that sorts
  wrongly, silently lets a Tuesday know what Wednesday did. This is the same
  invariant tests/test_features_forward.py enforces for the index modules, so
  it is enforced the same way: build the panel twice, once on a truncated
  history, and demand the overlapping rows be identical.

  A WRONG CIRCUIT LIMIT. NEPSE widened the daily scrip limit from 10% to 15% on
  2026-04-20 (configs/market_params.yaml). A hardcoded 10% marks every ordinary
  9-12% move after that date as a limit day; a hardcoded 15% marks no limit days
  at all before it. Both produce a plausible-looking dashboard that is wrong.

Run: .venv/bin/pytest tests/test_scrip.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.eval.costs import Params  # noqa: E402
from nepselab.features import scrip  # noqa: E402


def panel_rows(symbol: str, dates: list[str], closes: list[float],
               qty: list[float] | None = None,
               trades: list[int] | None = None) -> pd.DataFrame:
    n = len(dates)
    closes = list(closes)
    prev = [closes[0]] + closes[:-1]
    return pd.DataFrame({
        "businessDate": pd.to_datetime(dates),
        "symbol": [symbol] * n,
        "closePrice": closes,
        "previousDayClosePrice": prev,
        "totalTradedQuantity": qty if qty is not None else [1000.0] * n,
        "totalTradedValue": [(q or 0) * c for q, c in
                             zip(qty if qty is not None else [1000.0] * n, closes)],
        "totalTrades": trades if trades is not None else [100] * n,
    })


def steady(symbol: str = "AAA", n: int = 30, qty: float = 1000.0,
           start: str = "2026-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d").tolist()
    return panel_rows(symbol, dates, [100.0] * n, [qty] * n)


class TestNoLeak:
    def test_truncating_the_future_does_not_change_the_past(self):
        """The whole leak test: session t must not depend on session t+1."""
        rng = np.random.default_rng(0)
        n = 60
        dates = pd.bdate_range("2026-01-01", periods=n).strftime("%Y-%m-%d").tolist()
        closes = list(100 * np.cumprod(1 + rng.normal(0, 0.02, n)))
        qty = list(rng.lognormal(7, 1, n))
        full = panel_rows("AAA", dates, closes, qty)

        built_full = scrip.build_panel(full)
        built_short = scrip.build_panel(full.iloc[:40].copy())

        cols = ["ret", "vol_ratio", "vol_median_20", "is_thin", "quadrant"]
        a = built_full.iloc[:40][cols].reset_index(drop=True)
        b = built_short[cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)

    def test_rows_arriving_out_of_order_still_produce_the_same_panel(self):
        # The archive is not guaranteed sorted, and a groupby that trusts input
        # order would compute a rolling window over shuffled sessions.
        df = steady(n=30, qty=1000.0)
        df.loc[df.index[-1], "totalTradedQuantity"] = 9000.0
        shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)

        a = scrip.build_panel(df).sort_values("businessDate").reset_index(drop=True)
        b = scrip.build_panel(shuffled).sort_values("businessDate").reset_index(drop=True)
        pd.testing.assert_series_equal(a["vol_ratio"], b["vol_ratio"])


class TestCircuitLimitIsDateAware:
    """The 2026-04-20 widening, from both sides."""

    @pytest.fixture
    def params(self) -> Params:
        return Params()

    def test_twelve_percent_is_not_a_limit_day_before_the_change(self, params):
        # 12% under the old 10% cap is impossible, but if it appears in the data
        # it is a corporate action, not a limit -- and after the change it is an
        # ordinary move. Either way the flag must follow the date.
        df = panel_rows("AAA", ["2026-03-02"], [112.0])
        df["previousDayClosePrice"] = [100.0]
        out = scrip.build_panel(df, params=params)
        assert out["circuit_limit"].iloc[0] == 0.10
        assert bool(out["at_limit_up"].iloc[0]) is True

    def test_twelve_percent_is_an_ordinary_move_after_the_change(self, params):
        df = panel_rows("AAA", ["2026-06-01"], [112.0])
        df["previousDayClosePrice"] = [100.0]
        out = scrip.build_panel(df, params=params)
        assert out["circuit_limit"].iloc[0] == 0.15
        assert bool(out["at_limit_up"].iloc[0]) is False

    def test_fifteen_percent_is_a_limit_day_only_after_the_change(self, params):
        after = panel_rows("AAA", ["2026-06-01"], [115.0])
        after["previousDayClosePrice"] = [100.0]
        assert bool(scrip.build_panel(after, params=params)["at_limit_up"].iloc[0]) is True

    def test_limit_down_uses_the_same_dated_limit(self, params):
        df = panel_rows("AAA", ["2026-03-02"], [90.0])
        df["previousDayClosePrice"] = [100.0]
        out = scrip.build_panel(df, params=params)
        assert bool(out["at_limit_down"].iloc[0]) is True
        assert bool(out["at_limit_up"].iloc[0]) is False

    def test_without_params_no_limit_is_invented(self):
        df = panel_rows("AAA", ["2026-06-01"], [115.0])
        out = scrip.build_panel(df, params=None)
        assert "at_limit_up" not in out.columns


class TestVolumeBaseline:
    def test_one_block_trade_does_not_redefine_normal(self):
        """Median, not mean: a single huge day must not raise the baseline."""
        df = steady(n=30, qty=1000.0)
        df.loc[df.index[10], "totalTradedQuantity"] = 500_000.0
        out = scrip.build_panel(df)
        # The baseline on the last day should still be ~1000, so an ordinary
        # 1000-share day reads as ordinary rather than as a collapse.
        assert out["vol_median_20"].iloc[-1] == pytest.approx(1000.0)
        assert out["vol_ratio"].iloc[-1] == pytest.approx(1.0)

    def test_a_new_listing_cannot_spike_on_day_three(self):
        df = steady(n=3, qty=1000.0)
        df.loc[df.index[-1], "totalTradedQuantity"] = 50_000.0
        out = scrip.build_panel(df)
        assert out["vol_ratio"].isna().all(), "needs 10 sessions before a ratio"

    def test_each_scrip_gets_its_own_baseline(self):
        big = steady("BIG", n=25, qty=1_000_000.0)
        small = steady("SMALL", n=25, qty=500.0)
        out = scrip.build_panel(pd.concat([big, small], ignore_index=True))
        last = out.groupby("symbol")["vol_ratio"].last()
        # A million-share scrip and a 500-share scrip are both simply "normal".
        assert last["BIG"] == pytest.approx(1.0)
        assert last["SMALL"] == pytest.approx(1.0)


class TestThinness:
    def test_flag_fires_below_ten_trades(self):
        df = panel_rows("AAA", ["2026-06-01", "2026-06-02"], [100.0, 110.0],
                        qty=[1000.0, 1000.0], trades=[9, 10])
        out = scrip.build_panel(df)
        assert list(out["is_thin"]) == [True, False]

    def test_average_trade_size_survives_a_zero_trade_day(self):
        df = panel_rows("AAA", ["2026-06-01"], [100.0], qty=[0.0], trades=[0])
        out = scrip.build_panel(df)
        assert pd.isna(out["avg_trade_size"].iloc[0])


class TestQuadrant:
    def test_labels_are_neutral_and_correct(self):
        df = steady(n=25, qty=1000.0)
        # Last day: price down, volume 3x -> the pattern from the video.
        df.loc[df.index[-1], ["closePrice", "totalTradedQuantity"]] = [95.0, 3000.0]
        out = scrip.build_panel(df)
        assert out["quadrant"].iloc[-1] == "heavy_down"

    def test_an_unchanged_price_is_flat_not_down(self):
        df = steady(n=25, qty=1000.0)
        out = scrip.build_panel(df)
        assert out["quadrant"].iloc[-1] == "flat"


class TestBreadth:
    def test_untraded_scrips_are_not_counted_as_unchanged(self):
        """A scrip that did not trade did not decline; it did not participate."""
        traded = panel_rows("AAA", ["2026-06-01"], [110.0])
        traded["previousDayClosePrice"] = [100.0]
        idle = panel_rows("ZZZ", ["2026-06-01"], [100.0], qty=[0.0], trades=[0])

        out = scrip.market_breadth(
            scrip.build_panel(pd.concat([traded, idle], ignore_index=True)))
        assert out["traded"].iloc[0] == 1
        assert out["advancers"].iloc[0] == 1
        assert out["unchanged"].iloc[0] == 0


class TestConcentration:
    def test_share_is_of_traded_value_and_bounded(self):
        rows = [panel_rows(f"S{i}", ["2026-06-01"], [100.0], qty=[float(i + 1)])
                for i in range(20)]
        out = scrip.turnover_concentration(pd.concat(rows, ignore_index=True), top_n=10)
        share = out["top10_share"].iloc[0]
        assert 0.5 < share <= 1.0
        assert "S19" in out["top10_symbols"].iloc[0]


def test_missing_columns_say_which_ones():
    with pytest.raises(ValueError, match="totalTrades"):
        scrip.build_panel(pd.DataFrame({
            "businessDate": pd.to_datetime(["2026-06-01"]),
            "symbol": ["AAA"], "closePrice": [1.0],
            "previousDayClosePrice": [1.0], "totalTradedQuantity": [1.0],
        }))
