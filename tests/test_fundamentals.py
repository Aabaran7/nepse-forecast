"""Parsing company fundamentals, and the two ways they mislead.

FIRST: A RATIO WITHOUT ITS REPORTING DATE IS A LOOK-AHEAD LEAK. MeroLagani's
page says EPS 28.36 today. Using that for a date last March means trading on
earnings nobody had. The page labels the period -- "28.36 (FY:082-083, Q:4)" --
so the parser must keep it, and `snapshot_date` records when we looked. A row
means "as of this date, the latest reported EPS was X for period P".

SECOND: A NEGATIVE P/E IS NOT A CHEAP STOCK. It is a loss-making company, and it
sorts as if it were the cheapest thing on the exchange. Including them dragged
the raw sector medians to Hotels -123.7 and Others -16.1, which invites exactly
the wrong conclusion. Excluding them moved Hydro Power from 37.4 to 59.7 and
revealed that 31 of its 110 companies lose money -- the genuinely useful fact.

Run: .venv/bin/pytest tests/test_fundamentals.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.ingest import fundamentals as f  # noqa: E402

PAGE = """
<table class="table">
  <tr><td>Sector</td><td>Commercial Banks</td></tr>
  <tr><td>Shares Outstanding</td><td>270,569,970.00</td></tr>
  <tr><td>Market Price</td><td>556.00</td></tr>
  <tr><td>52 Weeks High - Low</td><td>568.00-471.00</td></tr>
  <tr><td>180 Day Average</td><td>528.76</td></tr>
  <tr><td>EPS</td><td>28.36 (FY:082-083, Q:4)</td></tr>
  <tr><td>P/E Ratio</td><td>19.61</td></tr>
  <tr><td>Book Value</td><td>247.28</td></tr>
  <tr><td>PBV</td><td>2.25</td></tr>
  <tr><td>% Dividend</td><td>12.50 (FY:081-082)</td></tr>
  <tr><td>1.</td><td>12.50%</td><td>(FY: 081-082)</td></tr>
  <tr><td>2.</td><td>10.00%</td><td>(FY: 080-081)</td></tr>
  <tr><td>3.</td><td>0.00%</td><td>(FY: 079-080)</td></tr>
</table>"""


class TestParsing:
    @pytest.fixture
    def rec(self) -> dict:
        return f.parse(PAGE, "nabil")

    def test_symbol_is_normalised(self, rec):
        assert rec["symbol"] == "NABIL"

    def test_numbers_lose_their_commas_and_percent_signs(self, rec):
        assert rec["shares_outstanding"] == 270569970.0
        assert rec["market_price"] == 556.0
        assert rec["pe_ratio"] == 19.61

    def test_the_52_week_range_splits(self, rec):
        assert rec["week52_high"] == 568.0
        assert rec["week52_low"] == 471.0

    def test_text_fields_stay_text(self, rec):
        assert rec["sector"] == "Commercial Banks"

    def test_the_reporting_period_travels_with_the_value(self, rec):
        """The whole point. Without this, EPS cannot be used for any dated work."""
        assert rec["eps"] == 28.36
        assert rec["eps_fy"] == "082-083"
        assert rec["eps_quarter"] == 4

    def test_a_dividend_carries_its_fiscal_year_too(self, rec):
        assert rec["dividend_pct"] == 12.50
        assert rec["dividend_pct_fy"] == "081-082"

    def test_a_missing_field_is_absent_not_zero(self):
        """A bank with no reported EPS has not earned nothing."""
        rec = f.parse("<table><tr><td>Sector</td><td>Finance</td></tr></table>", "X")
        assert "eps" not in rec
        assert rec.get("pe_ratio") is None

    def test_an_unrecognisable_page_yields_only_the_symbol(self):
        rec = f.parse("<html><body>site redesigned</body></html>", "X")
        assert set(rec) == {"symbol"}


class TestDividendHistory:
    def test_it_reads_the_numbered_rows(self):
        d = f.parse_dividend_history(PAGE, "NABIL")
        assert len(d) == 3
        assert set(d["symbol"]) == {"NABIL"}
        assert d.iloc[0]["fiscal_year"] == "081-082"
        assert d.iloc[0]["dividend_pct"] == 12.50

    def test_a_zero_dividend_year_is_kept(self):
        """Paying nothing is information; dropping it fakes an unbroken record."""
        d = f.parse_dividend_history(PAGE, "NABIL")
        assert (d["dividend_pct"] == 0.0).any()

    def test_an_empty_page_gives_an_empty_frame(self):
        d = f.parse_dividend_history("<html></html>", "X")
        assert d.empty
        assert list(d.columns) == ["symbol", "fiscal_year", "dividend_pct"]


class TestSectorMedians:
    @pytest.fixture
    def df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"symbol": "A", "sector": "Hotels", "pe_ratio": -120.0, "pbv": 6.0},
            {"symbol": "B", "sector": "Hotels", "pe_ratio": -40.0, "pbv": 7.0},
            {"symbol": "C", "sector": "Hotels", "pe_ratio": 80.0, "pbv": 6.5},
            {"symbol": "D", "sector": "Banks", "pe_ratio": 18.0, "pbv": 1.3},
            {"symbol": "E", "sector": "Banks", "pe_ratio": 20.0, "pbv": 1.4},
        ])

    def test_loss_makers_are_excluded_from_the_pe_median(self, df):
        med = f.sector_medians(df).set_index("sector")
        # With them in, the median is -40. That reads as the cheapest sector
        # on the exchange, which is the opposite of the truth.
        assert med.loc["Hotels", "pe_ratio"] == 80.0

    def test_loss_makers_are_counted_rather_than_hidden(self, df):
        med = f.sector_medians(df).set_index("sector")
        assert med.loc["Hotels", "n_loss"] == 2
        assert med.loc["Hotels", "n"] == 3, "they still count towards sector size"
        assert med.loc["Banks", "n_loss"] == 0

    def test_pbv_keeps_every_company(self, df):
        # Book value stays meaningful when earnings are negative, so excluding
        # loss-makers there would discard good data.
        med = f.sector_medians(df).set_index("sector")
        assert med.loc["Hotels", "pbv"] == 6.5

    def test_an_empty_frame_is_safe(self):
        assert f.sector_medians(pd.DataFrame()).empty

    def test_a_frame_without_sector_is_safe(self):
        assert f.sector_medians(pd.DataFrame({"pe_ratio": [1.0]})).empty


def test_url_is_built_from_the_symbol():
    assert f.url_for("nabil").endswith("symbol=NABIL")
