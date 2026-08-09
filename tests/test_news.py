"""Headline scraping, and the one part of it that can leak.

Parsing is the visible half of this module and the boring half: if a selector
breaks, the scrape returns nothing and `SourceEmpty` says so loudly. The half
worth testing hard is `session_for()`, because its failure mode is silent and
directional -- attributing an after-close headline to the session that already
closed hands a model information it could not have had, and the result is a
backtest that looks better than the strategy is. Plan §7 calls that the exact
failure the forward log exists to detect; catching it here is cheaper.

The calendar cases are not hypothetical. NEPSE's trading week changed twice
inside the sample (configs/market_params.yaml): a six-day week in mid-2022, and
a Sun-Thu -> Mon-Fri switch in April 2026. Any rule that hardcodes "next weekday"
is wrong on both sides of those boundaries, so attribution reads the calendar.

Run: .venv/bin/pytest tests/test_news.py -q
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.ingest import news  # noqa: E402

# A Sun-Thu week (the pre-2026-04 regime), with Thursday 2026-01-08 followed by
# Sunday 2026-01-11 -- a three-day gap that a weekday-arithmetic rule gets wrong.
SUN_THU = [date(2026, 1, 4), date(2026, 1, 5), date(2026, 1, 6),
           date(2026, 1, 7), date(2026, 1, 8), date(2026, 1, 11)]


def ktm(day: str, hour: int, minute: int = 0) -> pd.Timestamp:
    """A Kathmandu wall-clock time, as the timezone-aware stamp the scraper sees."""
    return pd.Timestamp(f"{day} {hour:02d}:{minute:02d}", tz=news.KTM)


class TestSessionAttribution:
    def test_before_close_is_same_session(self):
        assert news.session_for(ktm("2026-01-06", 11), SUN_THU) == date(2026, 1, 6)

    def test_at_the_close_is_already_too_late(self):
        # 15:00 sharp: the close has happened, so the news cannot have moved it.
        # The boundary goes this way deliberately -- the conservative direction is
        # forward, never back into a session that already printed.
        assert news.session_for(ktm("2026-01-06", 15), SUN_THU) == date(2026, 1, 7)

    def test_after_close_rolls_to_next_session(self):
        assert news.session_for(ktm("2026-01-06", 17, 30), SUN_THU) == date(2026, 1, 7)

    def test_weekend_gap_uses_the_calendar_not_the_weekday(self):
        # Thursday after close, and the next session is SUNDAY, not Friday.
        assert news.session_for(ktm("2026-01-08", 18), SUN_THU) == date(2026, 1, 11)
        # A story filed on the closed Friday belongs to Sunday too.
        assert news.session_for(ktm("2026-01-09", 10), SUN_THU) == date(2026, 1, 11)

    def test_holiday_is_skipped_because_it_is_not_in_the_calendar(self):
        calendar = [d for d in SUN_THU if d != date(2026, 1, 7)]
        assert news.session_for(ktm("2026-01-06", 17), calendar) == date(2026, 1, 8)

    def test_unknown_future_session_is_none_not_a_guess(self):
        # Scraped after the last session we know about. The honest answer is "not
        # yet", resolved on a later run -- not an invented next trading day.
        assert news.session_for(ktm("2026-01-11", 17), SUN_THU) is None

    def test_naive_timestamps_are_read_as_utc(self):
        # 12:00 UTC is 17:45 NPT -- after the close, so the NEXT session.
        # Read as local time it would be before the close, and would leak.
        assert news.session_for(pd.Timestamp("2026-01-06 12:00"), SUN_THU) == date(2026, 1, 7)


class TestAttributeFrame:
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"source": "s", "url_hash": "a", "url": "u1", "title": "t1",
             "published": pd.Timestamp("2026-01-06 05:00", tz="UTC"),   # 10:45 NPT
             "first_seen_utc": "2026-01-06T12:00:00+00:00"},
            {"source": "s", "url_hash": "b", "url": "u2", "title": "t2",
             "published": pd.NaT,
             "first_seen_utc": "2026-01-06T12:00:00+00:00"},            # 17:45 NPT
        ])

    def test_published_wins_over_first_seen(self):
        out = news.attribute(self.frame(), SUN_THU)
        # Published intraday -> same session; unknown publication falls back to
        # the scrape time, which is after the close -> next session.
        assert list(out["session"]) == [date(2026, 1, 6), date(2026, 1, 7)]

    def test_empty_frame_survives(self):
        assert news.attribute(pd.DataFrame(), SUN_THU).empty


class TestParsing:
    def test_url_hash_ignores_www_query_and_trailing_slash(self):
        h = news.url_hash("https://www.sharesansar.com/newsdetail/abc")
        assert h == news.url_hash("https://sharesansar.com/newsdetail/abc/")
        assert h == news.url_hash("https://www.sharesansar.com/newsdetail/abc?utm=x")
        assert h != news.url_hash("https://www.sharesansar.com/newsdetail/abd")

    def test_sharesansar_layout(self):
        html = """
        <div class="featured-news-list margin-bottom-15">
          <div class="col-md-10">
            <a href="/newsdetail/hulas-finserv-profit">
              <h4 class="featured-news-title">Hulas Finserv Reports 84.45% Surge</h4>
            </a>
            <p><span class="text-org">Sunday, August 9, 2026</span></p>
          </div>
        </div>"""
        rows = news.parse(news.SOURCES["sharesansar"], html, "https://www.sharesansar.com/x")
        assert len(rows) == 1
        assert rows[0]["title"] == "Hulas Finserv Reports 84.45% Surge"
        assert rows[0]["url"] == "https://www.sharesansar.com/newsdetail/hulas-finserv-profit"
        assert rows[0]["published"] == pd.Timestamp("2026-08-09")

    def test_nepali_headline_survives_parsing(self):
        # MeroLagani is Devanagari. Nothing may normalise, strip or mangle it --
        # the sentiment pass reads exactly what is stored here.
        title = "सुनचाँदीको मूल्यमा भारि वृद्धि"
        html = f'<h4 class="media-title"><a href="/NewsDetail.aspx?newsID=1">{title}</a></h4>'
        rows = news.parse(news.SOURCES["merolagani"], html, "https://merolagani.com/x")
        assert rows[0]["title"] == title

    def test_empty_parse_raises_rather_than_returning_nothing(self):
        # The whole point: a layout change must not be indistinguishable from a
        # quiet news day. Zero rows is an error, not a result.
        class Never:
            def get(self, url): return "<html><body>redesigned</body></html>"

        with pytest.raises(news.SourceEmpty):
            news.scrape_source(news.SOURCES["sharesansar"], Never(), pages=1)

    def test_one_source_failing_does_not_kill_the_others(self):
        class OnlySharesansar:
            def get(self, url):
                if "sharesansar" in url:
                    return ('<div class="featured-news-list">'
                            '<a href="/newsdetail/x">'
                            '<h4 class="featured-news-title">A headline here</h4></a></div>')
                raise ConnectionError("down")

        df, errors = news.scrape(fetcher=OnlySharesansar())
        assert len(df) == 1
        assert set(errors) == {"merolagani", "himalayantimes"}
