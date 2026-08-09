"""Which headlines reach the paid model, and which get filed without scoring.

This filter spends money when it says yes and loses information when it says no,
so the two errors are not symmetric and the tests are not either.

A false positive costs a fraction of a cent. A false negative silently removes
the one story that mattered, and nothing downstream can distinguish "filtered
out" from "never happened". So the rule errs towards including, and the tag
never deletes -- every headline is archived either way, and a filtering decision
that turns out to be wrong can be rescored later.

The single-token company matcher this replaced is the cautionary tale. Listed
names in Nepal decompose into ordinary English (Premier Insurance, Central
Finance, First Micro Finance), so "Gold falls Rs 5,800" matched a company
containing "Falls" and "Three arrested in Kathmandu" matched one containing
"Three". 12 of 14 company hits were garbage. Hence: tickers only as uppercase
words, company names only as two-word phrases.

Run: .venv/bin/pytest tests/test_relevance.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.ingest import relevance  # noqa: E402


@pytest.fixture
def securities() -> pd.DataFrame:
    """A few real listings, including the ones that broke the first version."""
    return pd.DataFrame({
        "symbol": ["NABIL", "SCB", "API", "PLIC", "SHINE", "HURJA"],
        "securityName": [
            "Nabil Bank Limited",
            "Standard Chartered Bank Nepal Limited",
            "Api Power Company Limited",
            "Premier Insurance Company Nepal Limited",
            "Shine Resunga Development Bank Limited",
            "Himalayan Urja Bikas Company Limited",
        ],
    })


@pytest.fixture
def names(securities: pd.DataFrame):
    return relevance.company_names(securities)


class TestMarketSources:
    def test_sharesansar_is_never_filtered(self, names):
        ok, why = relevance.score("Some entirely generic headline", "sharesansar", names)
        assert ok and "market-only" in why

    def test_merolagani_is_never_filtered(self, names):
        ok, _ = relevance.score("जथाभावी शीर्षक", "merolagani", names)
        assert ok


class TestGeneralNewspaper:
    @pytest.mark.parametrize("title", [
        "CDSC to suspend Mero Share services for accounts with unpaid annual fees",
        "Property transaction revenue hits record high ahead of capital gains tax hike",
        "From Rs 10 crore loss to Rs 3 crore profit: Nepal Aushadhi's turnaround",
        "NRB cuts lending rate as liquidity improves",
        "Hydropower company announces bonus share for shareholders",
    ])
    def test_market_news_is_kept(self, title, names):
        ok, why = relevance.score(title, "himalayantimes", names)
        assert ok, f"wrongly skipped: {title} ({why})"

    @pytest.mark.parametrize("title", [
        "Third Nepal Premier League set for October 26-November 21",
        "Rain to continue across Nepal as far-west temperatures climb to 37C",
        "Seven arrested in Birgunj for allegedly stealing fuel from tankers",
        "Bodies spotted at 5,000m on Yalung Ri, weather halts recovery",
        "British Council to host South Asia English Teaching Conference",
    ])
    def test_obvious_noise_is_skipped(self, title, names):
        ok, why = relevance.score(title, "himalayantimes", names)
        assert not ok, f"wrongly kept: {title} ({why})"


class TestTheSingleTokenBug:
    """Each of these matched a real company token before the fix."""

    @pytest.mark.parametrize("title", [
        "Gold falls Rs 5,800, silver Rs 155 on Friday",          # "falls"
        "Three arrested in Kathmandu for online betting",         # "three"
        "The first few hours can decide a snakebite victim's fate",  # "first"
        "Nepal's Shinta Mani Mustang crowned world's best luxury hotel",  # "hotel"
    ])
    def test_generic_words_from_company_names_do_not_match(self, title, names):
        ok, why = relevance.score(title, "himalayantimes", names)
        assert not ok, f"single-token match is back: {title} ({why})"

    def test_a_real_company_phrase_still_matches(self, names):
        ok, why = relevance.score(
            "Standard Chartered reaffirms commitment to Nepal's growth",
            "himalayantimes", names)
        assert ok and "standard chartered" in why

    def test_a_ticker_matches_only_in_uppercase(self, names):
        # API is a listed symbol AND an everyday lowercase word.
        ok, why = relevance.score("NABIL posts record quarterly profit",
                                  "himalayantimes", names)
        assert ok
        ok2, _ = relevance.score("The api documentation was updated today",
                                 "himalayantimes", names)
        assert not ok2, "lowercase 'api' must not match the ticker API"


class TestTagging:
    def test_it_adds_columns_and_removes_nothing(self, securities):
        df = pd.DataFrame({
            "title": ["Third Nepal Premier League set for October",
                      "NRB cuts lending rate"],
            "source": ["himalayantimes", "himalayantimes"],
        })
        out = relevance.tag(df, securities)
        assert len(out) == len(df), "tagging must never drop a row"
        assert list(out["market_relevant"]) == [False, True]
        assert out["relevance_reason"].notna().all(), "every decision needs a reason"

    def test_an_empty_frame_survives(self, securities):
        assert relevance.tag(pd.DataFrame(), securities).empty

    def test_it_works_with_no_securities_snapshot(self):
        # A fresh checkout has no archive yet. Keyword matching must still run.
        df = pd.DataFrame({"title": ["NRB cuts lending rate"],
                           "source": ["himalayantimes"]})
        out = relevance.tag(df, pd.DataFrame())
        assert bool(out["market_relevant"].iloc[0])


def test_nepali_headlines_are_not_dismissed_for_being_nepali(names):
    ok, why = relevance.score("बैंकको नाफा बढ्यो", "himalayantimes", names)
    assert ok, f"Nepali market news skipped ({why})"
