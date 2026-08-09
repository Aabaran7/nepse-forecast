"""Turning what the model says into a ticker. No API calls here.

The model's job is to isolate the company from the sentence -- the part string
rules failed at (5 of 42 matched, and one match was "Mero Share", the CDSC
website, mapped to the ticker MERO). Deciding whether that company is LISTED is
not its job and it is bad at it: asking for "the listed company" made it return
null for Standard Chartered, which is listed as SCB.

So the split is: the model proposes a name, this code checks it against the
exchange's own securities list, and anything that does not match is dropped.
A hallucinated ticker cannot reach the panel because the model never supplies
one.

Run: .venv/bin/pytest tests/test_sentiment.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import score_sentiment as ss  # noqa: E402


@pytest.fixture
def securities() -> pd.DataFrame:
    """Real shapes, including the one-name-many-symbols case."""
    return pd.DataFrame({
        "symbol": ["KBL", "KBLD86", "KBLD89", "KEF", "KSY",
                   "SCB", "SCBD", "MLBL", "MLBLD89", "PMLI", "NABIL"],
        "securityName": [
            "Kumari Bank Limited", "Kumari Bank Limited", "Kumari Bank Limited",
            "Kumari Bank Limited", "Kumari Bank Limited",
            "Standard Chartered Bank  Nepal Limited",
            "Standard Chartered Bank  Nepal Limited",
            "Mahalaxmi Bikas Bank Ltd.", "Mahalaxmi Bikas Bank Ltd.",
            "Prabhu Mahalaxmi Life Insurance Limited",
            "Nabil Bank Limited",
        ],
    })


class TestResolvingASymbol:
    def test_a_plain_company_name_resolves(self, securities):
        assert ss.resolve_symbol("Nabil Bank", securities) == "NABIL"

    def test_standard_chartered_resolves(self, securities):
        # The case v1's prompt lost entirely by asking the model to judge listing.
        assert ss.resolve_symbol("Standard Chartered", securities) == "SCB"

    def test_equity_beats_debentures_and_funds(self, securities):
        """Kumari Bank is 5 symbols here. News about the bank means the shares."""
        assert ss.resolve_symbol("Kumari Bank", securities) == "KBL"

    def test_the_result_does_not_depend_on_row_order(self, securities):
        shuffled = securities.sample(frac=1.0, random_state=3).reset_index(drop=True)
        assert ss.resolve_symbol("Kumari Bank", shuffled) == "KBL"
        assert ss.resolve_symbol("Mahalaxmi Bikas Bank", shuffled) == "MLBL"

    def test_a_single_shared_word_is_not_enough(self, securities):
        # "Mahalaxmi" alone appears in both MLBL and PMLI; one token must not pick.
        assert ss.resolve_symbol("Mahalaxmi", securities) is None

    def test_more_specific_name_wins_over_a_shared_word(self, securities):
        assert ss.resolve_symbol("Prabhu Mahalaxmi Life Insurance", securities) == "PMLI"

    @pytest.mark.parametrize("company", [
        None, "", "Mero Share", "Hulas Finserv", "Greenply Nepal", "YADEA", "BYD",
    ])
    def test_unlisted_or_bogus_names_resolve_to_nothing(self, company, securities):
        """Mero Share is a website. The rest are real firms that are not listed."""
        assert ss.resolve_symbol(company, securities) is None

    def test_legal_suffixes_do_not_create_matches(self, securities):
        # "Limited" and "Nepal" appear in nearly every listed name; matching on
        # them would make any company resolve to an arbitrary one.
        assert ss.resolve_symbol("Some Unlisted Nepal Limited", securities) is None

    def test_an_empty_securities_list_is_safe(self):
        assert ss.resolve_symbol("Nabil Bank", pd.DataFrame()) is None


class TestCaching:
    def test_scored_hashes_are_scoped_to_the_version(self, tmp_path, monkeypatch):
        """A prompt change must re-score; a re-run must not."""
        from nepselab.ingest import archive
        root = tmp_path / "news"
        archive.merge("sentiment", pd.DataFrame([
            {"url_hash": "aaa", "scorer_version": "v1", "sentiment": "bullish"},
            {"url_hash": "bbb", "scorer_version": "v1", "sentiment": "neutral"},
            {"url_hash": "aaa", "scorer_version": "v2", "sentiment": "bullish"},
        ]), root=root)
        monkeypatch.setattr(ss, "NEWS_ROOT", root)

        assert ss.already_scored("v1") == {"aaa", "bbb"}
        assert ss.already_scored("v2") == {"aaa"}, "v2 must not inherit v1's work"
        assert ss.already_scored("v3") == set(), "a new version re-scores everything"


class TestEnvLoading:
    def test_it_reads_a_key_and_ignores_comments(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("# a comment\n\nOPENAI_API_KEY=sk-test-123\nBLANK=\n")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        ss.load_env(env)
        assert os_environ_get("OPENAI_API_KEY") == "sk-test-123"

    def test_a_real_environment_variable_wins(self, tmp_path, monkeypatch):
        # setdefault, not overwrite: CI supplies the key as a secret, and a
        # stray .env in a checkout must not silently replace it.
        env = tmp_path / ".env"
        env.write_text("OPENAI_API_KEY=from-file\n")
        monkeypatch.setenv("OPENAI_API_KEY", "from-environment")
        ss.load_env(env)
        assert os_environ_get("OPENAI_API_KEY") == "from-environment"

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        ss.load_env(tmp_path / "nope.env")


def os_environ_get(k: str) -> str | None:
    import os
    return os.environ.get(k)
