"""The today_price queue must be ordered by expiry risk (plan §3.4).

This is a test about a deadline, not about correctness in the usual sense: the
old newest-first queue fetched perfectly good data in a perfectly sensible
order, and would have quietly let the oldest sessions roll out of NEPSE's
window while it worked backwards toward them. Nothing would have errored.

Run: .venv/bin/pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nepselab.ingest import archive  # noqa: E402
import archive_pull  # noqa: E402


class FakeClient:
    """Stands in for NepseClient: index history is the trading calendar."""

    def __init__(self, dates: list[str]):
        self._dates = pd.to_datetime(dates)

    def index_history(self, *_args, **_kwargs) -> pd.DataFrame:
        return pd.DataFrame({"businessDate": self._dates})


@pytest.fixture
def no_archive(monkeypatch):
    monkeypatch.setattr(archive, "load", lambda *_a, **_k: pd.DataFrame())


@pytest.fixture
def archived(monkeypatch):
    def _set(dates: list[str]):
        monkeypatch.setattr(
            archive, "load",
            lambda *_a, **_k: pd.DataFrame({"businessDate": pd.to_datetime(dates)}))
    return _set


SESSIONS = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]


def test_newest_sessions_lead_so_a_backfill_never_delays_today(no_archive):
    out = archive_pull.sessions_to_pull(FakeClient(SESSIONS), backfill=False)
    assert out[:2] == ["2026-01-09", "2026-01-08"]


def test_remainder_is_oldest_first_because_that_is_expiry_order(no_archive):
    out = archive_pull.sessions_to_pull(FakeClient(SESSIONS), backfill=False)
    assert out[2:] == ["2026-01-05", "2026-01-06", "2026-01-07"]


def test_every_session_appears_exactly_once(no_archive):
    out = archive_pull.sessions_to_pull(FakeClient(SESSIONS), backfill=False)
    assert sorted(out) == SESSIONS
    assert len(out) == len(set(out))


def test_already_archived_sessions_are_skipped_in_incremental_mode(archived):
    archived(["2026-01-05", "2026-01-06"])
    out = archive_pull.sessions_to_pull(FakeClient(SESSIONS), backfill=False)
    assert out == ["2026-01-09", "2026-01-08", "2026-01-07"]


def test_backfill_requeues_everything_regardless_of_what_is_archived(archived):
    archived(SESSIONS)
    out = archive_pull.sessions_to_pull(FakeClient(SESSIONS), backfill=True)
    assert sorted(out) == SESSIONS


def test_the_oldest_session_is_never_last(no_archive):
    """The regression, stated directly. Under the old newest-first queue the
    oldest -- i.e. the one closest to falling out of the rolling window -- was
    fetched last, which is the worst possible order for a race against expiry."""
    out = archive_pull.sessions_to_pull(FakeClient(SESSIONS), backfill=False)
    assert out[-1] != "2026-01-05"
    assert out.index("2026-01-05") < out.index("2026-01-07")


def test_short_calendar_does_not_lose_sessions(no_archive):
    """fresh_first=2 against a 1-session calendar must not drop or duplicate."""
    out = archive_pull.sessions_to_pull(FakeClient(["2026-01-09"]), backfill=False)
    assert out == ["2026-01-09"]


def test_empty_calendar_yields_nothing(no_archive):
    assert archive_pull.sessions_to_pull(FakeClient([]), backfill=False) == []


def test_fresh_first_zero_gives_pure_oldest_first(no_archive):
    """Not used by callers, but `want[-0:]` is the whole list rather than an
    empty one, so the zero case silently duplicated every session."""
    out = archive_pull.sessions_to_pull(FakeClient(SESSIONS), backfill=False,
                                        fresh_first=0)
    assert out == SESSIONS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
