"""The archive's append-only guarantee, tested directly.

This is the highest-stakes invariant in the project. NEPSE retains one rolling
year (plan §3.4), so a bug that drops an archived row destroys data that cannot
be re-fetched from anywhere. Losing rows must be impossible, not merely unlikely
-- which is why these tests assert on the adversarial cases (upstream revising a
past session, upstream returning a truncated window) rather than the happy path.

Run: .venv/bin/pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.ingest import archive  # noqa: E402


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "archive"


def idx_rows(dates: list[str], close: float = 100.0, index_id: int = 58) -> pd.DataFrame:
    return pd.DataFrame({
        "businessDate": pd.to_datetime(dates),
        "exchangeIndexId": index_id,
        "closingIndex": close,
    })


def test_first_write_creates_dataset(root: Path) -> None:
    r = archive.merge("indices", idx_rows(["2026-01-01", "2026-01-02"]), root=root)
    assert (r.added, r.total) == (2, 2)


def test_reingesting_identical_data_adds_nothing(root: Path) -> None:
    """Idempotence: the daily job re-pulls the same window constantly."""
    df = idx_rows(["2026-01-01", "2026-01-02"])
    archive.merge("indices", df, root=root)
    r = archive.merge("indices", df, root=root)
    assert (r.added, r.total, len(r.conflicts)) == (0, 2, 0)


def test_new_sessions_append_without_disturbing_old(root: Path) -> None:
    archive.merge("indices", idx_rows(["2026-01-01"]), root=root)
    r = archive.merge("indices", idx_rows(["2026-01-02", "2026-01-03"]), root=root)
    assert (r.added, r.total) == (2, 3)
    assert set(archive.load("indices", root=root)["businessDate"].astype(str).str[:10]) == {
        "2026-01-01", "2026-01-02", "2026-01-03"}


def test_rolled_off_sessions_survive_a_truncated_upstream(root: Path) -> None:
    """The whole point of the archive.

    Upstream's window advances and no longer contains the oldest sessions. A
    merge of that shorter window must not remove them -- this is precisely how
    the five sessions lost between 2026-07-25 and 2026-08-02 would have been
    saved, and it is the case a naive overwrite-on-write cache gets wrong.
    """
    archive.merge("indices", idx_rows(["2026-01-01", "2026-01-02", "2026-01-03"]), root=root)
    archive.merge("indices", idx_rows(["2026-01-03", "2026-01-04"]), root=root)
    got = set(archive.load("indices", root=root)["businessDate"].astype(str).str[:10])
    assert "2026-01-01" in got and "2026-01-02" in got
    assert len(got) == 4


def test_upstream_revision_is_rejected_and_reported(root: Path) -> None:
    """Ours wins; the disagreement is surfaced rather than silently applied."""
    archive.merge("indices", idx_rows(["2026-01-01"], close=100.0), root=root)
    r = archive.merge("indices", idx_rows(["2026-01-01"], close=999.0), root=root)

    assert r.added == 0
    assert len(r.conflicts) == 1, "a changed value must be flagged, not absorbed"
    assert "closingIndex" in r.conflicts.iloc[0]["changed_columns"]
    stored = archive.load("indices", root=root)["closingIndex"].iloc[0]
    assert stored == 100.0, "the archived value must win over upstream"


def test_identical_values_are_not_flagged_as_conflicts(root: Path) -> None:
    """Guards the conflict detector against crying wolf on every daily run."""
    df = idx_rows(["2026-01-01"], close=100.0)
    archive.merge("indices", df, root=root)
    r = archive.merge("indices", df.copy(), root=root)
    assert len(r.conflicts) == 0


def test_timestamp_and_date_forms_are_the_same_session(root: Path) -> None:
    """today_price returns dates; history returns timestamps. One session, one row."""
    archive.merge("indices", idx_rows(["2026-01-01"]), root=root)
    same = pd.DataFrame({"businessDate": [pd.Timestamp("2026-01-01 15:30:00")],
                         "exchangeIndexId": [58], "closingIndex": [100.0]})
    r = archive.merge("indices", same, root=root)
    assert r.added == 0, "a timestamped pull must not duplicate a dated one"


def test_distinct_indices_do_not_collide_on_date(root: Path) -> None:
    """All 17 indices share a date axis; only (index, date) identifies a row."""
    archive.merge("indices", idx_rows(["2026-01-01"], index_id=58), root=root)
    r = archive.merge("indices", idx_rows(["2026-01-01"], index_id=57), root=root)
    assert (r.added, r.total) == (1, 2)


def test_empty_pull_is_a_no_op(root: Path) -> None:
    """A holiday, or a throttled run, must never empty the archive."""
    archive.merge("indices", idx_rows(["2026-01-01"]), root=root)
    r = archive.merge("indices", pd.DataFrame(), root=root)
    assert (r.added, r.total) == (0, 1)
    assert len(archive.load("indices", root=root)) == 1


def test_missing_key_column_is_rejected(root: Path) -> None:
    """Fail loudly rather than archive rows that can never be deduplicated."""
    bad = pd.DataFrame({"businessDate": pd.to_datetime(["2026-01-01"]), "closingIndex": [1.0]})
    with pytest.raises(ValueError, match="key column"):
        archive.merge("indices", bad, root=root)


def test_unknown_dataset_is_rejected(root: Path) -> None:
    with pytest.raises(KeyError):
        archive.merge("not_a_dataset", idx_rows(["2026-01-01"]), root=root)


def test_duplicate_keys_within_one_pull_collapse(root: Path) -> None:
    df = pd.concat([idx_rows(["2026-01-01"]), idx_rows(["2026-01-01"])], ignore_index=True)
    r = archive.merge("indices", df, root=root)
    assert (r.added, r.total) == (1, 1)
