"""In-progress sessions, and the one write that is allowed to remove data.

Both halves of a real incident, 2026-08-04. A pull ran while the session was
open. NEPSE returned open=high=low with `closingIndex` 0 for all 17 indices,
109 of ~350 scrips, and 1.2% of a normal day's turnover. All of it was archived
as final.

The archive then made it permanent. merge() keeps the row it already holds and
records the incoming one as a conflict (§3.4) -- correct against an upstream
revision, catastrophic when our own capture was the broken one. Every pull
afterwards re-fetched the true closes and threw them away, logging 17 conflicts
a run, and no amount of re-pulling could ever fix it.

So there are two things to test and they pull in opposite directions:

  PREVENTION must be conservative. Withholding a good session costs one day of
  lag; archiving a bad one costs forever.

  REPAIR must be hard to do by accident. It is the only code here that deletes,
  and an unexplained deletion is indistinguishable from the loss the archive
  exists to prevent.

Run: .venv/bin/pytest tests/test_repair.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nepselab.ingest import archive  # noqa: E402
import archive_pull  # noqa: E402


def idx_rows(date: str, close: float, open_: float = 100.0, n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "businessDate": pd.to_datetime([date] * n),
        "exchangeIndexId": list(range(51, 51 + n)),
        "openIndex": [open_] * n,
        "highIndex": [open_] * n,
        "lowIndex": [open_] * n,
        "closingIndex": [close] * n,
    })


class TestDetectingAnOpenSession:
    def test_zero_close_with_a_real_open_is_in_progress(self):
        df = idx_rows("2026-08-04", close=0.0, open_=2685.41)
        assert archive_pull.incomplete_sessions(df) == {pd.Timestamp("2026-08-04")}

    def test_a_finished_session_is_left_alone(self):
        df = idx_rows("2026-08-05", close=2667.97, open_=2666.95)
        assert archive_pull.incomplete_sessions(df) == set()

    def test_an_all_zero_row_is_not_treated_as_in_progress(self):
        # open 0 AND close 0 is a different animal -- a placeholder or a holiday,
        # not a session caught mid-flight. Withholding it forever would be its
        # own silent gap, so the open>0 condition is load-bearing.
        df = idx_rows("2026-08-04", close=0.0, open_=0.0)
        assert archive_pull.incomplete_sessions(df) == set()

    def test_only_the_open_session_is_withheld(self):
        df = pd.concat([idx_rows("2026-08-04", 0.0, 2685.41),
                        idx_rows("2026-08-05", 2667.97, 2666.95)], ignore_index=True)
        skip = archive_pull.incomplete_sessions(df)
        kept = archive_pull.drop_incomplete(df, skip, "businessDate", "indices")
        assert set(pd.to_datetime(kept["businessDate"])) == {pd.Timestamp("2026-08-05")}

    def test_the_whole_date_is_withheld_across_datasets(self):
        """A session is atomic. today_price has no tell of its own."""
        skip = {pd.Timestamp("2026-08-04")}
        tp = pd.DataFrame({
            "businessDate": pd.to_datetime(["2026-08-04", "2026-08-05"]),
            "securityId": [131, 131], "closePrice": [500.0, 505.0],
        })
        kept = archive_pull.drop_incomplete(tp, skip, "businessDate", "today_price")
        assert list(pd.to_datetime(kept["businessDate"])) == [pd.Timestamp("2026-08-05")]

    def test_no_skips_means_nothing_is_dropped(self):
        tp = pd.DataFrame({"businessDate": pd.to_datetime(["2026-08-05"]),
                           "securityId": [131], "closePrice": [1.0]})
        assert len(archive_pull.drop_incomplete(tp, set(), "businessDate", "x")) == 1


class TestRepair:
    @pytest.fixture
    def root(self, tmp_path: Path) -> Path:
        r = tmp_path / "archive"
        archive.merge("indices", pd.concat([
            idx_rows("2026-08-04", close=0.0, open_=2685.41),
            idx_rows("2026-08-05", close=2667.97, open_=2666.95),
        ], ignore_index=True), root=r)
        return r

    def test_it_replaces_only_the_named_date(self, root: Path):
        fixed = idx_rows("2026-08-04", close=2663.18, open_=2685.41)
        archive.repair("indices", fixed, [pd.Timestamp("2026-08-04")],
                       reason="captured mid-session", root=root)

        got = archive.load("indices", root=root)
        d4 = got[pd.to_datetime(got["businessDate"]) == "2026-08-04"]
        d5 = got[pd.to_datetime(got["businessDate"]) == "2026-08-05"]
        assert set(d4["closingIndex"]) == {2663.18}
        assert set(d5["closingIndex"]) == {2667.97}, "untouched date was modified"
        assert len(got) == 6, "row count changed"

    def test_the_removed_rows_are_written_down_first(self, root: Path):
        archive.repair("indices", idx_rows("2026-08-04", 2663.18),
                       [pd.Timestamp("2026-08-04")],
                       reason="captured mid-session; closingIndex 0", root=root)

        saved = list((root / "_repairs").glob("indices_*.csv"))
        assert len(saved) == 1, "the original observation was not preserved"
        df = pd.read_csv(saved[0])
        assert set(df["closingIndex"]) == {0.0}, "wrong rows preserved"
        assert "captured mid-session" in df["_reason"].iloc[0]

    def test_it_refuses_without_a_reason(self, root: Path):
        with pytest.raises(archive.RepairRefused, match="reason"):
            archive.repair("indices", idx_rows("2026-08-04", 1.0),
                           [pd.Timestamp("2026-08-04")], reason="  ", root=root)
        # and nothing moved
        got = archive.load("indices", root=root)
        assert set(got[pd.to_datetime(got["businessDate"]) == "2026-08-04"]["closingIndex"]) == {0.0}

    def test_there_is_no_bulk_repair(self, root: Path):
        with pytest.raises(archive.RepairRefused, match="explicit dates"):
            archive.repair("indices", idx_rows("2026-08-04", 1.0), [],
                           reason="fix everything", root=root)

    def test_incoming_rows_for_other_dates_are_ignored(self, root: Path):
        """A repair scoped to one date must not smuggle in another."""
        mixed = pd.concat([idx_rows("2026-08-04", 2663.18),
                           idx_rows("2026-08-06", 9999.0)], ignore_index=True)
        archive.repair("indices", mixed, [pd.Timestamp("2026-08-04")],
                       reason="mid-session capture", root=root)
        got = archive.load("indices", root=root)
        assert (pd.to_datetime(got["businessDate"]) == "2026-08-06").sum() == 0


def test_the_2026_08_04_pattern_end_to_end(tmp_path: Path):
    """The incident, start to finish: bad capture, then guard, then repair."""
    root = tmp_path / "archive"

    # 1. A mid-session pull archives zeros, exactly as happened.
    bad = idx_rows("2026-08-04", close=0.0, open_=2685.41)
    archive.merge("indices", bad, root=root)

    # 2. Every later pull is rejected by append-only -- the bug's real sting.
    good = idx_rows("2026-08-04", close=2663.18, open_=2685.41)
    res = archive.merge("indices", good, root=root)
    assert res.added == 0
    assert len(res.conflicts) == 3, "conflicts should be recorded, not applied"
    assert set(archive.load("indices", root=root)["closingIndex"]) == {0.0}

    # 3. The guard would have prevented the whole thing.
    assert archive_pull.incomplete_sessions(bad) == {pd.Timestamp("2026-08-04")}
    assert archive_pull.drop_incomplete(bad, {pd.Timestamp("2026-08-04")},
                                        "businessDate", "indices").empty

    # 4. And the repair undoes the damage already done.
    archive.repair("indices", good, [pd.Timestamp("2026-08-04")],
                   reason="mid-session capture", root=root)
    assert set(archive.load("indices", root=root)["closingIndex"]) == {2663.18}
