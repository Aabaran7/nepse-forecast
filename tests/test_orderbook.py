"""Order book capture: the only source in this project with no history at all.

`today_price` and the index histories serve a rolling year, which is bad enough
(§3.4). The book serves NOW -- NEPSE returns empty lists outside the session, so
a snapshot not taken never existed and no backfill can ever recover it.

Two consequences drive these tests.

  A PAYLOAD MUST SURVIVE A SCHEMA IT CANNOT PARSE. The field names were unknown
  when this was written: every probe hit a closed market. So the raw JSON is
  written BEFORE parsing, and an unrecognised schema degrades to "raw kept,
  table not updated". Being unable to read it today is recoverable; not having
  it is not.

  TWO CAPTURES ARE TWO OBSERVATIONS. The same scrip appears on both sides many
  times a session. Keying on the capture instant is what stops the second
  snapshot of the day from looking like a revision of the first.

Run: .venv/bin/pytest tests/test_orderbook.py -q
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import snapshot_orderbook as ob  # noqa: E402
from nepselab.ingest import archive  # noqa: E402

KTM = ob.KTM


def ktm(day: str, hh: int, mm: int = 0) -> datetime:
    return datetime.fromisoformat(f"{day}T{hh:02d}:{mm:02d}:00").replace(tzinfo=KTM)


class TestMarketHours:
    @pytest.mark.parametrize("hh,mm", [(11, 0), (12, 30), (14, 59), (15, 10)])
    def test_inside_the_session(self, hh, mm):
        assert ob.market_hours(ktm("2026-08-10", hh, mm))   # a Monday

    @pytest.mark.parametrize("hh,mm", [(1, 45), (9, 0), (16, 0), (22, 0)])
    def test_outside_the_session(self, hh, mm):
        assert not ob.market_hours(ktm("2026-08-10", hh, mm))

    def test_weekends_are_closed(self):
        # Trading week is Mon-Fri since 2026-04-10; 2026-08-08/09 is Sat/Sun.
        assert not ob.market_hours(ktm("2026-08-08", 12, 0))
        assert not ob.market_hours(ktm("2026-08-09", 12, 0))

    def test_the_bell_is_straddled_not_refused(self):
        # A capture landing a few minutes either side of the bell is worth
        # keeping; GitHub delays scheduled runs and a strict window would
        # silently drop the open and close snapshots.
        assert ob.market_hours(ktm("2026-08-10", 10, 50))
        assert ob.market_hours(ktm("2026-08-10", 15, 12))


class TestRawFirst:
    def test_the_payload_is_written_before_parsing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ob, "RAW_DIR", tmp_path / "_raw")
        payload = {"supplyList": [{"weird": 1}], "demandList": []}
        p = ob.save_raw(payload, "2026-08-10T06:30:00+00:00")
        assert p.exists()
        assert json.loads(p.read_text()) == payload

    def test_an_empty_book_is_still_recorded(self, tmp_path, monkeypatch):
        """Knowing the book WAS empty at 09:00 is itself an observation."""
        monkeypatch.setattr(ob, "RAW_DIR", tmp_path / "_raw")
        p = ob.save_raw({"supplyList": [], "demandList": []}, "2026-08-10T03:00:00+00:00")
        assert json.loads(p.read_text()) == {"supplyList": [], "demandList": []}

    def test_an_unknown_schema_keeps_the_raw_and_does_not_raise(
            self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(ob, "BOOK_DIR", tmp_path / "book")
        df = pd.DataFrame([{"totallyUnexpected": 1, "side": "supply",
                            "captured_utc": "2026-08-10T06:30:00+00:00"}])
        ok = ob.store(df, tmp_path / "raw.json")
        assert ok is False, "must not claim success on an unreadable schema"
        assert not (tmp_path / "book" / "orderbook.parquet").exists()


class TestFraming:
    def test_both_sides_are_labelled_and_stamped(self):
        payload = {
            "supplyList": [{"symbol": "NABIL", "qty": 100}],
            "demandList": [{"symbol": "NABIL", "qty": 250}],
        }
        df = ob.to_frame(payload, "2026-08-10T06:30:00+00:00")
        assert set(df["side"]) == {"supply", "demand"}
        assert (df["captured_utc"] == "2026-08-10T06:30:00+00:00").all()

    def test_an_empty_payload_frames_to_nothing(self):
        assert ob.to_frame({"supplyList": [], "demandList": []}, "x").empty

    def test_missing_keys_do_not_raise(self):
        assert ob.to_frame({}, "x").empty


class TestTwoCapturesAreTwoObservations:
    def test_a_later_snapshot_does_not_overwrite_an_earlier_one(self, tmp_path):
        root = tmp_path / "book"
        first = pd.DataFrame([{"captured_utc": "2026-08-10T06:30:00+00:00",
                               "side": "demand", "symbol": "NABIL", "qty": 100}])
        second = pd.DataFrame([{"captured_utc": "2026-08-10T07:30:00+00:00",
                                "side": "demand", "symbol": "NABIL", "qty": 400}])
        archive.merge("orderbook", first, root=root)
        res = archive.merge("orderbook", second, root=root)

        assert res.added == 1, "the second capture must be a new row"
        got = archive.load("orderbook", root=root)
        assert sorted(got["qty"]) == [100, 400]
        assert len(res.conflicts) == 0, "different instants are not a conflict"

    def test_re_storing_the_same_capture_is_a_no_op(self, tmp_path):
        root = tmp_path / "book"
        row = pd.DataFrame([{"captured_utc": "2026-08-10T06:30:00+00:00",
                             "side": "supply", "symbol": "NABIL", "qty": 100}])
        archive.merge("orderbook", row, root=root)
        res = archive.merge("orderbook", row, root=root)
        assert res.added == 0
