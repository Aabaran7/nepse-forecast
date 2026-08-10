"""The two pieces of real logic behind the dashboard.

Neither is cosmetic, and neither fails loudly.

  SOURCE PRECEDENCE. The index series is stitched from two stores that disagree
  in span and provenance (§3.5): data/deep/ reaches back to 2016 but is only
  refreshed by a manual run, while data/archive/ is refreshed daily but holds
  one rolling year. Pick deep alone and the forward log can never resolve,
  because the target session is missing from the actuals. Pick archive alone and
  nine years of history vanish from every chart.

  EXPOSURE ROWS ARE NOT FORECASTS. §6.6's volatility-target rows are logged at
  horizon 0 -- they carry today's exposure, not a direction. score_log() cannot
  know that: it computes close[i+0]/close[i] - 1, gets 0, reads "not up", and
  marks every one of them wrong. Left alone the dashboard would report an
  accuracy dragged down by rows that never made a claim.

Run: .venv/bin/pytest tests/test_dashboard_data.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.dashboard import data  # noqa: E402


class TestIndexStitching:
    def test_archive_wins_where_the_two_disagree(self, monkeypatch, tmp_path):
        """The exchange's own close beats the third-party scrape on shared dates."""
        deep = tmp_path / "deep.parquet"
        pd.DataFrame({"date": pd.to_datetime(["2026-08-01", "2026-08-02"]),
                      "close": [100.0, 200.0]}).to_parquet(deep)
        monkeypatch.setattr(data, "DEEP", deep)
        monkeypatch.setattr(data.archive, "load", lambda name, **kw: pd.DataFrame({
            "businessDate": pd.to_datetime(["2026-08-02"]),
            "exchangeIndexId": [58], "closingIndex": [999.0],
        }) if name == "indices" else pd.DataFrame())

        out = data.index_closes()
        assert out.loc[out["date"] == pd.Timestamp("2026-08-02"), "close"].iloc[0] == 999.0
        # ...and the deep-only history survives.
        assert out.loc[out["date"] == pd.Timestamp("2026-08-01"), "close"].iloc[0] == 100.0

    def test_archive_extends_the_series_past_a_stale_deep_file(self, monkeypatch, tmp_path):
        deep = tmp_path / "deep.parquet"
        pd.DataFrame({"date": pd.to_datetime(["2026-08-01"]),
                      "close": [100.0]}).to_parquet(deep)
        monkeypatch.setattr(data, "DEEP", deep)
        monkeypatch.setattr(data.archive, "load", lambda name, **kw: pd.DataFrame({
            "businessDate": pd.to_datetime(["2026-08-05"]),
            "exchangeIndexId": [58], "closingIndex": [120.0],
        }) if name == "indices" else pd.DataFrame())

        out = data.index_closes()
        assert out["date"].max() == pd.Timestamp("2026-08-05")

    def test_other_indices_are_not_mixed_into_the_nepse_series(self, monkeypatch, tmp_path):
        monkeypatch.setattr(data, "DEEP", tmp_path / "absent.parquet")
        monkeypatch.setattr(data.archive, "load", lambda name, **kw: pd.DataFrame({
            "businessDate": pd.to_datetime(["2026-08-05", "2026-08-05"]),
            "exchangeIndexId": [58, 62],          # 62 is a sector index
            "closingIndex": [120.0, 5000.0],
        }) if name == "indices" else pd.DataFrame())

        out = data.index_closes()
        assert len(out) == 1
        assert out["close"].iloc[0] == 120.0


class TestExposureRowsExcluded:
    @pytest.fixture
    def scored(self) -> pd.DataFrame:
        return pd.DataFrame({
            "as_of": pd.to_datetime(["2026-08-03"] * 3),
            "horizon": [0, 1, 5],
            "prediction": [1, 0, 0],
            "correct": [0.0, 1.0, None],       # the 0 is score_log's false negative
            "actual": [0.0, 0.0, None],
            "fwd_return": [0.0, -0.01, None],
            "model_version": ["voltarget-v2", "lr-v1", "lr-v1"],
        })

    def test_horizon_zero_is_labelled_exposure_and_its_score_cleared(
            self, monkeypatch, scored):
        monkeypatch.setattr(data, "index_closes", lambda: pd.DataFrame(
            {"date": pd.to_datetime(["2026-08-03"]), "close": [100.0]}))
        monkeypatch.setattr("nepselab.forward.log.score_log", lambda *a, **k: scored)

        out = data.forward_log()
        exposure = out[out["horizon"] == 0]
        assert exposure["kind"].iloc[0] == "exposure"
        # The bogus "wrong" must be gone, not merely ignored downstream.
        assert pd.isna(exposure["correct"].iloc[0])

    def test_accuracy_input_contains_only_directional_rows(self, monkeypatch, scored):
        monkeypatch.setattr(data, "index_closes", lambda: pd.DataFrame(
            {"date": pd.to_datetime(["2026-08-03"]), "close": [100.0]}))
        monkeypatch.setattr("nepselab.forward.log.score_log", lambda *a, **k: scored)

        directional = data.directional_log()
        assert set(directional["horizon"]) == {1, 5}
        resolved = directional[directional["correct"].notna()]
        # Without the split this would be 1/2 = 50%. It is 1/1.
        assert resolved["correct"].mean() == 1.0


class TestFreshness:
    def test_mixed_timezones_do_not_raise(self, monkeypatch, tmp_path):
        """News carries a UTC offset; exchange dates do not. Both must compare."""
        monkeypatch.setattr(data, "DEEP", tmp_path / "absent.parquet")
        monkeypatch.setattr(data.archive, "load", lambda name, **kw: pd.DataFrame({
            "businessDate": pd.to_datetime(["2026-08-07"]),
            "exchangeIndexId": [58], "closingIndex": [1.0],
        }) if name == "indices" else pd.DataFrame())
        monkeypatch.setattr(data, "headlines", lambda: pd.DataFrame(
            {"first_seen_utc": ["2026-08-09T06:54:09+00:00"]}))

        out = data.freshness()
        assert out["news"].tz is None
        assert (pd.Timestamp.today().normalize() - out["news"].normalize()).days >= 0


class TestJsonSafety:
    """Every value the export emits must survive json.dumps(allow_nan=False).

    A plain datetime.date is not an instance of datetime.datetime, so it slipped
    past the type check and reached the encoder. It could only ever fail in
    production: news.session_for() returns None until a headline's target
    session has actually traded, so every local run had nulls there and looked
    fine. The first day headlines resolved to real sessions, CI broke.
    """

    def test_every_type_the_pipeline_produces_serialises(self):
        import json
        import sys
        from datetime import date, datetime
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import export_dashboard as ed

        for v in (date(2026, 8, 10), datetime(2026, 8, 10, 5, 0),
                  pd.Timestamp("2026-08-10"), pd.NaT, None,
                  float("nan"), float("inf"), 3, 3.5, "x", True):
            json.dumps(ed.clean(v), allow_nan=False)   # must not raise

    def test_a_date_becomes_an_iso_string(self):
        import sys
        from datetime import date
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import export_dashboard as ed
        assert ed.clean(date(2026, 8, 10)) == "2026-08-10"
