"""Feature-layer and forward-log tests (plan §5, §7, §9).

The feature tests are almost all leak tests. §9's risk table rates "Reddit
attention leaks a time trend" as *High if unhandled*, and a leaked feature does
not raise -- it produces a model that works in backtest and dies live.

The forward-log tests exist because §7's guarantee ("never overwrite a past
prediction") is only worth stating if something enforces it. A log that can be
rewritten proves nothing, since the most likely reason to rewrite it is a bug
fix applied after seeing the prediction was wrong.

Run: .venv/bin/pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.features import base as fbase  # noqa: E402
from nepselab.features import price as fprice  # noqa: E402
from nepselab.forward import log as flog  # noqa: E402

DEEP = Path("data/deep/nepse_index_deep.parquet")


def sessions(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    close = 1000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "date": dates, "close": close,
        "high": close * 1.01, "low": close * 0.99, "open": close,
        "turnover": rng.lognormal(20, 0.5, n),
        "ohlc_consistent": True,
    })


# --- the leak test ----------------------------------------------------------

@pytest.mark.parametrize("module", [fprice.PriceFeatures(), fprice.TurnoverFeatures()])
def test_features_do_not_depend_on_the_future(module):
    """Truncating the series must not change features computed before the cut.

    This is the general form of a lookahead check: if a feature at row 200 is
    identical whether or not rows 201+ exist, it cannot be using them. It
    catches centred windows, unshifted rolling stats, and whole-series
    normalisation in one assertion, including cases nobody thought to test.
    """
    full = sessions(400)
    truncated = full.iloc[:300].reset_index(drop=True)

    f_full = module.build(full).iloc[:250]
    f_trunc = module.build(truncated).iloc[:250]

    cols = [c for c in f_full.columns if c != "date"]
    for c in cols:
        a, b = f_full[c].to_numpy(), f_trunc[c].to_numpy()
        both = ~(np.isnan(a) | np.isnan(b))
        assert np.allclose(a[both], b[both], equal_nan=True), (
            f"{module.name}.{c} changed when future rows were removed -- lookahead")


def test_turnover_features_are_masked_before_their_start_date():
    """The 2017 units break is a 420x step (§3.5). A turnover ratio computed
    across it reads a surge that never happened."""
    s = sessions(600)
    s["date"] = pd.bdate_range("2016-01-01", periods=600)
    out = fprice.TurnoverFeatures().build(s)
    early = out[out["date"] < pd.Timestamp("2017-01-01")]
    cols = [c for c in out.columns if c != "date"]
    assert early[cols].isna().all().all()


def test_range_features_are_nan_on_inconsistent_bars():
    s = sessions(200)
    s.loc[50, "ohlc_consistent"] = False
    out = fprice.PriceFeatures().build(s)
    assert pd.isna(out.loc[50, "intraday_range"])


# --- assembly ---------------------------------------------------------------

def test_assemble_refuses_a_start_its_modules_cannot_cover():
    """§8: the engine must refuse rather than silently return a short sample."""
    s = sessions(600)
    s["date"] = pd.bdate_range("2016-01-01", periods=600)
    with pytest.raises(ValueError, match="precedes these modules"):
        fbase.assemble(s, [fprice.TurnoverFeatures()],
                       start=pd.Timestamp("2016-01-01"))


def test_assemble_allows_a_start_all_modules_cover():
    s = sessions(600)
    s["date"] = pd.bdate_range("2017-06-01", periods=600)
    a = fbase.assemble(s, [fprice.PriceFeatures(), fprice.TurnoverFeatures()],
                       start=pd.Timestamp("2017-06-01"))
    assert len(a.feature_names) > 20
    assert a.frame[a.feature_names].notna().all().all()


def test_assemble_rejects_two_modules_claiming_the_same_column():
    s = sessions(300)
    with pytest.raises(ValueError, match="overwrite"):
        fbase.assemble(s, [fprice.PriceFeatures(), fprice.PriceFeatures()])


def test_assemble_drops_only_rows_with_missing_features():
    s = sessions(400)
    a = fbase.assemble(s, [fprice.PriceFeatures()])
    assert a.dropped_rows > 0          # rolling windows need warm-up
    assert a.frame[a.feature_names].notna().all().all()


# --- the real feature frame -------------------------------------------------

@pytest.mark.skipif(not DEEP.exists(), reason="deep series absent")
def test_real_features_have_no_lookahead():
    df = pd.read_parquet(DEEP).sort_values("date").reset_index(drop=True)
    cut = len(df) - 200
    mod = fprice.PriceFeatures()
    a = mod.build(df).iloc[:cut - 50]
    b = mod.build(df.iloc[:cut].reset_index(drop=True)).iloc[:cut - 50]
    for c in [x for x in a.columns if x != "date"]:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        m = ~(np.isnan(x) | np.isnan(y))
        assert np.allclose(x[m], y[m]), f"lookahead in {c}"


# --- forward log ------------------------------------------------------------

def test_log_appends_and_reloads(tmp_path):
    p = flog.append(pd.Timestamp("2026-01-05"),
                    [{"horizon": 1, "prediction": 1}], root=tmp_path)
    assert p.exists()
    df = flog.load_all(tmp_path)
    assert len(df) == 1
    assert {"as_of", "horizon", "prediction", "model_version",
            "git_hash", "logged_utc"} <= set(df.columns)


def test_log_refuses_to_overwrite_a_past_prediction(tmp_path):
    """§7's one absolute rule. The dangerous rewrite is the one applied AFTER
    the outcome is known, which is exactly when it looks most reasonable."""
    flog.append(pd.Timestamp("2026-01-05"), [{"horizon": 1, "prediction": 1}],
                root=tmp_path)
    with pytest.raises(flog.PredictionExists):
        flog.append(pd.Timestamp("2026-01-05"), [{"horizon": 1, "prediction": 0}],
                    root=tmp_path)


def test_a_new_model_version_may_coexist_on_the_same_date(tmp_path):
    """The escape hatch §7 allows: version the model, never edit history."""
    flog.append(pd.Timestamp("2026-01-05"), [{"horizon": 1, "prediction": 1}],
                root=tmp_path, model_version="v1")
    flog.append(pd.Timestamp("2026-01-05"), [{"horizon": 1, "prediction": 0}],
                root=tmp_path, model_version="v2")
    df = flog.load_all(tmp_path)
    assert len(df) == 2
    assert set(df["model_version"]) == {"v1", "v2"}


def test_a_different_horizon_on_the_same_date_is_fine(tmp_path):
    flog.append(pd.Timestamp("2026-01-05"), [{"horizon": 1, "prediction": 1}],
                root=tmp_path)
    flog.append(pd.Timestamp("2026-01-05"), [{"horizon": 5, "prediction": 0}],
                root=tmp_path)
    assert len(flog.load_all(tmp_path)) == 2


def test_scoring_resolves_only_predictions_whose_target_has_happened(tmp_path):
    dates = pd.bdate_range("2026-01-05", periods=4)
    actuals = pd.DataFrame({"date": dates, "close": [100.0, 101.0, 102.0, 103.0]})
    flog.append(dates[0], [{"horizon": 1, "prediction": 1}], root=tmp_path)
    flog.append(dates[2], [{"horizon": 5, "prediction": 1}], root=tmp_path)

    scored = flog.score_log(actuals, root=tmp_path)
    resolved = scored[scored["correct"].notna()]
    assert len(resolved) == 1
    assert resolved.iloc[0]["correct"] == 1        # 100 -> 101 is up, predicted up
    assert scored["correct"].isna().sum() == 1     # h=5 has no target yet


def test_scoring_an_empty_log_returns_empty(tmp_path):
    assert flog.score_log(pd.DataFrame({"date": [], "close": []}),
                          root=tmp_path).empty


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
