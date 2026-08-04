"""Walk-forward harness tests (plan §1, §2).

§1: "the harness is what makes any result believable." So the harness has to be
the most heavily tested thing in the repo, and the tests that matter are the
ones that would catch it finding an edge that is not there.

The load-bearing test here is `test_no_edge_on_a_random_walk`. Leakage does not
announce itself -- it looks like a good model. Feeding the whole pipeline a
series with no predictable structure whatsoever and asserting nothing beats a
coin is the only check that covers leaks nobody thought to write an assertion
for, including the off-by-one that would make every feature secretly contain
its own answer.

Run: .venv/bin/pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.eval import baselines, labels, metrics, splits  # noqa: E402


def series(n: int, seed: int = 0, start: str = "2016-01-04") -> pd.DataFrame:
    """A random walk on business days. No predictable structure, by construction."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    close = 1000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"date": dates, "close": close})


# --- labels -----------------------------------------------------------------

def test_forward_return_looks_forward_by_exactly_h_sessions():
    df = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=4),
                       "close": [100.0, 110.0, 121.0, 133.1]})
    r = labels.forward_return(df, horizon=1)
    assert r.iloc[0] == pytest.approx(0.10)
    assert pd.isna(r.iloc[-1])
    r2 = labels.forward_return(df, horizon=2)
    assert r2.iloc[0] == pytest.approx(0.21)


def test_label_is_one_when_price_rises():
    df = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=3),
                       "close": [100.0, 101.0, 100.0]})
    out = labels.make_labels(df, horizon=1)
    assert out["label"].tolist()[:2] == [1.0, 0.0]


def test_flat_days_are_marked_unusable_not_bucketed_as_up():
    """`>= 0` vs `> 0` differs by ~0.1% of days -- small next to intuition,
    large next to §6's 2pp threshold."""
    df = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=3),
                       "close": [100.0, 100.0, 101.0]})
    out = labels.make_labels(df, horizon=1)
    assert not out["usable"].iloc[0]
    assert out.attrs["dropped_flat"] == 1


def test_labels_spanning_a_long_closure_are_unusable():
    """A '5-session-ahead' label that actually spans NEPSE's 51-day COVID
    closure is not the same target as its neighbours."""
    dates = pd.to_datetime(["2020-03-09", "2020-03-10", "2020-05-12", "2020-05-13"])
    df = pd.DataFrame({"date": dates, "close": [100.0, 101.0, 90.0, 91.0]})
    out = labels.make_labels(df, horizon=1)
    assert not out["usable"].iloc[1]          # spans 63 days
    assert out["usable"].iloc[0]
    assert out.attrs["dropped_long_span"] == 1


def test_horizon_must_be_positive():
    with pytest.raises(ValueError):
        labels.make_labels(series(10), horizon=0)


def test_weekly_blocks_are_non_overlapping():
    b = labels.weekly_blocks(pd.DataFrame(index=range(12)), horizon=5)
    assert b.tolist() == [0] * 5 + [1] * 5 + [2] * 2


# --- splits and the embargo -------------------------------------------------

def test_train_always_precedes_test():
    df = series(800)
    sp = splits.walk_forward(df["date"], horizon=1, initial_train=300)
    assert sp
    for s in sp:
        assert s.train[-1] < s.test[0]
        assert not (set(s.train) & set(s.test))


def test_embargo_removes_exactly_horizon_rows():
    df = series(800)
    for h in (1, 5, 10):
        sp = splits.walk_forward(df["date"], horizon=h, initial_train=300)
        for s in sp:
            # the last training row's label must close strictly before the test
            assert s.train[-1] + h < s.test[0] + 1
            assert s.test[0] - s.train[-1] >= h


def test_leakage_assertion_catches_a_deliberately_bad_split():
    df = series(200)
    bad = [splits.Split(train=np.arange(0, 100), test=np.arange(100, 120),
                        train_start=df.date.iloc[0], train_end=df.date.iloc[99],
                        test_start=df.date.iloc[100], test_end=df.date.iloc[119],
                        embargoed=0)]
    # horizon 5 means row 99's label closes at 104, inside the test window
    with pytest.raises(AssertionError):
        splits.assert_no_leakage(bad, df["date"], horizon=5)
    # horizon 1 is fine: row 99's label closes at 100... which is still the
    # first test row, so this must ALSO raise.
    with pytest.raises(AssertionError):
        splits.assert_no_leakage(bad, df["date"], horizon=1)


def test_rolling_window_is_bounded_expanding_is_not():
    df = series(1200)
    roll = splits.walk_forward(df["date"], horizon=1, initial_train=300,
                               expanding=False)
    grow = splits.walk_forward(df["date"], horizon=1, initial_train=300,
                               expanding=True)
    assert max(len(s.train) for s in roll) <= 300
    assert len(grow[-1].train) > len(grow[0].train)


def test_unsorted_dates_are_rejected():
    df = series(600)
    with pytest.raises(ValueError):
        splits.walk_forward(df["date"][::-1], horizon=1, initial_train=300)


def test_initial_train_larger_than_series_is_rejected():
    with pytest.raises(ValueError):
        splits.walk_forward(series(100)["date"], horizon=1, initial_train=500)


# --- metrics ----------------------------------------------------------------

def test_accuracy_and_f1_basics():
    y = np.array([1, 1, 0, 0])
    assert metrics.accuracy(y, np.array([1, 1, 0, 0])) == 1.0
    assert metrics.accuracy(y, np.array([0, 0, 1, 1])) == 0.0
    assert metrics.f1(y, np.array([1, 1, 0, 0])) == 1.0
    assert metrics.f1(y, np.array([0, 0, 0, 0])) == 0.0


def test_majority_class_is_learned_from_training_labels_only():
    assert metrics.majority_class(np.array([0, 0, 0, 1])) == 0
    assert metrics.majority_class(np.array([1, 1, 1, 0])) == 1


def test_edge_is_measured_against_the_baselines_actual_predictions():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 1, 0, 0, 1])       # perfect
    y_base = np.array([0, 0, 0, 0, 0])       # always-down: 2/5
    s = metrics.score(y_true, y_pred, y_base, n_boot=200)
    assert s.accuracy == 1.0
    assert s.baseline_majority == pytest.approx(0.4)
    assert s.edge_vs_majority == pytest.approx(0.6)


def test_block_bootstrap_widens_the_interval_when_errors_are_correlated():
    """This is the whole reason §2 demands blocks for h=5.

    Blocking is not a penalty applied for its own sake -- on independent data it
    changes nothing, which an earlier version of this test got backwards. It
    bites only when the correctness sequence is serially correlated, which is
    exactly what overlapping 5-day labels produce. Simulated here as runs of
    right-then-wrong; the naive interval is ~2x too narrow at block=5."""
    y_true = np.tile(np.r_[np.ones(20, int), np.zeros(20, int)], 25)
    y_pred = np.ones(len(y_true), int)
    lo1, hi1 = metrics.block_bootstrap_ci(y_true, y_pred, block=1, n_boot=1500)
    lo5, hi5 = metrics.block_bootstrap_ci(y_true, y_pred, block=5, n_boot=1500)
    assert (hi5 - lo5) > 1.5 * (hi1 - lo1)


def test_block_bootstrap_is_neutral_on_independent_data():
    """The converse, stated so the guarantee is not mistaken for a free penalty."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 1000)
    y_pred = rng.integers(0, 2, 1000)
    w1 = np.subtract(*reversed(metrics.block_bootstrap_ci(
        y_true, y_pred, block=1, n_boot=1500)))
    w5 = np.subtract(*reversed(metrics.block_bootstrap_ci(
        y_true, y_pred, block=5, n_boot=1500)))
    assert abs(w5 - w1) < 0.02


def test_paired_edge_ci_is_zero_width_when_predictors_are_identical():
    y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    lo, hi = metrics.paired_edge_ci(y_true, y_true, y_true, n_boot=200)
    assert lo == 0.0 and hi == 0.0


# --- the harness end to end -------------------------------------------------

FEATURES = ["prev_return"]


def framed(n: int = 900, seed: int = 0, horizon: int = 1) -> pd.DataFrame:
    df = series(n, seed=seed)
    df["prev_return"] = df["close"].pct_change()
    lab = labels.make_labels(df, horizon=horizon)
    return lab[lab["usable"] & lab["prev_return"].notna()].reset_index(drop=True)


def test_majority_baseline_ties_itself_exactly():
    frame = framed()
    preds, sp = baselines.run_walk_forward(frame, baselines.MajorityClass(),
                                           horizon=1, feature_cols=FEATURES,
                                           initial_train=300)
    y = preds["y_pred"].to_numpy()
    s = metrics.score(preds["y_true"].to_numpy(), y, y, n_boot=200)
    assert s.edge_vs_majority == 0.0


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_no_edge_on_a_random_walk(seed):
    """THE test. A random walk has no predictable sign, so every predictor here
    must land within noise of its own base rate. If the harness ever reports a
    real edge on this input, it has a leak -- and a leak looks exactly like a
    discovery, which is why this runs across several seeds rather than one."""
    frame = framed(900, seed=seed)
    base, _ = baselines.run_walk_forward(frame, baselines.MajorityClass(),
                                         horizon=1, feature_cols=FEATURES,
                                         initial_train=300)
    y_base = base["y_pred"].to_numpy()

    for model in (baselines.Momentum(), baselines.Coin(seed=seed)):
        preds, _ = baselines.run_walk_forward(frame, model, horizon=1,
                                              feature_cols=FEATURES,
                                              initial_train=300)
        s = metrics.score(preds["y_true"].to_numpy(), preds["y_pred"].to_numpy(),
                          y_base, horizon=1, n_boot=600, seed=seed)
        lo, hi = s.extra["edge_ci"]
        assert lo <= 0.0 <= hi, (
            f"{model.name} shows edge {s.edge_vs_majority * 100:+.2f}pp "
            f"CI [{lo * 100:+.2f},{hi * 100:+.2f}] on a random walk -- leak")


def test_a_cheating_predictor_is_detected_as_such():
    """Sanity on the sanity check: if a predictor IS handed the answer, the
    harness must report ~100%. A test suite that only ever proves 'no edge'
    would also pass if the scorer were broken and always returned 0.5."""
    frame = framed()

    class Oracle:
        name = "oracle"

        def fit(self, X, y):
            return self

        def predict(self, X):
            return frame["label"].to_numpy().astype(int)[X.index]

    preds, _ = baselines.run_walk_forward(frame, Oracle(), horizon=1,
                                          feature_cols=FEATURES,
                                          initial_train=300)
    assert metrics.accuracy(preds["y_true"], preds["y_pred"]) == 1.0


def test_every_test_row_is_predicted_exactly_once():
    frame = framed()
    preds, sp = baselines.run_walk_forward(frame, baselines.Coin(), horizon=1,
                                           feature_cols=FEATURES,
                                           initial_train=300)
    assert len(preds) == sum(len(s.test) for s in sp)
    assert preds["date"].is_unique
    assert preds["date"].is_monotonic_increasing


def test_folds_do_not_start_before_the_initial_training_window():
    frame = framed()
    _, sp = baselines.run_walk_forward(frame, baselines.Coin(), horizon=1,
                                       feature_cols=FEATURES, initial_train=300)
    assert sp[0].test[0] >= 300


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
