"""Baseline predictors, and the harness that walks them forward.

Baselines are not filler. §6 abandons the project unless a model beats the
majority class by 2pp, so the majority-class predictor IS the thing to beat,
and it has to run through the identical pipeline as any real model -- same
splits, same embargo, same scoring. If the baseline took a shortcut the
comparison would be measuring the shortcut.

Running the harness on baselines alone, before any model exists, is also the
cheapest possible leak test: a "predictor" that ignores its features entirely
must score at its own base rate. If a coin flip comes out at 56% here, the bug
is in the harness and it would otherwise have been credited to the model.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from . import metrics
from .splits import Split, assert_no_leakage, walk_forward


class Predictor(Protocol):
    """Anything the harness can walk forward. Phase 4's models satisfy this."""

    name: str

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Predictor": ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class MajorityClass:
    """Always predict whichever class dominated the training window."""

    name = "majority-class"

    def __init__(self) -> None:
        self.cls = 1

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MajorityClass":
        self.cls = metrics.majority_class(y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.cls, dtype=int)


class AlwaysUp:
    """Buy-and-hold's directional twin: the index goes up, always."""

    name = "always-up"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "AlwaysUp":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X), dtype=int)


class Coin:
    """Fair coin. Must land at 50% +- noise; anything else means a harness bug."""

    name = "coin"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Coin":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.rng.integers(0, 2, len(X))


class Momentum:
    """Tomorrow repeats today. The simplest thing that actually uses a feature.

    Included because it is the cheapest check that features are wired to the
    right rows: it needs `prev_return` to be genuinely lagged, and if the
    feature frame is off by one it will score suspiciously well.
    """

    name = "momentum"

    def __init__(self, col: str = "prev_return") -> None:
        self.col = col

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Momentum":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (X[self.col].to_numpy() > 0).astype(int)


def run_walk_forward(frame: pd.DataFrame, predictor: Predictor, horizon: int,
                     feature_cols: list[str], initial_train: int = 500,
                     retrain_freq: str = "MS", expanding: bool = True,
                     ) -> tuple[pd.DataFrame, list[Split]]:
    """Walk `predictor` forward over `frame`, returning per-row predictions.

    `frame` must already be labelled (see labels.make_labels) and filtered to
    usable rows. Returns one row per test prediction, with the fold it came
    from, so scoring can be pooled or split by regime afterwards.
    """
    frame = frame.reset_index(drop=True)
    splits = walk_forward(frame["date"], horizon=horizon,
                          initial_train=initial_train,
                          retrain_freq=retrain_freq, expanding=expanding)
    if not splits:
        raise ValueError("no walk-forward folds produced; check initial_train")
    assert_no_leakage(splits, frame["date"], horizon)

    X_all = frame[feature_cols]
    y_all = frame["label"].to_numpy().astype(int)

    out = []
    for fold, sp in enumerate(splits):
        model = predictor.fit(X_all.iloc[sp.train], y_all[sp.train])
        pred = np.asarray(model.predict(X_all.iloc[sp.test])).astype(int)
        out.append(pd.DataFrame({
            "fold": fold,
            "date": frame["date"].iloc[sp.test].to_numpy(),
            "y_true": y_all[sp.test],
            "y_pred": pred,
            "fwd_return": frame["fwd_return"].iloc[sp.test].to_numpy(),
            "train_n": len(sp.train),
        }))
    return pd.concat(out, ignore_index=True), splits


def pooled_train_labels(frame: pd.DataFrame, splits: list[Split]) -> np.ndarray:
    """Training labels of the FIRST fold -- the only window a live model would
    have had before predicting anything. Used for the pooled baseline so it
    never sees a label it could not have seen."""
    return frame["label"].to_numpy().astype(int)[splits[0].train]
