"""Logistic regression and gradient boosting (plan §6). Nothing deeper.

§6: "Logistic regression and gradient boosting only. 2,434 daily observations
does not support anything deeper." §1 rules out sequence models outright.

Both wrappers satisfy the `Predictor` protocol in `nepselab.eval.baselines`, so
they walk forward through the identical harness the baselines did -- same folds,
same embargo, same paired scoring. That is the point of §1's "models are
interchangeable": if a model needed a different pipeline, its result would not
be comparable to the baseline it has to beat.

**Scaling lives inside the model, not the feature layer.** §2 requires all
scaling and feature fitting to happen inside the training window only. A
StandardScaler fitted on the whole series before splitting leaks test-set
distribution into training -- a small leak, but a real one, and invisible. Here
the scaler is a pipeline step, so `fit` sees only the fold's training rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Logistic:
    """L2 logistic regression on standardised features."""

    def __init__(self, C: float = 0.1, seed: int = 0, name: str | None = None):
        self.C = C
        self.seed = seed
        self.name = name or f"logistic(C={C})"
        self.pipe: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Logistic":
        self.pipe = Pipeline([
            # Impute inside the fold too: a median computed over the whole
            # series would carry future information into the training window.
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=self.C, max_iter=2000,
                                       random_state=self.seed)),
        ])
        # A fold can be single-class early in a strong trend; sklearn raises.
        if len(np.unique(y)) < 2:
            self._constant = int(y[0]) if len(y) else 0
            self.pipe = None
            return self
        self._constant = None
        self.pipe.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipe is None:
            return np.full(len(X), self._constant, dtype=int)
        return self.pipe.predict(X).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipe is None:
            return np.full(len(X), float(self._constant))
        return self.pipe.predict_proba(X)[:, 1]


class GBM:
    """LightGBM, deliberately small.

    Defaults are shrunk hard against the sample size: ~1,900 training rows and
    40 features is a setting where a stock-default GBM memorises the training
    window and reports a beautiful in-sample number. Depth and leaf count are
    the two knobs that matter most here, and both are pinned low.
    """

    def __init__(self, seed: int = 0, n_estimators: int = 200,
                 learning_rate: float = 0.03, num_leaves: int = 7,
                 max_depth: int = 3, min_child_samples: int = 40,
                 name: str | None = None):
        self.seed = seed
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                           num_leaves=num_leaves, max_depth=max_depth,
                           min_child_samples=min_child_samples,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           reg_lambda=1.0, random_state=seed, verbose=-1)
        self.name = name or f"gbm(seed={seed})"
        self.model = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GBM":
        from lightgbm import LGBMClassifier

        if len(np.unique(y)) < 2:
            self._constant = int(y[0]) if len(y) else 0
            self.model = None
            return self
        self._constant = None
        self.model = LGBMClassifier(**self.params)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), self._constant, dtype=int)
        return self.model.predict(X).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), float(self._constant))
        return self.model.predict_proba(X)[:, 1]


def seeded(factory, seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> list:
    """§6: "Multiple seeds where stochastic; report mean ± sd."

    Logistic regression is deterministic here so one seed suffices; GBM is not
    (subsampling and feature sampling), and a single-seed GBM result is a draw
    from a distribution that is usually wider than the effect being measured.
    """
    return [factory(s) for s in seeds]
