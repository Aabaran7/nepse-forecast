"""Scoring, with the baselines and intervals §2 requires.

Two rules from §2 are enforced here rather than left to discipline:

**The majority-class baseline is fitted on TRAIN.** It is tempting to report the
test set's own majority share as "the baseline" -- it is one line of code and it
looks like the same number. It is not: it uses the test labels to decide what to
predict, which is the leak the rest of this package exists to prevent, and it
flatters or punishes the model depending on which way the regime broke. The
honest baseline is a predictor that saw only the training window.

**h=5 labels overlap, so naive intervals lie.** Consecutive 5-day labels share
four days; treating 2,400 of them as independent gives a CI far too narrow. The
bootstrap here resamples contiguous blocks (§2) -- and the block has to be
*longer* than the horizon, because the market's persistence outlives the label
overlap. `recommended_block` explains why, with the measurement that forced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Score:
    n: int
    accuracy: float
    f1_up: float
    baseline_majority: float
    baseline_majority_class: int
    baseline_coin: float = 0.5
    edge_vs_majority: float = 0.0
    edge_vs_coin: float = 0.0
    ci: tuple[float, float] | None = None
    extra: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        ci = f" CI[{self.ci[0]:.3f},{self.ci[1]:.3f}]" if self.ci else ""
        return (f"<n={self.n} acc={self.accuracy:.4f}{ci} "
                f"maj={self.baseline_majority:.4f} "
                f"edge={self.edge_vs_majority * 100:+.2f}pp>")


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else float("nan")


def f1(y_true: np.ndarray, y_pred: np.ndarray, positive: int = 1) -> float:
    """F1 for the `positive` class. Reported alongside accuracy because a
    long-only strategy cares specifically about calling up-days."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_pred == positive) & (y_true == positive)).sum())
    fp = int(((y_pred == positive) & (y_true != positive)).sum())
    fn = int(((y_pred != positive) & (y_true == positive)).sum())
    if tp == 0:
        return 0.0
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return float(2 * prec * rec / (prec + rec))


def block_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, block: int = 1,
                       n_boot: int = 5000, alpha: float = 0.05,
                       seed: int = 0) -> tuple[float, float]:
    """Percentile CI for accuracy, resampling contiguous blocks.

    `block` is left explicit because the answer is sensitive to it -- see
    `recommended_block`, which is what `score` uses. The horizon alone is too
    short: it covers the label overlap but not the persistence of the market
    itself.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return (float("nan"), float("nan"))
    correct = (y_true == y_pred).astype(float)

    block = max(1, int(block))
    starts = np.arange(0, n, block)
    blocks = [correct[s:s + block] for s in starts]
    rng = np.random.default_rng(seed)

    n_blocks = len(blocks)
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, n_blocks, n_blocks)
        means[i] = np.concatenate([blocks[j] for j in pick]).mean()
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def recommended_block(n: int, horizon: int) -> int:
    """Block length for the bootstrap: `max(horizon, n**(1/3))`.

    Setting block = horizon is the obvious choice and it is not enough. Two
    sources of dependence stack up, and only the first is the horizon:

      1. Overlapping labels -- consecutive h=5 rows share four days.
      2. The predictions and the market itself are persistent over far longer
         spans than h. Trends and regimes do not reset every five sessions.

    Measured on the h=5 momentum result (n=1911), the paired edge CI runs
    [+1.20, +10.13]pp at block=5 and [-0.26, +11.57]pp at block=25 -- i.e. the
    edge is "significant" or not depending purely on this choice. That is a
    reason to pick conservatively and to report the sensitivity, not to pick the
    number that gives the nicer answer.

    n**(1/3) is the standard growth rate for block bootstrap consistency; the
    max() keeps it from ever dropping below the label overlap.
    """
    return max(int(horizon), int(np.ceil(n ** (1 / 3)))) if n > 0 else max(1, horizon)


def majority_class(y_train: np.ndarray) -> int:
    """The class to beat, learned from the TRAINING window only."""
    y = np.asarray(y_train)
    return int(round(float((y == 1).mean()) > 0.5))


def paired_edge_ci(y_true: np.ndarray, y_pred: np.ndarray, y_base: np.ndarray,
                   block: int = 1, n_boot: int = 5000, alpha: float = 0.05,
                   seed: int = 0) -> tuple[float, float]:
    """CI for the ACCURACY DIFFERENCE, resampling both predictors together.

    §6's test is "does the model beat the majority class by 2pp", which is a
    statement about a DIFFERENCE. Checking whether two separately-computed CIs
    overlap is a different, and wrong, question.

    Resampling the pair together keeps whatever correlation the two predictors
    actually have, and the sign of that correlation is not always the flattering
    one. Two similar models are right and wrong on the same days, so their
    difference is pinned down far more tightly than either accuracy. A model
    against a *constant* predictor is the opposite case -- when always-down is
    right, anything predicting up is wrong -- so the difference is noisier than
    either marginal. On this data the momentum-vs-majority edge CI comes out
    ~6pp wide against ~4.5pp for either accuracy alone. Wider, and correct.
    """
    y_true = np.asarray(y_true)
    d = (y_true == np.asarray(y_pred)).astype(float) - \
        (y_true == np.asarray(y_base)).astype(float)
    n = len(d)
    if n == 0:
        return (float("nan"), float("nan"))

    block = max(1, int(block))
    blocks = [d[s:s + block] for s in range(0, n, block)]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        means[i] = np.concatenate([blocks[j] for j in pick]).mean()
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def score(y_true: np.ndarray, y_pred: np.ndarray, y_base: np.ndarray,
          horizon: int = 1, n_boot: int = 5000, seed: int = 0) -> Score:
    """Full scorecard, measured against a baseline's ACTUAL predictions.

    `y_base` is what the majority-class predictor produced on these same rows
    while being walked forward -- not a majority recomputed from a fixed
    training window. The distinction is not pedantic: the majority-class
    predictor is refitted at every retrain, so it changes its mind when the
    regime turns, and a static baseline both misstates the bar and breaks the
    pairing that makes the edge measurable.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    y_base = np.asarray(y_base)

    acc = accuracy(y_true, y_pred)
    base_acc = accuracy(y_true, y_base)
    block = recommended_block(len(y_true), horizon)
    return Score(
        n=len(y_true),
        accuracy=acc,
        f1_up=f1(y_true, y_pred, positive=1),
        baseline_majority=base_acc,
        baseline_majority_class=int(round(float(y_base.mean()))) if len(y_base) else 0,
        edge_vs_majority=acc - base_acc,
        edge_vs_coin=acc - 0.5,
        ci=block_bootstrap_ci(y_true, y_pred, block=block,
                              n_boot=n_boot, seed=seed),
        extra={"edge_ci": paired_edge_ci(y_true, y_pred, y_base, block=block,
                                         n_boot=n_boot, seed=seed),
               "block": block},
    )


def by_regime(dates: pd.Series, y_true: np.ndarray, y_pred: np.ndarray,
              y_base: np.ndarray, regimes: dict[str, tuple[str, str]],
              horizon: int = 1, n_boot: int = 5000) -> pd.DataFrame:
    """Score each regime separately (§2, §3.6).

    Necessary because the majority class FLIPS between them -- down 52.2%
    pooled, up 56.6% in the mania window -- so a single pooled baseline sets
    the wrong bar in every regime at once. The baseline here is the walked-
    forward majority predictor restricted to the regime, i.e. what a live
    model would genuinely have been competing with at the time.
    """
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    rows = []
    for name, (lo, hi) in regimes.items():
        m = ((dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))).to_numpy()
        if m.sum() == 0:
            continue
        s = score(y_true[m], y_pred[m], y_base[m], horizon=horizon, n_boot=n_boot)
        lo_e, hi_e = s.extra["edge_ci"]
        rows.append({"regime": name, "n": s.n, "accuracy": s.accuracy,
                     "majority": s.baseline_majority,
                     "edge_pp": s.edge_vs_majority * 100,
                     "edge_lo_pp": lo_e * 100, "edge_hi_pp": hi_e * 100})
    return pd.DataFrame(rows)
