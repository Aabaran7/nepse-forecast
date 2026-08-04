"""Probability -> position, with hysteresis (plan §6.3).

§6.2's models were accurate enough and traded far too often: 55.5% directional
accuracy against a break-even of 85% at daily frequency. At 21-session holds
break-even falls to 51.7%, which the same models already clear. Nothing about
that arithmetic depends on §6.2's outcome -- it is the cost model and the
series' volatility, both known in advance.

So the failure under test is the DECISION RULE, and this module is that rule:

  **Deadband.** Go long above `0.5 + delta`, flat below `0.5 - delta`, and in
  between hold whatever you already have. A raw `p > 0.5` rule trades every time
  the probability wobbles across a knife edge, which is most days.

  **Minimum hold.** After a change, freeze for `m` sessions. Bounds turnover
  from above regardless of how the probability behaves.

Both parameters are fitted on training data inside each walk-forward fold (see
`select_params`), never chosen by looking at test results. That is the whole
difference between this and the tuning §9 forbids.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rule:
    """Deadband + minimum-hold, symmetric or asymmetric.

    `delta=0, min_hold=1, asymmetric=False` reproduces §6.2's rule.

    **Asymmetric mode (§6.4)** holds LONG by default and requires a strong
    signal to leave, rather than requiring a strong signal to enter. The
    motivation is structural, not empirical: NEPSE compounded at ~9.6%/yr net
    across this sample, so a symmetric rule that starts flat has to earn its way
    into a rising market and pays the drift as the price of every hour it spends
    in cash. With a weak signal on a drifting asset, the sound use of the signal
    is to reduce exposure occasionally, not to justify holding it at all.
    """

    delta: float = 0.0
    min_hold: int = 1
    asymmetric: bool = False
    exit_delta: float = 0.10

    def positions(self, prob_up: np.ndarray) -> np.ndarray:
        p = np.asarray(prob_up, dtype=float)
        pos = np.zeros(len(p), dtype=int)
        cur = 1 if self.asymmetric else 0
        last_change = -10**9

        for i, pi in enumerate(p):
            want = cur
            if self.asymmetric:
                # Long unless the model is confidently bearish; re-enter as soon
                # as it stops being so.
                if pi < 0.5 - self.exit_delta:
                    want = 0
                elif pi > 0.5 - self.exit_delta + self.delta:
                    want = 1
            elif pi > 0.5 + self.delta:
                want = 1
            elif pi < 0.5 - self.delta:
                want = 0
            # Freeze after a change. Also keeps the ledger's settlement rule
            # from being the only thing bounding turnover.
            if want != cur and (i - last_change) >= self.min_hold:
                cur = want
                last_change = i
            pos[i] = cur
        return pos


# Grid searched INSIDE each training fold. Deliberately coarse: a fine grid
# fitted per fold on a few hundred rows is itself an overfitting surface, and
# the point of this experiment is turnover, not precision in delta.
DELTAS = (0.0, 0.02, 0.05, 0.10, 0.15)
MIN_HOLDS = (1, 5, 21, 63)
# Rule FAMILY is also fitted per fold rather than chosen by hand. Adding the
# asymmetric family after seeing §6.3 fail is a researcher decision and is
# recorded as such in §6.4; which family gets used on any given test fold is
# not -- that is decided in-sample, like delta and min_hold.
ASYMMETRIC = (False, True)
EXIT_DELTAS = (0.05, 0.10)

# Selection runs a full ledger per grid point per fold, which is O(n) Python.
# The full grid over an expanding window is ~27,000 backtests and does not
# finish in reasonable time. Two compute decisions, neither of which touches a
# test row: the grid is coarse (above), and selection uses the most recent
# SELECT_WINDOW training sessions rather than the whole expanding history. The
# CLASSIFIER still trains on everything -- only the rule's parameters are
# chosen on the recent window, which is also the more defensible choice on its
# own terms, since the relevant question is what turnover the current regime
# supports rather than what 2017 supported.
SELECT_WINDOW = 750


def _net_sharpe(frame: pd.DataFrame, positions: np.ndarray, cost) -> float:
    from .portfolio import run_backtest

    if len(frame) < 10:
        return float("-inf")
    try:
        return run_backtest(frame, positions, cost).stats["net_sharpe"]
    except Exception:  # noqa: BLE001
        return float("-inf")


def select_params(train_frame: pd.DataFrame, train_prob: np.ndarray, cost,
                  deltas: tuple[float, ...] = DELTAS,
                  min_holds: tuple[int, ...] = MIN_HOLDS) -> Rule:
    """Pick (delta, min_hold) by in-sample net Sharpe on the TRAINING fold.

    Ties break toward the LONGER hold and the WIDER deadband. Both push turnover
    down, which is the direction the arithmetic in §6.3 says to prefer, and
    preferring it on ties means the choice is not driven by noise in a
    third decimal place of Sharpe.
    """
    if len(train_frame) > SELECT_WINDOW:
        train_frame = train_frame.iloc[-SELECT_WINDOW:].reset_index(drop=True)
        train_prob = np.asarray(train_prob)[-SELECT_WINDOW:]

    best, best_score = Rule(), float("-inf")
    for d, m, asym in product(deltas, min_holds, ASYMMETRIC):
        for xd in (EXIT_DELTAS if asym else (0.10,)):
            r = Rule(delta=d, min_hold=m, asymmetric=asym, exit_delta=xd)
            sc = _net_sharpe(train_frame, r.positions(train_prob), cost)
            if sc > best_score + 1e-9 or (
                    abs(sc - best_score) <= 1e-9
                    and (m, d) > (best.min_hold, best.delta)):
                best, best_score = r, sc
    return best


def run_walk_forward_strategy(frame: pd.DataFrame, model_factory, horizon: int,
                              feature_cols: list[str], cost,
                              initial_train: int = 400,
                              retrain_freq: str = "MS") -> pd.DataFrame:
    """Walk forward, fitting BOTH the classifier and the decision rule per fold.

    Returns one row per test session with the probability, the chosen rule, and
    the resulting position -- so the rule's stability across folds is visible
    rather than assumed. A rule that changes wildly fold to fold is fitting
    noise, and that shows up here.
    """
    from .splits import assert_no_leakage, walk_forward

    frame = frame.reset_index(drop=True)
    splits = walk_forward(frame["date"], horizon=horizon,
                          initial_train=initial_train, retrain_freq=retrain_freq)
    assert_no_leakage(splits, frame["date"], horizon)

    X = frame[feature_cols]
    y = frame["label"].to_numpy().astype(int)
    out = []

    for sp in splits:
        model = model_factory().fit(X.iloc[sp.train], y[sp.train])

        # In-sample probabilities drive parameter selection. They are optimistic
        # by construction -- the model has seen these rows -- but they are
        # TRAINING rows, so nothing from the test window leaks. The optimism
        # costs realism in the chosen delta, not validity of the test score.
        p_train = np.asarray(model.predict_proba(X.iloc[sp.train]), dtype=float)
        rule = select_params(frame.iloc[sp.train].reset_index(drop=True),
                             p_train, cost)

        p_test = np.asarray(model.predict_proba(X.iloc[sp.test]), dtype=float)
        out.append(pd.DataFrame({
            "date": frame["date"].iloc[sp.test].to_numpy(),
            "prob_up": p_test,
            "delta": rule.delta,
            "min_hold": rule.min_hold,
            "asymmetric": rule.asymmetric,
            "exit_delta": rule.exit_delta,
            "y_true": y[sp.test],
        }))

    preds = pd.concat(out, ignore_index=True)

    # Positions are generated ONCE over the stitched test series rather than
    # per fold. Per-fold generation would reset the position at every month
    # boundary and manufacture a trade there -- an artifact that would inflate
    # turnover by ~100 round trips and quietly sabotage the thing being tested.
    stitched = Rule(delta=float(preds["delta"].median()),
                    min_hold=int(preds["min_hold"].median()),
                    asymmetric=bool(preds["asymmetric"].mean() > 0.5),
                    exit_delta=float(preds["exit_delta"].median()))
    preds["position"] = stitched.positions(preds["prob_up"].to_numpy())
    preds.attrs["stitched_rule"] = stitched
    return preds
