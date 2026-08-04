"""Walk-forward splitting with an embargo (plan §2). No k-fold, ever.

The embargo is the part that does the work, and the part most often skipped.
A training row at index `i` carries a label built from the close at `i + h`.
If the test window starts at `s`, then every training row with `i + h >= s`
had its *label* formed from prices inside the test period. Those rows are not
merely adjacent to the test set; they contain it. Training on them leaks the
answer, and the leak is completely invisible in the output -- it shows up as a
model that works in backtest and dies live, which §7 says the forward log exists
to catch after the fact. Better to make it impossible here.

So the last `h` training rows before every test window are dropped. For h=1 that
is one row and feels like pedantry; for h=5 it is five, and the overlap it
removes is exactly the correlation that makes h=5 CIs untrustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    """One walk-forward fold. Indices are positional into the labelled frame."""
    train: np.ndarray
    test: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargoed: int

    def __repr__(self) -> str:
        return (f"<Split train {self.train_start.date()}..{self.train_end.date()} "
                f"({len(self.train)}) test {self.test_start.date()}.."
                f"{self.test_end.date()} ({len(self.test)}) "
                f"embargo -{self.embargoed}>")


def walk_forward(dates: pd.Series, horizon: int, initial_train: int = 500,
                 retrain_freq: str = "MS", expanding: bool = True,
                 min_test: int = 1) -> list[Split]:
    """Retrain on a calendar schedule, predict forward, never look back.

    `retrain_freq` is a pandas offset alias; "MS" is §2's monthly retrain. Each
    fold trains on everything up to the retrain date (minus the embargo) and
    tests on the rows until the next retrain date.

    `expanding=True` grows the training window; False makes it a rolling window
    of `initial_train` rows. Expanding is the default because ~2,400 sessions is
    already small and throwing away the early years costs more than the
    non-stationarity it buys.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    if not dates.is_monotonic_increasing:
        raise ValueError("dates must be sorted ascending")
    n = len(dates)
    if initial_train >= n:
        raise ValueError(f"initial_train={initial_train} but only {n} rows")

    # Retrain boundaries: the first is the end of the initial training window,
    # then every period start after it.
    first = dates.iloc[initial_train]
    boundaries = pd.date_range(first.normalize(), dates.iloc[-1], freq=retrain_freq)
    boundaries = [b for b in boundaries if b > first]
    edges = [first] + list(boundaries) + [dates.iloc[-1] + pd.Timedelta(days=1)]

    splits: list[Split] = []
    for start, stop in zip(edges[:-1], edges[1:]):
        test_mask = (dates >= start) & (dates < stop)
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if len(test_idx) < min_test:
            continue

        s = int(test_idx[0])
        # THE EMBARGO. Row i is safe only if its label (formed at i + horizon)
        # closes strictly before the test window opens.
        train_hi = s - horizon
        if train_hi <= 0:
            continue
        train_lo = 0 if expanding else max(0, train_hi - initial_train)
        train_idx = np.arange(train_lo, train_hi)
        if len(train_idx) == 0:
            continue

        splits.append(Split(
            train=train_idx, test=test_idx,
            train_start=dates.iloc[train_idx[0]], train_end=dates.iloc[train_idx[-1]],
            test_start=dates.iloc[test_idx[0]], test_end=dates.iloc[test_idx[-1]],
            embargoed=horizon,
        ))
    return splits


def assert_no_leakage(splits: list[Split], dates: pd.Series, horizon: int) -> None:
    """Raise if any fold's training labels reach into its test window.

    Called by the harness before it runs anything. §2's rules are only worth
    stating if something checks them, and this is the check.
    """
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    for sp in splits:
        if len(sp.train) == 0:
            continue
        last_train = int(sp.train[-1])
        first_test = int(sp.test[0])
        label_closes_at = last_train + horizon
        if label_closes_at >= first_test:
            raise AssertionError(
                f"leak: training row {last_train} ({dates.iloc[last_train].date()}) "
                f"has a label closing at row {label_closes_at}, but the test "
                f"window opens at row {first_test} "
                f"({dates.iloc[first_test].date()})")
        if set(sp.train) & set(sp.test):
            raise AssertionError("leak: train and test indices overlap")
