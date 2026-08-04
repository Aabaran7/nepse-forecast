"""Immutable forward prediction log (plan §7).

§7's rule is one line and it is absolute: **never overwrite a past prediction.**
The log accumulates indefinitely.

That rule is not bookkeeping. §7 is explicit that the forward run's purpose is
leak detection and pipeline validation, not evidence of an edge -- ~40 trading
days cannot distinguish 55% from 50%. It works as leak detection *only* if a
prediction cannot be quietly revised after the outcome is known. A log that can
be rewritten proves nothing at all, because the most likely thing to rewrite it
is a bug fix applied after seeing that the prediction was wrong.

So: append-only, one file per date, and a re-run on a date that already has an
entry raises rather than replacing. Each row carries the model version and the
git hash, so a prediction can be tied to the exact code that made it.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LOG_DIR = Path("predictions")


class PredictionExists(RuntimeError):
    """A prediction for this date and model already exists. §7 forbids replacing it."""


def git_hash() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def log_path(as_of: pd.Timestamp, root: Path = LOG_DIR) -> Path:
    return root / f"{pd.Timestamp(as_of).date()}.csv"


def append(as_of: pd.Timestamp, rows: list[dict], root: Path = LOG_DIR,
           model_version: str = "v1") -> Path:
    """Write predictions made AT `as_of`, for horizons ahead of it.

    `rows` each need at least `horizon` and `prediction`. Everything else --
    features, probability, target date -- is carried through as given.
    """
    root.mkdir(parents=True, exist_ok=True)
    path = log_path(as_of, root)

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gh = git_hash()
    new = pd.DataFrame(rows)
    new.insert(0, "as_of", pd.Timestamp(as_of).date())
    new["model_version"] = model_version
    new["git_hash"] = gh
    new["logged_utc"] = stamp

    if path.exists():
        old = pd.read_csv(path)
        key = ["horizon", "model_version"]
        clash = old.merge(new[key], on=key)
        if len(clash):
            raise PredictionExists(
                f"{path} already holds predictions for "
                f"{sorted(set(clash['horizon']))} at model_version="
                f"{model_version}. §7 forbids overwriting a past prediction; "
                f"if the model changed, bump model_version instead.")
        new = pd.concat([old, new], ignore_index=True)

    new.to_csv(path, index=False)
    return path


def load_all(root: Path = LOG_DIR) -> pd.DataFrame:
    files = sorted(root.glob("*.csv")) if root.exists() else []
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["as_of"] = pd.to_datetime(df["as_of"])
    return df.sort_values(["as_of", "horizon"]).reset_index(drop=True)


def score_log(actuals: pd.DataFrame, root: Path = LOG_DIR) -> pd.DataFrame:
    """Join logged predictions to outcomes as they arrive (§7's scoring script).

    `actuals` needs `date` and `close`. Predictions whose target session has not
    happened yet are returned with a null outcome rather than dropped -- how many
    are still open is itself information about whether the job is running.
    """
    log = load_all(root)
    if log.empty:
        return log

    a = actuals.sort_values("date").reset_index(drop=True)
    dates = a["date"].to_numpy()
    close = a["close"].to_numpy()
    pos = {pd.Timestamp(d): i for i, d in enumerate(dates)}

    outcomes = []
    for _, r in log.iterrows():
        i = pos.get(pd.Timestamp(r["as_of"]))
        h = int(r["horizon"])
        if i is None or i + h >= len(close):
            outcomes.append({"actual": None, "fwd_return": None, "correct": None})
            continue
        fwd = close[i + h] / close[i] - 1.0
        actual = int(fwd > 0)
        outcomes.append({"actual": actual, "fwd_return": float(fwd),
                         "correct": int(actual == int(r["prediction"]))})
    return pd.concat([log, pd.DataFrame(outcomes)], axis=1)
