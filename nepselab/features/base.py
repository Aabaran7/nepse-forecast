"""Feature module interface (plan §5).

§5: "Each source is an independent module with a common interface, individually
toggleable, so an ablation is one config change." That is the whole design.

Every module declares the window it can actually cover, because they are not the
same and §5's table makes the differences load-bearing: index OHLC from 2016,
turnover from 2017 (a 420x units break at the boundary), Reddit from its own
dump start. §8 requires the engine to REFUSE a config whose features do not
cover the requested window rather than silently returning a short sample --
quietly handing back 1,400 rows when 2,400 were asked for is how a sample
shrinks without anyone noticing.

One invariant applies to every module and is checked in tests: a feature at
session t may use data from t and earlier, never later. `shift()` discipline is
the difference between a result and a leak.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd


class FeatureModule(Protocol):
    name: str
    available_from: pd.Timestamp

    def build(self, sessions: pd.DataFrame) -> pd.DataFrame:
        """Return a frame indexed like `sessions` with this module's columns."""
        ...


@dataclass
class Assembled:
    frame: pd.DataFrame
    columns: dict[str, list[str]]        # module name -> its columns
    dropped_rows: int
    coverage: dict[str, pd.Timestamp] = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return [c for cols in self.columns.values() for c in cols]


def assemble(sessions: pd.DataFrame, modules: list[FeatureModule],
             start: pd.Timestamp | None = None,
             require_full_coverage: bool = True) -> Assembled:
    """Join every enabled module onto `sessions` and drop incomplete rows.

    Rows are dropped only at the ends, where a module has not started yet or a
    rolling window has not filled. `require_full_coverage` makes a module whose
    data begins after `start` an error rather than a silent truncation.
    """
    sessions = sessions.sort_values("date").reset_index(drop=True)
    out = sessions.copy()
    cols: dict[str, list[str]] = {}
    coverage: dict[str, pd.Timestamp] = {}

    for m in modules:
        built = m.build(sessions)
        if len(built) != len(sessions):
            raise ValueError(f"module {m.name} returned {len(built)} rows for "
                             f"{len(sessions)} sessions")
        new = [c for c in built.columns if c != "date"]
        overlap = set(new) & set(out.columns)
        if overlap:
            raise ValueError(f"module {m.name} would overwrite {sorted(overlap)}")
        out[new] = built[new].to_numpy()
        cols[m.name] = new
        coverage[m.name] = m.available_from

    if start is not None and require_full_coverage:
        late = {n: d for n, d in coverage.items() if pd.Timestamp(d) > pd.Timestamp(start)}
        if late:
            raise ValueError(
                "requested start "
                f"{pd.Timestamp(start).date()} precedes these modules' data: "
                + ", ".join(f"{n} (from {pd.Timestamp(d).date()})"
                            for n, d in late.items())
                + ". Either move the start date or disable the module -- "
                  "silently shortening the sample is not an option (§8).")

    feature_cols = [c for cols_ in cols.values() for c in cols_]
    before = len(out)
    out = out.dropna(subset=feature_cols).reset_index(drop=True)
    return Assembled(frame=out, columns=cols, dropped_rows=before - len(out),
                     coverage=coverage)
