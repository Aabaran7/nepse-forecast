"""Append-only parquet archive.

NEPSE retains a rolling ~1 year on every endpoint (see
scripts/phase1_probe_depth.py): the history APIs ignore their date parameters
entirely and hand back the same ~225-session window whatever you ask for, and
`today_price` refuses dates before that window starts. Sessions that age out are
gone -- five of them fell off between the Phase 0 pull and 2026-08-02.

So the archive is not a cache we can rebuild; it is the only copy. Two rules
follow, and both are enforced here rather than left to callers:

  1. Writes only ever add rows. Nothing in this module deletes or overwrites.
  2. When the upstream feed contradicts a row we already hold, we keep OURS and
     record the conflict. A silent upstream revision to a past session is a
     data-integrity event worth seeing, not something to merge away.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

ARCHIVE = Path("data/archive")
MANIFEST = ARCHIVE / "_manifest.csv"

# dataset -> key columns identifying one observation
KEYS: dict[str, list[str]] = {
    "indices": ["exchangeIndexId", "businessDate"],
    "market_summary": ["businessDate"],
    "today_price": ["businessDate", "securityId"],
    "securities": ["snapshot_date", "securityId"],
    "ticker_history": ["businessDate", "symbol"],
}


class MergeResult:
    def __init__(self, dataset: str, added: int, total: int, conflicts: pd.DataFrame):
        self.dataset = dataset
        self.added = added
        self.total = total
        self.conflicts = conflicts

    def __repr__(self) -> str:
        c = len(self.conflicts)
        return (f"<{self.dataset}: +{self.added} -> {self.total}"
                + (f", {c} CONFLICTS" if c else "") + ">")


def _normalise(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    df = df.copy()
    for k in keys:
        if k.endswith("Date") or k.endswith("_date"):
            # Normalise to midnight so a timestamped pull and a date-only pull
            # of the same session don't land as two different rows.
            df[k] = pd.to_datetime(df[k], errors="coerce").dt.normalize()
    return df


def _find_conflicts(old: pd.DataFrame, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Rows present in both, keyed identically, but differing in some value."""
    shared = [c for c in old.columns if c in new.columns and c not in keys]
    if not shared or old.empty or new.empty:
        return pd.DataFrame()
    o = old[keys + shared].merge(new[keys + shared], on=keys, suffixes=("_archived", "_upstream"))
    if o.empty:
        return pd.DataFrame()
    diff = pd.Series(False, index=o.index)
    changed_cols: list[str] = []
    for c in shared:
        a, b = o[f"{c}_archived"], o[f"{c}_upstream"]
        # NaN == NaN counts as equal; float noise below 1e-9 does not count.
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            d = ~((a - b).abs() <= 1e-9) & ~(a.isna() & b.isna())
        else:
            d = (a.astype(str) != b.astype(str)) & ~(a.isna() & b.isna())
        if d.any():
            changed_cols.append(c)
        diff |= d
    if not diff.any():
        return pd.DataFrame()
    out = o[diff].copy()
    out["changed_columns"] = ", ".join(changed_cols)
    return out


def merge(dataset: str, new: pd.DataFrame, root: Path = ARCHIVE) -> MergeResult:
    """Fold `new` into the archived copy of `dataset`. Never removes a row."""
    if dataset not in KEYS:
        raise KeyError(f"unknown dataset {dataset!r}; add it to archive.KEYS")
    keys = KEYS[dataset]
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{dataset}.parquet"

    if new is None or new.empty:
        existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        return MergeResult(dataset, 0, len(existing), pd.DataFrame())

    missing = [k for k in keys if k not in new.columns]
    if missing:
        raise ValueError(f"{dataset}: incoming frame lacks key column(s) {missing}")

    new = _normalise(new, keys).drop_duplicates(subset=keys, keep="last")

    if not path.exists():
        new.to_parquet(path, index=False)
        return MergeResult(dataset, len(new), len(new), pd.DataFrame())

    old = _normalise(pd.read_parquet(path), keys)
    conflicts = _find_conflicts(old, new, keys)

    # keep="first" with old concatenated first is what makes this append-only:
    # an incoming row that collides with an archived one is discarded, not applied.
    combined = (pd.concat([old, new], ignore_index=True)
                  .drop_duplicates(subset=keys, keep="first")
                  .sort_values(keys)
                  .reset_index(drop=True))
    combined.to_parquet(path, index=False)
    return MergeResult(dataset, len(combined) - len(old), len(combined), conflicts)


def record(results: list[MergeResult], root: Path = ARCHIVE) -> None:
    """Append this run's outcome to the manifest, so provenance is reconstructable."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [{"run_utc": ts, "dataset": r.dataset, "rows_added": r.added,
             "rows_total": r.total, "conflicts": len(r.conflicts)} for r in results]
    df = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST, mode="a", header=not MANIFEST.exists(), index=False)

    for r in results:
        if len(r.conflicts):
            out = root / "_conflicts" / f"{r.dataset}_{ts.replace(':', '')}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            r.conflicts.to_csv(out, index=False)
            log.warning("%s: %d rows conflict with the archive; wrote %s",
                        r.dataset, len(r.conflicts), out)


def load(dataset: str, root: Path = ARCHIVE) -> pd.DataFrame:
    path = root / f"{dataset}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()
