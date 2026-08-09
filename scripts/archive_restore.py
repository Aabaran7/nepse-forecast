"""Rebuild data/archive and data/deep from the committed CSV mirror.

The exact inverse of `archive_backup.py`'s export(), and it exists for one
reason: a GitHub runner is wiped after every job. Without a restore step the
daily CI job would start from an empty data/ every morning, `archive_pull.py`
would see no history, and the append-only store would be re-created from
whatever NEPSE happens to serve that day -- silently throwing away every session
older than the rolling window. That is the §3.4 failure mode with extra steps.

Two rules, both of which exist because a restore runs unattended:

  IT REFUSES TO SHRINK THE ARCHIVE. If a parquet already on disk holds more rows
  than the CSV being restored, the restore stops rather than overwriting. A
  restore is meant to rebuild an empty machine, not to roll a good archive back
  to a stale mirror. `--force` exists for the deliberate case.

  DTYPES ARE READ, NOT GUESSED. CSV has no types, and these types cannot be
  inferred from the column name: `securityId` is int64 in today_price but object
  in securities, and `id` is int64 in indices but object in today_price. Guessing
  wrong makes archive.merge() raise on the key join -- or worse, silently stop
  matching keys and append a second copy of the entire history. So
  archive_backup.py writes `_dtypes.json` beside the CSVs and this reads it.
  Mirrors written before that existed fall back to coercing dates only, which is
  the one rule that is always right.

Usage:
    .venv/bin/python scripts/archive_restore.py --from data-mirror
    .venv/bin/python scripts/archive_restore.py --from data-mirror --force
    .venv/bin/python scripts/archive_restore.py --from data-mirror --check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("archive_restore")

# Mirrors archive_backup.SOURCES / EXTRA_DIRS. Kept as a literal rather than
# imported so a restore never depends on the backup script parsing cleanly --
# this is the recovery path, and it should have as few moving parts as possible.
DATASETS = {"archive": Path("data/archive"), "deep": Path("data/deep")}
VERBATIM = {"predictions": Path("predictions")}

class RestoreWouldShrink(RuntimeError):
    """The mirror holds fewer rows than the archive already on disk."""


def coerce(df: pd.DataFrame, schema: dict[str, str] | None) -> pd.DataFrame:
    """Put back the types the CSV threw away.

    `schema` is the recorded parquet dtype per column. Applied column by column
    rather than in one astype() call: one unconvertible column must not abort the
    whole restore, because a restore usually runs when nothing else is left.
    """
    for c in df.columns:
        want = (schema or {}).get(c)
        try:
            if want and want.startswith("datetime"):
                # to_datetime always lands on ns; the archive holds some columns
                # at ms because that is what the original parquet write chose.
                # Values compare equal either way, so this is cosmetic -- but an
                # exact restore is a far easier invariant to check than "exact
                # except for the bits that don't matter".
                df[c] = pd.to_datetime(df[c], errors="coerce").astype(want)
            elif want == "object":
                # NOT astype(object). A CSV column of digits reads back as int64,
                # and astype(object) only boxes the ints -- pyarrow then re-infers
                # int64 on write and the restore silently undoes itself. This is
                # real: securities.securityId is stored as the STRING "131", and
                # a merge of int 131 against string "131" matches nothing.
                # astype("string") keeps nulls as <NA> where astype(str) would
                # write the literal "nan"; the trailing astype(object) drops the
                # StringDtype label so the restored column is object, exactly as
                # the archive holds it, rather than merely equal in value.
                df[c] = df[c].astype("string").astype(object)
            elif want:
                df[c] = df[c].astype(want)
            elif c.lower().endswith(("date", "_date")) or c == "date":
                # No schema (an old mirror). Dates are the only safe guess, and
                # also the one that matters most -- every archive key is dated.
                df[c] = pd.to_datetime(df[c], errors="coerce")
        except (ValueError, TypeError) as exc:
            log.warning("could not restore %s to %s (%s); leaving as %s",
                        c, want, type(exc).__name__, df[c].dtype)
    return df


def restore_dir(mirror: Path, target: Path, force: bool, check: bool) -> list[str]:
    out: list[str] = []
    if not mirror.is_dir():
        log.warning("no mirror at %s; skipping", mirror)
        return out

    schemas: dict[str, dict[str, str]] = {}
    dt_file = mirror / "_dtypes.json"
    if dt_file.exists():
        schemas = json.loads(dt_file.read_text())
    else:
        log.warning("%s has no _dtypes.json; restoring dates only. Re-run "
                    "archive_backup.py to record the schema.", mirror)

    target.mkdir(parents=True, exist_ok=True)
    for csv in sorted(mirror.glob("*.csv")):
        if csv.name.startswith("_"):        # _manifest.csv and friends
            (target / csv.name).write_bytes(csv.read_bytes())
            continue

        pq = target / f"{csv.stem}.parquet"
        df = coerce(pd.read_csv(csv), schemas.get(csv.stem))

        if pq.exists():
            have = len(pd.read_parquet(pq))
            if have > len(df) and not force:
                raise RestoreWouldShrink(
                    f"{pq} holds {have:,} rows but the mirror has only "
                    f"{len(df):,}. Refusing to overwrite -- the archive on disk "
                    f"is AHEAD of the mirror, which usually means a pull ran "
                    f"and was never backed up. Back it up first, or pass "
                    f"--force if you are certain the mirror is correct.")
            if have == len(df):
                out.append(f"{target.name}/{csv.stem}: {len(df):,} rows (unchanged)")
                continue

        if not check:
            df.to_parquet(pq, index=False)
        out.append(f"{target.name}/{csv.stem}: {len(df):,} rows")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="mirror", type=Path, default=Path("data-mirror"),
                    help="the committed CSV mirror (default: data-mirror)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite even when the archive on disk is larger")
    ap.add_argument("--check", action="store_true",
                    help="report what would be restored; write nothing")
    args = ap.parse_args()

    if not args.mirror.exists():
        # Not an error. The very first CI run has no mirror yet, and the pull
        # that follows will create the archive from scratch -- which is correct
        # exactly once. Failing here would block that first run forever.
        log.warning("%s does not exist. If this is the first run, that is "
                    "expected; the archive will be created by the pull.", args.mirror)
        return 0

    summary: list[str] = []
    for name, target in DATASETS.items():
        summary += restore_dir(args.mirror / name, target, args.force, args.check)

    for name, target in VERBATIM.items():
        src = args.mirror / name
        if not src.is_dir():
            continue
        target.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(src.glob("*.csv")):
            dst = target / f.name
            # §7: a logged prediction is never rewritten, not even by a restore.
            # If it is already here it is authoritative; the mirror is a copy.
            if dst.exists():
                continue
            if not args.check:
                dst.write_bytes(f.read_bytes())
            n += 1
        if n:
            summary.append(f"{name}: {n} new file(s)")

    if not summary:
        log.info("nothing to restore")
        return 0

    verb = "would restore" if args.check else "restored"
    for line in summary:
        log.info("%s %s", verb, line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
