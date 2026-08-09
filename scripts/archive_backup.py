"""Push the archive off this machine. Runs from cron, after every pull.

The archive is the only asset in this project that cannot be rebuilt (plan
§3.4): NEPSE serves a rolling year, so a session that ages out exists nowhere
except here. Until now "here" meant one directory on one disk with no
redundancy -- a failure mode strictly worse than the five sessions already lost.

Backs up as CSV rather than parquet, which is a deliberate choice on two counts:

  SIZE. Parquet is a compressed binary blob rewritten whole on every merge, so
  git stores a fresh copy each day. today_price alone reaches ~8 MB, which is
  ~2 GB/yr of commits. The CSVs are sorted by the same keys the archive sorts
  on, so a day's rows land as an insertion git deltas down to almost nothing.

  RECOVERABILITY. A backup that needs pyarrow and a matching pandas to open is
  a backup with a dependency. CSV opens in anything, forever, which matters more
  for the one irreplaceable thing here than column types do.

Idempotent and quiet: if nothing changed, it commits nothing and exits 0.

Usage:
    .venv/bin/python scripts/archive_backup.py [--dir PATH] [--no-push]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("archive_backup")

DEFAULT_DIR = Path.home() / ".local/share/nepse-archive-backup"
# data/orderbook is here despite being the fastest-growing of the three: it is
# the only source with NO history at all -- NEPSE serves an empty book outside
# the session, so a snapshot not backed up is gone in a way even the rolling-year
# data is not. If it outgrows the repo, the answer is to thin old snapshots
# deliberately, not to leave the recent ones unbacked.
SOURCES = [Path("data/archive"), Path("data/deep"), Path("data/orderbook"),
           Path("data/fundamentals"), Path("data/news")]
# The forward log is small, text, and irreplaceable in a different way from
# the archive: §7 forbids rewriting it, so losing it cannot be repaired by
# re-running anything. It is already CSV, so it is copied verbatim.
EXTRA_DIRS = [Path("predictions")]


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=check)


def export(repo: Path) -> list[str]:
    """Mirror every parquet into the backup repo as CSV. Returns dataset names."""
    written = []
    for src in SOURCES:
        if not src.exists():
            continue
        out_dir = repo / src.name
        out_dir.mkdir(parents=True, exist_ok=True)
        # CSV carries no types, and the types here cannot be guessed: securityId
        # is int64 in today_price but object in securities; `id` is int64 in
        # indices but object in today_price. A restore that guesses wrong makes
        # archive.merge() fail outright, or worse, silently stop matching keys
        # and append a duplicate copy of the whole history. So the schema rides
        # along beside the data. Read by scripts/archive_restore.py.
        dtypes: dict[str, dict[str, str]] = {}
        for pq in sorted(src.glob("*.parquet")):
            df = pd.read_parquet(pq)
            dtypes[pq.stem] = {c: str(t) for c, t in df.dtypes.items()}
            # Sort by whatever date-ish columns exist so a backfilled session
            # inserts in place rather than reshuffling the file. Without this
            # the diff is the whole file and the size argument above collapses.
            keys = [c for c in df.columns
                    if c.lower().endswith(("date", "_date")) or c == "date"]
            keys += [c for c in ("securityId", "symbol", "exchangeIndexId") if c in df.columns]
            if keys:
                df = df.sort_values(keys, kind="stable")
            df.to_csv(out_dir / f"{pq.stem}.csv", index=False)
            written.append(f"{src.name}/{pq.stem}: {len(df)} rows")

        if dtypes:
            (out_dir / "_dtypes.json").write_text(json.dumps(dtypes, indent=1, sort_keys=True))

        # The manifest is the provenance trail for every pull ever made.
        man = src / "_manifest.csv"
        if man.exists():
            (out_dir / "_manifest.csv").write_bytes(man.read_bytes())

        # Raw payloads, where a store keeps them. The order book saves its JSON
        # before parsing precisely because the schema is unknown, and that
        # safeguard is worthless if the raw files are the one thing not backed
        # up. Copied verbatim -- they are already small and already text.
        raw = src / "_raw"
        if raw.is_dir():
            out_raw = out_dir / "_raw"
            out_raw.mkdir(parents=True, exist_ok=True)
            n = 0
            for j in sorted(raw.glob("*.json")):
                dst = out_raw / j.name
                if not dst.exists():          # payloads are immutable once written
                    dst.write_bytes(j.read_bytes())
                    n += 1
            if n:
                written.append(f"{src.name}/_raw: {n} new payload(s)")

    for src in EXTRA_DIRS:
        if not src.exists():
            continue
        out_dir = repo / src.name
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(src.glob("*.csv")):
            (out_dir / f.name).write_bytes(f.read_bytes())
            n += 1
        if n:
            written.append(f"{src.name}: {n} file(s)")
    return written


def write_readme(repo: Path, summary: list[str]) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    body = [
        "# NEPSE archive backup",
        "",
        "Automated off-machine copy of `data/archive/` and `data/deep/` from the",
        "NEPSE forecasting project. **Do not edit by hand.**",
        "",
        "NEPSE's API serves a rolling ~1 year and silently ignores date",
        "parameters, so any trading session not captured on the day is destroyed",
        "permanently. The rows here cannot be re-fetched from upstream.",
        "",
        "- `archive/` — exchange-sourced, append-only. Irreplaceable.",
        "- `deep/` — third-party index history from 2016. Re-downloadable;",
        "  included only so a restore is complete.",
        "",
        "CSV rather than parquet so restores need nothing but a text reader,",
        "and so daily commits stay small.",
        "",
        f"Last updated: {ts}",
        "",
        "```",
        *summary,
        "```",
    ]
    (repo / "README.md").write_text("\n".join(body) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path,
                    default=Path(os.environ.get("NEPSE_BACKUP_DIR", DEFAULT_DIR)))
    ap.add_argument("--no-push", action="store_true",
                    help="commit locally but do not push (for testing)")
    ap.add_argument("--no-git", action="store_true",
                    help="export the CSVs and stop; the caller owns the commit")
    args = ap.parse_args()
    repo = args.dir

    # --no-git exists because the mirror moved INSIDE this repository
    # (data-mirror/) when the daily job moved to GitHub Actions. It is no longer
    # a separate clone, so there is no .git of its own to commit into, and the
    # outer repo's own commit step does that work. Without this the daily job
    # would have failed on its very first run at the step whose entire purpose
    # is protecting the one irreplaceable thing here.
    if not args.no_git and not (repo / ".git").is_dir():
        log.error("%s is not a git repo. Either clone one there, or pass "
                  "--no-git if it lives inside this repository and the caller "
                  "commits it.", repo)
        sys.exit(1)

    summary = export(repo)
    if not summary:
        log.warning("nothing to back up -- no parquet found under %s",
                    ", ".join(str(s) for s in SOURCES))
        sys.exit(0)

    if args.no_git:
        write_readme(repo, summary)
        log.info("exported (no commit): %s", "; ".join(summary))
        return

    # Decide on the DATA before touching the README. The README carries a
    # timestamp, so writing it first dirties the tree unconditionally and turns
    # "commit only when something changed" into a commit every single run --
    # twice a day forever, mostly saying nothing.
    git(repo, "add", "-A")
    if not git(repo, "status", "--porcelain").stdout.strip():
        log.info("archive unchanged; nothing to commit")
        return

    write_readme(repo, summary)
    git(repo, "add", "-A")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    msg = f"archive {stamp}\n\n" + "\n".join(summary)
    git(repo, "commit", "-q", "-m", msg)
    log.info("committed: %s", "; ".join(summary))

    if args.no_push:
        log.info("--no-push set; leaving commit local")
        return

    # A failed push must not look like a failed backup: the commit is already
    # safe on disk and the next run pushes both. Cron should not get mail for a
    # dropped wifi connection.
    r = git(repo, "push", "-q", check=False)
    if r.returncode:
        log.warning("push failed (%s); commit is local and will go up next run",
                    r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "unknown")
    else:
        log.info("pushed to remote")


if __name__ == "__main__":
    main()
