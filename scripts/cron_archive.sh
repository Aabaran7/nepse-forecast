#!/usr/bin/env bash
# Daily archival run, for cron. See plan §3.4.
#
# NEPSE retains a rolling ~1 year, so a trading day that is never archived is
# destroyed permanently. This wrapper exists because that failure is SILENT:
# nothing errors when a day is missed, the archive just quietly has a hole.
#
# Two design points follow from that:
#   - It runs twice a day. archive_pull.py is idempotent and incremental, so the
#     second run costs almost nothing when the first succeeded, and rescues the
#     day when the first hit NEPSE's throttling.
#   - It never returns nonzero for a normal miss (holiday, closure, throttle).
#     Only setup failures are errors worth a cron mail.

set -uo pipefail

PROJECT="/home/aabaran/Desktop/NEPSE"
PY="$PROJECT/.venv/bin/python"
LOG="$PROJECT/data/archive/_cron.log"

cd "$PROJECT" || { echo "FATAL: cannot cd to $PROJECT"; exit 1; }
[ -x "$PY" ] || { echo "FATAL: no interpreter at $PY"; exit 1; }

mkdir -p "$(dirname "$LOG")"

# Serialise against a manual run or a slow previous cron: archive.merge() is a
# read-modify-write of a whole parquet file, so two concurrent pulls can lose
# one side's rows entirely. -n skips rather than queues -- if a pull is already
# in flight, this run has nothing to add.
exec 9>"$PROJECT/data/archive/.lock"
if ! flock -n 9; then
  echo "$(date -Is) another archive run holds the lock; skipping" >> "$LOG"
  exit 0
fi

{
  echo "=== $(date -Is) (KTM $(TZ=Asia/Kathmandu date '+%F %T')) ==="
  "$PY" scripts/archive_pull.py 2>&1
  echo "--- exit=$? ---"
} >> "$LOG" 2>&1

# Keep the log from growing without bound; the archive itself is the record.
if [ -f "$LOG" ] && [ "$(wc -l < "$LOG")" -gt 20000 ]; then
  tail -n 5000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi

exit 0
