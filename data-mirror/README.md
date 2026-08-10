# NEPSE archive backup

Automated off-machine copy of `data/archive/` and `data/deep/` from the
NEPSE forecasting project. **Do not edit by hand.**

NEPSE's API serves a rolling ~1 year and silently ignores date
parameters, so any trading session not captured on the day is destroyed
permanently. The rows here cannot be re-fetched from upstream.

- `archive/` — exchange-sourced, append-only. Irreplaceable.
- `deep/` — third-party index history from 2016. Re-downloadable;
  included only so a restore is complete.

CSV rather than parquet so restores need nothing but a text reader,
and so daily commits stay small.

Last updated: 2026-08-10T09:59:40+00:00

```
predictions: 1 file(s)
```
