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

Last updated: 2026-08-09T17:59:00+00:00

```
archive/indices: 3915 rows
archive/market_summary: 229 rows
archive/securities: 3965 rows
archive/ticker_history: 1120 rows
archive/today_price: 76195 rows
deep/nepse_index_deep: 2434 rows
deep/reddit_daily: 2376 rows
predictions: 1 file(s)
```
