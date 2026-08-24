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

Last updated: 2026-08-24T09:06:31+00:00

```
archive/company_info: 645 rows
archive/corporate_actions: 171 rows
archive/indices: 4085 rows
archive/market_summary: 239 rows
archive/securities: 8404 rows
archive/ticker_history: 1120 rows
archive/today_price: 79658 rows
deep/nepse_index_deep: 2434 rows
deep/reddit_daily: 2376 rows
orderbook/orderbook: 481 rows
orderbook/_raw: 1 new payload(s)
fundamentals/dividend_history: 1204 rows
fundamentals/fundamentals: 278 rows
news/headlines: 240 rows
news/sentiment: 182 rows
predictions: 1 file(s)
```
