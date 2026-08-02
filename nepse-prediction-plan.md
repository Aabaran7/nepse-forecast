# NEPSE Direction Forecasting Simulator — Project Plan

**What it is:** a personal forecasting simulator. Predict the sign of the NEPSE index return 1 day and 1 week ahead, score it honestly on history, then run it forward on live data for several months, logging predictions daily. Capital is deployed only if the forward log and the backtest agree.

**Not a paper.** No replication, no literature survey, no dataset release, no publication target.

---

## 1. Objectives

**Primary:**
1. **A walk-forward evaluation harness** with a correct cost model and fill logic. Models are interchangeable; the harness is what makes any result believable.
2. **Directional forecasts of the NEPSE index**, h=1 day and h=5 days, scored against 50% and majority-class baselines plus net-of-cost PnL.
3. **An immutable forward prediction log** that accumulates indefinitely and serves as leak detection and pipeline validation.
4. **A documented "no edge found" outcome** — thresholds fixed in §6 *before* results are seen.

**Non-goals:**
- Index *price level* prediction (lag-1 wins; dead end). Never report RMSE on price level.
- Sequence models — ~1500 usable daily observations is far too few. No LSTM/GRU/transformer.
- Ticker-level cross-sectional prediction (Reddit EDA killed the alt-data case; see §5).
- Publication of any kind.

**Success criteria:** one command from raw data to cost-adjusted, regime-split results; every model benchmarked against 50%, majority-class, and buy-and-hold; a forward job that has never overwritten a past prediction.

---

## 2. Target and scoring protocol

**Target:** sign of the NEPSE index return at h=1 trading day and h=5 trading days.

**Validation:**
- **Walk-forward only.** Train strictly on data before each prediction date; retrain monthly; embargo gap between train and test to kill leakage from overlapping labels. No k-fold, ever.
- All scaling and feature fitting inside the training window only.
- h=5 labels overlap — use non-overlapping weekly blocks for significance, and block bootstrap for confidence intervals. Naive CIs on overlapping labels are far too tight.

**Metrics:**
- Directional accuracy and F1, vs. **both** 50% and the majority class (NEPSE has long directional runs; majority-class is the harder baseline and the one that matters). In the verified 2025-07→2026-07 window only **45.5%** of index days were up — so the majority class is *down* at 54.5%, and the §6 "+2pp over majority-class" bar means clearing ~56.5%, not ~52%. Compute the majority baseline per regime; do not assume it sits near 50%.
- Net-of-cost simulated PnL: net Sharpe at a stated capital base, net CAGR, max drawdown, turnover, % of PnL from top-5 days.
- Never RMSE on price level.

**Regime split — report every result twice:**
- **2020-06 → 2021-12 (mania).**
- **2022-01 → present (chop).**

A signal that exists in only one regime is labelled as such in the results table, not averaged away. Pooled numbers are reported alongside, never instead.

**Power, computed before any model runs.** Roughly ~1500 usable daily observations, of which the walk-forward test set is ~1000 after the initial training window; non-overlapping weekly gives ~200 test blocks. That supports detecting roughly a 4pp daily edge and a 7pp weekly edge at α=0.05, 80% power — **TODO: compute exactly and put the numbers in the results table header.** Anything smaller than the detectable edge cannot be claimed, whatever the point estimate says.

---

## 3. Data correctness

The part most likely to silently break everything.

### 3.1 Sources
| Data | Source | Notes |
|---|---|---|
| OHLCV, index + scrips | `polymorphisma/nepse_scraper` (PyPI) | Use the library's session, not raw HTTP: plain `curl` cannot complete a TLS handshake with NEPSE at all, so a connection failure is **not** evidence the API is down. History endpoints return Spring-Data page envelopes, not the bare lists the type hints claim — unwrap `content` and follow `totalPages`, or you silently get 20 rows of 225. Ticker history takes the security id as a **path** segment; `symbol` as a query param returns 400. |
| Index, sector indices, market summary | Same scraper | Exchange-computed and continuity-adjusted; the target and most features come from here. |
| Breadth (advancers/decliners) | Dated `today_price` snapshots | One call per session, ~350 rows. Only feature touching raw scrip prices. |
| Corporate actions | — | **Descoped at the Phase 0 gate** (§3.2). Sourcing tested and poor; correction does not move an index-level target. |
| Mergers & suspensions | NEPSE securities list | Only to exclude affected tickers from the breadth denominator. |
| Floorsheet | Same scraper | **TODO: establish history depth** — decides whether any intraday-derived feature is possible. Not required for any current feature. |
| Macro | NRB (rates, CPI, remittances, liquidity) | Publication-lagged — see §5. |
| Reddit | Arctic Shift dump, r/NepalStock | Already pulled; EDA complete (§5). |
| Market structure & fees | SEBON / NEPSE / CDSC circulars | Feeds `market_params` (§4). |

### 3.2 Requirements
- **Corporate action adjustment — descoped at the Phase 0 gate.** The original requirement was inherited from the cross-sectional design that has since been cut, and it does not survive the change of target. Everything this project consumes is already exchange-computed and continuity-adjusted: the **NEPSE index** (the target), the **sector indices**, and market-wide **turnover/participation**. The only feature touching raw scrip prices is breadth (advancers/decliners), where an unadjusted ex-date miscounts **one scrip out of ~350 on scattered days** — noise, not systematic bias, and independently detectable via the circuit cross-check below.
  Sourcing was also tested and is poor: NEPSE's own disclosure endpoint returns 0 rows; MeroLagani publishes bonus % by Nepali fiscal year **with no ex-date** (the field the adjustment layer actually needs); ShareSansar carries actions as prose news and JS-loaded panels, not a clean dated table. Building a validated layer would be days of brittle scraping for a correction that does not move an index-level target.
  **Decision: no adjustment layer.** Revisit only if a future feature depends on scrip-level returns.
- **Scrip state machine — downgraded with the adjustment layer.** Suspensions and BFI-merger ticker discontinuities distort scrip-level returns badly, but the only consumer left is the breadth count, where a suspended or merged ticker is one name out of ~350. Track state (`active | suspended | delisted | merged_into(X)`) well enough to *exclude* affected tickers from the daily breadth denominator; do not build the full continuous-entity resolution.
- **Trading calendar — the week changed inside the sample.** Sun–Thu through **2026-04-05**; Mon–Fri from **2026-04-10** (verified in pulled data: clean switch, no overlap). The calendar is therefore date-indexed like every other constant, not a fixed rule. This is not cosmetic: h=5 "one week ahead" spans a different weekday set either side of the boundary, day-of-week features are not comparable across it, and Reddit weekly aggregation must align to trading weeks. Frequent unscheduled closures on top (6 gaps >4 calendar days in the verified year).
- **Circuit breakers block fills at both levels — both limits now confirmed empirically.** Scrip-level is **±10%** (daily returns cluster at ±9.99% and stop); index-level is a hard **±6%** daily cap (observed extremes +6.0070% / −6.0002%). Scrip-level matters more and is the single most likely source of phantom alpha — the days a signal fires hardest are the days you cannot get filled. On daily bars, detect via `close == prev_close × (1 ± limit)` and/or `high == low`.
- **Free corporate-action detector, retained as a data check:** any `|return| > circuit limit` is *impossible* as an ordinary price move and must reconcile to a bonus, rights, or split. Unexplained breaches mean the price feed is wrong and the sanity suite should fail. (Two in the verified year: NRIC +14.99% on 2026-06-01, HDL −14.18% on 2025-11-06.)
- SQLite or parquet store; ingestion idempotent and resumable. Frozen dataset v1.0, checksummed; everything downstream reads only that.

### 3.3 Sanity suite (write tests first)
Implemented in `scripts/phase0_quality.py`; extend rather than replace.
- No negative prices/volumes; OHLC consistency; no duplicate dates.
- Calendar assertions: sessions fall only on the weekdays valid for that date's `trading_week` entry; the Sun–Thu → Mon–Fri boundary is exercised explicitly.
- No index return outside the ±6% cap; no scrip return outside ±10% that isn't flagged as a corporate-action candidate.
- Breadth reconciles with index direction (a limit-up day cannot show majority decliners).
- Tickers with a state transition are excluded from the breadth denominator on affected days.

---

## 4. Costs and market structure

**Every constant is a date-indexed lookup, never a scalar.** The settlement cycle moved T+3 → T+2 inside our sample, and CGT rates and commission tiers have also changed. Build a `market_params` table keyed by effective date; fill logic and cost model read it per trade date. Lives in `configs/market_params.yaml`, one source-citation field per row.

Fields: `settlement_cycle`, `broker_commission` (tiered by turnover band), `sebon_fee_rate`, `dp_charge_npr`, `cgt_rate` (holding period × entity type), `scrip_circuit_pct`, `index_circuit_thresholds`, `trading_hours`.

**TODO — verify each against primary sources; where a date is unconfirmed, mark it TODO rather than guessing:**
- Settlement: T+3 → T+2 effective date (NEPSE / CDSC circulars).
- Commission: equity tiers and every revision date (SEBON).
- SEBON regulatory fee rate, current and historical.
- DP charge: amount, and whether levied buy-side, sell-side, or both, per scrip per settlement (CDSC).
- CGT: rates by holding period and entity type (individual vs. institutional), with Finance Act year and effective date per change.
- Circuits: scrip daily limit, index thresholds and halt durations, plus any changes over the sample.
- Trading hours by era (session length has changed).

**Cost model — three different animals, never one round-trip percentage:**

```
per-trade  = notional × (commission_rate(notional, date) + sebon_fee_rate(date))
           + dp_charge(date)              # FLAT, per scrip per settlement
on sale    = max(0, realized_gain) × cgt_rate(holding_days, entity_type, date)
```

- The flat DP charge means results depend on the **declared capital base** — fixed costs don't scale. Declare it and state it wherever a Sharpe appears.
- CGT hits *realized gains*, not turnover. A flat percentage overcharges losing trades and hides the 365-day holding cliff.
- Report at 0×, realistic, and 2× realistic frictions.

**Fill logic:**
- **Settlement constrains exit.** A buy on *t* isn't sellable until `t + settlement_cycle(t)` trading days — enforced in the portfolio ledger.
- No fill in the signal direction when a scrip or the index sits at its limit (§3.2).
- **Manual order entry via TMS — no retail API exists.** Penalize high turnover accordingly and design toward low-frequency rebalancing; a strategy needing intraday execution is not implementable and should not be evaluated as if it were.
- Long-only (no shorting on NEPSE) — exposure expressed as index-vs-cash weight.

---

## 5. Features

Each source is an **independent module with a common interface, individually toggleable**, so an ablation is one config change. Required output: an ablation table of **price-only vs. +Reddit vs. +news vs. all**.

**1. Price/volume.** Returns at multiple lags, realized vol, breadth (advancers/decliners), turnover trends, distance from 52w high, sector index spreads. Sources, all confirmed working in Phase 0: index OHLC + 52w high from `index/history/58`; market-wide turnover, traded shares, transactions and `tradedScrips` from `market-summary-history`; sector indices from the sector index endpoint; **breadth from dated `today_price` snapshots** (one call per session, ~350 rows, `closePrice` vs `previousDayClosePrice`). Breadth self-validates against the index — on the two limit days in the verified year it read 260 up / 1 down and 10 up / 228 down.

**2. Reddit — index-level attention only.** EDA on the r/NepalStock Arctic Shift dump is **complete; do not redo it**:
- Usable window 2020-06 onward.
- Median 39 comments/day since 2021; 1233 of 2032 days have ≥30.
- 285 weeks with ≥50 comments.
- Only 15 tickers exceed 500 total mentions.

Conclusions carried forward: usable as an **index-level attention and sentiment series only**; ticker-level cross-section is dead; weekly frequency is solid, daily workable but thinner. Features: daily and weekly post/comment counts, unique authors, normalized attention z-score, comment-to-post ratio. **Attention must be normalized as a share of total subreddit activity or z-scored against a trailing 90-day window** — the sub grew across the sample, so raw counts trend mechanically and will leak a time trend into any model.

**3. News.** ShareSansar / MeroLagani headline counts plus LLM-scored sentiment.

**Sentiment scoring (Reddit and news).** The corpus is code-switched English / romanized Nepali — **do not use VADER or TextBlob**; they will score romanized Nepali as neutral noise. Use an LLM scoring pass with a constrained label schema (structured outputs, so every response parses). **Cache all scores to disk keyed by comment/headline id** — the pass runs once per document, ever. Route it through the Message Batches API (50% of standard token price) and cache the shared few-shot prefix. Note the prompt-cache minimum is model-dependent — 512 tokens on Opus 5, 1024 on Sonnet 5, 4096 on Haiku 4.5 — so a short few-shot prefix silently won't cache on the cheapest model. **TODO: pick the model and estimate total cost** from the actual document count before committing; validate whichever model is chosen against a hand-labelled sample of ~200 code-switched comments.

**4. Macro.** NRB rates, CPI, remittance inflows, liquidity. **Must use publication lags, not reference dates.** These publish weeks to months late; store `reference_date` *and* `publication_date`, and let features read only rows where `publication_date <= t`. Reference-dated macro at time *t* is lookahead and will manufacture an edge that does not exist.

---

## 6. Models and abandonment thresholds

**Logistic regression and gradient boosting only.** ~1500 daily observations does not support anything deeper. Multiple seeds where stochastic; report mean ± sd.

**Abandonment thresholds — fixed now, before any results are seen.** If the backtest fails these, the project is abandoned and written up as a null result rather than tuned further:

- **h=5 directional accuracy ≤ (majority-class baseline + 2pp)** on the pooled walk-forward test → abandon.
- **Net Sharpe < 0.4** after realistic costs at the declared capital base → no capital, regardless of accuracy.
- **Edge present only in the 2020-06 → 2021-12 mania regime** → abandon; that is a bull-market artifact, not a signal.
- **Alt-data gate:** if +Reddit and +news each add <1pp over price-only in the ablation, drop those modules. The project may continue on price-only if it still clears the above.
- **Capital gate:** deploy only after ≥60 forward trading days logged *and* forward accuracy not more than 5pp below backtest accuracy.

---

## 7. Forward run

- **Daily job:** pull new data, generate predictions for t+1 and t+5, append to an **immutable prediction log** — timestamp, features, model version, git hash. **Never overwrite a past prediction.** The log accumulates indefinitely.
- **Scoring script:** reads the log, reports rolling accuracy and simulated PnL as outcomes arrive.
- **What the forward run is for.** ~40 trading days cannot distinguish 55% accuracy from 50%. Its purpose is **leak detection and pipeline validation** — catching a feature that silently used future information, a data source that changed shape, a fill rule that never fires in live data. **The backtest remains the primary evidence.** A forward log that agrees with the backtest confirms the pipeline; it does not independently prove an edge.

---

## 8. Phases and repo

| Phase | Work | Effort |
|---|---|---|
| 0 | ~~Scraper running; 1yr index + 5 scrips; corporate-action sourcing; seed `market_params`~~ | **DONE** |
| 1 | Full ingestion (index, breadth, sector, market summary), calendar, circuit flags, macro with publication dates, sanity suite, frozen dataset v1.0 | **5 days** (was 10; adjustment layer descoped) |
| 2 | Walk-forward engine, cost model, fill logic (settlement + circuits), metrics, regime split, baselines, power calculation | **6 days** |
| 3 | Four feature modules behind the common interface; LLM sentiment pass with on-disk cache | **8 days** |
| 4 | Logistic regression + GBM; ablation table; regime table; apply §6 thresholds | **6 days** |
| 5 | Daily forward job, immutable log, scoring script | **3 days to build**, then months of runtime |

**Total ≈ 28 effort-days remaining**, plus the forward run's calendar time. Phase 0 came in on estimate and removed ~8 days from Phase 1 by descoping the adjustment layer (§3.2) — the largest single item in the original plan.

```
nepse-lab/
├── data/                  # gitignored; parquet + sqlite
├── nepselab/
│   ├── ingest/            # scrapers, corporate actions, calendar, macro vintages
│   ├── adjust/            # price adjustment, scrip state machine + tests
│   ├── features/          # price.py, reddit.py, news.py, macro.py — toggleable
│   ├── models/            # logistic, gbm, baselines
│   ├── eval/              # walk-forward, cost model, fills, metrics, regimes
│   └── forward/           # daily job, prediction log, scoring
├── configs/               # yaml per experiment + market_params.yaml
├── predictions/           # append-only forward log; never rewritten
├── tests/                 # sanity suite runs in CI
└── results/               # one dir per run, config + metrics + plots frozen
```

**Rules:** config-driven experiments; no hardcoded fees or market-structure constants outside `market_params.yaml`; every result dir carries its config + git hash; sanity tests pass before any experiment runs. Python: pandas/polars, scikit-learn, lightgbm/xgboost, statsmodels, matplotlib.

---

## 9. Risks and kill criteria

| Risk | Likelihood | Mitigation |
|---|---|---|
| Corporate action data unavailable/dirty | Medium | Phase 0 gate; fall back to index-only |
| Merger/suspension handling wrong → phantom returns | Med-high | State machine; validate vs. ShareSansar charts |
| Market-structure constants unverifiable early in sample | Medium | TODOs tracked; start sample at first fully-sourced date |
| Reddit attention leaks a time trend | High if unhandled | Normalize as share of sub activity or trailing-90d z-score |
| Macro lookahead via reference dates | High if unhandled | Publication-date gating, enforced in the feature layer |
| NEPSE endpoints change/break mid-project | Medium | Freeze dataset v1.0 early; ingestion resumable |
| No edge found | High-ish | Thresholds in §6 fixed in advance; abandon rather than tune |
| Overfitting via repeated threshold-adjacent tuning | High (know thyself) | §6 numbers are frozen; changing them invalidates the run |

**Kill criteria:**
- ~~If the adjusted-price layer can't be validated, restrict to index-only.~~ **Resolved at the Phase 0 gate: index-only is now the design**, not a fallback (§3.2). The criterion is retired.
- If the backtest fails any §6 threshold, stop. Write up the null result and do not re-tune.
- If the forward log diverges from the backtest by more than 5pp accuracy over ≥60 trading days, treat it as a pipeline bug, find it, and re-run the backtest — do not deploy capital on the more favourable of the two.
