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

**Regime split — ~~report every result twice~~ NOT POSSIBLE (2026-08-02).**
- ~~**2020-06 → 2021-12 (mania).**~~ **No data exists.** NEPSE's earliest servable session is 2025-07-30 (§3.4). The mania regime predates the retained window by four years and cannot be reconstructed from any endpoint.
- ~~**2022-01 → present (chop).**~~ Only 2025-07 → present is available — a fragment of the chop regime, not the regime.

The whole sample is ~12 months inside one regime. The split cannot be performed, which also **retires the §6 kill criterion** that abandons on "edge present only in the mania regime" — there is no mania data to detect that artifact with. That criterion existed precisely because a bull-market artifact is the most likely false positive here, and we have now lost the ability to test for it. That is a loss of a safeguard, not a simplification.

**Power, computed before any model runs. — COMPUTED 2026-08-02; the answer is fatal to the design as written.** Implemented in `scripts/phase1_power.py` (α=0.05 one-sided, 80% power, baseline = majority class at 54.5%).

The original estimate assumed ~1500 usable daily observations. **NEPSE serves 225** (§3.4). The methodology was right — its 7pp weekly guess matches the 8.7pp this computes for the n=200 it assumed — but the sample was overstated ~7×:

| Scenario | n | Min detectable edge |
|---|---|---|
| h=1 daily, full available sample | 225 | **8.2pp** |
| h=1 daily, walk-forward test set | 75 | **14.0pp** |
| h=5 weekly, non-overlapping blocks | 45 | **17.8pp** |
| h=5 weekly, walk-forward test blocks | 15 | **29.2pp** |
| *(§2's assumption, not achievable)* | *200* | *8.7pp* |

Detecting the **2pp** edge that §6 abandons the project over requires **3,821 weekly blocks**. We have 45 — an **85× shortfall**, or ~85 years of archiving at 225 sessions/yr.

**This is not "the edge is probably absent."** It is that no result, in either direction, would be informative: the §6 test cannot be run as written, because the threshold it tests sits an order of magnitude below the sample's noise floor. Any backtest accuracy this project reports on 45 weekly blocks is a draw from a distribution roughly ±18pp wide. A 60% point estimate would be entirely consistent with a coin.

---

## 3. Data correctness

The part most likely to silently break everything. **It did — see §3.4, which now dominates every other consideration in this section.**

### 3.1 Sources
| Data | Source | Notes |
|---|---|---|
| OHLCV, index + scrips | `polymorphisma/nepse_scraper` (PyPI) | **Serves a rolling ~1 year only; `startDate`/`endDate` are accepted and ignored — see §3.4.** Use the library's session, not raw HTTP: plain `curl` cannot complete a TLS handshake with NEPSE at all, so a connection failure is **not** evidence the API is down. History endpoints return Spring-Data page envelopes, not the bare lists the type hints claim — unwrap `content` and follow `totalPages`, or you silently get 20 rows of 225. Ticker history takes the security id as a **path** segment; `symbol` as a query param returns 400. |
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

### 3.4 NEPSE retains one rolling year, and silently lies about it

**Established 2026-08-02 by `scripts/phase1_probe_depth.py`. This is the single most consequential fact about the project and it invalidates §2's sample assumption.**

The history endpoints **accept `startDate`/`endDate` and ignore them completely.** Probing one window per year from 2012 to 2026 returned *the identical 225 rows every time* — `totalElements: 225` on all fifteen. Requesting 2018 does not error, does not return empty, and does not warn: it returns last year's data with an HTTP 200. Alternative spellings (`fromDate`/`toDate`, `from`/`to`, `businessDate`) behave the same. Confirmed on index history, ticker history, and sector index history.

**Phase 0 did not catch this because the coincidence was perfect.** It asked for 365 days and got 225 rows — exactly what one year of NEPSE sessions looks like. A successful-looking pull and a completely ignored parameter are indistinguishable at that one window. Any future endpoint that takes a date range is now assumed to ignore it until proven otherwise.

**The window rolls, and aged-out sessions are unrecoverable.** Measured directly:

| | |
|---|---|
| Phase 0 archive (pulled 2026-07-25) | 225 sessions, 2025-07-23 → 2026-07-24 |
| API serves (2026-08-02) | 225 sessions, 2025-07-30 → 2026-07-31 |
| **Sessions lost in 8 days of not archiving** | **5** (2025-07-23 … 2025-07-29) |

Five in, five out, held constant at 225. The five oldest sessions in `data/raw/` **no longer exist anywhere upstream** — that pull is now the only copy. `today_price` is bounded identically (works to 2025-07-30, fails before it), so there is no backfill route around this on any endpoint.

**Consequences, in order of severity:**
1. The ~1500-observation sample is unobtainable. We have 225 and gain ~225/yr.
2. The regime split (§2) is impossible — the mania regime predates the window by four years.
3. **Every day not archived is a trading day destroyed.** This is the only irreversible item in the project.

**Response — `data/archive/`, append-only, superseding "frozen dataset v1.0":** the frozen-dataset instinct in §3.2 was right for the wrong reason. The store is not frozen, it *accumulates*, and it is the only asset here that cannot be recreated. Implemented in `nepselab/ingest/archive.py` + `scripts/archive_pull.py`:
- Writes only add rows; nothing deletes or overwrites.
- Where upstream contradicts an archived row, **ours wins and the conflict is logged** — a silent upstream revision to a settled session is a data-integrity event, not something to merge away.
- `scripts/archive_seed.py` folded `data/raw/` in first, preserving the five otherwise-lost sessions.
- Captures all 17 indices, market summary, per-session `today_price` (breadth), and dated securities snapshots.

**Throttling — the backfill is slow by necessity.** NEPSE tolerates a sustained sweep for roughly 75 calls at 0.7s spacing, then returns `HTTPError` for everything after. Those failures are **not** missing data: dates that failed mid-sweep returned ~340 rows each when retried in isolation minutes later. Two rules follow, both now in the ingest: **a run of failures means back off, never "no data"**, and the sweep commits every 10 sessions rather than at the end — an interrupted backfill must keep what it paid for. Defaults are 1.6s spacing with a 120s cool-down; a full backfill takes several runs and that is fine, because incremental mode re-queues whatever is still missing.

**`scripts/archive_pull.py` must run every trading day.** It is no longer a Phase 5 convenience; it is the data collection strategy, and it is the one task with a real deadline.

**UNRESOLVED — the archive has no backup.** `data/` is gitignored (correctly; parquet does not belong in git), so the only copy of the only irreplaceable asset in this project currently exists on one disk, in one directory, with no redundancy. A disk failure destroys strictly more than the five sessions already lost. The §8.1 decision does not affect this: **the archive needs an off-machine copy under every option**, and it needs one now rather than after the sample is large enough to matter. Cheapest sufficient fix is a periodic copy to cloud storage or a separate physical disk, versioned by pull date. **TODO: pick a target and wire it into the daily job.**

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

### 6.1 The thresholds are not testable on 225 sessions — and they stay frozen anyway (2026-08-02)

§9 names the exact failure mode this section is about to invite: *"overfitting via repeated threshold-adjacent tuning — High (know thyself)."* So, explicitly:

**The §6 numbers are not being changed.** The finding in §3.4 is a reason the project may be unable to *run* its test, not a reason to move the bar until the test passes. Relaxing "+2pp over majority-class" to something a 45-block sample can resolve (~18pp) would not be a weaker threshold — it would be a threshold no real equity-index signal has ever met, i.e. an unfalsifiable one dressed as a strict one. Tightening or loosening either way after seeing §3.4 is the tuning §9 forbids.

**What actually changed** is that three of the five thresholds have lost the data needed to evaluate them:

| §6 threshold | Status after §3.4 |
|---|---|
| h=5 accuracy ≤ majority + 2pp → abandon | **Untestable.** Needs 3,821 weekly blocks; 45 exist. |
| Net Sharpe < 0.4 after costs | **Untestable for a second, independent reason** — every cost constant in `market_params.yaml` is still `null` (§4). |
| Edge only in the mania regime → abandon | **Retired.** No mania data exists (§2). Safeguard lost. |
| Alt-data gate: <1pp from Reddit/news → drop | Testable as a *point estimate*, but 1pp is far below the noise floor; the ablation cannot distinguish 1pp from 0. |
| Capital gate: ≥60 forward days | Unaffected — and now the only threshold that still does its job. |

**The honest position:** on the sample NEPSE will give us, a backtest cannot produce evidence that clears §6. That is a finding about feasibility, not about NEPSE's predictability, and it is the kind of null result §1 committed to documenting. **The open decision is in §8.1 — it is about the project's scope, not about these numbers.**

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
| 1a | ~~Depth probe; exact power calculation; append-only archive + daily job~~ | **DONE 2026-08-02** |
| 1b | **Daily `archive_pull.py` run.** Not effort — calendar time, indefinitely, starting now | **continuous** |
| 1c | Deep-history sourcing probe (§8.1 option D) — decides everything below | **1 day** |
| — | *everything below is **blocked** on the §8.1 decision* | — |
| 2 | Walk-forward engine, cost model, fill logic, metrics, baselines, ~~regime split~~ | 6 days |
| 3 | Four feature modules; LLM sentiment pass with on-disk cache | 8 days |
| 4 | Logistic regression + GBM; ablation table; ~~regime table~~; apply §6 thresholds | 6 days |
| 5 | ~~Daily forward job~~ — **merged into 1b**; log + scoring script remain | 2 days |

Phase 1a came in well under its 5-day estimate because it stopped at the first thing that mattered. The effort-days below are unchanged and still accurate; what changed is that **spending them is no longer obviously worthwhile** — see §8.1.

### 8.1 OPEN DECISION: what this project is now

§3.4 removed ~6/7ths of the assumed sample and §2 showed the remainder cannot resolve the effect §6 tests for. Four coherent responses; **they are not mutually exclusive, and (D) should be resolved before choosing among the rest.**

**(D) Source deep history elsewhere — do this first, ~1 day.** NEPSE's API is not the only place NEPSE history exists. ShareSansar, MeroLagani, Nepse Alpha and assorted Kaggle dumps all publish index history well before 2025, and the Reddit dump already covers 2020-06 onward. If a clean, verifiable daily index series back to 2020 can be obtained, **the original plan survives essentially intact** and every number in §2 reverts. This is the only branch that recovers the project as designed, it is cheap to test, and it is therefore the next task. Caveats: provenance and continuity-adjustment become the user's problem rather than the exchange's, and any such series must be reconciled against our 225 archived sessions before it is trusted — an unverifiable series is worse than no series.

**(A) Continue as an engineering exercise; defer modelling.** Build Phases 2–4 against the honest 225-session sample, report every result with its ±18pp interval, and treat the harness — not the results — as the deliverable. §1 already frames the harness as the primary objective ("*the harness is what makes any result believable*"), so this is defensible. Risk: §9's "know thyself" — running models on a sample that cannot resolve them, month after month, is precisely the setup where a 60% point estimate quietly becomes a belief.

**(B) Stop modelling; write up the feasibility null.** §1 committed to documenting a "no edge found" outcome; this is the adjacent, more honest result — *no edge is testable*. Keep 1b running forever, revisit in a few years when the archive is large enough. Cheapest, and intellectually clean.

**(C) Change the target to fit the sample.** More observations per unit time, not more calendar time: floorsheet/intraday data (§3.1 flags depth as an open TODO) would multiply n substantially. But §4 is blunt that manual TMS execution makes intraday strategies unimplementable, so this risks building something well-powered and untradeable.

**Recommendation: (D) now, then (B) if D fails.** (D) is one day and either restores the project or definitively closes the question of whether the sample can be fixed. If it fails, (B) is the outcome the plan already committed in advance to accepting — and (A) is the option §9 warns about by name.

**Not yet decided. The archive (1b) runs regardless of the outcome**, since it is the only irreversible item and it costs one cron entry.

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
| ~~NEPSE endpoints change/break mid-project~~ **MATERIALISED, worse than written** | ~~Medium~~ **Certain** | The risk was framed as endpoints *breaking*. What happened is subtler and was not on this table: they work, return 200, and silently ignore the date range (§3.4). "Freeze dataset v1.0 early" was the right mitigation for the wrong reason — the fix is a *continuously accumulating* archive, not a frozen one, and the freeze framing would have left the window rolling while we built. |
| **Sample too small to resolve the effect under test** — **not on the original table** | **Certain** | Unmitigable; §2 power table. The original plan carried a power calculation as a TODO rather than a gate. Had it been computed at Phase 0 against real row counts instead of an assumption, §3.4 surfaces a week earlier and five sessions are not lost. **Compute power before building, not after.** |
| No edge found | High-ish | Thresholds in §6 fixed in advance; abandon rather than tune |
| Overfitting via repeated threshold-adjacent tuning | High (know thyself) | §6 numbers are frozen; changing them invalidates the run |

**Kill criteria:**
- ~~If the adjusted-price layer can't be validated, restrict to index-only.~~ **Resolved at the Phase 0 gate: index-only is now the design**, not a fallback (§3.2). The criterion is retired.
- If the backtest fails any §6 threshold, stop. Write up the null result and do not re-tune.
- If the forward log diverges from the backtest by more than 5pp accuracy over ≥60 trading days, treat it as a pipeline bug, find it, and re-run the backtest — do not deploy capital on the more favourable of the two.
