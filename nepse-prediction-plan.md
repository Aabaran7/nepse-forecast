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
- Sequence models — 2,434 usable daily observations (§3.5) is far too few. No LSTM/GRU/transformer. *Unchanged by the deep-history find: the sample grew ~10× and is still two orders of magnitude short of what a sequence model on noisy daily returns would need.*
- Ticker-level cross-sectional prediction (Reddit EDA killed the alt-data case; see §5).
- Publication of any kind.

**Success criteria — all three met, 2026-08-04:**
1. *One command from raw data to cost-adjusted, regime-split results* — `run_all.py`. Runs the gates first (§8), then Phases 2a/2b/4/5 at both horizons, in ~2 minutes.
2. *Every model benchmarked against 50%, majority-class, and buy-and-hold* — §6.2. The majority-class baseline is walked forward and refit at every retrain, not computed once, and edges are measured paired on identical rows.
3. *A forward job that has never overwritten a past prediction* — §7, enforced by `PredictionExists` rather than by intent.

The fourth objective, a documented "no edge found" outcome, is **§6.2**. It is the result the project got.

---

## 2. Target and scoring protocol

**Target:** sign of the NEPSE index return at h=1 trading day and h=5 trading days.

**Validation:**
- **Walk-forward only.** Train strictly on data before each prediction date; retrain monthly; embargo gap between train and test to kill leakage from overlapping labels. No k-fold, ever.
- All scaling and feature fitting inside the training window only.
- h=5 labels overlap — use non-overlapping weekly blocks for significance, and block bootstrap for confidence intervals. Naive CIs on overlapping labels are far too tight.

**Metrics:**
- Directional accuracy and F1, vs. **both** 50% and the majority class (NEPSE has long directional runs; majority-class is the harder baseline and the one that matters). In the verified 2025-07→2026-07 window only **45.5%** of index days were up. *(Superseded as the operative figure — on the 2016+ deep sample it is 47.8% up, majority down at 52.2%; see below and §3.6.)* Compute the majority baseline **per regime**: §3.6 measured it flipping to *up at 56.6%* inside the mania window, so a single pooled number sets the wrong bar in every regime at once.
- Net-of-cost simulated PnL: net Sharpe at a stated capital base, net CAGR, max drawdown, turnover, % of PnL from top-5 days.
- Never RMSE on price level.

**Regime split — ~~NOT POSSIBLE (2026-08-02)~~ RESTORED (2026-08-04).** §3.5 obtained a cross-validated daily index series back to 2016-01, so both regimes are in sample again:
- **2020-06 → 2021-12 (mania).** 386 sessions, real OHLC, both deep sources in exact agreement on every one. Recovered.
- **2022-01 → present (chop).** ~1,120 sessions.
- Plus **2016-01 → 2020-02**, a pre-mania stretch the original plan never had, which is the most useful of the three for checking that an edge is not a 2020s artifact.

The COVID closure sits inside this: NEPSE ran only 184 sessions in 2020, with gaps of 51 and 47 calendar days (2020-03 → 2020-06). Real, not missing data, but it means "2020" is not a normal year for any rolling feature.

The §6 kill criterion that abandons on "edge present only in the mania regime" is therefore **reinstated**. It was retired on 2026-08-02 for want of data; the data now exists.

**Power, computed before any model runs. — recomputed 2026-08-04 on the deep sample.** Implemented in `scripts/phase1_power.py`, applied to the surviving window in `scripts/phase1c_deep_history.py` (α=0.05 one-sided, 80% power, baseline = majority class).

The original estimate assumed ~1500 usable daily observations. The rolling-window finding (§3.4) cut that to 225; the deep-history finding (§3.5) restores **2,434** — more than the plan originally assumed.

| Scenario | n | Min detectable edge | (on 225 sessions) |
|---|---|---|---|
| h=1 daily, full usable sample | 2,434 | **2.5pp** | *8.2pp* |
| h=1 daily, walk-forward test set | 1,934 | **2.8pp** | *14.0pp* |
| h=5 weekly, non-overlapping blocks | 486 | **5.6pp** | *17.8pp* |
| h=5 weekly, walk-forward test blocks | 386 | **6.3pp** | *29.2pp* |

The majority class on the deep sample is **down at 52.3%** (47.7% of days up, 2016-01→2026-08), not the 54.5% measured on the 225-session window. The §6 bar is therefore ~54.3%, not ~56.5%. Substituting one baseline for the other moves every figure above by <0.05pp, so the power conclusions do not depend on which is used — but the *accuracy target* does, and 52.3% is the honest one now.

**What is and is not testable.** Detecting a 2pp edge on non-overlapping weekly blocks still needs **3,821 blocks**; we have 486, a **7.9× shortfall** (was 85×). So:

- **h=1 daily is now nearly resolvable** — 2.5pp against a 2pp bar. Close enough that a real result would be interpretable, with the caveat that it is marginal and the CI will straddle the threshold.
- **h=5 weekly, which is what §6 actually specifies, is still not.** Dividing by 5 for non-overlapping blocks is the entire reason. Deep history rescued the sample; it did not rescue this threshold, and no amount of archiving forward fixes it in a useful timeframe (~76 years at 225 sessions/yr).

This is a much better position than 2026-08-02 and it is still not the position §6 was written for. §6.1 tracks the consequences.

---

## 3. Data correctness

The part most likely to silently break everything. **It did — see §3.4, which now dominates every other consideration in this section.**

### 3.1 Sources
| Data | Source | Notes |
|---|---|---|
| OHLCV, index + scrips | `polymorphisma/nepse_scraper` (PyPI) | **Serves a rolling ~1 year only; `startDate`/`endDate` are accepted and ignored — see §3.4.** Use the library's session, not raw HTTP: plain `curl` cannot complete a TLS handshake with NEPSE at all, so a connection failure is **not** evidence the API is down. History endpoints return Spring-Data page envelopes, not the bare lists the type hints claim — unwrap `content` and follow `totalPages`, or you silently get 20 rows of 225. Ticker history takes the security id as a **path** segment; `symbol` as a query param returns 400. |
| Index, sector indices, market summary | Same scraper | Exchange-computed and continuity-adjusted; the target and most features come from here. **Rolling 1 year only.** |
| **Index OHLC, 2016 → present** | **MeroLagani chart handler** | **The deep sample (§3.5).** 2,434 sessions. Accepted only because a second independent scrape agrees with it on 99.92% of return signs from 2016; the two disagree on 2.4% before that, so pre-2016 is rejected. Index only — no breadth, no sector indices, no scrips. |
| Cross-check for the above | `menaceXnadin/nepse-historical-market-data-csv` (git-LFS) | Not a data source in its own right; its job is to adjudicate MeroLagani. Fetch via `media.githubusercontent.com`, not `raw.` |
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
- **Trading calendar — the week changed *three times*, and two of the changes were invisible in the 225-session window.** Sun–Thu; then **six days (Sun–Fri), 2022-05-20 → 2022-09-16**; then Sun–Thu again; then Mon–Fri from **2026-04-10**. The 2022 era was found only after the deep series arrived (§3.6) — the original two-era reading was not wrong about what it could see, it was wrong to generalise from one year. The calendar is date-indexed like every other constant, not a fixed rule, and the *number of eras* is not known either. This is not cosmetic: h=5 "one week ahead" spans a different weekday set either side of the boundary, day-of-week features are not comparable across it, and Reddit weekly aggregation must align to trading weeks. Frequent unscheduled closures on top (6 gaps >4 calendar days in the verified year).
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

Extended for the deep series in `tests/test_deep_history.py` (26 tests pass as of 2026-08-04). These target *silent* failures — each one produces a well-formed series that is wrong, rather than an error:
- Bar timestamps normalise in UTC; the Kathmandu-local misreading that shifts every post-2020 bar forward a day is asserted against explicitly.
- Source comparison scores **return-sign** disagreement, not just level difference: with matching endpoints one bad close can flip two labels.
- `usable_start` is a suffix rule — a clean year stranded inside dirty ones must not be selected.
- Turnover units-break detection ignores zero rows and short months, so the COVID closure cannot be mistaken for a change of units.
- `save()` cannot write into `data/archive/`.

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
1. ~~The ~1500-observation sample is unobtainable. We have 225 and gain ~225/yr.~~ **Superseded by §3.5** — obtainable elsewhere, back to 2016. Still true of every dataset except index OHLC.
2. ~~The regime split (§2) is impossible — the mania regime predates the window by four years.~~ **Superseded by §3.5** — restored for the index; still impossible for breadth and sector features.
3. **Every day not archived is a trading day destroyed.** This is the only irreversible item in the project. **§3.5 raised its stakes rather than lowering them:** deep history covers the index and nothing else, so `today_price` is the only route to breadth that will ever exist.

**Response — `data/archive/`, append-only, superseding "frozen dataset v1.0":** the frozen-dataset instinct in §3.2 was right for the wrong reason. The store is not frozen, it *accumulates*, and it is the only asset here that cannot be recreated. Implemented in `nepselab/ingest/archive.py` + `scripts/archive_pull.py`:
- Writes only add rows; nothing deletes or overwrites.
- Where upstream contradicts an archived row, **ours wins and the conflict is logged** — a silent upstream revision to a settled session is a data-integrity event, not something to merge away.
- `scripts/archive_seed.py` folded `data/raw/` in first, preserving the five otherwise-lost sessions.
- Captures all 17 indices, market summary, per-session `today_price` (breadth), and dated securities snapshots.

**Throttling — the backfill is slow by necessity.** NEPSE tolerates a sustained sweep for roughly 75 calls at 0.7s spacing, then returns `HTTPError` for everything after. Those failures are **not** missing data: dates that failed mid-sweep returned ~340 rows each when retried in isolation minutes later. Two rules follow, both now in the ingest: **a run of failures means back off, never "no data"**, and the sweep commits every 10 sessions rather than at the end — an interrupted backfill must keep what it paid for. Defaults are 1.6s spacing with a 120s cool-down; a full backfill takes several runs and that is fine, because incremental mode re-queues whatever is still missing.

**`scripts/archive_pull.py` must run every trading day.** It is no longer a Phase 5 convenience; it is the data collection strategy, and it is the one task with a real deadline.

**~~UNRESOLVED — the archive has no backup.~~ RESOLVED 2026-08-04.** `data/` is gitignored (correctly; parquet does not belong in git), so the only copy of the only irreplaceable asset in this project existed on one disk with no redundancy. Fixed: `scripts/archive_backup.py` mirrors `data/archive/` and `data/deep/` to a **private GitHub repo** (`Aabaran7/nepse-archive`), called from `cron_archive.sh` on every run, inside the same lock as the pull.

Two design points, both about the backup being genuinely useful rather than merely existing:
- **CSV, not parquet.** Parquet is a compressed blob rewritten whole on each merge, so git stores a fresh copy daily — `today_price` alone reaches ~8 MB, i.e. ~2 GB/yr of commits. The CSVs are sorted on the archive's own keys, so a day's rows land as an insertion git deltas to near-nothing. CSV also restores without pyarrow or a matching pandas; a backup with dependencies is a weaker backup.
- **It runs even when the pull fails**, since a partial pull still added rows worth protecting, and a failed push leaves the commit local rather than reporting a failed backup — the next run carries both.

First push 2026-08-04: 6 datasets, 36,525 rows.

**Breadth backfill is the live race (2026-08-04).** `today_price` — the only breadth source, and the one dataset that must be pulled one session at a time — was archived for just 48 of 231 sessions. Nothing else in the project has a deadline like this: §3.5 restores index OHLC back to 2016, but **it carries no breadth at all**, so any advancers/decliners feature is capped forever at what this backfill saves.

Two things surfaced while running it, both worse than "it is slow":

- **The queue was ordered newest-first, i.e. exactly backwards.** `sessions_to_pull` sorted descending, which is right for the daily incremental (get today) and wrong for a backlog (the oldest sessions are the ones expiring). With NEPSE throttling each run down to ~15 fetches, that ordering would have spent weeks on sessions in no danger while the at-risk ones fell off the back of the window. **Nothing would have errored.** Fixed: the newest 2 sessions go first so a long backfill never delays today's data, then strictly oldest-first. Regression test in `tests/test_pull_order.py`.
- **7 sessions are permanently unrecoverable** — 2025-07-23 → 2025-07-31 sit outside the API's rolling window. They exist in `data/archive/indices` (seeded from the Phase 0 pull) but their `today_price` was never captured and cannot be. Breadth for those dates is gone.

**BACKFILL COMPLETE 2026-08-04: 225 of 232 sessions.** The 7 missing are exactly the 7 above — the sweep converged on them and then stopped making progress, which is the confirmation that they are gone rather than merely throttled. Everything from 2025-08-03 onward is archived. Breadth is now available for the whole servable window and accumulates daily.

**Throttling is harsher than §3.4's original measurement suggests once a sweep has been running all day**: runs degrade from ~15 successful fetches to *zero*, returning `HTTPError` on every call. The catch-up loop therefore spaces runs 25 minutes apart rather than 5. Expect the 146 recoverable sessions to take days, not hours.

### 3.5 Deep history exists elsewhere, from 2016, and only because two sources agree

**Established 2026-08-04 by `scripts/phase1c_deep_history.py` (§8.1 option D). This reverses most of §3.4's damage.**

Two independent sources publish daily NEPSE index bars from 1997-07-20:

| Source | Access | Span |
|---|---|---|
| **MeroLagani** | `handlers/TechnicalChartHandler.ashx`, TradingView-UDF style | 6,678 sessions, 1997-07-20 → present |
| **GitHub dump** (`menaceXnadin/nepse-historical-market-data-csv`) | git-LFS CSV, a separate scrape | 6,564 sessions, same nominal span |

**Both reproduce the exchange almost exactly over the window we can check.** Against our 231 archived sessions: MeroLagani matches all 231 with close/high/low exact to the cent (open differs on 130 by ≤0.01, rounding); the GitHub dump has 230 of 231 — missing 2025-12-02 outright — with open/high/low exact and two closes off by up to 0.43.

**That verification proves nothing about the years we actually need**, which is the whole difficulty: 2016–2024 is precisely the data NEPSE no longer serves, so there is no authority to check either source against. The substitute test is to check them against *each other*, and specifically on the sign of the daily return, because that is the target:

| Era | Sessions | Days the two sources disagree on return sign |
|---|---|---|
| 1997 → 2015 | 4,120 | **98 (2.38%)** |
| 2016 → 2026 | 2,429 | **2 (0.08%)** |

Pre-2016 the two scrapes also disagree about *which days were sessions at all* — 129 dates in one and not the other, concentrated in 2002–2015 — and 4,243 of those 4,244 bars are flat (`open=high=low=close`), i.e. close-only, so that era carries no intraday range even where the closes agree.

**A 2.4% sign disagreement rate is disqualifying, not cosmetic.** The edge this project is looking for is ~2pp. Pre-2016, the choice of *which source to use* perturbs the labels by more than the effect size. Either source would produce a confident backtest; they would not produce the same one, and nothing available can say which is right. Depth that cannot be adjudicated is worse than no depth, exactly as §8.1 warned.

**Decision: the usable series starts 2016-01-01** — 2,434 sessions, the first year from which every subsequent year stays under 0.5% disagreement. MeroLagani is the accepted source (it matched the exchange on all 231 sessions; the GitHub dump lost one and missed two closes). The rule is a *suffix* condition, not per-year: a clean 2007 inside a dirty decade buys nothing, since the series has to be contiguous to model on.

**Turnover: available from 2017, and it is not the column it looks like.** Two separate traps.

*What it is.* Both deep sources carry a `volume` field that disagrees with `indices.turnoverValue` by a median 1.7% and up to 137% — which reads as a broken feed and is not one. It is **market-wide turnover**, and it reconciles to `market_summary.totalTurnover` almost exactly (213/224 sessions within 0.1%). Define turnover features against market summary, not the index endpoint.

*When it starts.* **Turnover has a units break at 2017-01-01 — a 420× step with no corresponding move in the index** (2016 medians ~1.4e6, 2017 ~5.5e8, and 2016 carries literal zeros). Prices are continuous across it; turnover is not. Because every §5 turnover feature is a ratio or z-score against a trailing window, one computed across the boundary reads a 420× surge that never happened, in a year that is otherwise perfectly good for price features. **Price features start 2016-01; turnover features start 2017-01.** The two dates are different and the difference is not cosmetic.

`turnover_scale_breaks()` detects this rather than hardcoding the date, and reports *every* break so they can be told apart: a naive version flagged 2020-05 instead, which is not a units change at all but NEPSE genuinely reopening into a boom after the two-month COVID closure. Only units breaks move a start date. Judge each one.

**What deep history does not give us:** breadth (advancers/decliners), sector indices, per-scrip data, or anything else requiring `today_price`. Those remain limited to the archived window — see the backfill note in §3.4. The §5 feature list has to be read with that split in mind: **index OHLC + market turnover from 2016; everything else from 2025-07 only.**

**Provenance and storage.** The accepted series lands in `data/deep/nepse_index_deep.parquet`, deliberately **not** in `data/archive/`. The archive holds exchange-sourced rows and is irreplaceable; this series is third-party and re-downloadable, and mixing them would let a scrape contaminate the one asset that cannot be rebuilt. Two traps are recorded in `nepselab/ingest/deep_history.py` because both fail silently: MeroLagani's bar timestamps switch time-of-day mid-series (05:45 UTC early, 20:45 UTC later), so normalising them in Kathmandu local time shifts every modern bar a day forward and misaligns the entire recent half against the archive; and `raw.githubusercontent.com` serves the 131-byte LFS *pointer* rather than the CSV, which parses as a valid 3-row file.

**Re-verify on every re-pull.** The reconciliation is not a one-off gate that has now been passed — it is the only check standing between the model and a silently revised third-party series. `scripts/phase1c_deep_history.py` should be re-run whenever the deep series is refreshed, and its agreement numbers should not get worse.

### 3.6 What the sanity suite found in the deep series (2026-08-04)

`scripts/phase1d_deep_quality.py` + `nepselab/quality.py`, tested in `tests/test_quality.py`. **All checks now pass**; three of them only pass because something was fixed or given a policy first.

**1. The trading calendar had a third era nobody knew about.** §4 recorded two eras (Sun–Thu, then Mon–Fri from 2026-04-10) because that is all the 225-session window could show. Over 2016–2026 there are **15 Friday sessions in one contiguous block, 2022-05-20 → 2022-09-16, with Sundays continuing throughout** — a **six-day trading week**, 95 sessions in ~17 weeks. `market_params.yaml` now carries all four eras. This is not cosmetic: **h=5 "one week ahead" spans six sessions of calendar time inside that block**, and day-of-week features are incomparable across its boundaries. It is also the case for keeping the check permanently — a new era is invisible in the price data and announces itself only here.

**2. The ±6% index circuit holds for a decade, which is evidence *for* the deep series.** Max |return| across 2016–2026 is **6.0610%**, with **zero** days beyond 6.1% and 12 sitting in the 6.0–6.1% overshoot band. The exchange-verified window shows the same shape (+6.0070%). A third-party feed that respects NEPSE's circuit across ten years is behaving like real NEPSE data — an independent corroboration the §3.5 cross-check could not provide. Any check must use a tolerance; asserting a hard 6% reports a dozen false positives.

**3. 16 bars are internally inconsistent, and the policy is flag, never repair.** Of 2,434 sessions, 83 nominally violate OHLC ordering, but the deep feed stores high/low to 2dp while open/close carry more digits — at a 1e-4 tolerance that representation noise drops out and **16 real cases remain** (2 with a close outside its own [low, high]).

Adjudicated against the second source: **the close agrees on all 16**, and on 13 the two sources carry the *identical* bad bar — so the defect is upstream in NEPSE's published data, not a scraping artifact. Consequences:
- **The label is safe.** The target is close-to-close sign; no close is in question.
- **Range-derived features are not** — realized vol from high−low, distance-from-high — on 0.66% of days.
- The series carries an **`ohlc_consistent`** column. Feature modules must read it; nothing repairs the bars, because a silently repaired bar is indistinguishable from a correct one downstream.

The check asserts the violations are *flagged*, not that they are absent. "Zero violations" is a bar this data will never clear, and a permanently-red gate is an ignored gate.

**4. The majority class flips between regimes — do not use one pooled baseline.**

| Window | n | Up share | Majority class |
|---|---|---|---|
| Full 2016+ | 2,433 | 47.8% | **down, 52.2%** |
| Pre-mania 2016 → 2020-02 | 988 | 47.3% | down, 52.7% |
| Mania 2020-06 → 2021-12 | 364 | 56.6% | **up, 56.6%** |
| Chop 2022+ | 1,063 | 45.5% | down, 54.5% |

§2 warned about this; here is the magnitude. §6's bar is "majority + 2pp", so it is **54.2% pooled but 58.6% in the mania window** — a pooled baseline sets the wrong bar in every regime, and the mania bar is high enough that clearing it is a serious ask. Report accuracy against the regime's own majority class, always.

**5. Closures.** 32 gaps >4 calendar days; the long ones are 2020-05-12 (51 days) and 2020-06-29 (47 days) — COVID — plus 2023-11-20 (11 days). Real closures, not missing data.

---

### 3.7 Phase 2a: the harness, and what it says before any model exists (2026-08-04)

`nepselab/eval/{labels,splits,metrics,baselines}.py`, driven by `scripts/phase2_walkforward.py`, tested in `tests/test_eval.py`. **Phase 2a is done; Phase 2b (cost model, fill logic, PnL) remains blocked on §4.**

**The harness passes its own self-checks.** 100 monthly folds at h=1, 1,930 pooled test rows, embargo enforced and asserted per fold. A coin scores 0.4896 with a CI covering 0.50; the majority baseline ties itself at exactly 0.00pp. The load-bearing test is `test_no_edge_on_a_random_walk` over five seeds — leakage does not announce itself, it looks like a discovery, and a synthetic series with no predictable sign is the only check that covers leaks nobody wrote an assertion for.

Four design decisions that turned out to matter more than expected:

1. **h is measured in sessions, not calendar days, and labels spanning a closure are dropped.** A "5-session-ahead" label across the 51-day COVID gap is not the same target as its neighbours. 17 such labels at h=5, 2 at h=1.
2. **Flat closes are dropped, not bucketed as up.** `>` vs `>=` differs by ~0.1% of days — negligible next to intuition, not negligible next to a 2pp threshold.
3. **The baseline is the majority-class predictor *walked forward*, not a majority recomputed from a fixed window.** It refits at every retrain, so it changes its mind when the regime turns. Edges are measured paired, on identical rows, because §6 asks about a *difference* — comparing two separate CIs is a different and wrong question.
4. **The bootstrap block must be longer than the horizon.** This is the finding worth carrying forward.

**Block length decides significance, so it is fixed in advance.** For the h=5 momentum baseline (edge +5.60pp, n=1,911):

| Block | Edge 95% CI | Excludes 0? |
|---|---|---|
| 1 | [+2.51, +8.69]pp | yes |
| 5 (= horizon) | [+1.25, +10.12]pp | yes |
| **13 (= n^⅓, used)** | **[+0.10, +10.83]pp** | **barely** |
| 25 | [−0.21, +11.49]pp | no |

Setting `block = horizon` covers the label overlap but not the persistence of the market itself — trends and regimes do not reset every five sessions. `recommended_block = max(horizon, n^⅓)` is now the default and is **fixed before Phase 4 runs**, for the same reason §6's thresholds are: choosing it after seeing a result would be picking the number that gives the nicer answer. **Any significance claim must state its block length.**

**A naive momentum rule already beats the majority class**, which is worth stating plainly and not overstating:

| | h=1 | h=5 |
|---|---|---|
| Majority-class accuracy | 0.5259 | 0.4976 |
| Momentum accuracy | 0.5591 | 0.5536 |
| Edge (paired 95% CI) | +3.32pp [+0.31, +6.32] | +5.60pp [+0.10, +10.83] |
| Mania-regime edge | +13.15pp | +10.14pp |

This is **not** evidence of a tradeable edge, and three things stand between it and one:
- **No costs.** §4's constants are all `null`; a daily-rebalancing rule facing flat DP charges and manual TMS execution is exactly what §4 predicts will not survive them. This is the whole point of Phase 2b.
- **The mania concentration is the §6 kill criterion, visible in the baselines.** The edge is largest in 2020-06→2021-12 in both horizons — precisely the bull-market artifact §6 abandons on. The criterion was retired on 2026-08-02 for want of data and reinstated by §3.5; here is the first evidence it will have real work to do.
- **The interval straddles the bar.** ±2.1–2.3pp against a 2pp threshold.

**Phase 2a's actual deliverable is that last row.** Any Phase 4 accuracy will carry a ~±2pp interval at h=1 and a block-sensitive one at h=5. That is now measured rather than projected, and it is the number §6 has to be read against.

---

### 3.8 Phase 3: feature modules (2026-08-04)

`nepselab/features/{base,price,reddit}.py`. Three toggleable modules behind one interface, so the §5 ablation is a config change.

| Module | Columns | From | Notes |
|---|---|---|---|
| `price` | 25 | 2016-01 | returns/momentum at several lags, realized vol and vol-ratio, distance from 52w high/low, position-in-range, MA ratios, run length, intraday range (masked on the 16 bad bars) |
| `turnover` | 6 | **2017-01** | log-ratios to trailing windows, z-score, turnover×return interaction. Separate module purely so it cannot be dragged across the 420× units break |
| `reddit` | 9 | 2020-06 | attention z-scores and ratios, comments-per-post, acceleration |

**Reddit is attention only, and every column is scale-free.** §9 rates "Reddit attention leaks a time trend" as *High if unhandled*: the subreddit grew across the sample, so a raw count encodes the calendar and a model will happily use the calendar to predict a market that also trended. Nothing in the module emits a level — only z-scores against a trailing 90-session window, log-ratios to that window, and composition ratios.

Two alignment details that would each have leaked silently:
- **Reddit's clock is not NEPSE's.** `created_utc` is UTC; NEPSE closes 15:00 NPT = 09:15 UTC. Attention for session *t* counts only what was posted before that session's close — otherwise the feature knows how the day went. Only **44%** of daily comment volume falls before the close, so this is not a rounding detail.
- **Non-trading-day activity is carried forward, not dropped.** A closure does not stop people posting, and the loudest days on a stock subreddit are often the ones the market is shut.

**Sentiment is not built.** §5 requires an LLM pass with a constrained schema, an on-disk cache, and validation against ~200 hand-labelled code-switched comments before use, and explicitly forbids VADER/TextBlob (they score romanized Nepali as neutral noise). None of that is done, so the ablation measures **attention only** and the §6(d) gate is read on that basis.

**A bug worth recording, because it cost 37% of the sample and nothing errored.** Rolling windows were written with default `min_periods`, which returns NaN if *any* input in the window is NaN. The 16 masked OHLC bars (§3.6) therefore poisoned 63 rows each — 2018 lost 117 of 240 sessions, 2017–2019 lost ~60% — and the survivors looked entirely healthy. It was caught only because the regime table came back missing its pre-mania row, i.e. because §6(c) needed data that had quietly disappeared. Fixing `min_periods` recovered the price frame from 1,721 to **2,352** sessions.

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

**Every module must declare the span it can actually cover — they are not the same, and this now drives sample size (§3.5).**

| Feature group | Available from | Source |
|---|---|---|
| Index OHLC, returns, realized vol, 52w distance | **2016-01** | `data/deep/` |
| Market-wide turnover, and anything derived from it | **2017-01** — *not* 2016; 420× units break (§3.5) | `data/deep/` `turnover` = `market_summary.totalTurnover` |
| Traded shares, transactions, `tradedScrips` | 2025-07 | `data/archive/market_summary` |
| **Breadth (advancers/decliners)** | 2025-07, and only as far back as the backfill saves | `data/archive/today_price` |
| Sector index spreads | 2025-07 | `data/archive/indices` |
| Reddit attention | 2020-06 | Arctic Shift dump |

A config mixing a 2016 feature with a 2025 one silently yields a 2025 sample. Phase 2's engine must reject that rather than run it (§8).

**1. Price/volume.** Returns at multiple lags, realized vol, breadth (advancers/decliners), turnover trends, distance from 52w high, sector index spreads. Sources, all confirmed working in Phase 0: index OHLC + 52w high from `index/history/58`; market-wide turnover, traded shares, transactions and `tradedScrips` from `market-summary-history`; sector indices from the sector index endpoint; **breadth from dated `today_price` snapshots** (one call per session, ~350 rows, `closePrice` vs `previousDayClosePrice`). Breadth self-validates against the index — on the two limit days in the verified year it read 260 up / 1 down and 10 up / 228 down.

**2. Reddit — index-level attention only.** EDA on the r/NepalStock Arctic Shift dump is **complete; do not redo it**:
- Usable window 2020-06 onward.
- Median 39 comments/day since 2021; 1233 of 2032 days have ≥30.
- 285 weeks with ≥50 comments.
- Only 15 tickers exceed 500 total mentions.

Conclusions carried forward: usable as an **index-level attention and sentiment series only**; ticker-level cross-section is dead; weekly frequency is solid, daily workable but thinner. Features: daily and weekly post/comment counts, unique authors, normalized attention z-score, comment-to-post ratio. **Attention must be normalized as a share of total subreddit activity or z-scored against a trailing 90-day window** — the sub grew across the sample, so raw counts trend mechanically and will leak a time trend into any model.

**3. News. NOT BUILT.** ShareSansar / MeroLagani headline counts plus LLM-scored sentiment. Never reached: §6(d) dropped the alt-data branch on Reddit attention alone (−2.34pp), and §6(b)/(c) abandoned the project before news could change the answer.

**Sentiment scoring (Reddit and news).** The corpus is code-switched English / romanized Nepali — **do not use VADER or TextBlob**; they will score romanized Nepali as neutral noise. Use an LLM scoring pass with a constrained label schema (structured outputs, so every response parses). **Cache all scores to disk keyed by comment/headline id** — the pass runs once per document, ever. Route it through the Message Batches API (50% of standard token price) and cache the shared few-shot prefix. Note the prompt-cache minimum is model-dependent — 512 tokens on Opus 5, 1024 on Sonnet 5, 4096 on Haiku 4.5 — so a short few-shot prefix silently won't cache on the cheapest model. **TODO: pick the model and estimate total cost** from the actual document count before committing; validate whichever model is chosen against a hand-labelled sample of ~200 code-switched comments.

**4. Macro. NOT BUILT** — same reason as news. **If ever revisited:** NRB rates, CPI, remittance inflows, liquidity. **Must use publication lags, not reference dates.** These publish weeks to months late; store `reference_date` *and* `publication_date`, and let features read only rows where `publication_date <= t`. Reference-dated macro at time *t* is lookahead and will manufacture an edge that does not exist.

---

## 6. Models and abandonment thresholds

**Logistic regression and gradient boosting only.** 2,434 daily observations (§3.5) does not support anything deeper — and note that the *walk-forward test set* is ~1,900 of those, while any Reddit-inclusive model is capped at the ~1,430 sessions from 2020-06. Multiple seeds where stochastic; report mean ± sd.

**Abandonment thresholds — fixed now, before any results are seen.** If the backtest fails these, the project is abandoned and written up as a null result rather than tuned further:

- **PRIMARY (decided 2026-08-04, before any model ran — see §6.1): h=1 directional accuracy ≤ (majority-class baseline + 2pp)** on the pooled walk-forward test → abandon. Majority class on the deep sample is **52.3% (down)**, so the bar is **54.3%**.
- **h=5 directional accuracy ≤ (majority-class baseline + 2pp)** on the pooled walk-forward test → abandon. Still binding, still reported, but its confidence interval (~±6pp) cannot resolve 2pp and never will — so a *pass* here is not evidence of an edge, only an absence of contrary evidence.
- **Net Sharpe < 0.4** after realistic costs at the declared capital base → no capital, regardless of accuracy.
- **Edge present only in the 2020-06 → 2021-12 mania regime** → abandon; that is a bull-market artifact, not a signal.
- **Alt-data gate:** if +Reddit and +news each add <1pp over price-only in the ablation, drop those modules. The project may continue on price-only if it still clears the above.
- **Capital gate:** deploy only after ≥60 forward trading days logged *and* forward accuracy not more than 5pp below backtest accuracy.

### 6.1 Which thresholds the sample can actually test — and why they stay frozen either way (2026-08-02, revised 2026-08-04)

> **Revision after §3.5.** The deep-history series changes the *inputs* to this section, not its argument. Three of the five thresholds were listed below as untestable or retired; two of those recover. The numbers themselves are still not being changed, for the reason given below, which is unaffected by having more data. Updated status table follows the original reasoning — read both.


§9 names the exact failure mode this section is about to invite: *"overfitting via repeated threshold-adjacent tuning — High (know thyself)."* So, explicitly:

**The §6 numbers are not being changed.** The finding in §3.4 is a reason the project may be unable to *run* its test, not a reason to move the bar until the test passes. Relaxing "+2pp over majority-class" to something a 45-block sample can resolve (~18pp) would not be a weaker threshold — it would be a threshold no real equity-index signal has ever met, i.e. an unfalsifiable one dressed as a strict one. Tightening or loosening either way after seeing §3.4 is the tuning §9 forbids.

**What actually changed** is that three of the five thresholds have lost the data needed to evaluate them:

| §6 threshold | Status after §3.4 | Status after §3.5 (2026-08-04) |
|---|---|---|
| h=5 accuracy ≤ majority + 2pp → abandon | **Untestable.** Needs 3,821 weekly blocks; 45 exist. | **Still untestable.** 486 blocks — 7.9× short, down from 85×. Better, not sufficient. |
| Net Sharpe < 0.4 after costs | **Untestable for a second, independent reason** — every cost constant in `market_params.yaml` is still `null` (§4). | **Unchanged and now the binding blocker.** Deep history does nothing for this; §4's TODOs are sourcing work nobody has done. |
| Edge only in the mania regime → abandon | **Retired.** No mania data exists (§2). Safeguard lost. | **Reinstated.** 2020-06→2021-12 recovered in full, plus a pre-mania 2016–2020 stretch the plan never had. |
| Alt-data gate: <1pp from Reddit/news → drop | Testable as a *point estimate*, but 1pp is far below the noise floor; the ablation cannot distinguish 1pp from 0. | Improved but still below the floor (~2.5pp at h=1). Reddit's own usable window starts 2020-06, so the ablation runs on ~1,430 sessions, not 2,434. |
| Capital gate: ≥60 forward days | Unaffected — and now the only threshold that still does its job. | Unaffected. |

**The honest position, restated for the deep sample.** §6 as written specifies h=5. That test needs ~3,800 weekly blocks and will not be runnable in this project's lifetime. What *is* now runnable is the **h=1 daily** test at 2.5pp against a 2pp bar — marginal, but interpretable, which it was never going to be at 14pp.

This is a real temptation and worth naming: the obvious move is to quietly promote h=1 to the primary threshold because it is the one the data can support. That is §9's "threshold-adjacent tuning" wearing a different hat — choosing the test by what the sample can pass. If h=1 becomes primary it must be **because it was always a stated target (§2 names h=1 and h=5 equally), decided now, before any model runs**, with the h=5 result still reported and still governed by the same +2pp rule even though its CI will be uninformative. Recording the decision here, in advance, is what makes it legitimate; discovering it after seeing results would not be.

**DECIDED 2026-08-04: h=1 is the primary threshold.** Taken on the terms above — before any model has been fitted, before any accuracy number exists, and with the h=5 test retained rather than dropped. What this buys is a primary test whose noise floor (2.5pp) sits below the effect it tests for (2pp), instead of one eight times above it.

Three commitments come with it, and they are the price of the promotion being honest rather than convenient:
1. **h=5 is still reported, still governed by +2pp, and still capable of abandoning the project.** It is demoted in precedence, not deleted.
2. **A pass on h=5 is not evidence.** Its CI is ~±6pp; it cannot distinguish +2pp from 0. Any write-up must say so next to the number rather than in a footnote.
3. **This is the last time either threshold moves.** The §6 numbers were frozen on 2026-08-02 and remain frozen; what changed here is precedence between two pre-existing targets, decided in advance and recorded. A further revision after results exist would invalidate the run — that is §9's kill criterion, not a guideline.

**The numbers stay frozen.** More data is not a reason to move a threshold, in either direction.

---

### 6.2 RESULTS (2026-08-04) — the thresholds fire, and the project abandons

§6's numbers were frozen on 2026-08-02, before any model ran. They are applied here unchanged and the outcome is reported as found.

**Ablation, h=1, common window (from 2020-06, n≈1,400). Edge = accuracy − walked-forward majority-class predictor, paired, block bootstrap.**

| Feature set | LR edge | LR 95% CI | GBM edge (5 seeds) |
|---|---|---|---|
| price-only | **+5.16pp** | [+0.48, +10.06] | +6.52 ± 0.41 |
| +turnover | +4.87pp | [+0.00, +9.96] | +6.72 ± 0.30 |
| +reddit | +2.82pp | [−1.77, +7.88] | +5.40 ± 0.40 |
| all | +2.20pp | [−2.41, +7.22] | +5.40 ± 0.44 |

**The regime table is the finding.** Price-only, logistic, on the widest window (2016-04 → 2026-07):

| Regime | n | Edge (h=1) | 95% CI | Edge (h=5) |
|---|---|---|---|---|
| pre-mania 2016 → 2020-02 | 511 | **+0.20pp** | [−5.47, +5.69] | −1.96pp |
| **mania 2020-06 → 2021-12** | 364 | **+13.19pp** | [+5.16, +22.01] | **+15.93pp** |
| chop 2022+ | 1,063 | **+0.56pp** | [−3.20, +4.52] | −4.58pp |

**Every point of edge lives in the mania regime.** Outside it the model is indistinguishable from the majority class at h=1 and *worse than it* at h=5. This is exactly the bull-market artifact §6(c) abandons on — the criterion that was retired on 2026-08-02 for want of data and reinstated by §3.5. It earned its place back on the first run that could test it.

**Net-of-cost PnL, h=1, NPR 1,000,000, 1× friction:**

| Model | Gross Sharpe | Net Sharpe | Net CAGR | Max DD |
|---|---|---|---|---|
| LR (all features) | 0.66 | **−1.36** | −19.1% | −58.8% |
| GBM (all features) | 1.32 | **−0.69** | −11.2% | −41.6% |
| LR (price-only) | 0.78 | **−1.06** | −15.7% | −53.2% |
| **buy-and-hold** | 0.32 | **+0.30** | +4.0% | −20.2% |

**The thresholds, applied:**

| §6 threshold | Result | Verdict |
|---|---|---|
| (a) h=1 accuracy > majority + 2pp | best of 8 combos: GBM/price-only **+7.30pp**, CI [+2.61, +11.98] | point estimate and CI both clear — **but see below** |
| (b) Net Sharpe ≥ 0.4 after costs | −1.36 / −0.69 / −1.06 | **FAIL** |
| (c) Edge only in the mania regime | +13.19pp mania vs +0.20 / +0.56 elsewhere | **TRIGGERS — abandon** |
| (d) Reddit adds ≥ 1pp | **−2.34pp** | **DROP the Reddit module** |
| (e) ≥ 60 forward days | 1 | not met (Phase 5 just built) |

**(a) is the one that must not be over-read, and the reason is §9's "know thyself".** The +7.30pp is the *maximum over 8 model × feature-set combinations*, and its interval is not corrected for that selection. A best-of-8 point estimate is biased upward by construction; the honest reading is the per-set table above, where the single pre-specified configuration (price-only, LR) gives +5.16pp with a CI whose lower bound is +0.48pp — i.e. barely distinguishable from zero, let alone from +2pp. And (c) explains where even that comes from.

**Verdict: the project abandons on §6(b) and §6(c).** No capital is deployed. This is the "documented no edge found" outcome §1 committed to in advance, and it arrives with the mechanism identified rather than as a shrug:

1. **The apparent edge is a regime artifact.** It is a 2020–21 bull-market phenomenon and does not exist in the four years before or the four years after.
2. **What edge there is cannot survive costs.** Gross Sharpe 0.66–1.32 becomes net −0.69 to −1.36, and buy-and-hold beats every model.

   **Correction (2026-08-04, after first writing this): it is not the DP charge.** §4 predicted the flat per-scrip charge would kill a high-turnover signal, and that framing was carried into the first draft of this section. Decomposed, it is wrong at any realistic capital base:

   | Capital | Round trip | commission + SEBON | DP charge | DP share |
   |---|---|---|---|---|
   | 50,000 | 93.0 bps | 83.0 | 10.0 | 10.8% |
   | 1,000,000 | 71.5 bps | 71.0 | 0.5 | **0.7%** |

   Over 371 round trips at NPR 1,000,000 the variable cost is ~263% of capital and the DP charge ~1.85%. **The killer is ordinary commission × turnover**, not a Nepal-specific flat fee. Two consequences: §4's insistence that results depend on the declared capital base is much weaker than it claimed (net Sharpe moves only −0.57 → −0.85 from 10M down to 50k), and the finding generalises — any market charging ~32 bps a side destroys a signal that trades 371 times, which is not an exotic result. §4's DP argument survives only for very small accounts, where it is still a minority of the cost.
3. **Alt-data adds nothing.** Reddit attention *subtracts* 2.34pp. §5's own EDA pointed this way; the ablation confirms it.

**What is NOT claimed:** that NEPSE is unpredictable. What is shown is that *this* target, on *this* sample, with *these* features, does not clear a bar fixed in advance. §6.1's h=5 caveat still stands — its intervals remain too wide to resolve 2pp either way.

---

### 6.3 PRE-REGISTRATION: low-turnover decision rule (written 2026-08-04, BEFORE running)

§6.2 stands as recorded. This is a **new experiment**, not a re-run, and it is written down before it executes precisely because §9 names "overfitting via repeated threshold-adjacent tuning" as this project's top risk and because everything below is being proposed by someone who has already seen §6.2. Pre-registration is the only thing separating this from the failure mode.

**Hypothesis, derived from arithmetic rather than from results.** §6.2's models lose to costs, not to inaccuracy. Break-even directional accuracy against a 65.5 bps round trip at NEPSE's 0.94% mean daily move:

| Holding period | Break-even accuracy | Model has |
|---|---|---|
| 1 session | 85.0% | 55.5% |
| 5 sessions | 57.0% | 55.5% |
| **21 sessions** | **51.7%** | **55.5%** |
| 63 sessions | 50.6% | 55.5% |

The 30pp gap at daily frequency is unreachable by any classifier. At 21 sessions the gap reverses. **This table depends on no result from §6.2** — it is the cost model and the volatility of the series, both known before any model ran. The claim under test is therefore that the *decision rule*, not the classifier, is what failed.

**The change.** A position layer between probability and trade:
- **Deadband δ:** go long when `p_up > 0.5 + δ`, go flat when `p_up < 0.5 − δ`, otherwise **hold the current position**. Hysteresis is what collapses turnover — a signal oscillating around 0.5 currently generates a trade every time it crosses.
- **Minimum holding period:** once a position changes, it cannot change again for `m` sessions.

**How δ and m are chosen, and why this is not peeking.** They are selected **inside each walk-forward fold, on that fold's training window only**, by maximising in-sample net-of-cost Sharpe. They are model parameters, not researcher choices. No test row influences any parameter, and I do not see a test result before the choice is made. A grid scanned by hand against the pooled test set would be the forbidden thing; this is not that.

**What is NOT changing:** the classifiers, the features, the folds, the embargo, the cost model, the capital base, and §6's thresholds. Same harness throughout.

**Success criterion — §6(b) unchanged: net Sharpe ≥ 0.4** at NPR 1,000,000, 1× friction.

**Two additional pre-registered guards, because the obvious way to pass is to cheat toward buy-and-hold:**
1. The rule must **beat buy-and-hold** on the same window (+0.30 net Sharpe). Merely matching it means the model added nothing.
2. The rule must be **meaningfully different from buy-and-hold** — time-in-market below 90% and at least 6 round trips. A rule that holds 99.5% of the time *is* buy-and-hold wearing a classifier, and passing that way would be a measurement artifact, not a result.

**Declared in advance: if this fails, it fails.** §6.2's verdict stands, §6.3 is written up as a second null, and there is no third attempt. A rule that needs a third redesign to clear a fixed bar is fitting the bar, not the market.

**RESULT (2026-08-04, run once as pre-registered). The hypothesis was right and it is still not enough.**

| Model / rule | Net Sharpe | Net CAGR | Max DD | Trades | Time in mkt | Costs |
|---|---|---|---|---|---|---|
| logistic / naive `p>0.5` | −0.46 | −8.0% | −54.3% | 515 | 43.1% | 162% |
| **logistic / deadband+hold** | **+0.10** | +0.5% | −37.4% | 111 | 27.4% | 58% |
| gbm / naive `p>0.5` | −0.66 | −11.5% | −61.6% | 638 | 49.5% | 160% |
| **gbm / deadband+hold** | **+0.29** | +3.3% | −31.2% | 452 | 40.8% | 214% |
| **buy-and-hold** | **+0.54** | +9.6% | −43.3% | 2 | 99.9% | 9% |

**The turnover hypothesis is confirmed.** Hysteresis moved logistic from −0.46 to +0.10 and GBM from −0.66 to +0.29 — swings of 0.56 and 0.95 Sharpe from changing nothing but *when to act on the same probabilities*. Costs fell from 162% to 58% of capital. §6.3 predicted the decision rule was the binding constraint and it was.

**And it fails anyway, on both pre-registered criteria:**
- Net Sharpe 0.10 / 0.29 against the 0.4 bar. **FAIL.**
- Buy-and-hold returns **0.54** on the same window. **No model beats simply owning the index.** FAIL.

(Guards 3 and 4 pass — the rules are genuinely different from buy-and-hold, at 27–41% time in market. They did not pass by degenerating into it.)

**Why it fails, and it is not subtle.** The models sit in cash 59–73% of the time. NEPSE compounded at ~9.6%/yr net across this window; being out of the market means forgoing that drift, and 55.5% daily directional accuracy does not identify *which* days to skip well enough to pay for missing the rest. The strategy is not losing to costs any more — it is losing to the index.

**Per §6.3, this is the second null and there is no third attempt.** Recorded as found.

*One honest note on what a third attempt would have tried, so the option is documented rather than quietly available: the rule is symmetric and starts flat, so it must earn its way into the market. An asymmetric rule — hold long by default, exit only on a strong down signal — would capture the drift and use the model only to sidestep drawdowns. That is a materially different and more plausible design. It is also exactly the "one more idea after seeing the result" that §6.3 pre-committed against, and the reason to write it down here rather than run it is that the pre-commitment is worth more than the idea.*



### 6.4 Attempt three: the asymmetric rule — and what its result can and cannot mean

**Disclosure first, because it determines how everything below should be read.** §6.3 pre-committed to no third attempt, in writing, before running. This section breaks that commitment. It was requested after §6.3's result was known, and the rule family it adds — hold long by default, exit only on a confident bearish signal — is precisely the idea §6.3 named as "what a third attempt would have tried" and declined to run.

That history cannot be undone by careful methodology afterwards, so it is recorded rather than smoothed over. **§6.2 and §6.3 remain exactly as written; neither was re-run, re-scored, or edited.**

**What was done to preserve what discipline remains:**
- The rule *family* is added to the same per-fold grid as `delta` and `min_hold`, and is **selected in-sample on training data**. Which family a given test fold uses is a fitted parameter, not a researcher choice.
- The classifier, features, folds, embargo, cost model, capital base and thresholds are all unchanged.
- The regime breakdown is reported alongside, because §6.2's central finding was that the classifier's edge lived entirely in 2020–21. A rule that only works there is the same artifact wearing new clothes.

**Two compute decisions, neither touching test data:** the grid is coarse, and rule selection uses the most recent 750 training sessions rather than the full expanding window (the classifier still trains on everything). The full grid over an expanding window is ~27,000 ledger runs and does not terminate in reasonable time.

**The honest discount.** This is the third rule tried against a fixed bar on one dataset by someone who has seen the first two fail. Even with in-sample parameter selection, the *family* was chosen with knowledge of what failed. A pass here is **weaker evidence than a pass in §6.2 would have been**, and the appropriate response to a pass is not deployment but the §6(e) capital gate: ≥60 forward trading days, logged prospectively, with forward accuracy within 5pp of backtest. That gate exists exactly for results like this one, and the forward log (§7) has been running since 2026-08-04.

**RESULT (2026-08-04). The best rule yet, and it still never beats owning the index.**

| Model / rule | Net Sharpe | Net CAGR | Max DD | Trades | Time in mkt |
|---|---|---|---|---|---|
| logistic / naive `p>0.5` | −0.46 | −8.0% | −54.3% | 515 | 43.1% |
| logistic / symmetric deadband (§6.3) | +0.10 | +0.5% | −37.4% | 111 | 27.4% |
| **logistic / asymmetric (§6.4)** | **+0.35** | +3.8% | **−21.5%** | **43** | 29.2% |
| gbm / asymmetric (§6.4) | +0.20 | +1.8% | −27.0% | 320 | 30.3% |
| **buy-and-hold** | **+0.54** | **+9.6%** | −43.3% | 2 | 99.9% |

Each redesign improved it — −0.46 → +0.10 → +0.35, max drawdown halving from −54% to −21.5%, trades falling from 515 to 43. The direction of travel was right every time. **It still fails both criteria:** 0.35 against the 0.4 bar, and 0.54 for buy-and-hold.

**The regime table ends this, and it is worse than §6.2's.**

| Model | Regime | Strategy CAGR | Buy-and-hold CAGR | Trades |
|---|---|---|---|---|
| logistic | pre-mania 2017→2020-02 | **0.0%** | **+19.1%** | **0** |
| logistic | mania 2020-06→2021-12 | +43.9% | +61.9% | 15 |
| logistic | chop 2022+ | +0.3% | +0.7% | 29 |
| gbm | pre-mania 2017→2020-02 | −6.4% | +19.1% | 50 |
| gbm | mania 2020-06→2021-12 | +13.8% | +61.9% | 94 |
| gbm | chop 2022+ | +1.6% | +0.7% | 176 |

**The strategy loses to buy-and-hold in five of six regime-model cells**, and the one it wins (gbm/chop, +1.6% vs +0.7%) is a rounding error on a flat market. In the pre-mania stretch the logistic rule **sat in cash for three years and made 0% while the index compounded at 19.1%.** It did not mistime the market; it never entered.

That is the finding, and it is no longer a cost or turnover problem. **A 55.5% daily directional signal cannot decide when to be OUT of a market that drifts up 9.6%/yr.** The cost of being wrong about absence — forgone drift — exceeds anything the signal recovers, at every rule frequency tried.


### 6.6 PRE-REGISTRATION: volatility targeting (written 2026-08-05, BEFORE running)

**This is a different signal, not a fourth decision rule.** §6.2–§6.4 all predicted *direction* and differed only in when to act on it. Three designs agreeing that a 55.5% directional signal cannot pay is a result, and re-skinning it a fourth time would be the search §9 forbids. This section changes what is being predicted.

**Why volatility, decided from properties of the series rather than from what failed:**

| Quantity | Autocorrelation, lag 1 | lag 5 | lag 21 |
|---|---|---|---|
| daily return (what §6.2–6.4 predicted) | +0.103 | +0.017 | +0.005 |
| **absolute return (volatility)** | **+0.232** | **+0.187** | +0.060 |

Trailing 21-day vol predicts the next 21 days' vol at **r = 0.41**. And the payoff to knowing it is large: sorting sessions into trailing-vol quartiles gives annualised Sharpe **1.04** in the lowest quartile against 0.26 / 0.09 / 0.61 in the rest. Volatility is roughly twice as autocorrelated as direction and the low-vol state is where the return-per-unit-risk lives.

**The mechanism needs no directional skill at all**, which is the point. Exposure is `clip(target_vol / predicted_vol, 0, 1)` — long-only, never levered. When the market is calm you own it; when it is violent you own less. Nothing here claims to know which way it goes.

**This required a real extension, not a parameter:** `run_backtest_weighted` supports continuous exposure in [0,1], charges costs on the *traded* notional `|target − current|`, uses weighted-average cost basis for partial-sale CGT, and keeps the settlement and circuit rules. Verified against the binary ledger at w=1.

**Parameters** — vol window ∈ {21, 63}, target vol ∈ {10%, 15%, 20%}, rebalance band ∈ {10%, 20%} — are selected **per fold on training data only**, exactly as in §6.3. A hand-picked best-of-18 would be worthless; a feasibility scan showing 18 combinations was run before this was written and is explicitly NOT the result.

**Criteria, unchanged from §6.3:** net Sharpe ≥ **0.4** at NPR 1,000,000, 1× friction, **and** it must beat buy-and-hold. Plus the regime check: it must not depend on 2020–21.

**Declared in advance.** If volatility targeting fails too, then across four attempts spanning two different predicted quantities and two position spaces, nothing beats owning the index — and that is the answer, not a prompt for a fifth.

### 6.5 Conclusion: the tradeable answer is buy-and-hold

Three independent rule designs, each better than the last, none beating a two-trade strategy. Combined with §6.2's finding that the classifier's *accuracy* edge was itself confined to 2020–21, the evidence is consistent and points one way.

**What this project produces as investment guidance: own the index.** Net Sharpe **0.54**, net CAGR **+9.6%** after realistic costs, and it beat every timing model in every regime tested. That is a real, actionable, cost-verified result. It is simply not a *predictive* one.

**With a risk warning the Sharpe hides: maximum drawdown −43.3%.** Buy-and-hold on NEPSE means sitting through roughly a halving. The timing rules cut that to −21.5%, and if drawdown rather than Sharpe is the binding constraint for a given investor, §6.4's logistic rule is defensible on those grounds alone — at a cost of ~5.8pp/yr of return. **That is a risk preference, not an edge, and must never be reported as one.**

**No fourth attempt.** The result is stable across three designs differing substantially in mechanism. Continuing would be searching for the rule that happens to clear 0.4 on this particular decade — §9's named failure mode, and the thing this project was built to avoid.

---

## 7. Forward run

**BUILT 2026-08-04** — `nepselab/forward/log.py` + `scripts/phase5_forward.py`, wired into `cron_archive.sh` after the pull and before the backup.

- **Daily job:** pull new data, generate predictions for t+1 and t+5, append to an **immutable prediction log** — timestamp, features, model version, git hash. **Never overwrite a past prediction.** The log accumulates indefinitely. Enforced, not merely intended: a second run on the same date raises `PredictionExists` rather than replacing. The escape hatch is to bump `model_version`, never to edit history — because the rewrite that matters is the one applied *after* the outcome is known, which is exactly when it looks most reasonable.
- **Scoring script:** `--score` reads the log, joins outcomes as they arrive, and reports resolved/open counts and rolling accuracy per horizon.
- The log is backed up off-machine with the archive (§3.4). It is irreplaceable in a different way: §7 forbids rewriting it, so losing it cannot be repaired by re-running anything.
- **It keeps running despite §6.2.** The project abandons on the backtest, but the forward job costs one cron line and is the only thing that would ever detect the backtest having been wrong.
- **What the forward run is for.** ~40 trading days cannot distinguish 55% accuracy from 50%. Its purpose is **leak detection and pipeline validation** — catching a feature that silently used future information, a data source that changed shape, a fill rule that never fires in live data. **The backtest remains the primary evidence.** A forward log that agrees with the backtest confirms the pipeline; it does not independently prove an edge.

---

## 8. Phases and repo

| Phase | Work | Effort |
|---|---|---|
| 0 | ~~Scraper running; 1yr index + 5 scrips; corporate-action sourcing; seed `market_params`~~ | **DONE** |
| 1a | ~~Depth probe; exact power calculation; append-only archive + daily job~~ | **DONE 2026-08-02** |
| 1b | **Daily `archive_pull.py` run.** Not effort — calendar time, indefinitely, starting now. Cron installed 2026-08-02, 15:30 + 20:30 local. **Open item: the `today_price` breadth backfill (§3.4) is racing the rolling window.** | **continuous** |
| 1c | ~~Deep-history sourcing probe (§8.1 option D)~~ | **DONE 2026-08-04** |
| 2a | ~~Walk-forward engine, metrics, baselines, regime split~~ | **DONE 2026-08-04** (§3.7) |
| 2b | Cost model, fill logic, net PnL/Sharpe | **BLOCKED on §4** — every cost constant is still `null` |
| 3 | ~~Feature modules~~ **DONE 2026-08-04** (§3.8): price, turnover, Reddit attention. **LLM sentiment pass NOT built** — §5 requires validation against ~200 hand-labelled comments first; the §6(d) gate dropped Reddit on attention alone, so it was never reached | — |
| 4 | ~~Logistic + GBM; ablation; regime table; §6 thresholds~~ | **DONE 2026-08-04** (§6.2) |
| 5 | ~~Daily forward job, immutable log, scoring script~~ | **DONE 2026-08-04** (§7) |

Phase 1a came in well under its 5-day estimate because it stopped at the first thing that mattered; 1c came in under its 1-day estimate for the same reason. Phases 2–5 are **unblocked** as of 2026-08-04 — see §8.1.

**Phase 2 has a prerequisite 1c created:** the walk-forward engine now reads from two stores with different spans and different provenance (`data/deep/` index OHLC from 2016; `data/archive/` everything else from 2025-07). Feature modules must declare which they need, and the engine must refuse to run a config whose features are unavailable over its requested window rather than silently producing a short sample. That is a §2-style correctness requirement, not a convenience.

### 8.1 ~~OPEN~~ RESOLVED 2026-08-04: (D) succeeded, and the project continues

**Outcome: (D) worked.** A daily index series back to **2016-01** is obtained and cross-validated (§3.5). §2's regime split is restored, its power table improves by roughly 3× on every line, and the sample (2,434 sessions) now exceeds the ~1500 the plan assumed before any of this went wrong. **Phases 2–5 are unblocked and the plan proceeds essentially as designed.**

Three qualifications, none fatal, all of which change what gets built:

1. **2016, not 2020.** The probe found history back to 1997, and rejected everything before 2016 because two independent scrapes disagree on 2.4% of daily return signs there — larger than the effect under test. The rejection is the substantive finding; the depth is the easy part.
2. **Index only.** Breadth, sector indices and per-scrip data are not in any deep source. §5's feature modules split across two spans (§3.5), and the breadth backfill is now racing the rolling window (§3.4).
3. **h=5 is still underpowered.** 486 weekly blocks against the ~3,800 §6 needs. (D) restored the sample; it did not restore that threshold. §6.1 covers what follows.

The decision below stood on the assumption that the sample could not be fixed. It could. Recorded as written for the record:

**(D) Source deep history elsewhere — do this first, ~1 day.** NEPSE's API is not the only place NEPSE history exists. ShareSansar, MeroLagani, Nepse Alpha and assorted Kaggle dumps all publish index history well before 2025, and the Reddit dump already covers 2020-06 onward. If a clean, verifiable daily index series back to 2020 can be obtained, **the original plan survives essentially intact** and every number in §2 reverts. This is the only branch that recovers the project as designed, it is cheap to test, and it is therefore the next task. Caveats: provenance and continuity-adjustment become the user's problem rather than the exchange's, and any such series must be reconciled against our 225 archived sessions before it is trusted — an unverifiable series is worse than no series.

> *Post-hoc note: the caveat did the work.* "An unverifiable series is worse than no series" is exactly what killed 1997–2015 and kept 2016–2026. Had the probe stopped at "history exists back to 1997", the project would have gained 4,000 sessions of labels that are wrong ~2.4% of the time in the direction the model is trying to predict — a bigger sample and a worse project. Sourcing was never the hard part; adjudication was.
>
> *Which sources did not work, so nobody re-tries them:* **Nepse Alpha** sits behind a Cloudflare challenge (403 to anything scripted). **ShareSansar** exposes `/index-history-data`, a DataTables endpoint taking `index_id`/`from`/`to`, but returns HTTP 202 with an empty result set for every query, with or without session cookies — a soft block, not a parameter mistake.

**(A) Continue as an engineering exercise; defer modelling.** Build Phases 2–4 against the honest 225-session sample, report every result with its ±18pp interval, and treat the harness — not the results — as the deliverable. §1 already frames the harness as the primary objective ("*the harness is what makes any result believable*"), so this is defensible. Risk: §9's "know thyself" — running models on a sample that cannot resolve them, month after month, is precisely the setup where a 60% point estimate quietly becomes a belief.

**(B) Stop modelling; write up the feasibility null.** §1 committed to documenting a "no edge found" outcome; this is the adjacent, more honest result — *no edge is testable*. Keep 1b running forever, revisit in a few years when the archive is large enough. Cheapest, and intellectually clean.

**(C) Change the target to fit the sample.** More observations per unit time, not more calendar time: floorsheet/intraday data (§3.1 flags depth as an open TODO) would multiply n substantially. But §4 is blunt that manual TMS execution makes intraday strategies unimplementable, so this risks building something well-powered and untradeable.

**Recommendation: (D) now, then (B) if D fails.** (D) is one day and either restores the project or definitively closes the question of whether the sample can be fixed. If it fails, (B) is the outcome the plan already committed in advance to accepting — and (A) is the option §9 warns about by name.

**Resolution.** (D) succeeded, so (B) does not fire and (C) is not needed — the sample was fixable by calendar depth, without changing the target to something §4 says is untradeable. **(A) is now the plan of record, but on its own terms rather than as a consolation:** Phases 2–4 run against 2,434 cross-validated sessions instead of 225, so the harness *and* the results are deliverables. §9's warning still applies to h=5, where the interval remains ~±6pp and a point estimate will still look more convincing than it is.

**The archive (1b) runs regardless**, since it is the only irreversible item and it costs one cron entry — and §3.5 raised its value rather than lowering it: deep history covers the index and nothing else, so `today_price` remains the sole route to breadth, forever.

```
nepse-lab/
├── data/                  # gitignored; parquet
│   ├── archive/           # append-only, exchange-sourced, IRREPLACEABLE (§3.4)
│   ├── deep/              # third-party index history from 2016, re-downloadable (§3.5)
│   └── raw/               # Phase 0 pull; folded into archive/, kept as provenance
├── nepselab/
│   ├── ingest/            # scrapers, archive, deep_history, calendar, macro vintages
│   ├── adjust/            # price adjustment, scrip state machine + tests
│   ├── features/          # base.py, price.py, reddit.py — toggleable
│   │                      #   (news.py, macro.py never built — §5)
│   ├── models/            # classifiers.py: logistic, gbm
│   ├── eval/              # labels, splits, metrics, baselines, costs, portfolio
│   ├── quality.py         # the §3.3 sanity suite, shared by script and tests
│   └── forward/           # log.py: immutable prediction log + scoring
├── configs/               # yaml per experiment + market_params.yaml
├── predictions/           # append-only forward log; never rewritten
├── tests/                 # sanity suite runs in CI
└── results/               # one dir per run, config + metrics + plots frozen
```

**Rules:** config-driven experiments; no hardcoded fees or market-structure constants outside `market_params.yaml`; every result dir carries its config + git hash; sanity tests pass before any experiment runs — enforced by `run_all.py`, which exits on a failing suite rather than continuing to results nobody should read. `results/` is deliberately not gitignored: a documented null result that exists on one disk is not documented. Python: pandas/polars, scikit-learn, lightgbm/xgboost, statsmodels, matplotlib.

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
| ~~**Sample too small to resolve the effect under test**~~ — **not on the original table; largely mitigated 2026-08-04** | ~~Certain~~ **Partial** | Was unmitigable against NEPSE's API; §3.5 sourced 2,434 cross-validated sessions elsewhere and §2's power table improved ~3×. **Still live for h=5** (486 blocks vs ~3,800 needed) and for every breadth feature (2025-07 onward only). The process lesson stands regardless: the original plan carried a power calculation as a TODO rather than a gate, and had it been computed at Phase 0 against real row counts, §3.4 surfaces a week earlier and five sessions are not lost. **Compute power before building, not after.** |
| **Third-party history is silently wrong** — new, and now load-bearing | **Certain in some era** | Realised immediately: two independent scrapes of the same index disagree on 2.4% of pre-2016 daily return signs, and there is no authority to adjudicate them. Mitigation is the cross-source check in `scripts/phase1c_deep_history.py`, re-run on every refresh — **never trust a single deep source, and never accept a window where two disagree at a rate near the effect size.** The 2025-07+ archive is the only part of the sample with exchange provenance. |
| ~~No edge found~~ **MATERIALISED 2026-08-04** | ~~High-ish~~ **Realised** | Thresholds in §6 fixed in advance; abandoned rather than tuned. §6.2 records the outcome and the mechanism: the edge is a 2020–21 regime artifact, and what remains does not survive costs. |
| Overfitting via repeated threshold-adjacent tuning | High (know thyself) | §6 numbers are frozen; changing them invalidates the run. **Survived the actual test:** §6.2 ran once, against thresholds set six days earlier, and reports a best-of-8 figure explicitly flagged as selection-biased rather than headlined. |

**Kill criteria:**
- ~~If the adjusted-price layer can't be validated, restrict to index-only.~~ **Resolved at the Phase 0 gate: index-only is now the design**, not a fallback (§3.2). The criterion is retired.
- If the backtest fails any §6 threshold, stop. Write up the null result and do not re-tune.
- If the forward log diverges from the backtest by more than 5pp accuracy over ≥60 trading days, treat it as a pipeline bug, find it, and re-run the backtest — do not deploy capital on the more favourable of the two.
