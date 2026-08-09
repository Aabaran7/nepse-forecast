# Figma prompt — NEPSE research dashboard

Paste everything below the line into Figma Make.

---

Design a web dashboard for a personal research project that studies whether the
Nepal Stock Exchange (NEPSE) is predictable. It is **not** a trading app. It is a
lab notebook made public: the author's own conclusion so far is that no reliable
edge was found, and the design must be comfortable saying that.

## Who reads it

The author, and people reviewing his work — recruiters, other researchers. It
should read as careful and honest, not as a product trying to sell a signal.
Think closer to a scientific instrument panel or a well-set research paper than
to a broker app or a crypto tracker.

## Tone

Sober, quiet, precise. Generous white space. Restrained color used only where it
carries meaning. Nothing that implies excitement, urgency, or a recommendation.
No glowing numbers, no gradients, no "BUY" energy, no rocket or flame icons, no
red/green profit-and-loss styling.

## The single most important design problem

The main screen reports how often the model's predictions were right. Right now
it has **one** resolved prediction, which was correct — so the honest reading is
"we know nothing yet," but a 100% bar makes it look like a triumph.

**Sample size must be as visually loud as the result itself.** Design for these
three states explicitly and show all three in the file:

1. **Not enough data** (under ~30 resolved). The accuracy number is visibly
   held back — greyed, or shown with the count directly attached, or replaced by
   a "needs N more" progress indicator. It must be impossible to screenshot the
   number alone and have it look like a finding.
2. **Enough data** (30+). Accuracy shown normally, always paired with a baseline
   comparison bar.
3. **Empty** (nothing resolved yet). A calm explanation, not an error.

Any accuracy figure must appear beside its sample count and its baseline. Never
alone.

## Screens

Four tabs sharing one header and one left sidebar.

**Header:** project title, a one-line subtitle stating this is descriptive and
not advice, and the four tabs.

**Sidebar — "Data freshness":** four rows (Exchange archive, Scrip prices, Deep
history, News), each showing a date and how many days old it is. Include a state
for "stale" — a gap here means permanently lost data, so it needs to be
noticeable without being alarming. Small, quiet, always visible.

**Tab 1 — Forward log.** The scoreboard.
- Four stat tiles: Directional predictions, Resolved, Still open, Exposure rows.
- One comparison chart: model accuracy vs. a "always guess the commoner
  direction" baseline, grouped by time horizon (t+1, t+5). Usually only one or
  two horizons, so design for **very few bars** — two or four total. Do not let
  two bars fill a huge canvas. Consider a horizontal comparison bar, a bullet
  chart, or a dot plot instead of tall vertical columns.
- A small table of the same numbers, with real percent signs.
- A notice block explaining what the log can and cannot prove. This is long-ish
  body text and appears on several screens — design it as a proper component,
  readable, not a cramped colored strip.

**Tab 2 — Market.**
- A line chart of the index, with a 1 year / 3 years / All toggle.
- A breadth chart: rising vs falling stocks per day, diverging from a zero
  centre line, about 60 days.
- A line chart of what share of trading is concentrated in the 10 busiest stocks.
- Each chart needs a caption explaining what it means.

**Tab 3 — Stocks.** A screener for the most recent trading day.
- A prominent notice at the top correcting a common misreading of volume. It is
  a paragraph, not one line, and it must not look like a warning error.
- A filter row: a toggle ("Hide thin stocks"), a multi-select ("Day type"), and
  a search box.
- Five small count tiles for day types: rose on heavy volume, rose on normal
  volume, fell on heavy volume, fell on normal volume, unchanged.
- A dense sortable table, roughly 350 rows, columns: Symbol, Close, Change,
  Volume vs normal, Trades, Avg trade, Thin, Day type, Limit up, Limit down,
  Turnover. Design the row height, number alignment, and a clear treatment for
  the boolean "Thin" flag — a flagged row means the day's move may have been a
  single participant, so it must be visible while scanning.

**Tab 4 — News.** A list of scraped headlines.
- Columns: trading session, source, headline, published date, link.
- Headlines are in **both English and Nepali (Devanagari)** — the type must
  handle both scripts at the same size without one looking broken.
- A state for "not scored yet," since the sentiment analysis is not built.
- A note explaining that some headlines have no trading session assigned yet,
  which is correct rather than an error.

## Color

These are already validated for colorblind readers and for contrast in both
light and dark. Use them as given; do not substitute.

| Role | Light | Dark |
|---|---|---|
| Up / positive | `#0d7d9e` | `#2ba3c4` |
| Down / negative | `#c2571a` | `#c9762a` |
| Series 1 | `#3b6fd4` | `#5b93e6` |
| Series 2 | `#c2571a` | `#c9762a` |
| Series 3 | `#8b5cf6` | `#9575f0` |
| Series 4 | `#3f8f5f` | `#4fa876` |
| Baseline / secondary | neutral grey | neutral grey |

Rules:
- Up/down is blue vs orange, deliberately **not** green vs red — green/red is
  the exact pair colorblind readers cannot separate.
- Direction is never shown by color alone; a sign or arrow always accompanies it.
- The "baseline" comparison bar stays neutral grey so it never competes with the
  model's bar.
- Grid lines and axes recede. Text uses text colors, never the series color.

## Type and layout

- Set a clear type scale: page title, tab labels, stat tile label, stat tile
  value, chart title, body, caption, table cell. Numbers in tables should be
  tabular/monospaced figures so columns align.
- Constrain content to a comfortable reading width — do not stretch tables and
  charts edge to edge on a wide monitor.
- Consistent spacing rhythm. Group related things into cards or clearly
  separated sections; right now everything floats at the same level with no
  hierarchy.
- Design **light and dark** as two deliberate themes, not an automatic inversion.
- Show a desktop layout and a narrow/tablet layout.

## Components to deliver

Stat tile (with and without a help icon), chart card with title + caption,
comparison bar chart, line chart, diverging bar chart, dense data table with
sort and a boolean flag column, filter row, notice/callout block (in a neutral
and an attention variant), sidebar freshness row (fresh and stale), tab bar,
empty state, and the "not enough data yet" state.

## Explicitly avoid

- Any layout where a chart title and its legend can overlap.
- Data labels that get clipped by the top of the plot area.
- Numbers shown without their unit (a percentage must carry `%`).
- Two vertical bars stretched across a full-width canvas.
- Dual y-axes, pie charts, 3D effects, drop shadows on data marks.
- Any element that reads as a buy/sell recommendation.
