/**
 * The real data contract. Replaces data/mock.ts.
 *
 * Every field here is produced by scripts/export_dashboard.py from the archive.
 * Nothing is computed in the browser except formatting -- accuracy, baselines
 * and the minimum sample size are decided once, in Python, so the page and the
 * Streamlit tool can never quietly disagree about what the numbers mean.
 *
 * The mock had three fields this project does not have: a per-prediction `note`,
 * an `id`, and a `scrip`. They are gone rather than faked. What the log actually
 * carries -- model version, probability, git hash -- is here instead.
 */

export type DayType =
  | 'heavy_up'
  | 'quiet_up'
  | 'heavy_down'
  | 'quiet_down'
  | 'flat'

/** Labels for DayType. Deliberately neutral: the panel describes, never advises. */
export const DAY_TYPE_LABEL: Record<DayType, string> = {
  heavy_up: 'Rose, heavy volume',
  quiet_up: 'Rose, normal volume',
  heavy_down: 'Fell, heavy volume',
  quiet_down: 'Fell, normal volume',
  flat: 'Unchanged',
}

export interface Stock {
  symbol: string
  close: number | null
  /** Percent, already multiplied out. */
  change: number | null
  /** Multiple of this scrip's own 20-session median volume. */
  volumeVsNormal: number | null
  trades: number | null
  avgTrade: number | null
  /** Under ~10 transactions: the day's move may be a single participant. */
  thin: boolean
  dayType: DayType
  limitUp: boolean
  limitDown: boolean
  turnover: number | null
}

export interface NewsItem {
  /** First trading session that could have traded on this. Null = not yet. */
  session: string | null
  source: string
  headline: string
  publishedAt: string | null
  url: string
  scored: boolean
}

export interface FreshnessItem {
  label: string
  date: string | null
  daysAgo: number | null
  stale: boolean
}

export interface HorizonAccuracy {
  horizon: string
  n: number
  /** Null when nothing has resolved at this horizon. */
  modelPct: number | null
  baselinePct: number
  minN: number
}

export interface Prediction {
  asOf: string
  /** 'direction' predicts up/down; 'exposure' is a §6.6 position size, not a forecast. */
  kind: 'direction' | 'exposure'
  horizon: string
  direction: 'up' | 'down' | null
  probUp: number | null
  exposure: number | null
  modelVersion: string
  gitHash: string
  outcome: 'correct' | 'incorrect' | null
  fwdReturn: number | null
}

export interface ForwardLog {
  totalPredictions: number
  resolved: number
  open: number
  exposureRows: number
  minN: number
  horizons: HorizonAccuracy[]
  predictions: Prediction[]
}

export interface Dashboard {
  generatedUtc: string
  gitHash: string
  freshness: FreshnessItem[]
  forwardLog: ForwardLog
  indexData: { date: string; value: number }[]
  breadthData: { date: string; rising: number; falling: number }[]
  concentrationData: { date: string; pct: number }[]
  stocks: Stock[]
  session: string | null
  news: NewsItem[]
}

/**
 * Fetched at runtime, not imported.
 *
 * An `import` would bake the numbers into the JS bundle, which means every
 * daily data refresh needs a rebuild and a redeploy. Fetching keeps the build
 * static and the data live: the daily job rewrites one JSON file and commits it.
 *
 * import.meta.env.BASE_URL respects the Vite `base` setting, so this resolves
 * correctly whether the site is served from a domain root or a /repo/ subpath.
 */
export async function loadDashboard(): Promise<Dashboard> {
  const url = `${import.meta.env.BASE_URL}data/dashboard.json`
  const res = await fetch(url, { cache: 'no-cache' })
  if (!res.ok) {
    throw new Error(
      `Could not load ${url} (HTTP ${res.status}). ` +
        `Run: .venv/bin/python scripts/export_dashboard.py`
    )
  }
  return res.json()
}
