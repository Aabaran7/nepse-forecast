import { useState, useMemo } from 'react'
import NoticeBlock from '../components/NoticeBlock'
import { useDashboard } from '../data/DashboardContext'
import { type DayType, type Stock } from '../data/dashboard'

// Keys match nepselab/features/scrip.py's `quadrant` values exactly. The design
// export invented its own (rose_heavy / fell_normal / unchanged); those are gone
// rather than translated, because a mapping layer between two naming schemes is
// a place for one of them to drift silently.
const DAY_TYPE_LABELS: Record<DayType, string> = {
  heavy_up: '↑ heavy vol',
  quiet_up: '↑ normal vol',
  heavy_down: '↓ heavy vol',
  quiet_down: '↓ normal vol',
  flat: '— unchanged',
}

const DAY_TYPE_COLORS: Record<DayType, string> = {
  heavy_up: 'var(--up)',
  quiet_up: 'var(--up)',
  heavy_down: 'var(--down)',
  quiet_down: 'var(--down)',
  flat: 'var(--text-faint)',
}

type SortKey = keyof Stock
type SortDir = 'asc' | 'desc'

const DAY_TYPE_OPTIONS: { value: DayType | 'all'; label: string }[] = [
  { value: 'all', label: 'All day types' },
  { value: 'heavy_up', label: '↑ Heavy vol' },
  { value: 'quiet_up', label: '↑ Normal vol' },
  { value: 'heavy_down', label: '↓ Heavy vol' },
  { value: 'quiet_down', label: '↓ Normal vol' },
  { value: 'flat', label: '— Unchanged' },
]

// Real data has gaps the mock never did: a scrip too newly listed to have a
// volume baseline, a session with no turnover figure. An em dash says "not
// known"; rendering 0.00 would say "measured, and it was zero".
function fmt(n: number | null, decimals = 2) {
  if (n === null || Number.isNaN(n)) return '—'
  return n.toLocaleString('en-IN', { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
}

function CountTile({ label, count, color }: { label: string; count: number; color: string }) {
  return (
    <div
      style={{
        background: 'var(--card)',
        border: '1px solid var(--card-border)',
        borderRadius: 4,
        padding: '12px 14px',
      }}
    >
      <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase', color: 'var(--text-faint)', marginBottom: 6 }}>
        {label}
      </div>
      <div
        style={{
          fontFamily: "'IBM Plex Serif', Georgia, serif",
          fontSize: 22,
          fontWeight: 400,
          color,
          fontVariantNumeric: 'tabular-nums',
        }}
      >
        {count}
      </div>
    </div>
  )
}

export default function Stocks() {
  const [hideThin, setHideThin] = useState(false)
  const [dayTypeFilter, setDayTypeFilter] = useState<DayType | 'all'>('all')
  const [search, setSearch] = useState('')
  const [sortKey, setSortKey] = useState<SortKey>('symbol')
  const [sortDir, setSortDir] = useState<SortDir>('asc')
  const { stocks: stockData, session } = useDashboard()

  const counts = useMemo(() => {
    const c: Record<DayType, number> = {
      heavy_up: 0,
      quiet_up: 0,
      heavy_down: 0,
      quiet_down: 0,
      flat: 0,
    }
    for (const s of stockData) if (s.dayType in c) c[s.dayType]++
    return c
  }, [stockData])

  const filtered = useMemo(() => {
    let rows = [...stockData]
    if (hideThin) rows = rows.filter((r) => !r.thin)
    if (dayTypeFilter !== 'all') rows = rows.filter((r) => r.dayType === dayTypeFilter)
    if (search) rows = rows.filter((r) => r.symbol.toLowerCase().includes(search.toLowerCase()))
    rows.sort((a, b) => {
      const av = a[sortKey]
      const bv = b[sortKey]
      // Real data has nulls where the mock never did (a scrip with no volume
      // baseline yet, a missing turnover). Sort them to the bottom instead of
      // letting Number(null) === 0 park them among the genuinely flat rows.
      if (av === null && bv === null) return 0
      if (av === null) return 1
      if (bv === null) return -1
      const cmp =
        typeof av === 'string' ? av.localeCompare(String(bv)) : Number(av) - Number(bv)
      return sortDir === 'asc' ? cmp : -cmp
    })
    return rows
  }, [stockData, hideThin, dayTypeFilter, search, sortKey, sortDir])

  function handleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('asc')
    }
  }

  const th = (key: SortKey, label: string, align: 'left' | 'right' = 'right') => (
    <th
      onClick={() => handleSort(key)}
      style={{
        padding: '6px 10px 8px 0',
        fontSize: 10,
        fontWeight: 600,
        letterSpacing: '0.06em',
        textTransform: 'uppercase',
        color: sortKey === key ? 'var(--text)' : 'var(--text-faint)',
        cursor: 'pointer',
        userSelect: 'none',
        textAlign: align,
        whiteSpace: 'nowrap',
        borderBottom: '1px solid var(--rule)',
      }}
    >
      {label}
      {sortKey === key && (
        <span style={{ marginLeft: 3, opacity: 0.6 }}>{sortDir === 'asc' ? '↑' : '↓'}</span>
      )}
    </th>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* Volume notice */}
      <NoticeBlock title="Reading volume in this screener">
        <p style={{ margin: 0 }}>
          "Volume vs. normal" is each scrip's traded quantity for this session divided by its trailing median daily volume. A ratio of 2.0 means twice the typical activity; 0.3 means a very thin day. The figure alone does not signal direction or intent — a single large block trade can multiply volume without reflecting broad participation.
        </p>
        <p style={{ margin: '8px 0 0' }}>
          Rows flagged "Thin" had fewer than 15 trades or volume below 25% of normal. A price move on a thin day may reflect a single participant rather than market consensus, and should not be read as a reliable signal.
        </p>
      </NoticeBlock>

      {/* Filter row */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          flexWrap: 'wrap',
        }}
      >
        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 7,
            fontSize: 12,
            color: 'var(--text-muted)',
            cursor: 'pointer',
          }}
        >
          <input
            type="checkbox"
            checked={hideThin}
            onChange={(e) => setHideThin(e.target.checked)}
            style={{ accentColor: 'var(--s1)', width: 13, height: 13 }}
          />
          Hide thin stocks
        </label>
        <select
          value={dayTypeFilter}
          onChange={(e) => setDayTypeFilter(e.target.value as DayType | 'all')}
          style={{
            fontSize: 12,
            padding: '5px 8px',
            borderRadius: 3,
            border: '1px solid var(--border)',
            background: 'var(--card)',
            color: 'var(--text)',
            outline: 'none',
          }}
        >
          {DAY_TYPE_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search symbol…"
          style={{
            fontSize: 12,
            padding: '5px 10px',
            borderRadius: 3,
            border: '1px solid var(--border)',
            background: 'var(--card)',
            color: 'var(--text)',
            outline: 'none',
            width: 160,
          }}
        />
        <span style={{ fontSize: 11, color: 'var(--text-faint)', marginLeft: 'auto' }}>
          {filtered.length} of {stockData.length} scrips
          {session && ` — session of ${session}`}
        </span>
      </div>

      {/* Day type count tiles */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(5, 1fr)',
          gap: 10,
        }}
      >
        <CountTile label="↑ heavy vol" count={counts.heavy_up} color="var(--up)" />
        <CountTile label="↑ normal vol" count={counts.quiet_up} color="var(--up)" />
        <CountTile label="↓ heavy vol" count={counts.heavy_down} color="var(--down)" />
        <CountTile label="↓ normal vol" count={counts.quiet_down} color="var(--down)" />
        <CountTile label="Unchanged" count={counts.flat} color="var(--text-faint)" />
      </div>

      {/* Dense table */}
      <div
        style={{
          background: 'var(--card)',
          border: '1px solid var(--card-border)',
          borderRadius: 4,
          overflow: 'auto',
        }}
      >
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead style={{ position: 'sticky', top: 0, background: 'var(--card)', zIndex: 1 }}>
            <tr>
              {th('symbol', 'Symbol', 'left')}
              {th('close', 'Close')}
              {th('change', 'Chg %')}
              {th('volumeVsNormal', 'Vol / norm')}
              {th('trades', 'Trades')}
              {th('avgTrade', 'Avg trade')}
              <th style={{ padding: '6px 10px 8px 0', fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--text-faint)', textAlign: 'center', borderBottom: '1px solid var(--rule)', whiteSpace: 'nowrap' }}>
                Thin
              </th>
              {th('dayType', 'Day type', 'left')}
              <th style={{ padding: '6px 10px 8px 0', fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--text-faint)', textAlign: 'center', borderBottom: '1px solid var(--rule)' }}>
                LU
              </th>
              <th style={{ padding: '6px 10px 8px 0', fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--text-faint)', textAlign: 'center', borderBottom: '1px solid var(--rule)' }}>
                LD
              </th>
              {th('turnover', 'Turnover (k)')}
            </tr>
          </thead>
          <tbody>
            {filtered.map((s, i) => (
              <tr
                key={s.symbol}
                style={{
                  background: s.thin
                    ? 'color-mix(in srgb, var(--thin-flag) 5%, transparent)'
                    : i % 2 === 1
                      ? 'color-mix(in srgb, var(--text) 2%, transparent)'
                      : 'transparent',
                  borderBottom: '1px solid var(--rule)',
                }}
              >
                <td
                  style={{
                    padding: '5px 10px 5px 0',
                    fontFamily: "'JetBrains Mono', monospace",
                    fontWeight: 500,
                    fontSize: 11,
                    color: 'var(--text)',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {s.symbol}
                </td>
                <td style={{ padding: '5px 10px 5px 0', textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontVariantNumeric: 'tabular-nums', color: 'var(--text)' }}>
                  {fmt(s.close)}
                </td>
                <td
                  style={{
                    padding: '5px 10px 5px 0',
                    textAlign: 'right',
                    fontFamily: "'JetBrains Mono', monospace",
                    fontVariantNumeric: 'tabular-nums',
                    color:
                      s.change === null
                        ? 'var(--text-faint)'
                        : s.change > 0
                          ? 'var(--up)'
                          : s.change < 0
                            ? 'var(--down)'
                            : 'var(--text-faint)',
                  }}
                >
                  {/* The sign is the secondary encoding: colour alone never
                      carries direction, so this reads correctly in greyscale
                      and to a colourblind reader. */}
                  {s.change !== null && s.change > 0 ? '+' : ''}
                  {fmt(s.change)}
                  {s.change === null ? '' : '%'}
                </td>
                <td style={{ padding: '5px 10px 5px 0', textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontVariantNumeric: 'tabular-nums', color: 'var(--text-muted)' }}>
                  {fmt(s.volumeVsNormal, 1)}×
                </td>
                <td style={{ padding: '5px 10px 5px 0', textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontVariantNumeric: 'tabular-nums', color: 'var(--text-muted)' }}>
                  {s.trades?.toLocaleString() ?? '—'}
                </td>
                <td style={{ padding: '5px 10px 5px 0', textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontVariantNumeric: 'tabular-nums', color: 'var(--text-muted)', fontSize: 11 }}>
                  {s.avgTrade?.toLocaleString() ?? '—'}
                </td>
                <td style={{ padding: '5px 10px 5px 0', textAlign: 'center' }}>
                  {s.thin && (
                    <span
                      title="Thin: fewer than 10 transactions. The day's move may be a single participant, and the volume figure cannot say which side they were on."
                      style={{
                        display: 'inline-block',
                        width: 7,
                        height: 7,
                        borderRadius: '50%',
                        background: 'var(--thin-flag)',
                        opacity: 0.8,
                      }}
                    />
                  )}
                </td>
                <td
                  style={{
                    padding: '5px 10px 5px 0',
                    fontSize: 11,
                    color: DAY_TYPE_COLORS[s.dayType],
                    whiteSpace: 'nowrap',
                  }}
                >
                  {DAY_TYPE_LABELS[s.dayType]}
                </td>
                <td style={{ padding: '5px 10px 5px 0', textAlign: 'center', fontSize: 11 }}>
                  {s.limitUp && <span style={{ color: 'var(--up)', fontWeight: 600 }}>↑</span>}
                </td>
                <td style={{ padding: '5px 10px 5px 0', textAlign: 'center', fontSize: 11 }}>
                  {s.limitDown && <span style={{ color: 'var(--down)', fontWeight: 600 }}>↓</span>}
                </td>
                <td style={{ padding: '5px 0', textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontVariantNumeric: 'tabular-nums', color: 'var(--text-muted)', paddingRight: 16 }}>
                  {s.turnover?.toLocaleString() ?? '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
