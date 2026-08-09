import { useState, useMemo } from 'react'
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import ChartCard from '../components/ChartCard'
import { useDashboard } from '../data/DashboardContext'

type Range = '1y' | '3y' | 'all'

function RangeToggle({ value, onChange }: { value: Range; onChange: (r: Range) => void }) {
  return (
    <div style={{ display: 'flex', gap: 2 }}>
      {(['1y', '3y', 'all'] as Range[]).map((r) => (
        <button
          key={r}
          onClick={() => onChange(r)}
          style={{
            fontSize: 11,
            padding: '3px 9px',
            borderRadius: 3,
            border: '1px solid',
            borderColor: r === value ? 'var(--text-muted)' : 'var(--border)',
            background: r === value ? 'var(--card)' : 'transparent',
            color: r === value ? 'var(--text)' : 'var(--text-faint)',
            cursor: 'pointer',
            fontWeight: r === value ? 600 : 400,
          }}
        >
          {r}
        </button>
      ))}
    </div>
  )
}

const tooltipStyle = {
  background: 'var(--card)',
  border: '1px solid var(--card-border)',
  borderRadius: 4,
  padding: '8px 12px',
  fontSize: 11,
  color: 'var(--text)',
  fontFamily: "'JetBrains Mono', monospace",
  boxShadow: 'none',
}

export default function Market() {
  const [range, setRange] = useState<Range>('1y')
  const { indexData, breadthData, concentrationData } = useDashboard()

  const filteredIndex = useMemo(() => {
    if (range === 'all') return indexData
    const days = range === '1y' ? 252 : 756
    return indexData.slice(-days)
  }, [range])

  const labeledIndex = useMemo(
    () =>
      filteredIndex.map((d, i) => ({
        ...d,
        label: i % Math.ceil(filteredIndex.length / 8) === 0 ? d.date.slice(0, 7) : '',
      })),
    [filteredIndex],
  )

  const labeledBreadth = useMemo(
    () =>
      breadthData.map((d, i) => ({
        ...d,
        label: i % 10 === 0 ? d.date.slice(5) : '',
      })),
    [],
  )

  const labeledConc = useMemo(
    () =>
      concentrationData.map((d, i) => ({
        ...d,
        label: i % Math.ceil(concentrationData.length / 8) === 0 ? d.date.slice(0, 7) : '',
      })),
    [],
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>

      {/* Index chart */}
      <ChartCard
        title="NEPSE composite index"
        caption="Daily closing value of the Nepal Stock Exchange composite index. This is a price-weighted index of all listed shares. No buy/sell implication."
        action={<RangeToggle value={range} onChange={setRange} />}
      >
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={labeledIndex} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 11, fill: 'var(--text-muted)' }}
              axisLine={false}
              tickLine={false}
              interval={0}
            />
            <YAxis
              tick={{ fontSize: 11, fill: 'var(--text-muted)', fontFamily: "'JetBrains Mono', monospace" }}
              axisLine={false}
              tickLine={false}
              width={50}
              tickFormatter={(v) => v.toLocaleString()}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              formatter={(v) => [Number(v).toLocaleString(), 'Index']}
              labelFormatter={(l) => l || ''}
            />
            <Line
              type="monotone"
              dataKey="value"
              stroke="var(--s1)"
              strokeWidth={1.5}
              dot={false}
              activeDot={{ r: 3, fill: 'var(--s1)', strokeWidth: 0 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* Breadth chart */}
      <ChartCard
        title="Market breadth — advancing vs. declining issues"
        caption="Number of listed stocks that rose (above zero) and fell (below zero) each day over the past 60 trading sessions. A positive-breadth day means more stocks advanced than declined. This does not capture magnitude or index weight."
      >
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={labeledBreadth} margin={{ top: 4, right: 8, bottom: 0, left: -10 }} barSize={4}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
              axisLine={false}
              tickLine={false}
              interval={0}
            />
            <YAxis
              tick={{ fontSize: 11, fill: 'var(--text-muted)', fontFamily: "'JetBrains Mono', monospace" }}
              axisLine={false}
              tickLine={false}
              width={36}
            />
            <ReferenceLine y={0} stroke="var(--border)" strokeWidth={1} />
            <Tooltip
              contentStyle={tooltipStyle}
              formatter={(v, name) => [
                Math.abs(Number(v)),
                name === 'rising' ? 'Advancing' : 'Declining',
              ]}
            />
            <Bar dataKey="rising" fill="var(--up)" opacity={0.75} radius={[1, 1, 0, 0]} />
            <Bar dataKey="falling" fill="var(--down)" opacity={0.75} radius={[0, 0, 1, 1]} />
          </BarChart>
        </ResponsiveContainer>
        <div style={{ display: 'flex', gap: 16, marginTop: 10 }}>
          {[
            { color: 'var(--up)', label: 'Advancing' },
            { color: 'var(--down)', label: 'Declining' },
          ].map(({ color, label }) => (
            <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: 'var(--text-muted)' }}>
              <span style={{ display: 'inline-block', width: 10, height: 3, background: color, borderRadius: 1, opacity: 0.75 }} />
              {label}
            </div>
          ))}
        </div>
      </ChartCard>

      {/* Concentration chart */}
      <ChartCard
        title="Volume concentration — top-10 scrips' share of total turnover"
        caption="Percentage of the day's total traded value accounted for by the 10 most-active scrips. High concentration means that most market turnover is driven by a small number of stocks, which limits how representative index moves are of the broader market."
      >
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={labeledConc} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 11, fill: 'var(--text-muted)' }}
              axisLine={false}
              tickLine={false}
              interval={0}
            />
            <YAxis
              tick={{ fontSize: 11, fill: 'var(--text-muted)', fontFamily: "'JetBrains Mono', monospace" }}
              axisLine={false}
              tickLine={false}
              width={40}
              domain={[20, 90]}
              tickFormatter={(v) => `${v}%`}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              formatter={(v) => [`${Number(v)}%`, 'Concentration']}
            />
            <Line
              type="monotone"
              dataKey="pct"
              stroke="var(--s3)"
              strokeWidth={1.5}
              dot={false}
              activeDot={{ r: 3, fill: 'var(--s3)', strokeWidth: 0 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  )
}
