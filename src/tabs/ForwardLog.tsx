import StatTile from '../components/StatTile'
import NoticeBlock from '../components/NoticeBlock'
import ChartCard from '../components/ChartCard'
import ComparisonBulletChart from '../components/ComparisonBulletChart'
import { useDashboard } from '../data/DashboardContext'

export default function ForwardLog() {
  const { forwardLog } = useDashboard()
  const s = forwardLog
  const rows = s.predictions
  const bulletRows = s.horizons
  // The design shipped a three-way "state demo" switcher so light/empty/full
  // states could be reviewed. It is gone: on a live page a control that fakes
  // the numbers is indistinguishable from one that filters them, and this is
  // the one page where a reader must be able to trust what is on screen.
  const isEmpty = s.resolved === 0 && s.totalPredictions === 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>

      {/* Stat tiles */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
          gap: 12,
        }}
      >
        <StatTile
          label="Directional predictions"
          value={s.totalPredictions}
          helpText="Total number of predictions logged, resolved or not"
        />
        <StatTile
          label="Resolved"
          value={s.resolved}
          helpText="Predictions where the target date has passed and the outcome is known"
        />
        <StatTile
          label="Still open"
          value={s.open}
          sub="awaiting resolution"
          helpText="Predictions not yet past their target date"
        />
        <StatTile
          label="Exposure rows"
          value={s.exposureRows}
          sub="not forecasts"
          helpText="Volatility-target rows: they record how much exposure the rule called for that day. They make no directional claim, so they are excluded from accuracy."
        />
      </div>

      {/* Scoreboard */}
      <ChartCard
        title="Directional accuracy vs. baseline"
        caption="Model accuracy is compared to a naive baseline that always guesses the more common historical direction. The baseline is the floor, not a benchmark — outperforming it would require consistency across at least 30 resolved predictions before the pattern can be distinguished from noise."
      >
        {isEmpty ? (
          <div
            style={{
              padding: '32px 0',
              textAlign: 'center',
              color: 'var(--text-faint)',
              fontSize: 13,
            }}
          >
            <div style={{ fontFamily: "'IBM Plex Serif', Georgia, serif", fontSize: 16, marginBottom: 8, color: 'var(--text-muted)' }}>
              No predictions resolved yet.
            </div>
            <div style={{ maxWidth: 400, margin: '0 auto', lineHeight: 1.6 }}>
              This chart will appear once at least one prediction has a known outcome. There is nothing to show, and that is correct — not an error.
            </div>
          </div>
        ) : (
          <div>
            {s.horizons.some((h) => h.n > 0 && h.n < h.minN) && (
              <div
                style={{
                  background: 'var(--stale-bg)',
                  border: '1px solid var(--stale)',
                  borderRadius: 4,
                  padding: '10px 14px',
                  marginBottom: 20,
                  fontSize: 12,
                  color: 'var(--stale)',
                  lineHeight: 1.55,
                }}
              >
                <strong>Sample size too small to interpret.</strong> The figures below are shown for completeness, not as findings. A minimum of ~30 resolved predictions per horizon is needed before accuracy diverges meaningfully from chance.
              </div>
            )}
            <ComparisonBulletChart rows={bulletRows} />
          </div>
        )}
      </ChartCard>

      {/* Numbers table */}
      {!isEmpty && (
        <ChartCard
          title="Accuracy by horizon — tabular"
          caption="Same data as above. Percentages are directional accuracy only."
        >
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--rule)' }}>
                {['Horizon', 'n (resolved)', 'Model accuracy', 'Baseline', 'Difference'].map((h) => (
                  <th
                    key={h}
                    style={{
                      textAlign: 'left',
                      padding: '6px 12px 8px 0',
                      fontSize: 10,
                      fontWeight: 600,
                      letterSpacing: '0.06em',
                      textTransform: 'uppercase',
                      color: 'var(--text-faint)',
                    }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {bulletRows.map((row) => {
                const diff =
                  row.modelPct !== null && row.n > 0
                    ? (row.modelPct - row.baselinePct).toFixed(1)
                    : null
                const insufficient = row.n < row.minN
                return (
                  <tr
                    key={row.horizon}
                    style={{ borderBottom: '1px solid var(--rule)' }}
                  >
                    <td style={{ padding: '8px 12px 8px 0', fontFamily: "'IBM Plex Serif', Georgia, serif" }}>
                      {row.horizon}
                    </td>
                    <td
                      style={{
                        padding: '8px 12px 8px 0',
                        fontFamily: "'JetBrains Mono', monospace",
                        fontVariantNumeric: 'tabular-nums',
                        color: insufficient ? 'var(--stale)' : 'var(--text)',
                      }}
                    >
                      {row.n}{insufficient && <span style={{ fontSize: 10, marginLeft: 4 }}>({row.minN - row.n} more needed)</span>}
                    </td>
                    <td
                      style={{
                        padding: '8px 12px 8px 0',
                        fontFamily: "'JetBrains Mono', monospace",
                        fontVariantNumeric: 'tabular-nums',
                        color: insufficient ? 'var(--text-faint)' : 'var(--text)',
                        opacity: insufficient ? 0.55 : 1,
                      }}
                    >
                      {row.modelPct !== null && row.n > 0 ? `${row.modelPct}%` : '—'}
                    </td>
                    <td
                      style={{
                        padding: '8px 12px 8px 0',
                        fontFamily: "'JetBrains Mono', monospace",
                        fontVariantNumeric: 'tabular-nums',
                        color: 'var(--text-muted)',
                      }}
                    >
                      {row.baselinePct}%
                    </td>
                    <td
                      style={{
                        padding: '8px 0',
                        fontFamily: "'JetBrains Mono', monospace",
                        fontVariantNumeric: 'tabular-nums',
                        color: insufficient ? 'var(--text-faint)' : 'var(--text-muted)',
                        opacity: insufficient ? 0.55 : 1,
                      }}
                    >
                      {diff !== null ? `${Number(diff) >= 0 ? '+' : ''}${diff} pp` : '—'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </ChartCard>
      )}

      {/* Prediction log */}
      {rows.length > 0 && (
        <ChartCard
          title="Prediction log"
          caption="Every prediction ever recorded, newest first. Entries are append-only: a logged prediction is never edited or removed, because the revision that matters is the one made after the outcome is known — exactly when it looks most reasonable. Each row carries the model version and the commit that produced it."
        >
          <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--rule)' }}>
                {['Made', 'Horizon', 'Call', 'P(up)', 'Outcome', 'Model version', 'Commit'].map((h) => (
                  <th
                    key={h}
                    style={{
                      textAlign: 'left',
                      padding: '6px 12px 8px 0',
                      fontSize: 10,
                      fontWeight: 600,
                      letterSpacing: '0.06em',
                      textTransform: 'uppercase',
                      color: 'var(--text-faint)',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((p, i) => {
                const mono = {
                  padding: '8px 12px 8px 0',
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 11,
                } as const
                return (
                  // No id in the log, so the key is what actually identifies a
                  // row: the date, the horizon and the model that made it.
                  <tr
                    key={`${p.asOf}-${p.horizon}-${p.modelVersion}-${i}`}
                    style={{ borderBottom: '1px solid var(--rule)' }}
                  >
                    <td style={{ ...mono, whiteSpace: 'nowrap' }}>{p.asOf}</td>
                    <td style={mono}>{p.horizon}</td>
                    <td style={{ padding: '8px 12px 8px 0', fontSize: 11, whiteSpace: 'nowrap' }}>
                      {p.kind === 'exposure' ? (
                        // Not a forecast. Labelled as what it is so it can never
                        // be read as a directional call that happened to be wrong.
                        <span style={{ color: 'var(--text-muted)' }}>
                          {p.exposure === null ? 'exposure' : `${Math.round(p.exposure * 100)}% exposure`}
                        </span>
                      ) : (
                        <span style={{ color: p.direction === 'up' ? 'var(--up)' : 'var(--down)' }}>
                          {p.direction === 'up' ? '↑ up' : '↓ down'}
                        </span>
                      )}
                    </td>
                    <td style={{ ...mono, color: 'var(--text-muted)' }}>
                      {p.probUp === null ? '—' : p.probUp.toFixed(3)}
                    </td>
                    <td style={{ padding: '8px 12px 8px 0', fontSize: 11, whiteSpace: 'nowrap' }}>
                      {p.kind === 'exposure' ? (
                        <span style={{ color: 'var(--text-faint)' }}>n/a</span>
                      ) : p.outcome === 'correct' ? (
                        <span style={{ color: 'var(--up)' }}>✓ Correct</span>
                      ) : p.outcome === 'incorrect' ? (
                        <span style={{ color: 'var(--down)' }}>✗ Incorrect</span>
                      ) : (
                        <span style={{ color: 'var(--text-faint)' }}>Open</span>
                      )}
                    </td>
                    <td style={{ ...mono, color: 'var(--text-muted)' }}>{p.modelVersion}</td>
                    <td style={{ ...mono, color: 'var(--text-faint)', paddingRight: 0 }}>
                      {p.gitHash}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          </div>
        </ChartCard>
      )}

      {/* Notice block */}
      <NoticeBlock title="What this log can and cannot establish">
        <p style={{ margin: 0 }}>
          This log records directional predictions — up or down — made before a trading session closes, using signals derived entirely from publicly available NEPSE data. A prediction is "resolved" once the target date passes and the outcome is observable in the exchange archive.
        </p>
        <p style={{ margin: '10px 0 0' }}>
          Accuracy here is directional only. The model guesses the sign of the next move, not its magnitude. It says nothing about expected returns, transaction costs, bid–ask spread, market impact, or whether any edge would survive real execution. Directional accuracy above 50% would be a necessary but not sufficient condition for a useful signal.
        </p>
        <p style={{ margin: '10px 0 0' }}>
          The current study has found no reliable edge. The sample is too small to claim either success or failure with any confidence. This page will continue to accumulate predictions; the conclusion remains open.
        </p>
      </NoticeBlock>
    </div>
  )
}
