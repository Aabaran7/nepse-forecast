interface HorizonRow {
  horizon: string
  modelPct: number | null
  baselinePct: number
  n: number
  minN: number
}

interface ComparisonBulletChartProps {
  rows: HorizonRow[]
  insufficient?: boolean
}

function BulletBar({ modelPct, baselinePct, n, minN, insufficient }: {
  modelPct: number | null
  baselinePct: number
  n: number
  minN: number
  insufficient: boolean
}) {
  const hasModel = modelPct !== null && n > 0
  const maxVal = 100
  const baseW = (baselinePct / maxVal) * 100
  const modelW = hasModel ? (modelPct! / maxVal) * 100 : 0
  const progressW = (n / minN) * 100

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {/* Baseline row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <div
          style={{
            fontSize: 10,
            fontWeight: 600,
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
            color: 'var(--text-faint)',
            width: 60,
            flexShrink: 0,
            textAlign: 'right',
          }}
        >
          Baseline
        </div>
        <div
          style={{
            flex: 1,
            height: 10,
            background: 'var(--rule)',
            borderRadius: 2,
            overflow: 'hidden',
            position: 'relative',
          }}
        >
          <div
            style={{
              position: 'absolute',
              left: 0,
              top: 0,
              height: '100%',
              width: `${baseW}%`,
              background: 'var(--baseline)',
              borderRadius: 2,
            }}
          />
        </div>
        <div
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 11,
            color: 'var(--text-muted)',
            width: 40,
            flexShrink: 0,
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          {baselinePct}%
        </div>
      </div>

      {/* Model row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <div
          style={{
            fontSize: 10,
            fontWeight: 600,
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
            color: insufficient ? 'var(--text-faint)' : 'var(--text-muted)',
            width: 60,
            flexShrink: 0,
            textAlign: 'right',
          }}
        >
          Model
        </div>
        <div
          style={{
            flex: 1,
            height: 10,
            background: 'var(--rule)',
            borderRadius: 2,
            overflow: 'hidden',
            position: 'relative',
          }}
        >
          {insufficient && hasModel && (
            /* Hatched / low-opacity bar for insufficient data */
            <div
              style={{
                position: 'absolute',
                left: 0,
                top: 0,
                height: '100%',
                width: `${modelW}%`,
                backgroundImage: `repeating-linear-gradient(
                  45deg,
                  var(--text-faint) 0px,
                  var(--text-faint) 2px,
                  transparent 2px,
                  transparent 6px
                )`,
                opacity: 0.5,
                borderRadius: 2,
              }}
            />
          )}
          {!insufficient && hasModel && (
            <div
              style={{
                position: 'absolute',
                left: 0,
                top: 0,
                height: '100%',
                width: `${modelW}%`,
                background: 'var(--s1)',
                borderRadius: 2,
              }}
            />
          )}
        </div>
        <div
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 11,
            color: insufficient ? 'var(--text-faint)' : 'var(--text)',
            width: 40,
            flexShrink: 0,
            fontVariantNumeric: 'tabular-nums',
            opacity: insufficient ? 0.6 : 1,
          }}
        >
          {hasModel ? `${modelPct}%` : '—'}
        </div>
      </div>

      {/* Progress toward minimum sample */}
      {insufficient && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 4 }}>
          <div style={{ width: 60, flexShrink: 0 }} />
          <div style={{ flex: 1 }}>
            <div
              style={{
                height: 4,
                background: 'var(--rule)',
                borderRadius: 2,
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  height: '100%',
                  width: `${Math.min(100, progressW)}%`,
                  background: 'var(--text-faint)',
                  borderRadius: 2,
                  transition: 'width 0.3s',
                }}
              />
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-faint)', marginTop: 4 }}>
              {n} of {minN} resolved — {minN - n} more before this figure is readable
            </div>
          </div>
          <div style={{ width: 40, flexShrink: 0 }} />
        </div>
      )}
    </div>
  )
}

export default function ComparisonBulletChart({ rows, insufficient }: ComparisonBulletChartProps) {
  // `insufficient` is decided per row below (a horizon can have enough resolved
  // predictions while its neighbour does not), so the chart-wide flag is only a
  // manual override for the demo states.
  void insufficient

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {rows.map((row) => {
        const rowInsufficient = row.n < row.minN
        return (
          <div key={row.horizon}>
            <div
              style={{
                display: 'flex',
                alignItems: 'baseline',
                gap: 8,
                marginBottom: 12,
              }}
            >
              <span
                style={{
                  fontFamily: "'IBM Plex Serif', Georgia, serif",
                  fontSize: 14,
                  fontWeight: 500,
                  color: 'var(--text)',
                }}
              >
                {row.horizon}
              </span>
              <span
                style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 11,
                  color: rowInsufficient ? 'var(--stale)' : 'var(--text-faint)',
                  fontVariantNumeric: 'tabular-nums',
                }}
              >
                n={row.n}
              </span>
              {rowInsufficient && (
                <span
                  style={{
                    fontSize: 10,
                    color: 'var(--stale)',
                    fontWeight: 500,
                  }}
                >
                  — too few to interpret
                </span>
              )}
            </div>
            {row.n === 0 ? (
              <div style={{ fontSize: 12, color: 'var(--text-faint)', paddingLeft: 70 }}>
                No predictions resolved at this horizon yet.
              </div>
            ) : (
              <BulletBar
                modelPct={row.modelPct}
                baselinePct={row.baselinePct}
                n={row.n}
                minN={row.minN}
                insufficient={rowInsufficient}
              />
            )}
          </div>
        )
      })}
    </div>
  )
}
