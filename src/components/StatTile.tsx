interface StatTileProps {
  label: string
  value: string | number
  suffix?: string
  sub?: string
  muted?: boolean
  helpText?: string
}

export default function StatTile({ label, value, suffix, sub, muted, helpText }: StatTileProps) {
  return (
    <div
      style={{
        background: 'var(--card)',
        border: '1px solid var(--card-border)',
        borderRadius: 4,
        padding: '18px 20px 16px',
      }}
    >
      <div
        style={{
          fontSize: 10,
          fontWeight: 600,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          color: 'var(--text-muted)',
          marginBottom: 10,
          display: 'flex',
          alignItems: 'center',
          gap: 6,
        }}
      >
        {label}
        {helpText && (
          <span
            title={helpText}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: 14,
              height: 14,
              borderRadius: '50%',
              border: '1px solid var(--border)',
              fontSize: 9,
              color: 'var(--text-faint)',
              cursor: 'help',
              fontWeight: 500,
              letterSpacing: 0,
              textTransform: 'none',
            }}
          >
            ?
          </span>
        )}
      </div>
      <div
        style={{
          fontFamily: "'IBM Plex Serif', Georgia, serif",
          fontSize: 28,
          fontWeight: 400,
          lineHeight: 1.1,
          color: muted ? 'var(--text-faint)' : 'var(--text)',
          fontVariantNumeric: 'tabular-nums',
        }}
      >
        {value}
        {suffix && (
          <span style={{ fontSize: 14, fontWeight: 400, marginLeft: 3, color: 'var(--text-muted)' }}>
            {suffix}
          </span>
        )}
      </div>
      {sub && (
        <div style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 5 }}>{sub}</div>
      )}
    </div>
  )
}
