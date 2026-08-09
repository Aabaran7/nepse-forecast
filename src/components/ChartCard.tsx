interface ChartCardProps {
  title: string
  caption: string
  children: React.ReactNode
  action?: React.ReactNode
}

export default function ChartCard({ title, caption, children, action }: ChartCardProps) {
  return (
    <div
      style={{
        background: 'var(--card)',
        border: '1px solid var(--card-border)',
        borderRadius: 4,
        padding: '20px 24px 20px',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          marginBottom: 4,
          gap: 12,
        }}
      >
        <div
          style={{
            fontSize: 12,
            fontWeight: 600,
            color: 'var(--text)',
            letterSpacing: '0.01em',
          }}
        >
          {title}
        </div>
        {action && <div style={{ flexShrink: 0 }}>{action}</div>}
      </div>
      <div
        style={{
          fontSize: 12,
          color: 'var(--text-muted)',
          lineHeight: 1.5,
          marginBottom: 20,
          maxWidth: 640,
        }}
      >
        {caption}
      </div>
      {children}
    </div>
  )
}
