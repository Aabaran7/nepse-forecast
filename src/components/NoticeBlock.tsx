interface NoticeBlockProps {
  variant?: 'neutral' | 'attention'
  title?: string
  children: React.ReactNode
}

export default function NoticeBlock({ variant = 'neutral', title, children }: NoticeBlockProps) {
  const isAttention = variant === 'attention'

  return (
    <div
      style={{
        background: isAttention ? 'var(--stale-bg)' : 'var(--card)',
        borderRadius: '0 4px 4px 0',
        padding: '16px 20px',
        border: `1px solid ${isAttention ? 'var(--stale)' : 'var(--card-border)'}`,
        borderLeft: `3px solid ${isAttention ? 'var(--stale)' : 'var(--text-faint)'}`,
      }}
    >
      {title && (
        <div
          style={{
            fontSize: 11,
            fontWeight: 600,
            letterSpacing: '0.07em',
            textTransform: 'uppercase',
            color: isAttention ? 'var(--stale)' : 'var(--text-muted)',
            marginBottom: 8,
          }}
        >
          {title}
        </div>
      )}
      <div
        style={{
          fontSize: 13,
          lineHeight: 1.65,
          color: 'var(--text-muted)',
          maxWidth: 720,
        }}
      >
        {children}
      </div>
    </div>
  )
}
