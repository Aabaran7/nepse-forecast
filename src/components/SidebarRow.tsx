import type { FreshnessItem } from '../data/dashboard'

interface SidebarRowProps {
  item: FreshnessItem
}

function formatDaysAgo(n: number | null): string {
  if (n === null) return 'never collected'
  if (n === 0) return 'today'
  if (n === 1) return '1 day ago'
  if (n < 30) return `${n} days ago`
  const months = Math.round(n / 30)
  return `${months} mo ago`
}

export default function SidebarRow({ item }: SidebarRowProps) {
  return (
    <div
      style={{
        padding: '10px 0',
        borderBottom: '1px solid var(--rule)',
      }}
    >
      <div
        style={{
          fontSize: 11,
          color: item.stale ? 'var(--stale)' : 'var(--text-muted)',
          fontWeight: item.stale ? 600 : 400,
          marginBottom: 3,
          display: 'flex',
          alignItems: 'center',
          gap: 5,
        }}
      >
        {item.stale && (
          <span
            style={{
              display: 'inline-block',
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: 'var(--stale)',
              flexShrink: 0,
            }}
          />
        )}
        {item.label}
      </div>
      <div
        style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 11,
          color: item.stale ? 'var(--stale)' : 'var(--text-faint)',
          lineHeight: 1.4,
        }}
      >
        {item.date ?? '—'}
      </div>
      <div
        style={{
          fontSize: 10,
          color: item.stale ? 'var(--stale)' : 'var(--text-faint)',
          marginTop: 1,
        }}
      >
        {formatDaysAgo(item.daysAgo)}
        {/* Not "the gap is permanent" -- being behind is not the same as having
            lost anything. It becomes permanent only once the missed sessions
            age out of the exchange's rolling year, which is exactly why a stale
            flag is worth acting on now rather than reporting after the fact. */}
        {item.stale && ' — at risk of permanent loss'}
      </div>
    </div>
  )
}
