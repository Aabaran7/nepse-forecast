import NoticeBlock from '../components/NoticeBlock'
import { useDashboard } from '../data/DashboardContext'

function formatDate(iso: string | null) {
  // Only ShareSansar publishes a date on its listing page; the other two give
  // nothing. Null is the honest answer and must not be rendered as a date.
  if (!iso) return 'no date'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return 'no date'
  return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })
}

// Detect Devanagari characters
function hasDevanagari(text: string) {
  return /[ऀ-ॿ]/.test(text)
}

export default function News() {
  const { news: newsData } = useDashboard()
  const noSession = newsData.filter((n) => n.session === null)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

      <NoticeBlock title="Session assignment">
        <p style={{ margin: 0 }}>
          Each headline is assigned to the nearest trading session that it plausibly preceded. Headlines with no assigned session were published outside any clear session window — on a public holiday, late at night, or too far from open — and could not be reliably attached to a single trading day. This is correct, not an error. There are currently {noSession.length} headlines in this state.
        </p>
        <p style={{ margin: '8px 0 0' }}>
          Sentiment scoring is not yet implemented. The "Scored" column will populate once a scoring pass is added. Until then, all headlines are marked as pending.
        </p>
      </NoticeBlock>

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
            <tr style={{ borderBottom: '1px solid var(--rule)' }}>
              {[
                { label: 'Session', align: 'left' as const },
                { label: 'Source', align: 'left' as const },
                { label: 'Headline', align: 'left' as const },
                { label: 'Published', align: 'left' as const },
                { label: 'Scored', align: 'center' as const },
                { label: 'Link', align: 'center' as const },
              ].map(({ label, align }) => (
                <th
                  key={label}
                  style={{
                    padding: '8px 12px 10px 0',
                    fontSize: 10,
                    fontWeight: 600,
                    letterSpacing: '0.07em',
                    textTransform: 'uppercase',
                    color: 'var(--text-faint)',
                    textAlign: align,
                    whiteSpace: 'nowrap',
                  }}
                >
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {newsData.map((item, i) => {
              const isDev = hasDevanagari(item.headline)
              return (
                <tr
                  key={i}
                  style={{
                    borderBottom: '1px solid var(--rule)',
                    background: item.session === null
                      ? 'color-mix(in srgb, var(--text) 1.5%, transparent)'
                      : 'transparent',
                  }}
                >
                  <td
                    style={{
                      padding: '9px 12px 9px 0',
                      fontFamily: "'JetBrains Mono', monospace",
                      fontSize: 11,
                      color: item.session ? 'var(--text-muted)' : 'var(--text-faint)',
                      whiteSpace: 'nowrap',
                      fontVariantNumeric: 'tabular-nums',
                    }}
                  >
                    {item.session ?? (
                      <span style={{ fontFamily: 'Inter, sans-serif', fontStyle: 'italic', fontSize: 11 }}>
                        no session
                      </span>
                    )}
                  </td>
                  <td
                    style={{
                      padding: '9px 12px 9px 0',
                      color: 'var(--text-muted)',
                      whiteSpace: 'nowrap',
                      fontSize: 11,
                    }}
                  >
                    {item.source}
                  </td>
                  <td
                    style={{
                      padding: '9px 12px 9px 0',
                      color: 'var(--text)',
                      lineHeight: isDev ? 1.8 : 1.5,
                      fontSize: isDev ? 13 : 12,
                      minWidth: 260,
                      maxWidth: 500,
                    }}
                  >
                    {item.headline}
                  </td>
                  <td
                    style={{
                      padding: '9px 12px 9px 0',
                      whiteSpace: 'nowrap',
                      color: 'var(--text-faint)',
                    }}
                  >
                    <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, fontVariantNumeric: 'tabular-nums' }}>
                      {formatDate(item.publishedAt)}
                    </div>
                    {/* No time is shown: none of the sources publish one on
                        their listing page, and a rendered 00:00 would look like
                        a midnight publication rather than a missing field. */}
                  </td>
                  <td style={{ padding: '9px 12px 9px 0', textAlign: 'center' }}>
                    {item.scored ? (
                      <span style={{ fontSize: 11, color: 'var(--up)' }}>✓</span>
                    ) : (
                      <span
                        style={{
                          fontSize: 10,
                          color: 'var(--text-faint)',
                          fontStyle: 'italic',
                        }}
                      >
                        pending
                      </span>
                    )}
                  </td>
                  <td style={{ padding: '9px 16px 9px 0', textAlign: 'center' }}>
                    <a
                      href={item.url}
                      style={{
                        fontSize: 11,
                        color: 'var(--s1)',
                        textDecoration: 'none',
                        borderBottom: '1px solid color-mix(in srgb, var(--s1) 40%, transparent)',
                      }}
                    >
                      ↗
                    </a>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
