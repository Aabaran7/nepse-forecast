import { useState } from 'react'
import SidebarRow from './components/SidebarRow'
import ForwardLog from './tabs/ForwardLog'
import Market from './tabs/Market'
import Stocks from './tabs/Stocks'
import News from './tabs/News'
import { DashboardProvider, useDashboard } from './data/DashboardContext'

type Tab = 'forward-log' | 'market' | 'stocks' | 'news'
type Theme = 'light' | 'dark'

const TABS: { id: Tab; label: string }[] = [
  { id: 'forward-log', label: 'Forward log' },
  { id: 'market', label: 'Market' },
  { id: 'stocks', label: 'Stocks' },
  { id: 'news', label: 'News' },
]

export default function App() {
  // The provider owns loading and failure, so everything below can assume data.
  return (
    <DashboardProvider>
      <Shell />
    </DashboardProvider>
  )
}

function Shell() {
  const [tab, setTab] = useState<Tab>('forward-log')
  const [theme, setTheme] = useState<Theme>('light')
  const { freshness, generatedUtc, gitHash } = useDashboard()

  return (
    <div
      className={theme}
      style={{
        minHeight: '100vh',
        background: 'var(--bg)',
        color: 'var(--text)',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Header */}
      <header
        style={{
          borderBottom: '1px solid var(--border)',
          background: 'var(--card)',
          position: 'sticky',
          top: 0,
          zIndex: 20,
        }}
      >
        <div
          style={{
            maxWidth: 1280,
            margin: '0 auto',
            padding: '0 24px',
            display: 'flex',
            alignItems: 'center',
            gap: 24,
            flexWrap: 'wrap',
          }}
        >
          {/* Wordmark */}
          <div style={{ padding: '14px 0', flexShrink: 0 }}>
            <div
              style={{
                fontFamily: "'IBM Plex Serif', Georgia, serif",
                fontSize: 15,
                fontWeight: 500,
                color: 'var(--text)',
                letterSpacing: '-0.01em',
                lineHeight: 1.2,
              }}
            >
              NEPSE predictability study
            </div>
            <div
              style={{
                fontSize: 10,
                color: 'var(--text-faint)',
                marginTop: 2,
                letterSpacing: '0.02em',
              }}
            >
              Descriptive research — not financial advice
            </div>
          </div>

          {/* Tabs */}
          <nav
            style={{
              display: 'flex',
              gap: 0,
              marginLeft: 16,
              overflowX: 'auto',
              flexShrink: 0,
            }}
          >
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                style={{
                  padding: '18px 16px 16px',
                  fontSize: 12,
                  fontWeight: tab === t.id ? 600 : 400,
                  color: tab === t.id ? 'var(--text)' : 'var(--text-muted)',
                  background: 'transparent',
                  border: 'none',
                  borderBottom: tab === t.id ? '2px solid var(--text)' : '2px solid transparent',
                  cursor: 'pointer',
                  letterSpacing: '0.01em',
                  whiteSpace: 'nowrap',
                  transition: 'color 0.15s',
                }}
              >
                {t.label}
              </button>
            ))}
          </nav>

          {/* Theme toggle */}
          <button
            onClick={() => setTheme((t) => (t === 'light' ? 'dark' : 'light'))}
            style={{
              marginLeft: 'auto',
              fontSize: 11,
              padding: '5px 10px',
              borderRadius: 3,
              border: '1px solid var(--border)',
              background: 'transparent',
              color: 'var(--text-faint)',
              cursor: 'pointer',
              flexShrink: 0,
            }}
          >
            {theme === 'light' ? 'Dark' : 'Light'}
          </button>
        </div>
      </header>

      {/* Body */}
      <div
        style={{
          flex: 1,
          maxWidth: 1280,
          width: '100%',
          margin: '0 auto',
          display: 'grid',
          gridTemplateColumns: '180px 1fr',
          gap: 0,
          alignItems: 'start',
        }}
      >
        {/* Sidebar */}
        <aside
          style={{
            borderRight: '1px solid var(--border)',
            padding: '24px 20px 32px 24px',
            minHeight: 'calc(100vh - 58px)',
            position: 'sticky',
            top: 58,
            alignSelf: 'start',
          }}
        >
          <div
            style={{
              fontSize: 10,
              fontWeight: 600,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: 'var(--text-faint)',
              marginBottom: 14,
            }}
          >
            Data freshness
          </div>
          {freshness.map((item) => (
            <SidebarRow key={item.label} item={item} />
          ))}
          <div style={{ marginTop: 20, fontSize: 10, color: 'var(--text-faint)', lineHeight: 1.6 }}>
            A stale source means data has not updated since that date. The
            exchange serves only a rolling year, so a session that is never
            archived is lost permanently. Deep history is refreshed by hand and
            is expected to lag.
          </div>
          <div
            style={{
              marginTop: 16,
              paddingTop: 12,
              borderTop: '1px solid var(--rule)',
              fontSize: 10,
              color: 'var(--text-faint)',
              fontFamily: "'JetBrains Mono', monospace",
              lineHeight: 1.7,
            }}
          >
            {/* Which code produced these numbers. The forward log records a git
                hash per prediction (§7); the page should be as traceable. */}
            <div>built {generatedUtc.slice(0, 16).replace('T', ' ')}Z</div>
            <div>{gitHash}</div>
          </div>
        </aside>

        {/* Main */}
        <main
          style={{
            padding: '28px 32px 48px',
            minWidth: 0,
          }}
        >
          {/* Tab heading */}
          <div style={{ marginBottom: 24 }}>
            <h1
              style={{
                fontFamily: "'IBM Plex Serif', Georgia, serif",
                fontSize: 20,
                fontWeight: 500,
                color: 'var(--text)',
                margin: 0,
                letterSpacing: '-0.01em',
              }}
            >
              {TABS.find((t) => t.id === tab)?.label}
            </h1>
            {tab === 'forward-log' && (
              <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: '5px 0 0', lineHeight: 1.5 }}>
                Every prediction made in this study, plus a comparison of model accuracy against a naive baseline.
              </p>
            )}
            {tab === 'market' && (
              <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: '5px 0 0', lineHeight: 1.5 }}>
                Historical context for the NEPSE composite index, breadth, and volume concentration.
              </p>
            )}
            {tab === 'stocks' && (
              <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: '5px 0 0', lineHeight: 1.5 }}>
                Screener for the most recent trading session. Descriptive only.
              </p>
            )}
            {tab === 'news' && (
              <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: '5px 0 0', lineHeight: 1.5 }}>
                Scraped headlines from Nepali financial news sources, pending sentiment scoring.
              </p>
            )}
          </div>

          {tab === 'forward-log' && <ForwardLog />}
          {tab === 'market' && <Market />}
          {tab === 'stocks' && <Stocks />}
          {tab === 'news' && <News />}
        </main>
      </div>

      {/* Narrow layout: sidebar below header on mobile */}
      <style>{`
        @media (max-width: 768px) {
          div[style*="grid-template-columns: 180px"] {
            grid-template-columns: 1fr !important;
          }
          aside {
            min-height: unset !important;
            position: static !important;
            border-right: none !important;
            border-bottom: 1px solid var(--border) !important;
            padding: 16px 20px !important;
          }
          aside > div[style*="min-height"] {
            display: none;
          }
          main {
            padding: 20px 16px 40px !important;
          }
          div[style*="grid-template-columns: repeat(5, 1fr)"] {
            grid-template-columns: repeat(3, 1fr) !important;
          }
        }
      `}</style>
    </div>
  )
}
