import { createContext, useContext, useEffect, useState } from 'react'
import { loadDashboard, type Dashboard } from './dashboard'

const Ctx = createContext<Dashboard | null>(null)

/**
 * Loads the JSON once and hands it to every tab.
 *
 * Loading and failure are rendered as real states rather than a blank page. A
 * dashboard whose data has not arrived should say so: the alternative is a page
 * of zeroes, which for THIS project reads as "no predictions, no stocks, no
 * news" -- indistinguishable from a pipeline that has quietly stopped running.
 */
export function DashboardProvider({ children }: { children: React.ReactNode }) {
  const [data, setData] = useState<Dashboard | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    loadDashboard()
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(String(e.message ?? e)))
    return () => {
      alive = false
    }
  }, [])

  if (error) {
    return (
      <div
        style={{
          maxWidth: 560,
          margin: '80px auto',
          padding: 24,
          border: '1px solid var(--stale)',
          background: 'var(--stale-bg)',
          color: 'var(--stale)',
          borderRadius: 4,
          fontSize: 13,
          lineHeight: 1.6,
        }}
      >
        <strong>The data file did not load.</strong>
        <div style={{ marginTop: 8, fontFamily: "'JetBrains Mono', monospace", fontSize: 11 }}>
          {error}
        </div>
      </div>
    )
  }

  if (!data) {
    return (
      <div
        style={{
          maxWidth: 560,
          margin: '80px auto',
          padding: 24,
          color: 'var(--text-faint)',
          fontSize: 13,
          textAlign: 'center',
        }}
      >
        Loading…
      </div>
    )
  }

  return <Ctx.Provider value={data}>{children}</Ctx.Provider>
}

export function useDashboard(): Dashboard {
  const d = useContext(Ctx)
  if (!d) throw new Error('useDashboard must be used inside <DashboardProvider>')
  return d
}
