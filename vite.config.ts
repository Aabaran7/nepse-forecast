import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// GitHub Pages serves a project site from /<repo>/, not from /. Without the
// matching base, every asset 404s on Pages while working perfectly in local
// dev -- which is the confusing way round to discover it.
//
// Overridable so the same build can be served from a domain root:
//   BASE_PATH=/ npm run build
const base = process.env.BASE_PATH ?? '/nepse-forecast/'

export default defineConfig({
  base,
  plugins: [react()],
  build: {
    outDir: 'dist',
    // The data file is written by scripts/export_dashboard.py into public/data/
    // and copied verbatim. It is deliberately NOT imported as a module: bundling
    // it would mean a rebuild for every daily data refresh, when the whole point
    // is that the page fetches fresh numbers without being rebuilt.
    assetsDir: 'assets',
    sourcemap: false,
  },
})
