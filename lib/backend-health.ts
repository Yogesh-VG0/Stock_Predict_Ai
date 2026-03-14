/**
 * Backend Health / Cold-Start Wake-Up Service
 *
 * Koyeb free-tier instances scale to zero after ~65 min of inactivity.
 * When a user visits the site after a sleep period the backend needs
 * 15-60 s to cold-start.  This module:
 *
 *  1. Pings a known-working API endpoint with exponential back-off until
 *     the backend responds with HTTP 200.
 *  2. Exposes a promise (`backendReady`) that other modules can await
 *     before firing real API calls.
 *  3. Provides a React context so the UI can show a "waking up" banner.
 *
 *  NOTE: We probe `/api/market/status` instead of `/health` because the
 *  `/health` rewrite returns 404 on Vercel (root-level rewrites don't
 *  reliably proxy to external backends).  `/api/market/:path*` works.
 */

// ── Singleton state (shared across the whole client bundle) ────────────

let _status: 'idle' | 'waking' | 'ready' | 'error' = 'idle'
let _resolveReady: (() => void) | null = null
let _readyPromise: Promise<void> | null = null
let _listeners: Array<(s: typeof _status) => void> = []

const MAX_RETRIES = 12          // ~60-90 s total depending on Vercel proxy timing
const INITIAL_DELAY_MS = 3000   // first retry after 3 s
const MAX_DELAY_MS = 6000       // cap back-off at 6 s
const HEALTH_TIMEOUT_MS = 15000 // 15 s — long enough for Vercel to proxy to waking Koyeb

function getProbeUrl(): string {
  if (typeof window === 'undefined') return ''
  // Localhost: use direct /health (fast, no proxy)
  // Production: use /api/market/status (goes through known-working Vercel rewrite)
  return window.location.hostname === 'localhost'
    ? 'http://localhost:5000/health'
    : '/api/market/status'
}

function notify() {
  _listeners.forEach(fn => fn(_status))
}

export function getBackendStatus() {
  return _status
}

export function onStatusChange(fn: (s: typeof _status) => void) {
  _listeners.push(fn)
  return () => {
    _listeners = _listeners.filter(l => l !== fn)
  }
}

/**
 * Returns a promise that resolves once the backend is confirmed healthy.
 * Safe to call multiple times — only one wake-up loop runs.
 */
export function backendReady(): Promise<void> {
  if (_status === 'ready') return Promise.resolve()
  if (_readyPromise) return _readyPromise
  _readyPromise = new Promise<void>(resolve => {
    _resolveReady = resolve
  })
  startWakeUp()
  return _readyPromise
}

async function pingHealth(): Promise<boolean> {
  const url = getProbeUrl()
  if (!url) return false
  try {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS)
    const res = await fetch(url, {
      cache: 'no-store',
      signal: controller.signal,
    })
    clearTimeout(timer)
    // ONLY 200 means the backend is truly alive and responding.
    // 404 = Vercel can't find route (rewrite broken), 502/504 = backend sleeping.
    return res.ok
  } catch {
    // Network error / abort = backend unreachable
    return false
  }
}

async function startWakeUp() {
  if (_status === 'ready' || _status === 'waking') return
  _status = 'waking'
  notify()

  let delay = INITIAL_DELAY_MS

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    const ok = await pingHealth()
    if (ok) {
      _status = 'ready'
      notify()
      _resolveReady?.()
      console.log(`✅ Backend ready after ${attempt} health check(s)`)
      return
    }
    console.log(`⏳ Backend waking up… attempt ${attempt}/${MAX_RETRIES}`)
    await sleep(delay)
    delay = Math.min(delay * 1.4, MAX_DELAY_MS)
  }

  // Exhausted retries — mark error but still resolve so the app doesn't hang
  _status = 'error'
  notify()
  _resolveReady?.()
  console.warn('⚠️ Backend did not respond after maximum retries')
}

function sleep(ms: number) {
  return new Promise(r => setTimeout(r, ms))
}

// ── fetchWithRetry: drop-in replacement for fetch that retries on cold start ──

const FETCH_RETRIES = 3
const FETCH_RETRY_DELAY = 3000

/**
 * Fetch wrapper that automatically retries when the backend is waking up.
 * If the backend is not yet ready it waits for the health-check loop first.
 */
export async function fetchWithRetry(
  url: string,
  options?: RequestInit,
  retries: number = FETCH_RETRIES,
): Promise<Response> {
  // Wait for backend to be alive before even trying
  await backendReady()

  let lastError: Error | null = null
  for (let i = 0; i <= retries; i++) {
    try {
      const res = await fetch(url, options)
      return res
    } catch (err) {
      lastError = err as Error
      if (i < retries) {
        console.log(`🔄 Retrying ${url} (${i + 1}/${retries})…`)
        await sleep(FETCH_RETRY_DELAY)
      }
    }
  }
  throw lastError ?? new Error(`fetchWithRetry failed for ${url}`)
}
