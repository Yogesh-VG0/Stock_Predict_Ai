'use client'

import { useEffect, useState } from 'react'
import { getBackendStatus, onStatusChange, backendReady } from '@/lib/backend-health'
import { Loader2, ServerCrash, Wifi } from 'lucide-react'

/**
 * A thin banner that appears at the top of the page while the Koyeb
 * backend is cold-starting.  Automatically disappears once `/health`
 * responds 200.
 */
export default function BackendWakingBanner() {
  const [status, setStatus] = useState<'idle' | 'waking' | 'ready' | 'error'>(
    () => getBackendStatus()
  )

  useEffect(() => {
    // Kick off the wake-up loop on mount (no-op if already running)
    backendReady()
    // Subscribe to status changes
    const unsub = onStatusChange(setStatus)
    return unsub
  }, [])

  if (status === 'ready' || status === 'idle') return null

  if (status === 'error') {
    return (
      <div className="w-full bg-red-950/80 border-b border-red-500/30 px-4 py-2 flex items-center justify-center gap-2 text-xs text-red-300 z-50">
        <ServerCrash className="h-3.5 w-3.5 shrink-0" />
        <span>Backend server is unreachable. Data may be unavailable — try refreshing in a minute.</span>
      </div>
    )
  }

  // status === 'waking'
  return (
    <div className="w-full bg-amber-950/70 border-b border-amber-500/30 px-4 py-2 flex items-center justify-center gap-2 text-xs text-amber-200 z-50 animate-in fade-in slide-in-from-top-1 duration-300">
      <Loader2 className="h-3.5 w-3.5 animate-spin shrink-0" />
      <span>
        Waking up the server — this may take up to 30 seconds on first visit…
      </span>
      <Wifi className="h-3.5 w-3.5 animate-pulse shrink-0 text-amber-400" />
    </div>
  )
}
