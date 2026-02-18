"use client"

import { useEffect, useRef } from 'react'

// ── In-memory data cache shared across the app ──
interface CacheEntry<T> {
  data: T
  expiry: number
}

const dataCache = new Map<string, CacheEntry<any>>()
const CACHE_TTL = 5 * 60 * 1000   // 5 minutes for most data
const INDICATOR_TTL = 10 * 60 * 1000 // 10 minutes for indicators (they change slowly)

// Priority stocks to pre-fetch on app load
const PREFETCH_STOCKS = [
  'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NFLX', 'JPM', 'V'
]

const getApiBaseUrl = () => {
  if (typeof window === 'undefined') return ''
  return window.location.hostname === 'localhost' ? 'http://localhost:5000' : ''
}

// ── Public cache API (used by components to get cached data instantly) ──

export function getCachedData<T>(key: string): T | null {
  const entry = dataCache.get(key)
  if (entry && Date.now() < entry.expiry) {
    return entry.data as T
  }
  dataCache.delete(key)
  return null
}

export function setCachedData<T>(key: string, data: T, ttl: number = CACHE_TTL): void {
  dataCache.set(key, { data, expiry: Date.now() + ttl })
}

// ── Fetch with cache (components use this instead of raw fetch) ──

export async function fetchWithCache<T>(
  url: string,
  cacheKey: string,
  ttl: number = CACHE_TTL
): Promise<T | null> {
  // Return cached data instantly if available
  const cached = getCachedData<T>(cacheKey)
  if (cached) return cached

  try {
    const response = await fetch(url, { cache: 'no-store' })
    if (!response.ok) return null

    const data = await response.json()
    setCachedData(cacheKey, data, ttl)
    return data
  } catch {
    return null
  }
}

// ── Pre-fetch hook: runs once on app mount, loads data in background ──

export function usePrefetch(enabled: boolean = true) {
  const hasPrefetched = useRef(false)

  useEffect(() => {
    if (!enabled) return
    if (hasPrefetched.current || typeof window === 'undefined') return
    hasPrefetched.current = true

    const baseUrl = getApiBaseUrl()

    // Phase 1: Pre-fetch stock details + explanations for top 5 (fastest, most likely visited)
    const phase1 = async () => {
      const top5 = PREFETCH_STOCKS.slice(0, 5)
      const promises = top5.flatMap(symbol => [
        fetchWithCache(`${baseUrl}/api/stock/${symbol}`, `stock-details-${symbol}`, CACHE_TTL),
        fetchWithCache(`${baseUrl}/api/stock/${symbol}/explanation?window=comprehensive`, `explanation-${symbol}`, CACHE_TTL),
      ])
      await Promise.allSettled(promises)
      console.log(`⚡ Pre-fetched details + explanations for ${top5.join(', ')}`)
    }

    // Phase 2: Pre-fetch technical indicators for top 5 (slightly slower endpoint)
    const phase2 = async () => {
      const top5 = PREFETCH_STOCKS.slice(0, 5)
      const promises = top5.map(symbol =>
        fetchWithCache(`${baseUrl}/api/stock/${symbol}/indicators`, `indicators-${symbol}`, INDICATOR_TTL)
      )
      await Promise.allSettled(promises)
      console.log(`⚡ Pre-fetched indicators for ${top5.join(', ')}`)
    }

    // Phase 3: Pre-fetch remaining 5 stocks after a delay
    const phase3 = async () => {
      const next5 = PREFETCH_STOCKS.slice(5)
      const promises = next5.flatMap(symbol => [
        fetchWithCache(`${baseUrl}/api/stock/${symbol}`, `stock-details-${symbol}`, CACHE_TTL),
        fetchWithCache(`${baseUrl}/api/stock/${symbol}/explanation?window=comprehensive`, `explanation-${symbol}`, CACHE_TTL),
        fetchWithCache(`${baseUrl}/api/stock/${symbol}/indicators`, `indicators-${symbol}`, INDICATOR_TTL),
      ])
      await Promise.allSettled(promises)
      console.log(`⚡ Pre-fetched all data for ${next5.join(', ')}`)
    }

    // Execute phases with staggered timing to avoid overwhelming the backend
    phase1()
    setTimeout(phase2, 1500)
    setTimeout(phase3, 4000)
  }, [enabled])
}
