"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import {
  TrendingUp,
  TrendingDown,
  Activity,
  BarChart2,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  RefreshCw,
  Info,
  Gauge,
  Layers,
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import { getCachedData, setCachedData } from "@/hooks/use-prefetch"

interface TechnicalIndicatorsProps {
  symbol: string
}

interface IndicatorData {
  symbol: string
  timestamp: string
  indicators: {
    rsi: {
      value: number | null
      signal: string
      window: number
      interpretation: string
    }
    macd: {
      value: number | null
      signal: number | null
      histogram: number | null
      trend: string
      interpretation: string
    }
    sma: {
      sma20: number | null
      sma50: number | null
      trend: string
    }
    ema: {
      ema12: number | null
      ema26: number | null
      trend: string
    }
  }
  summary: {
    signal: string
    strength: number
    description: string
  }
  source: string
}

export default function TechnicalIndicators({ symbol }: TechnicalIndicatorsProps) {
  const [indicators, setIndicators] = useState<IndicatorData | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchIndicators = async () => {
    // Check prefetch cache first for instant loading
    const cached = getCachedData<any>(`indicators-${symbol}`)
    if (cached && cached.success !== false) {
      setIndicators(cached.data || cached)
      setIsLoading(false)
      return
    }

    setIsLoading(true)
    setError(null)
    try {
      const response = await fetch(`/api/stock/${symbol}/indicators`)
      const data = await response.json()
      if (data.success && data.data) {
        setIndicators(data.data)
        // Cache the response for future visits
        setCachedData(`indicators-${symbol}`, data, 10 * 60 * 1000)
      } else {
        throw new Error(data.message || 'Failed to fetch indicators')
      }
    } catch (err: any) {
      console.error('Error fetching indicators:', err)
      setError(err.message || 'Failed to load technical indicators')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => { fetchIndicators() }, [symbol])

  // ── Signal helpers ──
  const signalColor = (s: string) => {
    const l = s.toLowerCase()
    if (l.includes('buy') || l.includes('bullish') || l.includes('oversold')) return 'text-emerald-400'
    if (l.includes('sell') || l.includes('bearish') || l.includes('overbought')) return 'text-red-400'
    return 'text-amber-400'
  }
  const signalBg = (s: string) => {
    const l = s.toLowerCase()
    if (l.includes('buy') || l.includes('bullish')) return 'bg-emerald-500/10 border-emerald-500/25'
    if (l.includes('sell') || l.includes('bearish')) return 'bg-red-500/10 border-red-500/25'
    return 'bg-amber-500/10 border-amber-500/25'
  }
  const signalBarColor = (s: string) => {
    const l = s.toLowerCase()
    if (l.includes('buy')) return 'from-emerald-500 to-emerald-400'
    if (l.includes('sell')) return 'from-red-500 to-red-400'
    return 'from-amber-500 to-amber-400'
  }
  const rsiColor = (v: number | null) => {
    if (v === null) return 'text-zinc-500'
    if (v >= 70) return 'text-red-400'
    if (v <= 30) return 'text-emerald-400'
    return 'text-white'
  }

  // ── Loading ──
  if (isLoading) {
    return (
      <Card className="overflow-hidden">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-sm">
            <div className="h-7 w-7 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-600 flex items-center justify-center">
              <Activity className="h-4 w-4 text-white" />
            </div>
            Technical Indicators
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-16 rounded-lg bg-zinc-800/40 animate-pulse" />
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  // ── Error ──
  if (error) {
    return (
      <Card className="overflow-hidden">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-sm">
            <div className="h-7 w-7 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-600 flex items-center justify-center">
              <Activity className="h-4 w-4 text-white" />
            </div>
            Technical Indicators
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-8 text-zinc-400">
            <Info className="h-7 w-7 mb-2 text-zinc-600" />
            <p className="text-xs text-zinc-500">{error}</p>
            <button onClick={fetchIndicators}
              className="mt-3 flex items-center gap-1.5 text-xs text-blue-400 hover:text-blue-300 transition-colors">
              <RefreshCw className="h-3 w-3" /> Try again
            </button>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!indicators) return null

  // ── Main ──
  return (
    <Card className="overflow-hidden">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-sm">
            <div className="h-7 w-7 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
              <Activity className="h-4 w-4 text-white" />
            </div>
            <span>Technical Indicators</span>
            <span className={cn("text-[10px] font-normal px-1.5 py-0.5 rounded-full border",
              indicators.source === 'mongodb_pipeline' ? 'text-emerald-400 border-emerald-500/30 bg-emerald-500/5' : 'text-zinc-500 border-zinc-700 bg-zinc-800')}>
              {indicators.source === 'mongodb_pipeline' ? 'Pipeline' : 'Live'}
            </span>
          </CardTitle>
          <button onClick={fetchIndicators} className="p-1.5 rounded-md hover:bg-zinc-800 transition-colors" title="Refresh">
            <RefreshCw className="h-3.5 w-3.5 text-zinc-500" />
          </button>
        </div>
      </CardHeader>

      <CardContent className="space-y-3 pt-0">
        {/* ── Overall Signal ── */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className={cn("rounded-lg p-3 border", signalBg(indicators.summary.signal))}
        >
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-1.5">
              <Gauge className="h-4 w-4 text-zinc-400" />
              <span className="text-xs text-zinc-400 uppercase tracking-wider">Overall Signal</span>
            </div>
            <span className={cn("text-sm font-bold", signalColor(indicators.summary.signal))}>
              {indicators.summary.signal}
            </span>
          </div>
          <div className="h-1.5 bg-zinc-800/80 rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${indicators.summary.strength}%` }}
              transition={{ duration: 0.8, ease: "easeOut" }}
              className={cn("h-full rounded-full bg-gradient-to-r", signalBarColor(indicators.summary.signal))}
            />
          </div>
          <p className="text-[10px] text-zinc-500 mt-1.5">{indicators.summary.description}</p>
        </motion.div>

        {/* ── RSI ── */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}
          className="rounded-lg p-3 bg-zinc-900/50 border border-zinc-800/50"
        >
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-1.5">
              <BarChart2 className="h-3.5 w-3.5 text-purple-400" />
              <span className="text-xs font-medium">RSI ({indicators.indicators.rsi.window})</span>
            </div>
            <span className={cn("text-base font-bold tabular-nums", rsiColor(indicators.indicators.rsi.value))}>
              {indicators.indicators.rsi.value?.toFixed(1) || '--'}
            </span>
          </div>

          {/* RSI gauge bar */}
          <div className="relative h-2 rounded-full bg-gradient-to-r from-emerald-600/60 via-amber-500/60 to-red-600/60 overflow-hidden">
            <div className="absolute inset-0 bg-zinc-900/30" />
            {indicators.indicators.rsi.value !== null && (
              <motion.div
                initial={{ left: '50%' }}
                animate={{ left: `${Math.max(2, Math.min(98, indicators.indicators.rsi.value))}%` }}
                transition={{ duration: 0.6 }}
                className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 h-3.5 w-1.5 bg-white rounded-full shadow-md shadow-white/30"
              />
            )}
          </div>
          <div className="flex justify-between mt-1 text-[9px] text-zinc-600">
            <span>Oversold &lt;30</span>
            <span>Neutral</span>
            <span>Overbought &gt;70</span>
          </div>
          <div className="flex items-center justify-between mt-2">
            <span className={cn("text-[10px] px-1.5 py-0.5 rounded border", signalBg(indicators.indicators.rsi.signal))}>
              {indicators.indicators.rsi.signal}
            </span>
            <span className="text-[10px] text-zinc-500">{indicators.indicators.rsi.interpretation}</span>
          </div>
        </motion.div>

        {/* ── MACD ── */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="rounded-lg p-3 bg-zinc-900/50 border border-zinc-800/50"
        >
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-1.5">
              <TrendingUp className="h-3.5 w-3.5 text-cyan-400" />
              <span className="text-xs font-medium">MACD</span>
            </div>
            <div className="flex items-center gap-1">
              {indicators.indicators.macd.trend === 'Bullish' ? (
                <ArrowUpRight className="h-3.5 w-3.5 text-emerald-400" />
              ) : indicators.indicators.macd.trend === 'Bearish' ? (
                <ArrowDownRight className="h-3.5 w-3.5 text-red-400" />
              ) : (
                <Minus className="h-3.5 w-3.5 text-amber-400" />
              )}
              <span className={cn("text-xs font-medium", signalColor(indicators.indicators.macd.trend))}>
                {indicators.indicators.macd.trend}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-1.5">
            {[
              { label: 'MACD', value: indicators.indicators.macd.value, color: '' },
              { label: 'Signal', value: indicators.indicators.macd.signal, color: '' },
              { label: 'Histogram', value: indicators.indicators.macd.histogram, color: (indicators.indicators.macd.histogram ?? 0) > 0 ? 'text-emerald-400' : 'text-red-400' },
            ].map((item) => (
              <div key={item.label} className="rounded-md bg-zinc-800/50 p-2 text-center border border-zinc-700/20">
                <div className="text-[9px] text-zinc-500 mb-0.5">{item.label}</div>
                <div className={cn("text-xs font-bold tabular-nums", item.color || 'text-zinc-200')}>
                  {item.value?.toFixed(3) ?? '--'}
                </div>
              </div>
            ))}
          </div>
          <p className="text-[10px] text-zinc-500 mt-2">{indicators.indicators.macd.interpretation}</p>
        </motion.div>

        {/* ── Moving Averages ── */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="grid grid-cols-2 gap-2"
        >
          {/* SMA */}
          <div className="rounded-lg p-3 bg-zinc-900/50 border border-zinc-800/50">
            <div className="flex items-center gap-1.5 mb-2">
              <Layers className="h-3.5 w-3.5 text-blue-400" />
              <span className="text-xs font-medium">SMA</span>
            </div>
            <div className="space-y-1.5">
              <div className="flex justify-between items-center">
                <span className="text-[10px] text-zinc-500">SMA(20)</span>
                <span className="text-xs font-bold tabular-nums">${indicators.indicators.sma.sma20?.toFixed(2) ?? '--'}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-[10px] text-zinc-500">SMA(50)</span>
                <span className="text-xs font-bold tabular-nums">${indicators.indicators.sma.sma50?.toFixed(2) ?? '--'}</span>
              </div>
            </div>
            <div className={cn("mt-2 text-[10px] px-1.5 py-0.5 rounded text-center border", signalBg(indicators.indicators.sma.trend))}>
              {indicators.indicators.sma.trend}
            </div>
          </div>

          {/* EMA */}
          <div className="rounded-lg p-3 bg-zinc-900/50 border border-zinc-800/50">
            <div className="flex items-center gap-1.5 mb-2">
              <Layers className="h-3.5 w-3.5 text-orange-400" />
              <span className="text-xs font-medium">EMA</span>
            </div>
            <div className="space-y-1.5">
              <div className="flex justify-between items-center">
                <span className="text-[10px] text-zinc-500">EMA(12)</span>
                <span className="text-xs font-bold tabular-nums">${indicators.indicators.ema.ema12?.toFixed(2) ?? '--'}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-[10px] text-zinc-500">EMA(26)</span>
                <span className="text-xs font-bold tabular-nums">${indicators.indicators.ema.ema26?.toFixed(2) ?? '--'}</span>
              </div>
            </div>
            <div className={cn("mt-2 text-[10px] px-1.5 py-0.5 rounded text-center border", signalBg(indicators.indicators.ema.trend))}>
              {indicators.indicators.ema.trend}
            </div>
          </div>
        </motion.div>

        {/* ── Timestamp ── */}
        <div className="text-[9px] text-zinc-600 text-right">
          Updated: {new Date(indicators.timestamp).toLocaleString()}
        </div>
      </CardContent>
    </Card>
  )
}
