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
  Info
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { cn } from "@/lib/utils"

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
    setIsLoading(true)
    setError(null)
    
    try {
      const response = await fetch(`/api/stock/${symbol}/indicators`)
      const data = await response.json()
      
      if (data.success && data.data) {
        setIndicators(data.data)
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

  useEffect(() => {
    fetchIndicators()
  }, [symbol])

  const getSignalColor = (signal: string) => {
    const lowerSignal = signal.toLowerCase()
    if (lowerSignal.includes('buy') || lowerSignal.includes('bullish') || lowerSignal.includes('oversold')) {
      return 'text-emerald-400'
    }
    if (lowerSignal.includes('sell') || lowerSignal.includes('bearish') || lowerSignal.includes('overbought')) {
      return 'text-red-400'
    }
    return 'text-amber-400'
  }

  const getSignalBg = (signal: string) => {
    const lowerSignal = signal.toLowerCase()
    if (lowerSignal.includes('buy') || lowerSignal.includes('bullish')) {
      return 'bg-emerald-500/10 border-emerald-500/30'
    }
    if (lowerSignal.includes('sell') || lowerSignal.includes('bearish')) {
      return 'bg-red-500/10 border-red-500/30'
    }
    return 'bg-amber-500/10 border-amber-500/30'
  }

  const getRSIColor = (value: number | null) => {
    if (value === null) return 'text-zinc-400'
    if (value >= 70) return 'text-red-400'
    if (value <= 30) return 'text-emerald-400'
    return 'text-amber-400'
  }

  const getRSIBarWidth = (value: number | null) => {
    if (value === null) return 50
    return Math.max(0, Math.min(100, value))
  }

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-blue-500" />
            Technical Indicators
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-20 bg-zinc-800 rounded-lg animate-pulse"></div>
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-blue-500" />
            Technical Indicators
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-8 text-zinc-400">
            <Info className="h-8 w-8 mb-2" />
            <p className="text-sm">{error}</p>
            <button
              onClick={fetchIndicators}
              className="mt-4 flex items-center gap-2 text-sm text-blue-400 hover:text-blue-300"
            >
              <RefreshCw className="h-4 w-4" />
              Try again
            </button>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!indicators) return null

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-blue-500" />
            Technical Indicators
          </CardTitle>
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500">
              {indicators.source === 'massive_api' ? 'Massive API' : 'Calculated'}
            </span>
            <button
              onClick={fetchIndicators}
              className="p-1 rounded hover:bg-zinc-800 transition-colors"
              title="Refresh indicators"
            >
              <RefreshCw className="h-3.5 w-3.5 text-zinc-400" />
            </button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Overall Signal Summary */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className={cn(
            "rounded-lg p-4 border",
            getSignalBg(indicators.summary.signal)
          )}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-zinc-300">Overall Signal</span>
            <span className={cn("text-lg font-bold", getSignalColor(indicators.summary.signal))}>
              {indicators.summary.signal}
            </span>
          </div>
          <p className="text-xs text-zinc-400">{indicators.summary.description}</p>
          <div className="mt-2 h-2 bg-zinc-800 rounded-full overflow-hidden">
            <div
              className={cn(
                "h-full rounded-full transition-all duration-500",
                indicators.summary.signal.includes('Buy') ? 'bg-emerald-500' :
                indicators.summary.signal.includes('Sell') ? 'bg-red-500' : 'bg-amber-500'
              )}
              style={{ width: `${indicators.summary.strength}%` }}
            />
          </div>
        </motion.div>

        {/* RSI Indicator */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-zinc-900/50 rounded-lg p-4 border border-zinc-800"
        >
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <BarChart2 className="h-4 w-4 text-purple-400" />
              <span className="font-medium">RSI ({indicators.indicators.rsi.window})</span>
            </div>
            <span className={cn("text-lg font-bold", getRSIColor(indicators.indicators.rsi.value))}>
              {indicators.indicators.rsi.value?.toFixed(1) || '--'}
            </span>
          </div>
          
          {/* RSI Visual Bar */}
          <div className="relative h-3 bg-gradient-to-r from-emerald-500 via-amber-500 to-red-500 rounded-full mb-2">
            <div
              className="absolute top-0 w-1 h-full bg-white rounded-full shadow-lg transition-all duration-500"
              style={{ left: `${getRSIBarWidth(indicators.indicators.rsi.value)}%`, transform: 'translateX(-50%)' }}
            />
            {/* Zone markers */}
            <div className="absolute top-full mt-1 left-0 text-[10px] text-emerald-400">Oversold</div>
            <div className="absolute top-full mt-1 left-1/2 -translate-x-1/2 text-[10px] text-zinc-500">Neutral</div>
            <div className="absolute top-full mt-1 right-0 text-[10px] text-red-400">Overbought</div>
          </div>
          
          <div className="mt-4 flex items-center justify-between">
            <span className={cn("text-xs px-2 py-1 rounded-full border", getSignalBg(indicators.indicators.rsi.signal))}>
              {indicators.indicators.rsi.signal}
            </span>
            <span className="text-xs text-zinc-400">{indicators.indicators.rsi.interpretation}</span>
          </div>
        </motion.div>

        {/* MACD Indicator */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-zinc-900/50 rounded-lg p-4 border border-zinc-800"
        >
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-cyan-400" />
              <span className="font-medium">MACD</span>
            </div>
            <div className="flex items-center gap-1">
              {indicators.indicators.macd.trend === 'Bullish' ? (
                <ArrowUpRight className="h-4 w-4 text-emerald-400" />
              ) : indicators.indicators.macd.trend === 'Bearish' ? (
                <ArrowDownRight className="h-4 w-4 text-red-400" />
              ) : (
                <Minus className="h-4 w-4 text-amber-400" />
              )}
              <span className={getSignalColor(indicators.indicators.macd.trend)}>
                {indicators.indicators.macd.trend}
              </span>
            </div>
          </div>
          
          <div className="grid grid-cols-3 gap-3 text-center">
            <div className="bg-zinc-800/50 rounded p-2">
              <div className="text-[10px] text-zinc-500 mb-1">MACD Line</div>
              <div className="text-sm font-medium">
                {indicators.indicators.macd.value?.toFixed(3) || '--'}
              </div>
            </div>
            <div className="bg-zinc-800/50 rounded p-2">
              <div className="text-[10px] text-zinc-500 mb-1">Signal Line</div>
              <div className="text-sm font-medium">
                {indicators.indicators.macd.signal?.toFixed(3) || '--'}
              </div>
            </div>
            <div className="bg-zinc-800/50 rounded p-2">
              <div className="text-[10px] text-zinc-500 mb-1">Histogram</div>
              <div className={cn(
                "text-sm font-medium",
                (indicators.indicators.macd.histogram || 0) > 0 ? 'text-emerald-400' : 'text-red-400'
              )}>
                {indicators.indicators.macd.histogram?.toFixed(3) || '--'}
              </div>
            </div>
          </div>
          
          <p className="text-xs text-zinc-400 mt-3">{indicators.indicators.macd.interpretation}</p>
        </motion.div>

        {/* Moving Averages */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="grid grid-cols-2 gap-3"
        >
          {/* SMA */}
          <div className="bg-zinc-900/50 rounded-lg p-4 border border-zinc-800">
            <div className="flex items-center gap-2 mb-3">
              <div className="h-2 w-2 rounded-full bg-blue-400"></div>
              <span className="font-medium text-sm">SMA</span>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-zinc-500">SMA(20)</span>
                <span className="font-medium">${indicators.indicators.sma.sma20?.toFixed(2) || '--'}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-zinc-500">SMA(50)</span>
                <span className="font-medium">${indicators.indicators.sma.sma50?.toFixed(2) || '--'}</span>
              </div>
            </div>
            <div className={cn(
              "mt-2 text-xs px-2 py-1 rounded text-center",
              getSignalBg(indicators.indicators.sma.trend)
            )}>
              {indicators.indicators.sma.trend}
            </div>
          </div>

          {/* EMA */}
          <div className="bg-zinc-900/50 rounded-lg p-4 border border-zinc-800">
            <div className="flex items-center gap-2 mb-3">
              <div className="h-2 w-2 rounded-full bg-orange-400"></div>
              <span className="font-medium text-sm">EMA</span>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-zinc-500">EMA(12)</span>
                <span className="font-medium">${indicators.indicators.ema.ema12?.toFixed(2) || '--'}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-zinc-500">EMA(26)</span>
                <span className="font-medium">${indicators.indicators.ema.ema26?.toFixed(2) || '--'}</span>
              </div>
            </div>
            <div className={cn(
              "mt-2 text-xs px-2 py-1 rounded text-center",
              getSignalBg(indicators.indicators.ema.trend)
            )}>
              {indicators.indicators.ema.trend}
            </div>
          </div>
        </motion.div>

        {/* Data freshness */}
        <div className="text-[10px] text-zinc-600 text-right">
          Last updated: {new Date(indicators.timestamp).toLocaleString()}
        </div>
      </CardContent>
    </Card>
  )
}
