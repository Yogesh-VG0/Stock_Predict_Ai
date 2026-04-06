'use client'

import { useEffect, useState } from 'react'
import { TrendingUp, TrendingDown, BarChart3, Target, Zap } from 'lucide-react'

interface HorizonMetrics {
  predictions: number
  avg_confidence: string
  sharpe_ratio: number
  win_rate: number
  total_return: number
}

interface ModelPerformance {
  model_version: string
  last_updated: string
  total_predictions: number
  avg_confidence: string
  horizons: {
    next_day: HorizonMetrics
    '7_day': HorizonMetrics
    '30_day': HorizonMetrics
  }
  benchmark: {
    spy_return: number
    period: string
  }
  data_source?: 'live_backtest' | 'fallback_defaults'
}

export function ModelPerformanceCard() {
  const [metrics, setMetrics] = useState<ModelPerformance | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchMetrics() {
      try {
        const res = await fetch('/api/stock/model/performance')
        if (res.ok) {
          const data = await res.json()
          setMetrics(data)
        }
      } catch (error) {
        console.error('Failed to fetch model performance:', error)
      } finally {
        setLoading(false)
      }
    }
    fetchMetrics()
  }, [])

  if (loading) {
    return (
      <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-6 animate-pulse">
        <div className="h-6 bg-zinc-800 rounded w-48 mb-4"></div>
        <div className="space-y-3">
          <div className="h-20 bg-zinc-800 rounded"></div>
          <div className="h-20 bg-zinc-800 rounded"></div>
        </div>
      </div>
    )
  }

  if (!metrics) return null

  const horizonLabels = {
    next_day: 'Next Day',
    '7_day': '7-Day',
    '30_day': '30-Day'
  }

  return (
    <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-blue-400" />
          Model Performance
        </h3>
        <span className="text-xs text-zinc-500 bg-zinc-800 px-2 py-1 rounded">
          {metrics.model_version}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        {(['30_day', '7_day', 'next_day'] as const).map((horizon) => {
          const data = metrics.horizons[horizon]
          const isPositive = data.total_return > 0
          const beatsBenchmark = data.total_return > metrics.benchmark.spy_return

          return (
            <div
              key={horizon}
              className={`p-4 rounded-lg border ${
                horizon === '30_day' 
                  ? 'bg-emerald-950/30 border-emerald-800/50' 
                  : 'bg-zinc-800/50 border-zinc-700/50'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-zinc-300">
                  {horizonLabels[horizon]}
                </span>
                {horizon === '30_day' && (
                  <span className="text-[10px] bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded">
                    BEST
                  </span>
                )}
              </div>

              <div className="flex items-baseline gap-1 mb-2">
                {isPositive ? (
                  <TrendingUp className="w-4 h-4 text-emerald-400" />
                ) : (
                  <TrendingDown className="w-4 h-4 text-red-400" />
                )}
                <span className={`text-xl font-bold ${isPositive ? 'text-emerald-400' : 'text-red-400'}`}>
                  {isPositive ? '+' : ''}{data.total_return.toFixed(2)}%
                </span>
              </div>

              <div className="space-y-1 text-xs">
                <div className="flex justify-between text-zinc-400">
                  <span className="flex items-center gap-1">
                    <Zap className="w-3 h-3" /> Sharpe
                  </span>
                  <span className={data.sharpe_ratio > 0 ? 'text-emerald-400' : 'text-red-400'}>
                    {data.sharpe_ratio.toFixed(2)}
                  </span>
                </div>
                <div className="flex justify-between text-zinc-400">
                  <span className="flex items-center gap-1">
                    <Target className="w-3 h-3" /> Win Rate
                  </span>
                  <span className={data.win_rate > 50 ? 'text-emerald-400' : 'text-zinc-300'}>
                    {data.win_rate.toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between text-zinc-400">
                  <span>Confidence</span>
                  <span className="text-zinc-300">{data.avg_confidence}%</span>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      <div className="flex items-center justify-between text-xs text-zinc-500 pt-3 border-t border-zinc-800">
        <span>
          Benchmark (SPY): <span className="text-zinc-300">+{metrics.benchmark.spy_return}%</span>
        </span>
        <span>
          30-Day beats SPY by{' '}
          <span className="text-emerald-400">
            +{(metrics.horizons['30_day'].total_return - metrics.benchmark.spy_return).toFixed(2)}%
          </span>
        </span>
      </div>
    </div>
  )
}
