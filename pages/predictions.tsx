"use client"

import { useState, useEffect } from "react"
import { useNavigate } from "react-router-dom"
import { motion } from "framer-motion"
import { TrendingUp, TrendingDown, Brain, Calendar, Target, Sparkles, Loader2, AlertCircle } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import {
  getAvailableStocksWithExplanations,
  getBatchExplanationStatus,
  getComprehensiveAIExplanation,
} from "@/lib/api"
import { getCachedData, setCachedData } from "@/hooks/use-prefetch"

interface StockPrediction {
  symbol: string
  name: string
  predictions: {
    '1_day': { predicted_price: number; confidence: number; price_change: number }
    '7_day': { predicted_price: number; confidence: number; price_change: number }
    '30_day': { predicted_price: number; confidence: number; price_change: number }
  }
}

const COMPANY_NAMES: Record<string, string> = {
  AAPL: "Apple Inc.", MSFT: "Microsoft Corp.", GOOGL: "Alphabet Inc.", AMZN: "Amazon.com Inc.",
  TSLA: "Tesla Inc.", NVDA: "NVIDIA Corp.", META: "Meta Platforms", NFLX: "Netflix Inc.",
  JPM: "JPMorgan Chase", V: "Visa Inc.", JNJ: "Johnson & Johnson", WMT: "Walmart Inc.",
  PG: "Procter & Gamble", UNH: "UnitedHealth", HD: "Home Depot", MA: "Mastercard",
  BAC: "Bank of America", XOM: "Exxon Mobil", LLY: "Eli Lilly", ABBV: "AbbVie Inc.",
  AVGO: "Broadcom Inc.", COST: "Costco Wholesale",
  ORCL: "Oracle Corp.", CRM: "Salesforce Inc.", KO: "Coca-Cola Co.",
}

export default function Predictions() {
  const navigate = useNavigate()
  const [selectedTimeframe, setSelectedTimeframe] = useState<'1_day' | '7_day' | '30_day'>('7_day')
  const [stocksWithExplanations, setStocksWithExplanations] = useState<string[]>([])
  const [batchStatus, setBatchStatus] = useState<any>(null)
  const [predictions, setPredictions] = useState<StockPrediction[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isGeneratingBatch, setIsGeneratingBatch] = useState(false)
  
  useEffect(() => {
    loadPredictions()
  }, [])

  const loadPredictions = async () => {
    setIsLoading(true)
    try {
      const [status, availableStocks] = await Promise.all([
        getBatchExplanationStatus().catch(() => null),
        getAvailableStocksWithExplanations().catch(() => []),
      ])
      setBatchStatus(status)
      setStocksWithExplanations(availableStocks)

      // Fetch real prediction data for available stocks
      const stockPredictions: StockPrediction[] = []
      const tickersToFetch = availableStocks.length > 0 ? availableStocks.slice(0, 20) : []

      const results = await Promise.allSettled(
        tickersToFetch.map(async (symbol: string) => {
          // Use prefetch cache if available
          const cached = getCachedData<any>(`explanation-${symbol}`)
          const explanation = cached || await getComprehensiveAIExplanation(symbol)
          if (!cached && explanation) setCachedData(`explanation-${symbol}`, explanation)
          return { symbol, explanation }
        })
      )

      for (const result of results) {
        if (result.status === 'fulfilled' && result.value.explanation) {
          const { symbol, explanation } = result.value
          const ps = explanation.prediction_summary
          if (!ps) continue

          const nextDay = ps['1_day'] || ps.next_day
          const sevenDay = ps['7_day']
          const thirtyDay = ps['30_day']

          stockPredictions.push({
            symbol,
            name: COMPANY_NAMES[symbol] || `${symbol}`,
            predictions: {
              '1_day': {
                predicted_price: nextDay?.predicted_price || 0,
                confidence: nextDay?.confidence || 0,
                price_change: nextDay?.price_change || 0,
              },
              '7_day': {
                predicted_price: sevenDay?.predicted_price || 0,
                confidence: sevenDay?.confidence || 0,
                price_change: sevenDay?.price_change || 0,
              },
              '30_day': {
                predicted_price: thirtyDay?.predicted_price || 0,
                confidence: thirtyDay?.confidence || 0,
                price_change: thirtyDay?.price_change || 0,
              },
            },
          })
        }
      }

      setPredictions(stockPredictions)
    } catch (error) {
      console.error('Error loading predictions:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleGenerateBatch = async () => {
    setIsGeneratingBatch(true)
    try {
      const { generateBatchAIExplanations } = await import('@/lib/api')
      const result = await generateBatchAIExplanations()
      if (result) {
        alert(`Batch generation started! Processing ${result.total_tickers} tickers...`)
        setTimeout(loadPredictions, 10000)
      }
    } catch (error) {
      console.error('Error generating batch explanations:', error)
      alert('Error starting batch generation. Please try again.')
    } finally {
      setIsGeneratingBatch(false)
    }
  }

  const getTimeframeLabel = (timeframe: string) => {
    switch (timeframe) {
      case '1_day': return '1 Day'
      case '7_day': return '7 Days'
      case '30_day': return '30 Days'
      default: return timeframe
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-emerald-500'
    if (confidence >= 0.7) return 'text-amber-500'
    return 'text-red-500'
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div className="space-y-1">
            <h1 className="text-2xl font-bold flex items-center gap-2">
              <Brain className="h-6 w-6 text-purple-500" />
              AI Stock Predictions
              <Badge variant="outline" className="ml-1">
                <Sparkles className="h-3 w-3 mr-1" /> Pipeline
              </Badge>
            </h1>
            <p className="text-xs md:text-sm text-zinc-400 max-w-xl">
              Real ML predictions from the daily CI/CD pipeline — no mock data
            </p>
          </div>
        </div>
      </motion.div>

      {/* AI Batch Status */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        <Card className="bg-gradient-to-br from-emerald-500/10 to-blue-500/10 border-emerald-500/20">
          <CardContent className="p-4">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-2 flex-wrap">
                <Brain className="h-5 w-5 text-emerald-400" />
                <span className="font-medium">AI Explanation Status</span>
                {batchStatus && (
                  <Badge variant="outline" className="text-emerald-400 border-emerald-400">
                    {batchStatus.coverage_percentage}% Coverage
                  </Badge>
                )}
              </div>
              <Button onClick={handleGenerateBatch} disabled={isGeneratingBatch} size="sm" className="bg-emerald-600 hover:bg-emerald-700">
                {isGeneratingBatch ? (
                  <><Loader2 className="h-4 w-4 mr-2 animate-spin" /> Generating...</>
                ) : (
                  <><Sparkles className="h-4 w-4 mr-2" /> Generate All</>
                )}
              </Button>
            </div>
            {batchStatus && (
              <div className="flex flex-wrap items-center gap-4 text-sm text-zinc-300 mt-3">
                <span>✅ {batchStatus.with_explanations} stocks have AI explanations</span>
                <span>⏳ {batchStatus.without_explanations} pending</span>
                <span>📊 Total: {batchStatus.total_tickers} tickers</span>
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Timeframe Selector */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
        <div className="flex items-center gap-2 mb-3">
          <Calendar className="h-5 w-5 text-blue-500" />
          <span className="font-medium">Prediction Timeframe</span>
        </div>
        <div className="flex gap-2">
          {(['1_day', '7_day', '30_day'] as const).map((timeframe) => (
            <Button
              key={timeframe}
              variant={selectedTimeframe === timeframe ? "default" : "outline"}
              size="sm"
              onClick={() => setSelectedTimeframe(timeframe)}
              className={selectedTimeframe === timeframe ? "bg-purple-600 hover:bg-purple-700" : ""}
            >
              {getTimeframeLabel(timeframe)}
            </Button>
          ))}
        </div>
      </motion.div>

      {/* Loading State */}
      {isLoading && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <Card key={i}>
              <CardHeader className="pb-3">
                <Skeleton className="h-6 w-24" />
                <Skeleton className="h-4 w-40 mt-1" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-24 w-full rounded-lg" />
                <div className="space-y-2 mt-4">
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-full" />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Empty State */}
      {!isLoading && predictions.length === 0 && (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16 text-center">
            <AlertCircle className="h-10 w-10 text-zinc-600 mb-4" />
            <h3 className="text-lg font-medium text-zinc-300 mb-2">No Predictions Available</h3>
            <p className="text-sm text-zinc-500 max-w-md mb-4">
              AI predictions are generated daily by the CI/CD pipeline. Click "Generate All" above to start the process.
            </p>
          </CardContent>
        </Card>
      )}

      {/* Predictions Grid */}
      {!isLoading && predictions.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {predictions.map((stock, index) => {
            const pred = stock.predictions[selectedTimeframe]
            if (!pred || pred.predicted_price === 0) return null

            return (
              <motion.div
                key={stock.symbol}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 + index * 0.05 }}
              >
                <Card className="hover:shadow-lg transition-shadow cursor-pointer group hover:border-zinc-600">
                  <CardHeader className="pb-3">
                    <div className="flex justify-between items-start">
                      <div>
                        <CardTitle className="flex items-center gap-2">
                          {stock.symbol}
                          <Target className="h-4 w-4 text-blue-500" />
                          <div className="flex items-center gap-1">
                            <Sparkles className="h-3 w-3 text-emerald-400" />
                            <span className="text-xs text-emerald-400 font-medium">AI</span>
                          </div>
                        </CardTitle>
                        <p className="text-sm text-zinc-400 mt-1">{stock.name}</p>
                      </div>
                      <Badge variant="outline" className={getConfidenceColor(pred.confidence)}>
                        {(pred.confidence * 100).toFixed(0)}% confidence
                      </Badge>
                    </div>
                  </CardHeader>

                  <CardContent>
                    <div className="space-y-4">
                      {/* Current Timeframe Prediction */}
                      <div className="bg-gradient-to-br from-purple-500/10 to-purple-600/5 rounded-lg p-4 border border-purple-500/20">
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-sm font-medium text-purple-400">
                            {getTimeframeLabel(selectedTimeframe)} Forecast
                          </span>
                          <span className="text-xs text-zinc-400">ML Pipeline</span>
                        </div>
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-2xl font-bold">
                            ${pred.predicted_price.toFixed(2)}
                          </span>
                          <div className={`flex items-center gap-1 text-sm font-medium px-2 py-1 rounded-md ${
                            pred.price_change >= 0
                              ? "bg-emerald-500/20 text-emerald-400"
                              : "bg-red-500/20 text-red-400"
                          }`}>
                            {pred.price_change >= 0 ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                            {pred.price_change >= 0 ? '+' : ''}{pred.price_change.toFixed(1)}%
                          </div>
                        </div>
                      </div>

                      {/* All Timeframes Overview */}
                      <div className="space-y-2">
                        <h4 className="text-sm font-medium text-zinc-300">All Predictions</h4>
                        {(['1_day', '7_day', '30_day'] as const).map((tf) => {
                          const p = stock.predictions[tf]
                          if (!p || p.predicted_price === 0) return null
                          return (
                            <div key={tf} className="flex justify-between items-center text-xs">
                              <span className="text-zinc-400">{getTimeframeLabel(tf)}</span>
                              <div className="flex items-center gap-2">
                                <span className="font-medium">${p.predicted_price.toFixed(2)}</span>
                                <span className={p.price_change >= 0 ? 'text-emerald-500' : 'text-red-500'}>
                                  {p.price_change >= 0 ? '+' : ''}{p.price_change.toFixed(1)}%
                                </span>
                              </div>
                            </div>
                          )
                        })}
                      </div>

                      {/* View Details */}
                      <Button
                        size="sm"
                        variant="outline"
                        className="w-full group-hover:bg-purple-600 group-hover:text-white transition-colors"
                        onClick={() => navigate(`/stock/${stock.symbol}`)}
                      >
                        View Comprehensive Analysis
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )
          })}
        </div>
      )}

      {/* Feature Highlight */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}>
        <Card className="bg-gradient-to-br from-purple-500/10 to-blue-500/10 border-purple-500/20">
          <CardContent className="p-6">
            <div className="flex items-center gap-3 mb-4">
              <Sparkles className="h-6 w-6 text-purple-400" />
              <h3 className="text-xl font-semibold">Enhanced AI Analysis</h3>
            </div>
            <p className="text-zinc-300 mb-4">
              Our advanced AI system provides comprehensive explanations using:
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
              {[
                { color: 'bg-emerald-500', label: 'Real-time sentiment analysis' },
                { color: 'bg-blue-500', label: 'Technical indicators' },
                { color: 'bg-purple-500', label: 'Fundamental analysis' },
                { color: 'bg-amber-500', label: 'Social media insights' },
                { color: 'bg-pink-500', label: 'News impact analysis' },
                { color: 'bg-cyan-500', label: 'Risk assessment' },
              ].map(({ color, label }) => (
                <div key={label} className="flex items-center gap-2">
                  <div className={`w-2 h-2 ${color} rounded-full`} />
                  <span className="text-sm">{label}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  )
}
