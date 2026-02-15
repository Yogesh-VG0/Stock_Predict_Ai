"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Search, TrendingUp, TrendingDown, ArrowRight, ExternalLink, Bot, Wifi, WifiOff } from "lucide-react"
import { cn } from "@/lib/utils"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { getPredictions, getStockPrice } from "@/lib/api"
import { useWebSocket, useStockPrice } from "@/hooks/use-websocket-context"

interface PredictionData {
  symbol: string
  currentPrice: number
  predictions: {
    "1_day"?: { predicted_price: number; predicted_change: number; current_price: number }
    "7_day"?: { predicted_price: number; predicted_change: number; current_price: number }
    "30_day"?: { predicted_price: number; predicted_change: number; current_price: number }
  }
  confidence: string
  lastUpdated: string
  isRealTimePrice?: boolean
  isRealMLPrediction?: boolean
}

interface RealtimeData {
  price: number
  volume: number
  timestamp: number
  change?: number
  changePercent?: number
}

const AVAILABLE_STOCKS = [
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
  'JPM', 'V', 'JNJ', 'WMT', 'PG', 'UNH', 'HD', 'MA', 'BAC', 'XOM', 'LLY', 'ABBV'
];

// Direct MongoDB prediction data as fallback
// Direct MongoDB prediction data as fallback
const REAL_MONGODB_PREDICTIONS: Record<string, any> = {};


export default function EnhancedQuickPredictionWidget() {
  const [symbol, setSymbol] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [prediction, setPrediction] = useState<PredictionData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selectedTimeframe, setSelectedTimeframe] = useState<"1_day" | "7_day" | "30_day">("1_day")
  const [isClient, setIsClient] = useState(false)

  // Use centralized WebSocket service
  const { isConnected: isWebSocketConnected, lastUpdate } = useWebSocket()
  const currentStockPrice = useStockPrice(prediction?.symbol || "")

  // Ensure we're on the client side to prevent hydration mismatches
  useEffect(() => {
    setIsClient(true)
  }, [])

  // Update prediction with real-time price when it changes
  useEffect(() => {
    if (currentStockPrice && prediction) {
      setPrediction(prev => {
        // Only update if the price has actually changed to prevent infinite loops
        if (!prev || prev.currentPrice === currentStockPrice.price) {
          return prev
        }

        return {
          ...prev,
          currentPrice: currentStockPrice.price,
          isRealTimePrice: true,
          predictions: {
            ...prev.predictions,
            // Update current_price in all predictions for dynamic percentage calculation
            "1_day": prev.predictions["1_day"] ? {
              ...prev.predictions["1_day"],
              current_price: currentStockPrice.price
            } : undefined,
            "7_day": prev.predictions["7_day"] ? {
              ...prev.predictions["7_day"],
              current_price: currentStockPrice.price
            } : undefined,
            "30_day": prev.predictions["30_day"] ? {
              ...prev.predictions["30_day"],
              current_price: currentStockPrice.price
            } : undefined,
          }
        }
      })
    }
  }, [currentStockPrice]) // Remove prediction dependency to prevent infinite loop

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!symbol) return

    setIsLoading(true)
    setError(null)

    try {
      // Get real-time price data - prefer WebSocket data if available
      let priceData = currentStockPrice ? {
        price: currentStockPrice.price,
        change: currentStockPrice.change,
        changePercent: currentStockPrice.changePercent
      } : await getStockPrice(symbol.toUpperCase())

      // If both WebSocket and API fail, create fallback data
      if (!priceData) {
        priceData = {
          price: 200, // Reasonable fallback for AAPL-like price
          change: 0,
          changePercent: 0
        }
        console.warn(`Using fallback price data for ${symbol.toUpperCase()}`)
      }

      console.log(`🔍 Fetching ML predictions for ${symbol.toUpperCase()}...`)



      // Try to get ML predictions from API backend
      const predictionData = await getPredictions(symbol.toUpperCase())

      console.log('📊 Prediction data received:', predictionData)

      if (predictionData && predictionData[symbol.toUpperCase()]) {
        // Use real ML predictions if available
        console.log(`✅ Using API ML predictions for ${symbol.toUpperCase()}`)
        const mlPredictions = predictionData[symbol.toUpperCase()]

        // Calculate confidence based on ML model data if available
        let confidence = "Medium"
        const nextDayPred = mlPredictions['1_day'] || mlPredictions.next_day;
        if (nextDayPred?.confidence !== undefined || mlPredictions['7_day']?.confidence !== undefined) {
          const confidenceValue = nextDayPred?.confidence || mlPredictions['7_day']?.confidence || 0.5
          confidence = confidenceValue > 0.7 ? "High" : confidenceValue > 0.4 ? "Medium" : "Low"
        }

        setPrediction({
          symbol: symbol.toUpperCase(),
          currentPrice: priceData.price,
          predictions: mlPredictions,
          confidence: confidence,
          lastUpdated: new Date().toISOString(),
          isRealTimePrice: false,
          isRealMLPrediction: true
        })
      } else {
        // No real ML predictions available
        console.log(`⚠️ No real ML predictions found for ${symbol.toUpperCase()}`)
        setPrediction(null)
        setError("No ML predictions available for this ticker yet. Predictions are generated daily.")
      }
    } catch (error) {
      console.error("Prediction error:", error)
      setError(error instanceof Error ? error.message : "Failed to generate prediction")
    } finally {
      setIsLoading(false)
    }
  }

  const timeframes = [
    { key: "1_day", label: "1 Day" },
    { key: "7_day", label: "7 Days" },
    { key: "30_day", label: "30 Days" }
  ]

  const getCurrentPrediction = () => {
    if (!prediction || !prediction.predictions[selectedTimeframe]) return null
    return prediction.predictions[selectedTimeframe]
  }

  const getConfidenceColor = (confidence: string) => {
    switch (confidence.toLowerCase()) {
      case "high": return "text-emerald-500"
      case "medium": return "text-amber-500"
      case "low": return "text-red-500"
      default: return "text-zinc-400"
    }
  }

  const currentPrediction = getCurrentPrediction()

  // Dynamically calculate change percentage
  const calculatedChange = currentPrediction && currentPrediction.current_price
    ? ((currentPrediction.predicted_price - currentPrediction.current_price) / currentPrediction.current_price) * 100
    : 0;

  return (
    <Card className="overflow-hidden">
      <CardHeader className="bg-gradient-to-r from-zinc-900 to-zinc-800 pb-4">
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="bg-blue-500 p-1 rounded">
              <Bot className="h-4 w-4 text-white" />
            </span>
            AI Stock Prediction
          </div>
          <div className="flex items-center gap-2">
            {isClient && (
              <>
                {isWebSocketConnected ? (
                  <div className="flex items-center gap-1 text-green-500 text-xs">
                    <Wifi className="h-3 w-3" />
                    <span>Live</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-1 text-red-500 text-xs">
                    <WifiOff className="h-3 w-3" />
                    <span>Offline</span>
                  </div>
                )}
              </>
            )}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="p-4">
        <form onSubmit={handleSubmit} className="mb-4">
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-2.5 h-4 w-4 text-zinc-400" />
              <input
                type="text"
                placeholder="Enter stock symbol (e.g. AAPL)"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-md py-2 pl-9 pr-4 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
            </div>
            <button
              type="submit"
              disabled={!symbol || isLoading}
              className="bg-blue-500 hover:bg-blue-600 text-white font-medium rounded-md px-4 py-2 text-sm disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isLoading ? (
                <div className="h-5 w-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                "Predict"
              )}
            </button>
          </div>
        </form>

        {error && (
          <div className="mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
            <p className="text-red-500 text-sm">{error}</p>
          </div>
        )}

        {prediction && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-zinc-900 rounded-lg p-4 border border-zinc-800"
          >
            <div className="mb-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-bold">{prediction.symbol}</h3>
              </div>

              <div className="text-sm text-zinc-400 mb-3">
                Current Price: <span className="text-white font-medium">${prediction.currentPrice.toFixed(2)}</span>
                {isClient && prediction.isRealTimePrice && lastUpdate && (
                  <div className="text-xs text-green-500 mt-1 flex items-center gap-1">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    Live Price • Updated: {lastUpdate}
                  </div>
                )}
              </div>

              {/* Timeframe Selector */}
              <div className="flex justify-center gap-2 mb-4">
                {timeframes.map((timeframe) => (
                  <button
                    key={timeframe.key}
                    type="button"
                    onClick={() => setSelectedTimeframe(timeframe.key as any)}
                    disabled={!prediction.predictions[timeframe.key as keyof typeof prediction.predictions]}
                    className={cn(
                      "px-4 py-1 rounded-full border text-xs font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed",
                      selectedTimeframe === timeframe.key
                        ? "bg-blue-500 text-white border-blue-500 shadow"
                        : "bg-zinc-800 text-zinc-300 border-zinc-700 hover:bg-zinc-700"
                    )}
                  >
                    {timeframe.label}
                  </button>
                ))}
              </div>

              {/* Prediction Display */}
              {currentPrediction ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 bg-zinc-800 rounded-lg">
                    <div>
                      <div className="text-sm text-zinc-400">
                        {(currentPrediction as any).is_market_neutral ? "Alpha-Implied Price" : "Predicted Price"}
                      </div>
                      <div className="text-lg font-bold text-white">
                        ${currentPrediction.predicted_price.toFixed(2)}
                      </div>
                      {(currentPrediction as any).prob_positive != null && (
                        <div className="text-xs text-zinc-500 mt-0.5">
                          P(up) {((currentPrediction as any).prob_positive * 100).toFixed(0)}%
                        </div>
                      )}
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-zinc-400">
                        {(currentPrediction as any).is_market_neutral ? "Alpha vs SPY" : "Change"}
                      </div>
                      <div className={cn("text-lg font-bold flex items-center gap-1",
                        calculatedChange >= 0 ? "text-green-500" : "text-red-500"
                      )}>
                        {calculatedChange >= 0 ? (
                          <TrendingUp className="h-4 w-4" />
                        ) : (
                          <TrendingDown className="h-4 w-4" />
                        )}
                        {calculatedChange >= 0 ? "+" : ""}{calculatedChange.toFixed(2)}%
                      </div>
                    </div>
                  </div>


                </div>
              ) : (
                <div className="text-center py-4 text-zinc-500">
                  No prediction available for this timeframe
                </div>
              )}


            </div>
          </motion.div>
        )}
      </CardContent>
    </Card>
  )
} 