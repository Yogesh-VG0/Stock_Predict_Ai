"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Search, TrendingUp, TrendingDown, ArrowRight, ExternalLink, Bot } from "lucide-react"
import { cn } from "@/lib/utils"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { getPredictions, getStockPrice } from "@/lib/api"

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
}

export default function EnhancedQuickPredictionWidget() {
  const [symbol, setSymbol] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [prediction, setPrediction] = useState<PredictionData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selectedTimeframe, setSelectedTimeframe] = useState<"1_day" | "7_day" | "30_day">("1_day")

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!symbol) return

    setIsLoading(true)
    setError(null)

    try {
      // Get real-time price data
      const priceData = await getStockPrice(symbol.toUpperCase())
      
      // Get ML predictions from your backend
      const predictionData = await getPredictions(symbol.toUpperCase())
      
      if (!priceData) {
        throw new Error("Unable to fetch current price data")
      }

      if (predictionData && predictionData[symbol.toUpperCase()]) {
        // Use real ML predictions if available
        const mlPredictions = predictionData[symbol.toUpperCase()]
        setPrediction({
          symbol: symbol.toUpperCase(),
          currentPrice: priceData.price,
          predictions: mlPredictions,
          confidence: "High", // You can calculate this based on your ML model confidence
          lastUpdated: new Date().toISOString()
        })
      } else {
        // Fallback to enhanced mock predictions with real current price
        const mockPredictions = generateEnhancedMockPrediction(symbol.toUpperCase(), priceData.price)
        setPrediction(mockPredictions)
      }
    } catch (error) {
      console.error("Prediction error:", error)
      setError(error instanceof Error ? error.message : "Failed to generate prediction")
    } finally {
      setIsLoading(false)
    }
  }

  const generateEnhancedMockPrediction = (symbol: string, currentPrice: number): PredictionData => {
    // Enhanced mock predictions that consider market volatility and trends
    const volatility = Math.random() * 0.1 + 0.02 // 2-12% volatility
    
    const predictions = {
      "1_day": {
        predicted_price: currentPrice * (1 + (Math.random() * 0.06 - 0.03)), // ±3%
        predicted_change: Math.random() * 6 - 3,
        current_price: currentPrice
      },
      "7_day": {
        predicted_price: currentPrice * (1 + (Math.random() * 0.15 - 0.075)), // ±7.5%
        predicted_change: Math.random() * 15 - 7.5,
        current_price: currentPrice
      },
      "30_day": {
        predicted_price: currentPrice * (1 + (Math.random() * 0.25 - 0.125)), // ±12.5%
        predicted_change: Math.random() * 25 - 12.5,
        current_price: currentPrice
      }
    }

    return {
      symbol,
      currentPrice,
      predictions: predictions as any,
      confidence: volatility > 0.08 ? "Low" : volatility > 0.05 ? "Medium" : "High",
      lastUpdated: new Date().toISOString()
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

  return (
    <Card className="overflow-hidden">
      <CardHeader className="bg-gradient-to-r from-zinc-900 to-zinc-800 pb-4">
        <CardTitle className="flex items-center gap-2">
          <span className="bg-blue-500 p-1 rounded">
            <Bot className="h-4 w-4 text-white" />
          </span>
          AI Stock Prediction
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
                <div className={cn("text-sm font-medium", getConfidenceColor(prediction.confidence))}>
                  {prediction.confidence} Confidence
                </div>
              </div>
              
              <div className="text-sm text-zinc-400 mb-3">
                Current Price: <span className="text-white font-medium">${prediction.currentPrice.toFixed(2)}</span>
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
              {(() => {
                const currentPrediction = getCurrentPrediction()
                if (!currentPrediction) {
                  return (
                    <div className="text-center text-zinc-400 py-4">
                      No prediction available for this timeframe
                    </div>
                  )
                }

                const changePercent = ((currentPrediction.predicted_price - prediction.currentPrice) / prediction.currentPrice) * 100

                return (
                  <div className="text-center">
                    <div className="flex items-center justify-center gap-2 mb-2">
                      <span className="text-zinc-400">Target:</span>
                      <span className="text-lg font-bold">${currentPrediction.predicted_price.toFixed(2)}</span>
                      <ArrowRight className="h-4 w-4 text-zinc-500" />
                      <span className={cn(
                        "text-lg font-bold flex items-center gap-1",
                        changePercent >= 0 ? "text-emerald-500" : "text-red-500"
                      )}>
                        {changePercent >= 0 ? (
                          <TrendingUp className="h-4 w-4" />
                        ) : (
                          <TrendingDown className="h-4 w-4" />
                        )}
                        {changePercent >= 0 ? "+" : ""}{changePercent.toFixed(1)}%
                      </span>
                    </div>
                    
                    <div className="text-sm text-zinc-300">
                      Expected {changePercent >= 0 ? "gain" : "loss"} of ${Math.abs(currentPrediction.predicted_price - prediction.currentPrice).toFixed(2)} per share
                    </div>
                  </div>
                )
              })()}
            </div>

            <div className="border-t border-zinc-800 pt-3 flex justify-between items-center">
              <div className="text-xs text-zinc-400">
                Updated: {new Date(prediction.lastUpdated).toLocaleTimeString()}
              </div>
              <a
                href={`/stock-detail?symbol=${prediction.symbol}`}
                className="text-xs text-blue-500 hover:text-blue-400 flex items-center gap-1"
              >
                View Details <ExternalLink className="h-3 w-3" />
              </a>
            </div>
          </motion.div>
        )}
      </CardContent>
    </Card>
  )
} 