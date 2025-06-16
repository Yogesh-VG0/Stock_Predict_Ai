"use client"

import type React from "react"

import { useState } from "react"
import { motion } from "framer-motion"
import { Search, TrendingUp, TrendingDown, ArrowRight, ExternalLink } from "lucide-react"
import { cn } from "@/lib/utils"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface PredictionWindow {
  label: string;
  targetPrice: number;
  percentChange: number;
  confidence: string;
  explanation: string;
}

interface QuickPredictionWidgetProps {
  className?: string
}

export default function QuickPredictionWidget({ className }: QuickPredictionWidgetProps) {
  const [symbol, setSymbol] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [prediction, setPrediction] = useState<
    | {
        symbol: string
        currentPrice: number
        windows: PredictionWindow[]
        analystContext?: string
        analystTargets?: { label: string; value: string }[]
        recentPerformance?: string
        marketDrivers?: string
      }
    | null
  >(null)
  const [selectedWindow, setSelectedWindow] = useState<number>(0)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!symbol) return

    setIsLoading(true)
    setSelectedWindow(0)

    setTimeout(() => {
      const upperSymbol = symbol.toUpperCase()
      if (upperSymbol === "AAPL") {
        setPrediction({
          symbol: "AAPL",
          currentPrice: 206.86,
          windows: [
            {
              label: "Next Day",
              targetPrice: 208.93,
              percentChange: 1.0,
              confidence: "Medium",
              explanation:
                "Modest upside is expected with a 1% gain, reflecting typical daily volatility and ongoing optimism around iPhone 15 sales and AI integration.",
            },
            {
              label: "30 Days",
              targetPrice: 217.20,
              percentChange: 5.0,
              confidence: "Medium",
              explanation:
                "The model projects a 5% increase, targeting $217.20, assuming continued positive sentiment and stable tech sector performance.",
            },
            {
              label: "90 Days",
              targetPrice: 231.68,
              percentChange: 12.0,
              confidence: "Medium",
              explanation:
                "The 12% upside to $231.68 factors in seasonal strength, further AI advancements, and potential new product cycles.",
            },
          ],
          analystContext:
            "Analyst Consensus: Wall Street analysts currently set a 12-month average target of $229.61 (+10.98%) and a 30-day consensus target of $251.88 (+21.76%), suggesting room for further upside in the medium term.",
          analystTargets: [
            { label: "12-month avg.", value: "$229.61 (+10.98%)" },
            { label: "30-day consensus", value: "$251.88 (+21.76%)" },
          ],
          recentPerformance:
            "Recent Performance: Over the past month, AAPL has risen 5.34% and is up 9.26% over the last year, indicating a rebound from earlier volatility.",
          marketDrivers:
            "Market Drivers: Strong iPhone sales, increasing AI integration, and robust growth in emerging markets (notably India) are key factors supporting the bullish outlook.",
        })
      } else {
        // fallback to old behavior for other symbols
        const mockPredictions = {
          MSFT: {
            symbol: "MSFT",
            currentPrice: 412.76,
            predictedChange: 3.8,
            explanation: "Azure cloud growth and AI investments position Microsoft for continued market leadership.",
            timeframe: "short",
          },
          TSLA: {
            symbol: "TSLA",
            currentPrice: 248.42,
            predictedChange: -2.1,
            explanation:
              "Production challenges and increased competition in the EV market may pressure Tesla's stock in the near term.",
            timeframe: "short",
          },
          AMZN: {
            symbol: "AMZN",
            currentPrice: 178.95,
            predictedChange: 5.3,
            explanation: "AWS growth and retail expansion into new markets suggest strong upside potential.",
            timeframe: "short",
          },
          GOOGL: {
            symbol: "GOOGL",
            currentPrice: 164.32,
            predictedChange: 2.7,
            explanation: "Ad revenue growth and AI advancements in search position Google for continued success.",
            timeframe: "short",
          },
        }
        const fallback = mockPredictions[upperSymbol as keyof typeof mockPredictions]
        if (fallback) {
          setPrediction({
            symbol: fallback.symbol,
            currentPrice: fallback.currentPrice,
            windows: [
              {
                label: fallback.timeframe === "short" ? "Short-Term" : "Long-Term",
                targetPrice: fallback.currentPrice * (1 + fallback.predictedChange / 100),
                percentChange: fallback.predictedChange,
                confidence: "Medium",
                explanation: fallback.explanation,
              },
            ],
          })
        } else {
          // Random prediction for other symbols
          const currentPrice = Math.random() * 500 + 50
          const predictedChange = Math.random() * 10 - 5
          setPrediction({
            symbol: upperSymbol,
            currentPrice,
            windows: [
              {
                label: "Short-Term",
                targetPrice: currentPrice * (1 + predictedChange / 100),
                percentChange: predictedChange,
                confidence: "Medium",
                explanation:
                  "Based on technical analysis and market trends, we expect moderate movement in the coming period.",
              },
            ],
          })
        }
      }
      setIsLoading(false)
    }, 1500)
  }

  return (
    <Card className={cn("overflow-hidden", className)}>
      <CardHeader className="bg-gradient-to-r from-zinc-900 to-zinc-800 pb-4">
        <CardTitle className="flex items-center gap-2">
          <span className="bg-emerald-500 p-1 rounded">
            <TrendingUp className="h-4 w-4 text-black" />
          </span>
          Quick AI Prediction
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
                onChange={(e) => setSymbol(e.target.value)}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-md py-2 pl-9 pr-4 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
              />
            </div>
            <button
              type="submit"
              disabled={!symbol || isLoading}
              className="bg-emerald-500 hover:bg-emerald-600 text-black font-medium rounded-md px-4 py-2 text-sm disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isLoading ? (
                <div className="h-5 w-5 border-2 border-black border-t-transparent rounded-full animate-spin" />
              ) : (
                "Predict"
              )}
            </button>
          </div>
        </form>

        {prediction && prediction.symbol === "AAPL" ? (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-zinc-900 rounded-lg p-4 border border-zinc-800"
          >
            <div className="mb-3">
              <h3 className="text-lg font-bold">AAPL</h3>
              <div className="text-sm text-zinc-400 mb-2">Current Price: ${prediction.currentPrice.toFixed(2)}</div>
              {/* Timeframe Selector */}
              <div className="flex justify-center gap-2 mb-4">
                {prediction.windows.map((w, i) => (
                  <button
                    key={w.label}
                    type="button"
                    onClick={() => setSelectedWindow(i)}
                    className={cn(
                      "px-4 py-1 rounded-full border text-xs font-semibold transition-colors",
                      selectedWindow === i
                        ? "bg-emerald-500 text-black border-emerald-500 shadow"
                        : "bg-zinc-800 text-zinc-300 border-zinc-700 hover:bg-zinc-700"
                    )}
                  >
                    {w.label}
                  </button>
                ))}
              </div>
              {/* Selected Prediction Window */}
              <div className="flex flex-col items-center gap-2 mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-zinc-400">Target:</span>
                  <span className="text-lg font-bold">${prediction.windows[selectedWindow].targetPrice.toFixed(2)}</span>
                  <ArrowRight className="h-4 w-4 text-zinc-500" />
                  <span className={cn(
                    "text-lg font-bold",
                    prediction.windows[selectedWindow].percentChange >= 0 ? "text-emerald-500" : "text-red-500"
                  )}>
                    {prediction.windows[selectedWindow].percentChange >= 0 ? "+" : ""}
                    {prediction.windows[selectedWindow].percentChange.toFixed(1)}%
                  </span>
                </div>
                <div className="text-xs text-zinc-400">AI Confidence: <span className="text-white">{prediction.windows[selectedWindow].confidence}</span></div>
              </div>
              <div className="text-sm text-zinc-300 text-center mb-2">{prediction.windows[selectedWindow].explanation}</div>
            </div>
            {prediction.analystContext && (
              <div className="mt-4 border-t border-zinc-800 pt-3">
                <div className="text-xs text-zinc-400 font-semibold mb-1">Analyst and AI Context</div>
                <div className="text-sm text-zinc-300 mb-1">{prediction.analystContext}</div>
                {prediction.analystTargets && (
                  <div className="flex gap-4 text-xs text-zinc-400 mb-1">
                    {prediction.analystTargets.map((t, i) => (
                      <div key={i}><span className="font-semibold text-white">{t.label}:</span> {t.value}</div>
                    ))}
                  </div>
                )}
                {prediction.recentPerformance && (
                  <div className="text-sm text-zinc-300 mb-1">{prediction.recentPerformance}</div>
                )}
                {prediction.marketDrivers && (
                  <div className="text-sm text-zinc-300">{prediction.marketDrivers}</div>
                )}
              </div>
            )}
            <div className="mt-3 pt-3 border-t border-zinc-800 flex justify-between items-center">
              <div className="text-xs text-zinc-400">
                AI Confidence: <span className="text-white">Medium</span>
              </div>
              <a
                href={`/stocks/${prediction.symbol}`}
                className="text-xs text-emerald-500 hover:text-emerald-400 flex items-center gap-1"
              >
                View Details <ExternalLink className="h-3 w-3" />
              </a>
            </div>
          </motion.div>
        ) : prediction ? (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-zinc-900 rounded-lg p-4 border border-zinc-800"
          >
            <div className="mb-3">
              <h3 className="text-lg font-bold">{prediction.symbol}</h3>
              <div className="text-sm text-zinc-400 mb-2">Current Price: ${prediction.currentPrice.toFixed(2)}</div>
              <div className="mb-3">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-zinc-400">Target:</span>
                  <span className="text-sm">${prediction.windows[0].targetPrice.toFixed(2)}</span>
                  <ArrowRight className="h-3.5 w-3.5 text-zinc-500" />
                  <span className={cn("text-sm", prediction.windows[0].percentChange >= 0 ? "text-emerald-500" : "text-red-500")}>{prediction.windows[0].percentChange >= 0 ? "+" : ""}{prediction.windows[0].percentChange.toFixed(1)}%</span>
                </div>
                <div className="text-sm text-zinc-300">{prediction.windows[0].explanation}</div>
              </div>
            </div>
            <div className="mt-3 pt-3 border-t border-zinc-800 flex justify-between items-center">
              <div className="text-xs text-zinc-400">
                AI Confidence: <span className="text-white">Medium</span>
              </div>
              <a
                href={`/stocks/${prediction.symbol}`}
                className="text-xs text-emerald-500 hover:text-emerald-400 flex items-center gap-1"
              >
                View Details <ExternalLink className="h-3 w-3" />
              </a>
            </div>
          </motion.div>
        ) : null}
      </CardContent>
    </Card>
  )
}
