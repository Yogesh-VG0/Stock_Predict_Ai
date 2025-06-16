"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Search, TrendingUp, BarChart3, CheckCircle, AlertCircle, Clock, Newspaper } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import TradingViewAdvancedChart from "@/components/tradingview/trading-view-advanced-chart"

export default function PredictionsPage() {
  const [isLoading, setIsLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedStock, setSelectedStock] = useState<string>("AAPL")
  const [predictionData, setPredictionData] = useState<any>(null)

  useEffect(() => {
    // Simulate API call
    setTimeout(() => {
      const mockPredictions = {
        AAPL: {
          symbol: "AAPL",
          name: "Apple Inc.",
          currentPrice: 187.68,
          shortTerm: {
            predictedPrice: 195.42,
            predictedChange: 4.12,
            confidence: "high",
            timeframe: "30 days",
          },
          longTerm: {
            predictedPrice: 210.25,
            predictedChange: 12.02,
            confidence: "medium",
            timeframe: "6 months",
          },
          accuracy: {
            overall: 78,
            recent: 82,
          },
          news: [
            {
              title: "Apple Unveils New AI Features for iPhone and iPad",
              date: "2023-06-05",
              source: "TechCrunch",
              sentiment: "positive",
            },
            {
              title: "Apple's Services Revenue Hits All-Time High",
              date: "2023-05-28",
              source: "Bloomberg",
              sentiment: "positive",
            },
            {
              title: "EU Fines Apple €500M Over Music Streaming Rules",
              date: "2023-05-15",
              source: "Financial Times",
              sentiment: "negative",
            },
          ],
        },
        MSFT: {
          symbol: "MSFT",
          name: "Microsoft Corporation",
          currentPrice: 412.76,
          shortTerm: {
            predictedPrice: 428.45,
            predictedChange: 3.8,
            confidence: "high",
            timeframe: "30 days",
          },
          longTerm: {
            predictedPrice: 455.2,
            predictedChange: 10.28,
            confidence: "high",
            timeframe: "6 months",
          },
          accuracy: {
            overall: 81,
            recent: 85,
          },
          news: [
            {
              title: "Microsoft Cloud Revenue Surges in Q2",
              date: "2023-06-02",
              source: "CNBC",
              sentiment: "positive",
            },
            {
              title: "Microsoft Expands AI Capabilities Across Product Line",
              date: "2023-05-25",
              source: "TechCrunch",
              sentiment: "positive",
            },
            {
              title: "Microsoft Faces Antitrust Scrutiny Over Cloud Practices",
              date: "2023-05-10",
              source: "Wall Street Journal",
              sentiment: "negative",
            },
          ],
        },
        TSLA: {
          symbol: "TSLA",
          name: "Tesla, Inc.",
          currentPrice: 248.42,
          shortTerm: {
            predictedPrice: 243.2,
            predictedChange: -2.1,
            confidence: "medium",
            timeframe: "30 days",
          },
          longTerm: {
            predictedPrice: 275.65,
            predictedChange: 10.96,
            confidence: "low",
            timeframe: "6 months",
          },
          accuracy: {
            overall: 72,
            recent: 68,
          },
          news: [
            {
              title: "Tesla Cybertruck Production Delayed Again",
              date: "2023-06-01",
              source: "Reuters",
              sentiment: "negative",
            },
            {
              title: "Tesla's China Sales Hit Record in May",
              date: "2023-05-22",
              source: "Bloomberg",
              sentiment: "positive",
            },
            {
              title: "Musk Announces New Tesla AI Day",
              date: "2023-05-15",
              source: "Electrek",
              sentiment: "neutral",
            },
          ],
        },
      }

      setPredictionData(mockPredictions)
      setIsLoading(false)
    }, 1500)
  }, [])

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (!searchQuery) return

    const upperQuery = searchQuery.toUpperCase()
    if (predictionData && upperQuery in predictionData) {
      setSelectedStock(upperQuery)
    } else {
      // Handle unknown stock
      alert(`No prediction data available for ${upperQuery}`)
    }
  }

  const getConfidenceColor = (confidence: string) => {
    switch (confidence) {
      case "high":
        return "text-emerald-500"
      case "medium":
        return "text-amber-500"
      case "low":
        return "text-red-500"
      default:
        return "text-zinc-400"
    }
  }

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case "positive":
        return <CheckCircle className="h-4 w-4 text-emerald-500" />
      case "negative":
        return <AlertCircle className="h-4 w-4 text-red-500" />
      default:
        return <Clock className="h-4 w-4 text-amber-500" />
    }
  }

  return (
    <div className="space-y-6">
      <motion.h1 initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="text-2xl font-bold">
        AI Stock Predictions
      </motion.h1>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.1 }}
        className="bg-zinc-900 rounded-lg p-4 border border-zinc-800"
      >
        <form onSubmit={handleSearch} className="flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-2.5 h-4 w-4 text-zinc-400" />
            <input
              type="text"
              placeholder="Enter stock symbol (e.g. AAPL, MSFT, TSLA)"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-zinc-800 border border-zinc-700 rounded-md py-2 pl-9 pr-4 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
            />
          </div>
          <button
            type="submit"
            disabled={!searchQuery}
            className="bg-emerald-500 hover:bg-emerald-600 text-black font-medium rounded-md px-4 py-2 text-sm disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Search
          </button>
        </form>

        {!isLoading && predictionData && (
          <div className="flex items-center gap-3 mt-4 overflow-x-auto pb-2">
            {Object.keys(predictionData).map((symbol) => (
              <button
                key={symbol}
                onClick={() => setSelectedStock(symbol)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                  selectedStock === symbol ? "bg-emerald-500 text-black" : "bg-zinc-800 text-white hover:bg-zinc-700"
                }`}
              >
                {symbol}
              </button>
            ))}
          </div>
        )}
      </motion.div>

      {isLoading ? (
        <div className="space-y-6">
          <div className="h-[500px] bg-zinc-900 animate-pulse rounded-lg"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="h-64 bg-zinc-900 animate-pulse rounded-lg"></div>
            <div className="h-64 bg-zinc-900 animate-pulse rounded-lg"></div>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          {predictionData && selectedStock && (
            <>
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
                <TradingViewAdvancedChart symbol={selectedStock} height={500} />
              </motion.div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                  className="space-y-6"
                >
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="flex items-center gap-2">
                        <TrendingUp className="h-5 w-5 text-emerald-500" />
                        AI Prediction
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="flex items-center justify-between mb-4">
                        <div>
                          <h3 className="text-lg font-bold">{predictionData[selectedStock].name}</h3>
                          <p className="text-sm text-zinc-400">{predictionData[selectedStock].symbol}</p>
                        </div>
                        <div className="text-xl font-bold">
                          ${predictionData[selectedStock].currentPrice.toFixed(2)}
                        </div>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <div className="bg-zinc-900 rounded-lg p-4 border border-zinc-800">
                          <div className="flex justify-between items-center mb-2">
                            <div className="text-sm font-medium">Short-Term</div>
                            <div className="text-xs text-zinc-400">
                              {predictionData[selectedStock].shortTerm.timeframe}
                            </div>
                          </div>

                          <div className="flex items-center gap-2 mb-2">
                            <div className="text-lg font-bold">
                              ${predictionData[selectedStock].shortTerm.predictedPrice.toFixed(2)}
                            </div>
                            <div
                              className={`text-sm font-medium px-2 py-0.5 rounded-md ${
                                predictionData[selectedStock].shortTerm.predictedChange >= 0
                                  ? "bg-emerald-500/10 text-emerald-500"
                                  : "bg-red-500/10 text-red-500"
                              }`}
                            >
                              {predictionData[selectedStock].shortTerm.predictedChange >= 0 ? "+" : ""}
                              {predictionData[selectedStock].shortTerm.predictedChange.toFixed(2)}%
                            </div>
                          </div>

                          <div className="flex items-center gap-1.5">
                            <span className="text-xs text-zinc-400">Confidence:</span>
                            <span
                              className={`text-xs font-medium capitalize ${getConfidenceColor(predictionData[selectedStock].shortTerm.confidence)}`}
                            >
                              {predictionData[selectedStock].shortTerm.confidence}
                            </span>
                          </div>
                        </div>

                        <div className="bg-zinc-900 rounded-lg p-4 border border-zinc-800">
                          <div className="flex justify-between items-center mb-2">
                            <div className="text-sm font-medium">Long-Term</div>
                            <div className="text-xs text-zinc-400">
                              {predictionData[selectedStock].longTerm.timeframe}
                            </div>
                          </div>

                          <div className="flex items-center gap-2 mb-2">
                            <div className="text-lg font-bold">
                              ${predictionData[selectedStock].longTerm.predictedPrice.toFixed(2)}
                            </div>
                            <div
                              className={`text-sm font-medium px-2 py-0.5 rounded-md ${
                                predictionData[selectedStock].longTerm.predictedChange >= 0
                                  ? "bg-emerald-500/10 text-emerald-500"
                                  : "bg-red-500/10 text-red-500"
                              }`}
                            >
                              {predictionData[selectedStock].longTerm.predictedChange >= 0 ? "+" : ""}
                              {predictionData[selectedStock].longTerm.predictedChange.toFixed(2)}%
                            </div>
                          </div>

                          <div className="flex items-center gap-1.5">
                            <span className="text-xs text-zinc-400">Confidence:</span>
                            <span
                              className={`text-xs font-medium capitalize ${getConfidenceColor(predictionData[selectedStock].longTerm.confidence)}`}
                            >
                              {predictionData[selectedStock].longTerm.confidence}
                            </span>
                          </div>
                        </div>
                      </div>

                      <div className="mt-4 pt-4 border-t border-zinc-800">
                        <div className="flex justify-between items-center mb-2">
                          <div className="text-sm font-medium">Historical Prediction Accuracy</div>
                          <div className="text-xs text-zinc-400">Last 12 months</div>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <div className="text-xs text-zinc-400 mb-1">Overall</div>
                            <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-emerald-500 rounded-full"
                                style={{ width: `${predictionData[selectedStock].accuracy.overall}%` }}
                              ></div>
                            </div>
                            <div className="text-sm font-medium mt-1">
                              {predictionData[selectedStock].accuracy.overall}%
                            </div>
                          </div>

                          <div>
                            <div className="text-xs text-zinc-400 mb-1">Recent (3 months)</div>
                            <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-emerald-500 rounded-full"
                                style={{ width: `${predictionData[selectedStock].accuracy.recent}%` }}
                              ></div>
                            </div>
                            <div className="text-sm font-medium mt-1">
                              {predictionData[selectedStock].accuracy.recent}%
                            </div>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="flex items-center gap-2">
                        <BarChart3 className="h-5 w-5 text-purple-500" />
                        Comparison Tool
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="flex items-center gap-2 mb-4">
                        <div className="relative flex-1">
                          <Search className="absolute left-3 top-2.5 h-4 w-4 text-zinc-400" />
                          <input
                            type="text"
                            placeholder="Compare with another stock..."
                            className="w-full bg-zinc-800 border border-zinc-700 rounded-md py-2 pl-9 pr-4 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
                          />
                        </div>
                        <button className="bg-zinc-800 hover:bg-zinc-700 text-white rounded-md px-3 py-2 text-sm transition-colors">
                          Compare
                        </button>
                      </div>

                      <div className="bg-zinc-900 rounded-lg p-4 border border-zinc-800 flex items-center justify-center h-32">
                        <p className="text-sm text-zinc-400">Select another stock to compare predictions</p>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>

                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Newspaper className="h-5 w-5 text-amber-500" />
                        Latest News Affecting Prediction
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {predictionData[selectedStock].news.map((item: any, index: number) => (
                          <div
                            key={index}
                            className="bg-zinc-900/50 rounded-md p-3 border border-zinc-800/50 hover:bg-zinc-900 transition-colors"
                          >
                            <div className="flex justify-between items-start mb-1">
                              <h3 className="text-sm font-medium">{item.title}</h3>
                              {getSentimentIcon(item.sentiment)}
                            </div>
                            <div className="flex items-center gap-2 text-xs text-zinc-400">
                              <span>{item.source}</span>
                              <span>•</span>
                              <span>
                                {new Date(item.date).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>

                      <div className="mt-4 pt-4 border-t border-zinc-800">
                        <h3 className="text-sm font-medium mb-3">AI Sentiment Analysis</h3>

                        <div className="flex items-center gap-2 mb-3">
                          <div className="h-2 flex-1 bg-zinc-800 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-red-500 via-amber-500 to-emerald-500 rounded-full"
                              style={{ width: "70%" }}
                            ></div>
                          </div>
                          <div className="text-xs font-medium">70%</div>
                        </div>

                        <div className="flex justify-between text-xs text-zinc-400">
                          <div>Bearish</div>
                          <div>Neutral</div>
                          <div>Bullish</div>
                        </div>

                        <p className="text-sm mt-3">
                          News sentiment for {selectedStock} is moderately bullish, with positive developments in
                          product launches and revenue growth outweighing regulatory concerns.
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
