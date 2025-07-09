"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { TrendingUp, TrendingDown, Brain, Calendar, Target, Sparkles } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

export default function Predictions() {
  const [selectedTimeframe, setSelectedTimeframe] = useState<'1_day' | '7_day' | '30_day'>('7_day')
  const [stocksWithExplanations, setStocksWithExplanations] = useState<string[]>([])
  const [batchStatus, setBatchStatus] = useState<any>(null)
  const [isGeneratingBatch, setIsGeneratingBatch] = useState(false)
  
  // Load batch status on component mount
  useEffect(() => {
    loadBatchStatus()
  }, [])

  const loadBatchStatus = async () => {
    try {
      const { getBatchExplanationStatus, getAvailableStocksWithExplanations } = await import('@/lib/api')
      const status = await getBatchExplanationStatus()
      const available = await getAvailableStocksWithExplanations()
      
      setBatchStatus(status)
      setStocksWithExplanations(available)
    } catch (error) {
      console.error('Error loading batch status:', error)
    }
  }

  const handleGenerateBatch = async () => {
    setIsGeneratingBatch(true)
    try {
      const { generateBatchAIExplanations } = await import('@/lib/api')
      const result = await generateBatchAIExplanations()
      
      if (result) {
        alert(`Batch generation started! Processing ${result.total_tickers} tickers...`)
        // Reload status after a delay
        setTimeout(loadBatchStatus, 10000)
      }
    } catch (error) {
      console.error('Error generating batch explanations:', error)
      alert('Error starting batch generation. Please try again.')
    } finally {
      setIsGeneratingBatch(false)
    }
  }
  
  const mockPredictions = [
    {
      symbol: 'AAPL',
      name: 'Apple Inc.',
      predictions: {
        '1_day': { predicted_price: 191.45, confidence: 0.85, price_change: 1.2 },
        '7_day': { predicted_price: 195.30, confidence: 0.78, price_change: 3.4 },
        '30_day': { predicted_price: 203.75, confidence: 0.72, price_change: 8.9 }
      }
    },
    {
      symbol: 'MSFT',
      name: 'Microsoft Corporation',
      predictions: {
        '1_day': { predicted_price: 378.20, confidence: 0.82, price_change: -0.5 },
        '7_day': { predicted_price: 385.50, confidence: 0.75, price_change: 1.8 },
        '30_day': { predicted_price: 398.40, confidence: 0.68, price_change: 5.2 }
      }
    },
    {
      symbol: 'NVDA',
      name: 'NVIDIA Corporation',
      predictions: {
        '1_day': { predicted_price: 875.60, confidence: 0.79, price_change: 2.1 },
        '7_day': { predicted_price: 892.30, confidence: 0.71, price_change: 4.2 },
        '30_day': { predicted_price: 925.80, confidence: 0.65, price_change: 8.1 }
      }
    }
  ]

  const getTimeframeLabel = (timeframe: string) => {
    switch(timeframe) {
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
    <div className="container mx-auto px-4 py-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center gap-3 mb-2">
          <Brain className="h-8 w-8 text-purple-500" />
          <h1 className="text-3xl font-bold">AI Stock Predictions</h1>
          <Badge variant="outline" className="ml-2">
            <Sparkles className="h-3 w-3 mr-1" />
            Enhanced
          </Badge>
        </div>
        <p className="text-zinc-400">
          Advanced AI-powered predictions with comprehensive market analysis
        </p>
      </motion.div>

      {/* AI Batch Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="mb-6"
      >
        <Card className="bg-gradient-to-br from-emerald-500/10 to-blue-500/10 border-emerald-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-emerald-400" />
                <span className="font-medium">AI Explanation Status</span>
                {batchStatus && (
                  <Badge variant="outline" className="text-emerald-400 border-emerald-400">
                    {batchStatus.coverage_percentage}% Coverage
                  </Badge>
                )}
              </div>
              <Button
                onClick={handleGenerateBatch}
                disabled={isGeneratingBatch}
                size="sm"
                className="bg-emerald-600 hover:bg-emerald-700"
              >
                {isGeneratingBatch ? (
                  <>
                    <span className="animate-spin mr-2">⚡</span>
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4 mr-2" />
                    Generate All
                  </>
                )}
              </Button>
            </div>
            
            {batchStatus && (
              <div className="flex items-center gap-4 text-sm text-zinc-300">
                <span>✅ {batchStatus.with_explanations} stocks have AI explanations</span>
                <span>⏳ {batchStatus.without_explanations} pending</span>
                <span>📊 Total: {batchStatus.total_tickers} tickers</span>
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Timeframe Selector */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="mb-6"
      >
        <div className="flex items-center gap-2 mb-4">
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

      {/* Predictions Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {mockPredictions.map((stock, index) => {
          const hasAIExplanation = stocksWithExplanations.includes(stock.symbol)
          
          // You can enhance this to show more stored data if available
          const enhancedMetrics = hasAIExplanation
          
          return (
            <motion.div
              key={stock.symbol}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 + index * 0.1 }}
            >
              <Card className="hover:shadow-lg transition-shadow cursor-pointer group">
                <CardHeader className="pb-3">
                  <div className="flex justify-between items-start">
                    <div>
                      <CardTitle className="flex items-center gap-2">
                        {stock.symbol}
                        <Target className="h-4 w-4 text-blue-500" />
                        {hasAIExplanation && (
                          <div className="flex items-center gap-1">
                            <Sparkles className="h-3 w-3 text-emerald-400" />
                            <span className="text-xs text-emerald-400 font-medium">AI</span>
                          </div>
                        )}
                      </CardTitle>
                      <p className="text-sm text-zinc-400 mt-1">{stock.name}</p>
                    </div>
                    <div className="flex flex-col items-end gap-1">
                      <Badge 
                        variant="outline" 
                        className={getConfidenceColor(stock.predictions[selectedTimeframe].confidence)}
                      >
                        {(stock.predictions[selectedTimeframe].confidence * 100).toFixed(0)}% confidence
                      </Badge>
                      {hasAIExplanation && (
                        <Badge variant="outline" className="text-emerald-400 border-emerald-400 text-xs">
                          Enhanced Analysis
                        </Badge>
                      )}
                    </div>
                  </div>
                </CardHeader>
              
              <CardContent>
                <div className="space-y-4">
                  {/* Current Timeframe Prediction */}
                  <div className="bg-gradient-to-br from-purple-500/10 to-purple-600/5 rounded-lg p-4 border border-purple-500/20">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium text-purple-400">
                        {getTimeframeLabel(selectedTimeframe)} Target
                      </span>
                      <span className="text-xs text-zinc-400">AI Prediction</span>
                    </div>
                    
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-2xl font-bold">
                        ${stock.predictions[selectedTimeframe].predicted_price.toFixed(2)}
                      </span>
                      <div className={`flex items-center gap-1 text-sm font-medium px-2 py-1 rounded-md ${
                        stock.predictions[selectedTimeframe].price_change >= 0
                          ? "bg-emerald-500/20 text-emerald-400"
                          : "bg-red-500/20 text-red-400"
                      }`}>
                        {stock.predictions[selectedTimeframe].price_change >= 0 ? (
                          <TrendingUp className="h-3 w-3" />
                        ) : (
                          <TrendingDown className="h-3 w-3" />
                        )}
                        {stock.predictions[selectedTimeframe].price_change >= 0 ? '+' : ''}
                        {stock.predictions[selectedTimeframe].price_change.toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  {/* All Timeframes Overview */}
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium text-zinc-300">All Predictions</h4>
                    {(['1_day', '7_day', '30_day'] as const).map((timeframe) => (
                      <div key={timeframe} className="flex justify-between items-center text-xs">
                        <span className="text-zinc-400">{getTimeframeLabel(timeframe)}</span>
                        <div className="flex items-center gap-2">
                          <span className="font-medium">
                            ${stock.predictions[timeframe].predicted_price.toFixed(2)}
                          </span>
                          <span className={`${
                            stock.predictions[timeframe].price_change >= 0 
                              ? 'text-emerald-500' 
                              : 'text-red-500'
                          }`}>
                            {stock.predictions[timeframe].price_change >= 0 ? '+' : ''}
                            {stock.predictions[timeframe].price_change.toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* View Details Button */}
                  <Button 
                    size="sm" 
                    variant="outline" 
                    className="w-full group-hover:bg-purple-600 group-hover:text-white transition-colors"
                    onClick={() => window.location.href = `/stock/${stock.symbol}`}
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

      {/* Feature Highlight */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="mt-12"
      >
        <Card className="bg-gradient-to-br from-purple-500/10 to-blue-500/10 border-purple-500/20">
          <CardContent className="p-6">
            <div className="flex items-center gap-3 mb-4">
              <Sparkles className="h-6 w-6 text-purple-400" />
              <h3 className="text-xl font-semibold">Enhanced AI Analysis</h3>
            </div>
            <p className="text-zinc-300 mb-4">
              Our advanced AI system now provides comprehensive explanations using:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-emerald-500 rounded-full"></div>
                <span className="text-sm">Real-time sentiment analysis</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                <span className="text-sm">Technical indicators</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                <span className="text-sm">Fundamental analysis</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-amber-500 rounded-full"></div>
                <span className="text-sm">Social media insights</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-pink-500 rounded-full"></div>
                <span className="text-sm">News impact analysis</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-cyan-500 rounded-full"></div>
                <span className="text-sm">Risk assessment</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  )
}
