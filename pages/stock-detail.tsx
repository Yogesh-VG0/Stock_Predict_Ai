"use client"

import { useState, useEffect, Suspense, lazy, memo } from "react"
import { useParams } from "react-router-dom"
import { motion } from "framer-motion"
import {
  Star,
  TrendingUp,
  TrendingDown,
  Info,
  DollarSign,
  BarChart3,
  CheckCircle,
  XCircle,
  Newspaper,
  Search,
  AlertCircle,
  Clock,
  Zap,
  Wifi,
  WifiOff,
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"

// Lazy load heavy components
const TradingViewAdvancedChart = lazy(() => import("@/components/tradingview/trading-view-advanced-chart"))
const TradingViewCompanyProfile = lazy(() => import("@/components/tradingview/TradingViewCompanyProfile"))
const AIExplanationWidget = lazy(() => import("@/components/market/AIExplanationWidget"))
const TechnicalIndicators = lazy(() => import("@/components/market/TechnicalIndicators"))

// Chart loading skeleton
const ChartSkeleton = memo(() => (
  <div className="w-full h-[500px] bg-zinc-900/50 rounded-lg animate-pulse flex items-center justify-center">
    <div className="text-zinc-600">Loading chart...</div>
  </div>
))
ChartSkeleton.displayName = "ChartSkeleton"
import {
  getStockDetails,
  StockDetails,
  getPredictions,
  getStockPrice,
  getSymbolFromCompanyName,
  PredictionTimeframes,
  getComprehensiveAIExplanation,
} from "@/lib/api"
import { useToast } from "@/hooks/use-toast"
import { useWebSocket, useStockPrice } from "@/hooks/use-websocket-context"
import { getCachedData, setCachedData } from "@/hooks/use-prefetch"

type StockDetailProps = {}

export default function StockDetail({ }: StockDetailProps) {
  const { symbol: urlSymbol } = useParams<{ symbol: string }>()
  const [isLoading, setIsLoading] = useState(true)
  const [stockData, setStockData] = useState<StockDetails | null>(null)
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedStock, setSelectedStock] = useState<string>(urlSymbol || "AAPL")
  const [predictionData, setPredictionData] = useState<PredictionTimeframes | null>(null)
  const [stockNews, setStockNews] = useState<any[]>([])
  const [currentPrice, setCurrentPrice] = useState<number>(0)
  const [isUsingRealData, setIsUsingRealData] = useState(false)
  const [isTrackedStock, setIsTrackedStock] = useState(true)
  // Top tracked stocks shown as quick-access buttons
  const [trackedStocks] = useState([
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX",
    "JPM", "V", "JNJ", "WMT", "PG", "UNH", "HD", "MA", "BAC", "XOM", "LLY", "ABBV"
  ])
  const { toast } = useToast();
  const [isFollowed, setIsFollowed] = useState(false);
  const userId = 'default';

  // Use centralized WebSocket service
  const { isConnected: isWebSocketConnected } = useWebSocket()
  const currentStockPrice = useStockPrice(selectedStock)
  const [isClient, setIsClient] = useState(false)

  // Ensure we're on the client side to prevent hydration mismatches
  useEffect(() => {
    setIsClient(true)
  }, [])

  useEffect(() => {
    if (urlSymbol) {
      setSelectedStock(urlSymbol)
    }
  }, [urlSymbol])

  useEffect(() => {
    loadStockData(selectedStock)
  }, [selectedStock])

  // Update current price when real-time data is available
  useEffect(() => {
    if (currentStockPrice) {
      setCurrentPrice(currentStockPrice.price)

      // Update prediction data with new real-time price for dynamic percentage calculation
      setPredictionData(prev => {
        if (!prev) return null;

        const updated = { ...prev };

        if (updated['1_day']) {
          updated['1_day'] = { ...updated['1_day'], current_price: currentStockPrice.price };
        }
        if (updated.next_day) {
          updated.next_day = { ...updated.next_day, current_price: currentStockPrice.price };
        }
        if (updated['7_day']) {
          updated['7_day'] = { ...updated['7_day'], current_price: currentStockPrice.price };
        }
        if (updated['30_day']) {
          updated['30_day'] = { ...updated['30_day'], current_price: currentStockPrice.price };
        }

        return updated;
      })
    }
  }, [currentStockPrice?.price])

  // Check if stock is in watchlist on load or when selectedStock changes
  useEffect(() => {
    async function checkWatchlist() {
      try {
        const { getWatchlist } = await import("@/lib/api");
        const data = await getWatchlist(userId);
        if (data && data.watchlist) {
          setIsFollowed(data.watchlist.some((item: any) => item.symbol === selectedStock));
        }
      } catch (error) {
        setIsFollowed(false);
      }
    }
    checkWatchlist();
  }, [selectedStock]);



  const loadStockData = async (symbol: string) => {
    setIsLoading(true)
    try {
      // Check prefetch cache first for instant loading
      const cachedDetails = getCachedData<StockDetails>(`stock-details-${symbol}`)

      // Load stock details (from cache or API)
      const details = cachedDetails || await getStockDetails(symbol)
      if (details) {
        // Cache for future use if fetched from API
        if (!cachedDetails) setCachedData(`stock-details-${symbol}`, details)

        // Track whether this stock has ML predictions
        const tracked = (details as any).isTracked !== false
        setIsTrackedStock(tracked)

        // Use real price from backend if available, otherwise fallback to details or mock
        const finalPrice = details.price || currentPrice || 0
        const stockDataWithPrice = {
          ...details,
          price: finalPrice,
          change: details.change !== undefined ? details.change : 0,
          changePercent: details.changePercent !== undefined ? details.changePercent : 0
        }
        setStockData(stockDataWithPrice)
        if (details.price) setCurrentPrice(details.price)
      }

      // Load prediction data from the same source as AIExplanationWidget
      let aiExplanation = null
      let priceData = null

      try {
        // Check prefetch cache first, then fall back to API
        const cachedExplanation = getCachedData<any>(`explanation-${symbol}`)
        aiExplanation = cachedExplanation || await getComprehensiveAIExplanation(symbol)
        if (!cachedExplanation && aiExplanation) setCachedData(`explanation-${symbol}`, aiExplanation)
        console.log(`✅ Loaded AI explanation for ${symbol}:`, aiExplanation ? (cachedExplanation ? 'Prefetch cache' : 'API') : 'No stored data')
      } catch (error) {
        console.log(`No AI explanation available for ${symbol}, will use enhanced mock`)
        aiExplanation = null
      }

      try {
        priceData = await getStockPrice(symbol)
        if (priceData) {
          setCurrentPrice(priceData.price)
        }
      } catch (error) {
        console.log(`Error fetching price for ${symbol}:`, error)
        priceData = null
      }

      // Fetch news
      try {
        await fetchStockNews(symbol)
      } catch (error) {
        console.log(`Error fetching news for ${symbol}:`, error)
        setStockNews([])
      }

      // Set prediction data - use real data from AI explanation if available
      if (aiExplanation && aiExplanation.prediction_summary && (aiExplanation.prediction_summary['1_day'] || aiExplanation.prediction_summary.next_day)) {
        // Transform AI explanation prediction data to match the expected format
        const realPredictionData: PredictionTimeframes = {
          '1_day': {
            predicted_price: aiExplanation.prediction_summary['1_day']?.predicted_price || aiExplanation.prediction_summary.next_day?.predicted_price || 0,
            predicted_change: aiExplanation.prediction_summary['1_day']?.price_change || aiExplanation.prediction_summary.next_day?.price_change || 0,
            current_price: priceData?.price || 0,
            confidence: aiExplanation.prediction_summary['1_day']?.confidence || aiExplanation.prediction_summary.next_day?.confidence || 0,
          },
          '7_day': {
            predicted_price: aiExplanation.prediction_summary['7_day']?.predicted_price || 0,
            predicted_change: aiExplanation.prediction_summary['7_day']?.price_change || 0,
            current_price: priceData?.price || 0,
            confidence: aiExplanation.prediction_summary['7_day']?.confidence || 0,
          },
          '30_day': {
            predicted_price: aiExplanation.prediction_summary['30_day']?.predicted_price || 0,
            predicted_change: aiExplanation.prediction_summary['30_day']?.price_change || 0,
            current_price: priceData?.price || 0,
            confidence: aiExplanation.prediction_summary['30_day']?.confidence || 0,
          }
        }
        console.log(`✅ Using real prediction data for ${symbol} from MongoDB explanation`)
        setPredictionData(realPredictionData)
        setIsUsingRealData(true)
      } else {
        // fallback: Try to get predictions independently
        console.log(`🔍 No explanation-based predictions, trying independent getPredictions for ${symbol}...`)
        const independentPredictions = await getPredictions(symbol)

        if (independentPredictions && independentPredictions[symbol]) {
          console.log(`✅ Using independent ML predictions for ${symbol}`)
          const mlPreds = independentPredictions[symbol]

          // Map to PredictionTimeframes
          const realPredictionData: PredictionTimeframes = {
            '1_day': {
              predicted_price: mlPreds['1_day']?.predicted_price || mlPreds.next_day?.predicted_price || 0,
              predicted_change: mlPreds['1_day']?.predicted_change || mlPreds.next_day?.predicted_change || 0,
              current_price: priceData?.price || 0,
              confidence: mlPreds['1_day']?.confidence || mlPreds.next_day?.confidence || 0
            },
            '7_day': mlPreds['7_day'] || { predicted_price: 0, predicted_change: 0, current_price: 0, confidence: 0 },
            '30_day': mlPreds['30_day'] || { predicted_price: 0, predicted_change: 0, current_price: 0, confidence: 0 }
          }

          setPredictionData(realPredictionData)
          setIsUsingRealData(true)
        } else {
          console.log(`⚠️ No real prediction data available for ${symbol}`)
          setPredictionData(null)
          setIsUsingRealData(false)
        }
      }
    } catch (error) {
      console.error('Error loading stock data:', error)
      setPredictionData(null)
      setStockNews([])
    } finally {
      setIsLoading(false)
    }
  }

  const fetchStockNews = async (symbol: string) => {
    try {
      const rssRes = await fetch(`/api/news/rss?symbol=${symbol}`)
      const rssData = await rssRes.json()

      if (rssData.data) {
        const rssNews = rssData.data.map((item: any) => ({
          ...item,
          sentiment: item.sentiment || 'neutral',
          provider: 'rss'
        }))

        rssNews.sort((a: any, b: any) => new Date(b.published_at).getTime() - new Date(a.published_at).getTime())
        setStockNews(rssNews.slice(0, 6))
      } else {
        setStockNews([])
      }
    } catch (error) {
      console.error('Error fetching RSS news:', error)
      setStockNews([])
    }
  }



  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (!searchQuery) return

    let symbol = getSymbolFromCompanyName(searchQuery) || searchQuery.toUpperCase()
    // Accept any valid-looking ticker symbol (1-5 uppercase letters)
    if (/^[A-Z]{1,5}$/.test(symbol) || /^[A-Z]+\.[A-Z]$/.test(symbol)) {
      setSelectedStock(symbol)
      setSearchQuery("")
    } else {
      alert(`Invalid symbol: ${searchQuery}. Enter a valid ticker like AAPL, PLTR, TSLA, etc.`)
    }
  }

  // Helper function to get company name from symbol
  const getCompanyName = (symbol: string): string => {
    const companies: Record<string, string> = {
      AAPL: "Apple Inc.",
      MSFT: "Microsoft Corporation",
      GOOGL: "Alphabet Inc.",
      AMZN: "Amazon.com Inc.",
      TSLA: "Tesla, Inc.",
      META: "Meta Platforms, Inc.",
      NVDA: "NVIDIA Corporation",
      JPM: "JPMorgan Chase & Co.",
      V: "Visa Inc.",
      WMT: "Walmart Inc.",
      NFLX: "Netflix, Inc.",
      DIS: "The Walt Disney Company",
    }

    return companies[symbol] || `${symbol} Inc.`
  }

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case "positive":
        return <CheckCircle className="h-4 w-4 text-emerald-500" />
      case "negative":
        return <XCircle className="h-4 w-4 text-red-500" />
      default:
        return <Info className="h-4 w-4 text-amber-500" />
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

  // Follow/unfollow logic
  const handleFollow = async () => {
    try {
      if (!isFollowed) {
        const { addToWatchlist } = await import("@/lib/api");
        const res = await addToWatchlist(userId, selectedStock);
        if (res.success) {
          setIsFollowed(true);
          toast({ title: `Added to Watchlist`, description: `${selectedStock} is now in your watchlist.` });
        } else {
          toast({ title: `Error`, description: res.error || 'Failed to add to watchlist.' });
        }
      } else {
        const { removeFromWatchlist } = await import("@/lib/api");
        const res = await removeFromWatchlist(userId, selectedStock);
        if (res.success) {
          setIsFollowed(false);
          toast({ title: `Removed from Watchlist`, description: `${selectedStock} was removed from your watchlist.` });
        } else {
          toast({ title: `Error`, description: res.error || 'Failed to remove from watchlist.' });
        }
      }
    } catch (error) {
      toast({ title: `Error`, description: 'Failed to update watchlist.' });
    }
  };

  if (isLoading || !stockData) {
    return (
      <div className="flex flex-col gap-6">
        <div className="h-20 bg-zinc-900 animate-pulse rounded-lg"></div>
        <div className="h-[500px] bg-zinc-900 animate-pulse rounded-lg"></div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="h-64 bg-zinc-900 animate-pulse rounded-lg"></div>
          <div className="h-64 bg-zinc-900 animate-pulse rounded-lg"></div>
          <div className="h-64 bg-zinc-900 animate-pulse rounded-lg"></div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-5 md:space-y-6">
      {/* Search Bar */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-zinc-900 rounded-lg p-4 border border-zinc-800"
      >
        <form onSubmit={handleSearch} className="flex gap-2 mb-4">
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

        <div className="flex items-center gap-2 sm:gap-3 overflow-x-auto pb-2 scrollbar-hide -mx-1 px-1">
          {trackedStocks.slice(0, 8).map((symbol) => (
            <button
              key={symbol}
              onClick={() => setSelectedStock(symbol)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors whitespace-nowrap ${selectedStock === symbol ? "bg-emerald-500 text-black" : "bg-zinc-800 text-white hover:bg-zinc-700"
                }`}
            >
              <img
                src={`https://raw.githubusercontent.com/davidepalazzo/ticker-logos/main/ticker_icons/${symbol}.png`}
                alt={symbol}
                className="h-4 w-4 object-contain rounded-sm"
                onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
              />
              {symbol}
            </button>
          ))}
        </div>
      </motion.div>

      {/* Stock Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-gradient-to-r from-zinc-900 to-black p-4 rounded-lg border border-zinc-800"
      >
        <div className="flex items-center gap-4">
          <div className="h-12 w-12 rounded-lg overflow-hidden bg-gradient-to-br from-zinc-700 to-zinc-800 flex items-center justify-center">
            <img
              src={`https://raw.githubusercontent.com/davidepalazzo/ticker-logos/main/ticker_icons/${selectedStock}.png`}
              alt={selectedStock}
              className="h-12 w-12 object-contain"
              onError={(e) => {
                // Fallback to first letter if logo not found
                const target = e.target as HTMLImageElement;
                target.style.display = 'none';
                if (target.parentElement) {
                  target.parentElement.innerHTML = `<span class="text-xl font-bold">${selectedStock.charAt(0)}</span>`;
                }
              }}
            />
          </div>
          <div>
            <h1 className="text-lg sm:text-2xl font-bold flex items-center gap-2 flex-wrap">
              <span className="truncate max-w-[200px] sm:max-w-none">{stockData.name}</span>
              <span className="text-sm sm:text-lg text-zinc-400">({selectedStock})</span>
            </h1>
            <div className="flex items-center gap-2 text-sm">
              <span className="text-zinc-400">{stockData.sector}</span>
              <span className="text-zinc-600">•</span>
              <span className="text-zinc-400">{stockData.industry}</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleFollow}
            className={`px-4 py-2 rounded-md flex items-center gap-2 font-medium transition-colors border focus:outline-none focus:ring-2 focus:ring-emerald-500
              ${isFollowed
                ? 'bg-amber-100 border-amber-300 text-amber-700 hover:bg-amber-200'
                : 'bg-zinc-800 border-zinc-700 text-white hover:bg-zinc-700'}`}
            aria-pressed={isFollowed}
          >
            <Star className={`h-5 w-5 ${isFollowed ? 'fill-amber-400 text-amber-400' : 'text-amber-400'}`} />
            <span className="text-sm font-semibold">
              {isFollowed ? 'Followed' : 'Follow'}
            </span>
          </button>
        </div>
      </motion.div>

      {/* TradingView Chart */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
        <Suspense fallback={<ChartSkeleton />}>
          <div className="h-[350px] sm:h-[500px]">
            <TradingViewAdvancedChart symbol={selectedStock} height={typeof window !== 'undefined' && window.innerWidth < 640 ? 350 : 500} />
          </div>
        </Suspense>
      </motion.div>

      {/* Main Content Grid - Top Row (Company, Predictions, News) */}
      {/* Slightly taller side cards for visual balance, without forcing full-height stretch */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Company Profile (TradingView Widget) */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <Card className="flex flex-col overflow-hidden">
            <CardHeader className="pb-0">
              <CardTitle className="flex items-center gap-2">
                <Info className="h-5 w-5 text-blue-500" />
                Company Profile
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0 pt-2">
              <Suspense fallback={
                <div className="h-[400px] flex items-center justify-center">
                  <div className="text-zinc-500 text-sm">Loading company profile...</div>
                </div>
              }>
                <TradingViewCompanyProfile symbol={selectedStock} height={400} />
              </Suspense>
            </CardContent>
          </Card>
        </motion.div>

        {/* Middle Column - AI Predictions (only for tracked stocks) */}
        {isTrackedStock && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
            <Card className="flex flex-col">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-emerald-500" />
                  AI Predictions
                  {isUsingRealData && (
                    <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-1 rounded-full border border-emerald-500/30">
                      💾 Real Data
                    </span>
                  )}
                  <span className="text-xs text-zinc-500 font-normal ml-1">(alpha vs SPY)</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="flex-1">
              {!predictionData ? (
                <div className="flex flex-col items-center justify-center py-8 text-center">
                  <AlertCircle className="h-10 w-10 text-zinc-600 mb-3" />
                  <p className="text-sm text-zinc-400 font-medium">No predictions available yet</p>
                  <p className="text-xs text-zinc-500 mt-1">Predictions are generated daily by the ML pipeline.</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 gap-4">
                  {/* 1 Day Prediction */}
                  {(() => {
                    const prediction = predictionData['1_day'] || predictionData.next_day
                    if (!prediction) return null;

                    const currentPriceForCalc = currentStockPrice?.price || currentPrice || prediction.current_price || 0
                    const predictedPrice = prediction.predicted_price
                    const calculatedChange = currentPriceForCalc > 0
                      ? ((predictedPrice - currentPriceForCalc) / currentPriceForCalc) * 100
                      : 0

                    return (
                      <div className="bg-gradient-to-br from-emerald-500/10 to-emerald-600/5 rounded-lg p-4 border border-emerald-500/20">
                        <div className="flex justify-between items-center mb-2">
                          <div className="text-sm font-medium text-emerald-400">Next Day</div>
                          <div className="text-xs text-zinc-400">1 Day</div>
                        </div>
                        <div className="flex items-center gap-2 mb-2">
                          <div className="text-lg font-bold">
                            ${predictedPrice.toFixed(2)}
                          </div>
                          <div
                            className={`flex items-center gap-1 text-sm font-medium px-2 py-0.5 rounded-md ${calculatedChange >= 0
                              ? "bg-emerald-500/20 text-emerald-400"
                              : "bg-red-500/20 text-red-400"
                              }`}
                          >
                            {calculatedChange >= 0 ? (
                              <TrendingUp className="h-3 w-3" />
                            ) : (
                              <TrendingDown className="h-3 w-3" />
                            )}
                            <span>
                              {calculatedChange >= 0 ? "+" : ""}{calculatedChange.toFixed(2)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    )
                  })()}

                  {/* 7 Day Prediction */}
                  {(() => {
                    const currentPriceForCalc = currentStockPrice?.price || currentPrice || predictionData['7_day'].current_price || 0
                    const predictedPrice = predictionData['7_day'].predicted_price
                    const calculatedChange = currentPriceForCalc > 0
                      ? ((predictedPrice - currentPriceForCalc) / currentPriceForCalc) * 100
                      : 0

                    return (
                      <div className="bg-gradient-to-br from-blue-500/10 to-blue-600/5 rounded-lg p-4 border border-blue-500/20">
                        <div className="flex justify-between items-center mb-2">
                          <div className="text-sm font-medium text-blue-400">1 Week</div>
                          <div className="text-xs text-zinc-400">7 Days</div>
                        </div>
                        <div className="flex items-center gap-2 mb-2">
                          <div className="text-lg font-bold">
                            ${predictedPrice.toFixed(2)}
                          </div>
                          <div
                            className={`flex items-center gap-1 text-sm font-medium px-2 py-0.5 rounded-md ${calculatedChange >= 0
                              ? "bg-emerald-500/20 text-emerald-400"
                              : "bg-red-500/20 text-red-400"
                              }`}
                          >
                            {calculatedChange >= 0 ? (
                              <TrendingUp className="h-3 w-3" />
                            ) : (
                              <TrendingDown className="h-3 w-3" />
                            )}
                            <span>
                              {calculatedChange >= 0 ? "+" : ""}{calculatedChange.toFixed(2)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    )
                  })()}

                  {/* 30 Day Prediction */}
                  {(() => {
                    const currentPriceForCalc = currentStockPrice?.price || currentPrice || predictionData['30_day'].current_price || 0
                    const predictedPrice = predictionData['30_day'].predicted_price
                    const calculatedChange = currentPriceForCalc > 0
                      ? ((predictedPrice - currentPriceForCalc) / currentPriceForCalc) * 100
                      : 0

                    return (
                      <div className="bg-gradient-to-br from-purple-500/10 to-purple-600/5 rounded-lg p-4 border border-purple-500/20">
                        <div className="flex justify-between items-center mb-2">
                          <div className="text-sm font-medium text-purple-400">1 Month</div>
                          <div className="text-xs text-zinc-400">30 Days</div>
                        </div>
                        <div className="flex items-center gap-2 mb-2">
                          <div className="text-lg font-bold">
                            ${predictedPrice.toFixed(2)}
                          </div>
                          <div
                            className={`flex items-center gap-1 text-sm font-medium px-2 py-0.5 rounded-md ${calculatedChange >= 0
                              ? "bg-emerald-500/20 text-emerald-400"
                              : "bg-red-500/20 text-red-400"
                              }`}
                          >
                            {calculatedChange >= 0 ? (
                              <TrendingUp className="h-3 w-3" />
                            ) : (
                              <TrendingDown className="h-3 w-3" />
                            )}
                            <span>
                              {calculatedChange >= 0 ? "+" : ""}{calculatedChange.toFixed(2)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    )
                  })()}
                </div>
              )}
              </CardContent>
            </Card>
        </motion.div>
        )}

        {/* Right Column - News */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <Card className="flex flex-col md:min-h-[280px]">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Newspaper className="h-5 w-5 text-amber-500" />
                Latest News & Sentiment
              </CardTitle>
            </CardHeader>
            <CardContent>
              {/* Compact news list - show ~2 items then scroll, without stretching the whole card too tall */}
              <div className="space-y-3 max-h-64 overflow-y-auto pr-2">
                {stockNews.map((item: any, index: number) => {
                  const sentiment = item.sentiment || "neutral"

                  return (
                    <div
                      key={`${item.provider}-${item.uuid || item.id || index}`}
                      className="bg-zinc-900/50 rounded-md p-3 border border-zinc-800/50 hover:bg-zinc-900 transition-colors cursor-pointer"
                      onClick={() => item.url && window.open(item.url, '_blank')}
                    >
                      <div className="flex justify-between items-start mb-1">
                        <h3 className="text-sm font-medium pr-2 flex-1 line-clamp-2">{item.title}</h3>
                        <div className="flex items-center gap-1 ml-2">
                          {getSentimentIcon(sentiment)}
                        </div>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-zinc-400 mb-2">
                        <span>
                          {new Date(item.published_at || item.date).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                        </span>
                        <span>•</span>
                        <span>{item.source}</span>
                      </div>
                      {item.snippet && (
                        <p className="text-xs text-zinc-300 line-clamp-2">
                          {item.snippet.length > 120 ? item.snippet.slice(0, 120) + "..." : item.snippet}
                        </p>
                      )}
                    </div>
                  )
                })}
              </div>

              {/* Sentiment Analysis */}
              <div className="mt-3 pt-3 border-t border-zinc-800">
                <h3 className="text-sm font-medium mb-3">News Sentiment</h3>

                {(() => {
                  const positiveSentiment = stockNews.filter(n => n.sentiment === 'positive').length
                  const negativeSentiment = stockNews.filter(n => n.sentiment === 'negative').length
                  const totalSentiment = stockNews.length || 1
                  const sentimentScore = ((positiveSentiment - negativeSentiment) / totalSentiment + 1) * 50

                  return (
                    <>
                      <div className="flex items-center gap-2 mb-2">
                        <div className="h-2 flex-1 bg-zinc-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-red-500 via-amber-500 to-emerald-500 rounded-full"
                            style={{ width: `${sentimentScore}%` }}
                          ></div>
                        </div>
                        <div className="text-xs font-medium">{Math.round(sentimentScore)}%</div>
                      </div>

                      <div className="flex justify-between text-xs text-zinc-400 mb-2">
                        <div>Bearish</div>
                        <div>Neutral</div>
                        <div>Bullish</div>
                      </div>

                      <p className="text-xs text-zinc-300">
                        Sentiment is {
                          sentimentScore > 60 ? 'bullish' :
                            sentimentScore > 40 ? 'neutral' : 'bearish'
                        } with {positiveSentiment} positive, {negativeSentiment} negative articles.
                      </p>
                    </>
                  )
                })()}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Advanced Technical & AI Analysis Header (only for tracked stocks) */}
      {isTrackedStock && (
      <>
      <div className="flex items-center justify-between pt-4">
        <h2 className="text-lg font-semibold text-zinc-200 flex items-center gap-2">
          <BarChart3 className="h-5 w-5 text-emerald-500" />
          Advanced Technical & AI Analysis
        </h2>
      </div>

      {/* Technical Indicators & AI Analysis Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Technical Indicators - Left side */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="lg:col-span-1"
        >
          <Suspense fallback={
            <Card className="p-6">
              <div className="animate-pulse space-y-4">
                <div className="h-6 bg-zinc-800 rounded w-1/2"></div>
                <div className="h-20 bg-zinc-800 rounded"></div>
                <div className="h-20 bg-zinc-800 rounded"></div>
                <div className="h-20 bg-zinc-800 rounded"></div>
              </div>
            </Card>
          }>
            <TechnicalIndicators symbol={selectedStock} />
          </Suspense>
        </motion.div>

        {/* AI Analysis - Right side (2 columns) */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
            className="lg:col-span-2"
          >
            <Suspense fallback={
              <Card className="p-6">
                <div className="animate-pulse space-y-4">
                  <div className="h-6 bg-zinc-800 rounded w-1/3"></div>
                  <div className="h-4 bg-zinc-800 rounded w-full"></div>
                  <div className="h-4 bg-zinc-800 rounded w-2/3"></div>
                </div>
              </Card>
            }>
              <AIExplanationWidget
                ticker={selectedStock}
                currentPrice={currentPrice}
              />
            </Suspense>
          </motion.div>
      </div>
      </>
      )}
    </div>
  )
}
