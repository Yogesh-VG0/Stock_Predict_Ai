"use client"

import { useState, useEffect } from "react"
import { useParams } from "react-router-dom"
import { motion } from "framer-motion"
import {
  Star,
  TrendingUp,
  TrendingDown,
  Info,
  DollarSign,
  BarChart3,
  Calendar,
  CheckCircle,
  XCircle,
  Building,
  Globe,
  Users,
  Newspaper,
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import TradingViewAdvancedChart from "@/components/tradingview/trading-view-advanced-chart"

type StockDetailProps = {}

export default function StockDetail({}: StockDetailProps) {
  const { symbol } = useParams<{ symbol: string }>()
  const [isLoading, setIsLoading] = useState(true)
  const [stockData, setStockData] = useState<any>(null)

  useEffect(() => {
    // Simulate API call
    setTimeout(() => {
      // Mock data for the stock
      const mockStockData = {
        symbol: symbol,
        name: getCompanyName(symbol || ""),
        price: 187.68,
        change: 4.23,
        changePercent: 2.31,
        marketCap: "2.94T",
        peRatio: 31.2,
        dividend: 0.92,
        volume: "58.4M",
        avgVolume: "62.1M",
        high52w: 198.23,
        low52w: 143.9,
        open: 184.32,
        previousClose: 183.45,
        sector: "Technology",
        industry: "Consumer Electronics",
        description:
          "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide. The company offers iPhone, a line of smartphones; Mac, a line of personal computers; iPad, a line of multi-purpose tablets; and wearables, home, and accessories comprising AirPods, Apple TV, Apple Watch, Beats products, and HomePod.",
        headquarters: "Cupertino, California",
        founded: 1976,
        employees: 154000,
        ceo: "Tim Cook",
        website: "www.apple.com",
        financials: {
          revenue: [394.33, 365.82, 274.52, 260.17, 265.6],
          netIncome: [94.32, 99.8, 57.41, 55.26, 59.53],
          years: ["2023", "2022", "2021", "2020", "2019"],
        },
        aiAnalysis: {
          positiveFactors: [
            "Strong brand loyalty and ecosystem integration",
            "Consistent revenue growth from services segment",
            "Robust cash position for investments and shareholder returns",
            "Expanding into new markets like AR/VR",
          ],
          negativeFactors: [
            "Increasing competition in smartphone market",
            "Regulatory scrutiny over App Store practices",
            "Supply chain dependencies and vulnerabilities",
            "Slowing growth in mature markets",
          ],
        },
        dividendHistory: [
          { year: "2023", amount: 0.92, yield: 0.54 },
          { year: "2022", amount: 0.88, yield: 0.58 },
          { year: "2021", amount: 0.85, yield: 0.62 },
          { year: "2020", amount: 0.82, yield: 0.68 },
          { year: "2019", amount: 0.77, yield: 0.72 },
        ],
        earnings: {
          nextDate: "2023-07-27",
          estimates: { eps: 1.32, revenue: "84.2B" },
          history: [
            {
              date: "2023-04-30",
              epsEstimate: 1.43,
              epsActual: 1.52,
              revenueEstimate: "93.5B",
              revenueActual: "94.8B",
              surprise: "+6.3%",
            },
            {
              date: "2023-01-31",
              epsEstimate: 1.94,
              epsActual: 1.88,
              revenueEstimate: "121.2B",
              revenueActual: "117.2B",
              surprise: "-3.1%",
            },
            {
              date: "2022-10-31",
              epsEstimate: 1.27,
              epsActual: 1.29,
              revenueEstimate: "88.9B",
              revenueActual: "90.1B",
              surprise: "+1.6%",
            },
            {
              date: "2022-07-31",
              epsEstimate: 1.16,
              epsActual: 1.2,
              revenueEstimate: "82.8B",
              revenueActual: "83.0B",
              surprise: "+3.4%",
            },
          ],
        },
        news: [
          {
            title: "Apple Unveils New AI Features for iPhone and iPad",
            date: "2023-06-05",
            source: "TechCrunch",
            sentiment: "positive",
            url: "#",
          },
          {
            title: "Apple's Services Revenue Hits All-Time High",
            date: "2023-05-28",
            source: "Bloomberg",
            sentiment: "positive",
            url: "#",
          },
          {
            title: "EU Fines Apple €500M Over Music Streaming Rules",
            date: "2023-05-15",
            source: "Financial Times",
            sentiment: "negative",
            url: "#",
          },
          {
            title: "Apple Supplier Reports Production Delays",
            date: "2023-05-10",
            source: "Reuters",
            sentiment: "negative",
            url: "#",
          },
        ],
      }

      setStockData(mockStockData)
      setIsLoading(false)
    }, 1500)
  }, [symbol])

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

  if (isLoading) {
    return (
      <div className="flex flex-col gap-6">
        <div className="h-16 bg-zinc-900 animate-pulse rounded-lg"></div>
        <div className="h-[500px] bg-zinc-900 animate-pulse rounded-lg"></div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="h-64 bg-zinc-900 animate-pulse rounded-lg"></div>
          <div className="h-64 bg-zinc-900 animate-pulse rounded-lg"></div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-gradient-to-r from-zinc-900 to-black p-4 rounded-lg border border-zinc-800"
      >
        <div className="flex items-center gap-4">
          <div className="h-12 w-12 bg-gradient-to-br from-zinc-700 to-zinc-800 rounded-lg flex items-center justify-center text-xl font-bold">
            {stockData.symbol.charAt(0)}
          </div>
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-2">
              {stockData.name}
              <span className="text-lg text-zinc-400">({stockData.symbol})</span>
            </h1>
            <div className="flex items-center gap-2 text-sm">
              <span className="text-zinc-400">{stockData.sector}</span>
              <span className="text-zinc-600">•</span>
              <span className="text-zinc-400">{stockData.industry}</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div
            className={`flex items-center gap-1 px-3 py-1.5 rounded-md ${stockData.changePercent >= 0 ? "bg-emerald-500/10 text-emerald-500" : "bg-red-500/10 text-red-500"}`}
          >
            {stockData.changePercent >= 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
            <span className="font-medium">
              {stockData.changePercent >= 0 ? "+" : ""}
              {stockData.changePercent.toFixed(2)}%
            </span>
          </div>

          <button className="bg-zinc-800 hover:bg-zinc-700 text-white rounded-md px-3 py-1.5 flex items-center gap-1.5 transition-colors">
            <Star className="h-4 w-4 text-amber-400" />
            <span className="text-sm font-medium">Follow</span>
          </button>
        </div>
      </motion.div>

      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}>
        <TradingViewAdvancedChart symbol={stockData.symbol} height={500} />
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Info className="h-5 w-5 text-blue-500" />
                Company Overview
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-zinc-300 leading-relaxed">{stockData.description}</p>

              <div className="grid grid-cols-2 gap-4 pt-2">
                <div className="flex items-start gap-2">
                  <Building className="h-4 w-4 text-zinc-400 mt-0.5" />
                  <div>
                    <div className="text-xs text-zinc-400">Headquarters</div>
                    <div className="text-sm">{stockData.headquarters}</div>
                  </div>
                </div>

                <div className="flex items-start gap-2">
                  <Calendar className="h-4 w-4 text-zinc-400 mt-0.5" />
                  <div>
                    <div className="text-xs text-zinc-400">Founded</div>
                    <div className="text-sm">{stockData.founded}</div>
                  </div>
                </div>

                <div className="flex items-start gap-2">
                  <Users className="h-4 w-4 text-zinc-400 mt-0.5" />
                  <div>
                    <div className="text-xs text-zinc-400">Employees</div>
                    <div className="text-sm">{stockData.employees.toLocaleString()}</div>
                  </div>
                </div>

                <div className="flex items-start gap-2">
                  <Globe className="h-4 w-4 text-zinc-400 mt-0.5" />
                  <div>
                    <div className="text-xs text-zinc-400">Website</div>
                    <div className="text-sm">{stockData.website}</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="mt-6">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <DollarSign className="h-5 w-5 text-emerald-500" />
              Financial Highlights
            </h2>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <Card className="bg-black/60 backdrop-blur-sm">
                <CardContent className="p-4">
                  <div className="text-xs text-zinc-400 mb-1">Market Cap</div>
                  <div className="text-lg font-bold">${stockData.marketCap}</div>
                </CardContent>
              </Card>

              <Card className="bg-black/60 backdrop-blur-sm">
                <CardContent className="p-4">
                  <div className="text-xs text-zinc-400 mb-1">P/E Ratio</div>
                  <div className="text-lg font-bold">{stockData.peRatio}</div>
                </CardContent>
              </Card>

              <Card className="bg-black/60 backdrop-blur-sm">
                <CardContent className="p-4">
                  <div className="text-xs text-zinc-400 mb-1">Dividend Yield</div>
                  <div className="text-lg font-bold">{stockData.dividend}%</div>
                </CardContent>
              </Card>

              <Card className="bg-black/60 backdrop-blur-sm">
                <CardContent className="p-4">
                  <div className="text-xs text-zinc-400 mb-1">Volume</div>
                  <div className="text-lg font-bold">{stockData.volume}</div>
                </CardContent>
              </Card>
            </div>
          </div>

          <div className="mt-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5 text-purple-500" />
                  Revenue Growth
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64 flex items-end gap-2">
                  {stockData.financials.revenue.map((value: number, index: number) => (
                    <div key={index} className="flex-1 flex flex-col items-center gap-1">
                      <div className="text-xs text-zinc-400">${(value / 1000).toFixed(1)}B</div>
                      <div
                        className="w-full bg-gradient-to-t from-emerald-600 to-emerald-400 rounded-t-sm"
                        style={{
                          height: `${(value / Math.max(...stockData.financials.revenue)) * 180}px`,
                        }}
                      ></div>
                      <div className="text-xs font-medium">{stockData.financials.years[index]}</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="space-y-6"
        >
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-emerald-500" />
                AI Stock Analysis
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <h3 className="text-sm font-medium text-emerald-500 flex items-center gap-1.5 mb-2">
                    <CheckCircle className="h-4 w-4" />
                    Positive Factors
                  </h3>
                  <ul className="space-y-2">
                    {stockData.aiAnalysis.positiveFactors.map((factor: string, index: number) => (
                      <li key={index} className="text-sm flex items-start gap-2">
                        <span className="text-emerald-500 mt-0.5">•</span>
                        <span>{factor}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div>
                  <h3 className="text-sm font-medium text-red-500 flex items-center gap-1.5 mb-2">
                    <XCircle className="h-4 w-4" />
                    Negative Factors
                  </h3>
                  <ul className="space-y-2">
                    {stockData.aiAnalysis.negativeFactors.map((factor: string, index: number) => (
                      <li key={index} className="text-sm flex items-start gap-2">
                        <span className="text-red-500 mt-0.5">•</span>
                        <span>{factor}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Calendar className="h-5 w-5 text-blue-500" />
                Earnings
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="bg-zinc-900 rounded-md p-3 mb-4 border border-zinc-800">
                <div className="flex justify-between items-center">
                  <div>
                    <div className="text-xs text-zinc-400">Next Earnings Date</div>
                    <div className="text-sm font-medium">
                      {new Date(stockData.earnings.nextDate).toLocaleDateString("en-US", {
                        month: "short",
                        day: "numeric",
                        year: "numeric",
                      })}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-zinc-400">EPS Estimate</div>
                    <div className="text-sm font-medium">${stockData.earnings.estimates.eps}</div>
                  </div>
                  <div>
                    <div className="text-xs text-zinc-400">Revenue Estimate</div>
                    <div className="text-sm font-medium">${stockData.earnings.estimates.revenue}</div>
                  </div>
                </div>
              </div>

              <div className="text-sm font-medium mb-2">Past Earnings</div>
              <div className="space-y-3">
                {stockData.earnings.history.map((earning: any, index: number) => (
                  <div key={index} className="bg-zinc-900/50 rounded-md p-3 border border-zinc-800/50">
                    <div className="flex justify-between items-center mb-2">
                      <div className="text-xs font-medium">
                        {new Date(earning.date).toLocaleDateString("en-US", {
                          month: "short",
                          day: "numeric",
                          year: "numeric",
                        })}
                      </div>
                      <div
                        className={`text-xs px-2 py-0.5 rounded-full ${earning.surprise.startsWith("+") ? "bg-emerald-500/10 text-emerald-500" : "bg-red-500/10 text-red-500"}`}
                      >
                        {earning.surprise}
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div>
                        <span className="text-zinc-400">EPS Est: </span>
                        <span>${earning.epsEstimate}</span>
                      </div>
                      <div>
                        <span className="text-zinc-400">EPS Act: </span>
                        <span>${earning.epsActual}</span>
                      </div>
                      <div>
                        <span className="text-zinc-400">Rev Est: </span>
                        <span>${earning.revenueEstimate}</span>
                      </div>
                      <div>
                        <span className="text-zinc-400">Rev Act: </span>
                        <span>${earning.revenueActual}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Newspaper className="h-5 w-5 text-amber-500" />
                Latest News
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {stockData.news.map((item: any, index: number) => (
                  <a
                    key={index}
                    href={item.url}
                    className="block bg-zinc-900/50 rounded-md p-3 border border-zinc-800/50 hover:bg-zinc-900 transition-colors"
                  >
                    <div className="flex justify-between items-start mb-1">
                      <h3 className="text-sm font-medium">{item.title}</h3>
                      {getSentimentIcon(item.sentiment)}
                    </div>
                    <div className="flex items-center gap-2 text-xs text-zinc-400">
                      <span>{item.source}</span>
                      <span>•</span>
                      <span>{new Date(item.date).toLocaleDateString("en-US", { month: "short", day: "numeric" })}</span>
                    </div>
                  </a>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  )
}
