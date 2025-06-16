"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Link } from "react-router-dom"
import { Star, TrendingUp, TrendingDown, Plus, MoreHorizontal, Bell, Trash2, MoveVertical } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import Sparkline from "@/components/ui/sparkline"
import { cn } from "@/lib/utils"

export default function WatchlistPage() {
  const [isLoading, setIsLoading] = useState(true)
  const [watchlist, setWatchlist] = useState<any[]>([])

  useEffect(() => {
    // Simulate API call
    setTimeout(() => {
      setWatchlist([
        {
          symbol: "AAPL",
          name: "Apple Inc.",
          price: 187.68,
          change: 4.23,
          changePercent: 2.31,
          sparklineData: [180, 182, 185, 183, 186, 188],
        },
        {
          symbol: "MSFT",
          name: "Microsoft Corporation",
          price: 412.76,
          change: 8.54,
          changePercent: 2.11,
          sparklineData: [400, 405, 410, 408, 415, 413],
        },
        {
          symbol: "TSLA",
          name: "Tesla, Inc.",
          price: 248.42,
          change: -7.85,
          changePercent: -3.06,
          sparklineData: [260, 255, 252, 250, 247, 248],
        },
        {
          symbol: "AMZN",
          name: "Amazon.com Inc.",
          price: 178.95,
          change: 3.45,
          changePercent: 1.97,
          sparklineData: [172, 175, 177, 176, 179, 179],
        },
        {
          symbol: "GOOGL",
          name: "Alphabet Inc.",
          price: 164.32,
          change: 2.87,
          changePercent: 1.78,
          sparklineData: [160, 162, 163, 162, 165, 164],
        },
        {
          symbol: "META",
          name: "Meta Platforms, Inc.",
          price: 474.36,
          change: -12.54,
          changePercent: -2.58,
          sparklineData: [490, 485, 480, 478, 475, 474],
        },
        {
          symbol: "NVDA",
          name: "NVIDIA Corporation",
          price: 938.35,
          change: 32.45,
          changePercent: 3.58,
          sparklineData: [890, 905, 920, 915, 925, 938],
        },
        {
          symbol: "NFLX",
          name: "Netflix, Inc.",
          price: 624.58,
          change: -15.32,
          changePercent: -2.39,
          sparklineData: [645, 640, 635, 630, 625, 625],
        },
      ])

      setIsLoading(false)
    }, 1000)
  }, [])

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <motion.h1
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-2xl font-bold flex items-center gap-2"
        >
          <Star className="h-6 w-6 text-amber-400" />
          My Watchlist
        </motion.h1>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="flex items-center gap-2"
        >
          <button className="bg-zinc-800 hover:bg-zinc-700 text-white rounded-md px-3 py-2 text-sm transition-colors flex items-center gap-1.5">
            <Bell className="h-4 w-4" />
            <span>Set Alert</span>
          </button>
          <button className="bg-emerald-500 hover:bg-emerald-600 text-black rounded-md px-3 py-2 text-sm font-medium transition-colors flex items-center gap-1.5">
            <Plus className="h-4 w-4" />
            <span>Add Stock</span>
          </button>
        </motion.div>
      </div>

      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle>Stocks</CardTitle>
            <div className="text-xs text-zinc-400">{watchlist.length} stocks • Drag to reorder</div>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="h-16 bg-zinc-900 animate-pulse rounded-md"></div>
              ))}
            </div>
          ) : (
            <div className="space-y-2">
              {watchlist.map((stock, index) => (
                <motion.div
                  key={stock.symbol}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="bg-zinc-900 rounded-md border border-zinc-800 p-3 flex items-center justify-between group hover:border-zinc-700 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div className="cursor-move opacity-0 group-hover:opacity-100 transition-opacity">
                      <MoveVertical className="h-4 w-4 text-zinc-500" />
                    </div>

                    <div className="flex flex-col">
                      <Link
                        to={`/stocks/${stock.symbol}`}
                        className="font-medium hover:text-emerald-500 transition-colors"
                      >
                        {stock.symbol}
                      </Link>
                      <span className="text-xs text-zinc-400">{stock.name}</span>
                    </div>
                  </div>

                  <div className="flex items-center gap-6">
                    <Sparkline
                      data={stock.sparklineData}
                      height={30}
                      width={80}
                      color={stock.changePercent >= 0 ? "#10b981" : "#ef4444"}
                    />

                    <div className="flex flex-col items-end">
                      <span className="font-medium">${stock.price.toFixed(2)}</span>
                      <span
                        className={cn(
                          "text-xs flex items-center gap-1",
                          stock.changePercent >= 0 ? "text-emerald-500" : "text-red-500",
                        )}
                      >
                        {stock.changePercent >= 0 ? (
                          <TrendingUp className="h-3 w-3" />
                        ) : (
                          <TrendingDown className="h-3 w-3" />
                        )}
                        {stock.changePercent >= 0 ? "+" : ""}
                        {stock.changePercent.toFixed(2)}%
                      </span>
                    </div>

                    <div className="flex items-center gap-1">
                      <button className="p-1.5 rounded-md hover:bg-zinc-800 transition-colors">
                        <Bell className="h-4 w-4 text-zinc-400 hover:text-white" />
                      </button>
                      <button className="p-1.5 rounded-md hover:bg-zinc-800 transition-colors">
                        <Trash2 className="h-4 w-4 text-zinc-400 hover:text-red-500" />
                      </button>
                      <button className="p-1.5 rounded-md hover:bg-zinc-800 transition-colors">
                        <MoreHorizontal className="h-4 w-4 text-zinc-400 hover:text-white" />
                      </button>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-emerald-500" />
              Performance Overview
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64 flex items-center justify-center bg-zinc-900 rounded-lg border border-zinc-800">
              <p className="text-zinc-400">Performance chart will be displayed here</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bell className="h-5 w-5 text-amber-500" />
              Active Alerts
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="bg-zinc-900 rounded-md p-3 border border-zinc-800">
                <div className="flex justify-between items-center mb-1">
                  <div className="font-medium">AAPL</div>
                  <div className="text-xs text-emerald-500">Above $190</div>
                </div>
                <div className="text-xs text-zinc-400">Triggers when price exceeds $190</div>
              </div>

              <div className="bg-zinc-900 rounded-md p-3 border border-zinc-800">
                <div className="flex justify-between items-center mb-1">
                  <div className="font-medium">TSLA</div>
                  <div className="text-xs text-red-500">Below $240</div>
                </div>
                <div className="text-xs text-zinc-400">Triggers when price falls below $240</div>
              </div>

              <div className="bg-zinc-900 rounded-md p-3 border border-zinc-800">
                <div className="flex justify-between items-center mb-1">
                  <div className="font-medium">NVDA</div>
                  <div className="text-xs text-amber-500">Earnings</div>
                </div>
                <div className="text-xs text-zinc-400">Triggers 1 day before earnings report</div>
              </div>

              <button className="w-full bg-zinc-800 hover:bg-zinc-700 text-white rounded-md px-3 py-2 text-sm transition-colors flex items-center justify-center gap-1.5 mt-2">
                <Plus className="h-4 w-4" />
                <span>Add New Alert</span>
              </button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
