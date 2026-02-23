"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Search, RefreshCw, AlertCircle, BarChart3 } from "lucide-react"
import Link from "next/link"
import { getSankeyData, getStockDetails, StockDetails } from "@/lib/api"
import SankeyChart from "@/components/market/sankey-chart"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"

export default function SankeyView({ symbol = "AAPL" }: { symbol?: string }) {
    const [currentSymbol, setCurrentSymbol] = useState(symbol)
    const [searchQuery, setSearchQuery] = useState("")
    const [isLoading, setIsLoading] = useState(true)
    const [sankeyData, setSankeyData] = useState<any>(null)
    const [stockData, setStockData] = useState<StockDetails | null>(null)
    const [error, setError] = useState<string | null>(null)

    const trackedStocks = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX",
        "JPM", "V", "WMT", "PG", "HD", "MA"
    ]

    useEffect(() => {
        loadData()
    }, [currentSymbol])

    const handleSearch = (e: React.FormEvent) => {
        e.preventDefault()
        if (searchQuery.trim()) {
            setCurrentSymbol(searchQuery.trim().toUpperCase())
        }
    }

    const loadData = async () => {
        setIsLoading(true)
        setError(null)
        try {
            const [stockDetails, sankeyRes] = await Promise.all([
                getStockDetails(currentSymbol),
                getSankeyData(currentSymbol)
            ])

            setStockData(stockDetails)

            if (!sankeyRes || sankeyRes.error) {
                setError(sankeyRes?.error || "Financial data unavailable for this stock.")
            } else {
                setSankeyData(sankeyRes)
            }
        } catch (err: any) {
            console.error("Failed to load sankey data:", err)
            setError("Failed to load financial flow data. The API limit may have been reached.")
        } finally {
            setIsLoading(false)
        }
    }

    // Helper for formatting large currency
    const formatCurrency = (value: number) => {
        if (!value) return "$0";
        if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
        if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
        return `$${value.toLocaleString()}`;
    };

    return (
        <div className="space-y-6">
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
                            placeholder="Enter stock symbol (e.g. AAPL, MSFT)"
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

                <div className="flex flex-wrap gap-1.5 sm:gap-2">
                    {trackedStocks.map((ticker) => (
                        <button
                            key={ticker}
                            onClick={() => setCurrentSymbol(ticker)}
                            className={`flex items-center gap-1.5 px-2 sm:px-3 py-1 sm:py-1.5 rounded-md text-xs sm:text-sm font-medium transition-colors whitespace-nowrap ${currentSymbol === ticker ? "bg-emerald-500 text-black" : "bg-zinc-800 text-white hover:bg-zinc-700"}`}
                        >
                            {ticker}
                        </button>
                    ))}
                </div>
            </motion.div>

            <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <div className="h-12 w-12 rounded-lg overflow-hidden bg-gradient-to-br from-zinc-700 to-zinc-800 flex items-center justify-center">
                        <img
                            src={`https://raw.githubusercontent.com/davidepalazzo/ticker-logos/main/ticker_icons/${currentSymbol}.png`}
                            alt={currentSymbol}
                            className="h-12 w-12 object-contain"
                            onError={(e) => {
                                const target = e.target as HTMLImageElement;
                                target.style.display = 'none';
                                if (target.parentElement) {
                                    target.parentElement.innerHTML = `<span class="text-xl font-bold">${currentSymbol.charAt(0)}</span>`;
                                }
                            }}
                        />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold flex items-center gap-2">
                            {stockData ? stockData.name : currentSymbol}
                            <span className="text-zinc-500 text-lg">({currentSymbol})</span>
                        </h1>
                        <p className="text-zinc-400 text-sm">Income Statement Flow Analysis</p>
                    </div>
                </div>

                <button
                    onClick={loadData}
                    disabled={isLoading}
                    className="flex items-center gap-2 px-4 py-2 bg-zinc-900 border border-zinc-800 rounded-md text-sm hover:bg-zinc-800 transition-colors disabled:opacity-50"
                >
                    <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
                    Refresh
                </button>
            </div>

            <Card className="border-zinc-800 bg-zinc-900/50">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <BarChart3 className="h-5 w-5 text-emerald-500" />
                        Financial Flow (TTM)
                    </CardTitle>
                    <CardDescription>
                        Visualizing how revenue breaks down into expenses and profits. Hover over nodes and links for exact values.
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    {isLoading ? (
                        <div className="h-[600px] flex items-center justify-center">
                            <div className="flex flex-col items-center gap-4 text-emerald-500">
                                <RefreshCw className="h-8 w-8 animate-spin" />
                                <span className="text-zinc-400">Analyzing financial pipelines...</span>
                            </div>
                        </div>
                    ) : error ? (
                        <div className="h-[400px] flex flex-col items-center justify-center text-center">
                            <AlertCircle className="h-12 w-12 text-zinc-600 mb-4" />
                            <h3 className="text-lg font-medium text-zinc-300 mb-2">Data Unavailable</h3>
                            <p className="text-zinc-500 max-w-md">{error}</p>
                        </div>
                    ) : sankeyData ? (
                        <div className="space-y-6">
                            {/* Summary Metrics */}
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                <div className="bg-zinc-900 rounded-lg p-4 border border-zinc-800">
                                    <div className="text-xs text-zinc-400 mb-1">Total Revenue</div>
                                    <div className="text-xl font-bold text-blue-400">
                                        {formatCurrency(sankeyData.financials.revenue)}
                                    </div>
                                </div>
                                <div className="bg-zinc-900 rounded-lg p-4 border border-zinc-800">
                                    <div className="text-xs text-zinc-400 mb-1">Gross Profit</div>
                                    <div className="text-xl font-bold text-emerald-400">
                                        {formatCurrency(sankeyData.financials.revenue * sankeyData.financials.grossProfitMargin)}
                                    </div>
                                    <div className="text-xs text-zinc-500 mt-1">
                                        {(sankeyData.financials.grossProfitMargin * 100).toFixed(1)}% Margin
                                    </div>
                                </div>
                                <div className="bg-zinc-900 rounded-lg p-4 border border-zinc-800">
                                    <div className="text-xs text-zinc-400 mb-1">Net Income</div>
                                    <div className="text-xl font-bold text-emerald-500">
                                        {formatCurrency(sankeyData.financials.netIncome)}
                                    </div>
                                </div>
                                <div className="bg-zinc-900 rounded-lg p-4 border border-zinc-800">
                                    <div className="text-xs text-zinc-400 mb-1">Period</div>
                                    <div className="text-xl font-medium text-zinc-200">
                                        {sankeyData.financials.period} {sankeyData.financials.fiscalYear}
                                    </div>
                                    <div className="text-xs text-zinc-500 mt-1">
                                        Ended {sankeyData.financials.date}
                                    </div>
                                </div>
                            </div>

                            {/* The Sankey Chart */}
                            <div className="p-3 sm:p-4 bg-black/40 rounded-xl border border-zinc-800/50">
                                <SankeyChart
                                    data={sankeyData.sankey}
                                    symbol={currentSymbol}
                                    height={typeof window !== 'undefined' && window.innerWidth < 640 ? 520 : 680}
                                />
                            </div>
                        </div>
                    ) : (
                        <div className="h-[400px] flex items-center justify-center">
                            <span className="text-zinc-500">No chart data available</span>
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    )
}
