"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Search, RefreshCw, AlertCircle, BarChart3 } from "lucide-react"
import Link from "next/link"
import { getSankeyData, getStockDetails, StockDetails } from "@/lib/api"
import SankeyChart from "@/components/market/sankey-chart"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { cn } from "@/lib/utils"

// FMP Free API symbol allowlist (only these return data)
const FMP_FREE_SYMBOLS = new Set([
    "AAPL", "TSLA", "AMZN", "MSFT", "NVDA", "GOOGL", "META", "NFLX", "JPM", "V", "BAC", "PYPL", "DIS", "T", "PFE",
    "COST", "INTC", "KO", "TGT", "NKE", "SPY", "BA", "BABA", "XOM", "WMT", "GE", "CSCO", "VZ", "JNJ", "CVX", "PLTR",
    "SQ", "SHOP", "SBUX", "SOFI", "HOOD", "RBLX", "SNAP", "AMD", "UBER", "FDX", "ABBV", "ETSY", "MRNA", "LMT", "GM",
    "F", "LCID", "CCL", "DAL", "UAL", "AAL", "TSM", "SONY", "ET", "MRO", "COIN", "RIVN", "RIOT", "CPRX", "VWO", "SPYG",
    "NOK", "ROKU", "VIAC", "ATVI", "BIDU", "DOCU", "ZM", "PINS", "TLRY", "WBA", "MGM", "NIO", "C", "GS", "WFC", "ADBE",
    "PEP", "UNH", "CARR", "HCA", "TWTR", "BILI", "SIRI", "FUBO", "RKT",
])

const PRESET_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "JPM", "V", "WMT", "PG", "HD", "MA"]
const trackedStocks = PRESET_TICKERS.filter((t) => FMP_FREE_SYMBOLS.has(t))

export default function SankeyView({ symbol = "AAPL" }: { symbol?: string }) {
    const [currentSymbol, setCurrentSymbol] = useState(symbol)
    const [searchQuery, setSearchQuery] = useState("")
    const [isLoading, setIsLoading] = useState(true)
    const [sankeyData, setSankeyData] = useState<any>(null)
    const [stockData, setStockData] = useState<StockDetails | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [isMobile, setIsMobile] = useState(false)

    useEffect(() => {
        const onResize = () => setIsMobile(window.innerWidth < 640)
        onResize()
        window.addEventListener("resize", onResize)
        return () => window.removeEventListener("resize", onResize)
    }, [])

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

    const formatCurrency = (value: number) => {
        if (!value) return "$0";
        if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
        if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
        return `$${value.toLocaleString()}`;
    };

    // Compute dynamic chart height based on node count
    const chartHeight = (() => {
        if (!sankeyData?.sankey?.nodes) return isMobile ? 520 : 680;
        const nodeCount = sankeyData.sankey.nodes.length;
        if (isMobile) return Math.max(480, Math.min(700, nodeCount * 52));
        return Math.max(680, Math.min(1000, nodeCount * 58));
    })();

    return (
        <div className="space-y-4 sm:space-y-6">
            {/* Search Bar */}
            <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-zinc-900 rounded-lg p-3 sm:p-4 border border-zinc-800"
            >
                <form onSubmit={handleSearch} className="flex gap-2 mb-3 sm:mb-4">
                    <div className="relative flex-1 group">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-500 group-focus-within:text-emerald-400 transition-colors" />
                        <input
                            type="text"
                            placeholder="Search by symbol or company name..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full bg-zinc-800/80 border border-zinc-700/60 rounded-lg py-2.5 pl-10 pr-4 text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/40 focus:border-emerald-500/50 transition-all"
                        />
                    </div>
                    <button
                        type="submit"
                        disabled={!searchQuery}
                        className="bg-emerald-500 hover:bg-emerald-400 text-black font-semibold rounded-lg px-4 sm:px-5 py-2.5 text-sm disabled:opacity-40 disabled:cursor-not-allowed transition-all shadow-sm shadow-emerald-500/20 hover:shadow-emerald-500/30"
                    >
                        Search
                    </button>
                </form>

                <div className="flex flex-wrap gap-1 sm:gap-1.5">
                    {trackedStocks.map((ticker) => (
                        <button
                            key={ticker}
                            type="button"
                            onClick={() => setCurrentSymbol(ticker)}
                            className={cn(
                                "flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium transition-colors whitespace-nowrap",
                                "sm:gap-1.5 sm:px-3 sm:py-1.5 sm:text-sm",
                                currentSymbol === ticker
                                    ? "bg-emerald-500 text-black"
                                    : "bg-zinc-800 text-white hover:bg-zinc-700"
                            )}
                        >
                            <img
                                src={`https://raw.githubusercontent.com/davidepalazzo/ticker-logos/main/ticker_icons/${ticker}.png`}
                                alt={ticker}
                                className="h-3.5 w-3.5 sm:h-4 sm:w-4 object-contain rounded-sm"
                                onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
                            />
                            {ticker}
                        </button>
                    ))}
                </div>
            </motion.div>

            {/* Stock header - responsive layout */}
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-center gap-3 sm:gap-4 min-w-0">
                    <div className="h-10 w-10 sm:h-12 sm:w-12 rounded-lg overflow-hidden bg-gradient-to-br from-zinc-700 to-zinc-800 flex items-center justify-center flex-shrink-0">
                        <img
                            src={`https://raw.githubusercontent.com/davidepalazzo/ticker-logos/main/ticker_icons/${currentSymbol}.png`}
                            alt={currentSymbol}
                            className="h-10 w-10 sm:h-12 sm:w-12 object-contain"
                            onError={(e) => {
                                const target = e.target as HTMLImageElement;
                                target.style.display = 'none';
                                if (target.parentElement) {
                                    target.parentElement.innerHTML = `<span class="text-lg sm:text-xl font-bold">${currentSymbol.charAt(0)}</span>`;
                                }
                            }}
                        />
                    </div>
                    <div className="min-w-0">
                        <h1 className="text-lg sm:text-2xl font-bold flex items-center gap-2 truncate">
                            <span className="truncate">{stockData ? stockData.name : currentSymbol}</span>
                            <span className="text-zinc-500 text-sm sm:text-lg flex-shrink-0">({currentSymbol})</span>
                        </h1>
                        <p className="text-zinc-400 text-xs sm:text-sm">Income Statement Flow Analysis</p>
                    </div>
                </div>

                <button
                    onClick={loadData}
                    disabled={isLoading}
                    className="flex items-center justify-center gap-2 px-3 py-1.5 sm:px-4 sm:py-2 bg-zinc-900 border border-zinc-800 rounded-md text-xs sm:text-sm hover:bg-zinc-800 transition-colors disabled:opacity-50 self-start sm:self-auto"
                >
                    <RefreshCw className={`h-3.5 w-3.5 sm:h-4 sm:w-4 ${isLoading ? 'animate-spin' : ''}`} />
                    Refresh
                </button>
            </div>

            <Card className="border-zinc-800 bg-zinc-900/50">
                <CardHeader className="px-4 py-3 sm:px-6 sm:py-4">
                    <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
                        <BarChart3 className="h-4 w-4 sm:h-5 sm:w-5 text-emerald-500" />
                        Financial Flow (TTM)
                    </CardTitle>
                    <CardDescription className="text-xs sm:text-sm">
                        Visualizing how revenue breaks down into expenses and profits. Hover over nodes and links for exact values.
                    </CardDescription>
                </CardHeader>
                <CardContent className="px-1.5 pb-3 sm:px-6 sm:pb-6">
                    {isLoading ? (
                        <div className="h-[400px] sm:h-[600px] flex items-center justify-center">
                            <div className="flex flex-col items-center gap-4 text-emerald-500">
                                <RefreshCw className="h-8 w-8 animate-spin" />
                                <span className="text-zinc-400 text-sm">Analyzing financial pipelines...</span>
                            </div>
                        </div>
                    ) : error ? (
                        <div className="h-[300px] sm:h-[400px] flex flex-col items-center justify-center text-center px-4">
                            <AlertCircle className="h-10 w-10 sm:h-12 sm:w-12 text-zinc-600 mb-4" />
                            <h3 className="text-base sm:text-lg font-medium text-zinc-300 mb-2">Data Unavailable</h3>
                            <p className="text-zinc-500 max-w-md text-sm">{error}</p>
                        </div>
                    ) : sankeyData ? (
                        <div className="space-y-4 sm:space-y-6">
                            {(() => {
                                const links = sankeyData?.sankey?.links ?? [];
                                const sumTo = (target: string) =>
                                    links
                                        .filter((l: any) => String(l.target) === target)
                                        .reduce((acc: number, l: any) => acc + Number(l.value || 0), 0);
                                const sumFromTo = (source: string, target: string) =>
                                    links
                                        .filter((l: any) => String(l.source) === source && String(l.target) === target)
                                        .reduce((acc: number, l: any) => acc + Number(l.value || 0), 0);

                                const totalRevenueFromGraph = sumTo("Total Revenue");
                                const grossProfitFromGraph = sumFromTo("Total Revenue", "Gross Profit");

                                const totalRevenue = totalRevenueFromGraph || sankeyData.financials?.revenue || 0;
                                const grossProfit =
                                    grossProfitFromGraph ||
                                    sankeyData.financials?.grossProfit ||
                                    (totalRevenue * Number(sankeyData.financials?.grossProfitMargin || 0));

                                return (
                                    <div className="grid grid-cols-2 gap-2 sm:gap-4 md:grid-cols-4">
                                        <div className="bg-zinc-900 rounded-lg p-3 sm:p-4 border border-zinc-800">
                                            <div className="text-[10px] sm:text-xs text-zinc-400 mb-1">Total Revenue</div>
                                            <div className="text-base sm:text-xl font-bold text-blue-400">
                                                {formatCurrency(totalRevenue)}
                                            </div>
                                        </div>
                                        <div className="bg-zinc-900 rounded-lg p-3 sm:p-4 border border-zinc-800">
                                            <div className="text-[10px] sm:text-xs text-zinc-400 mb-1">Gross Profit</div>
                                            <div className="text-base sm:text-xl font-bold text-emerald-400">
                                                {formatCurrency(grossProfit)}
                                            </div>
                                            <div className="text-[10px] sm:text-xs text-zinc-500 mt-0.5 sm:mt-1">
                                                {(() => {
                                                    const apiMargin = Number(sankeyData.financials.grossProfitMargin || 0);
                                                    const computedMargin = totalRevenue > 0 ? grossProfit / totalRevenue : 0;
                                                    const margin = apiMargin > 0 ? apiMargin : computedMargin;
                                                    return `${(margin * 100).toFixed(1)}% Margin`;
                                                })()}
                                            </div>
                                        </div>
                                        <div className="bg-zinc-900 rounded-lg p-3 sm:p-4 border border-zinc-800">
                                            <div className="text-[10px] sm:text-xs text-zinc-400 mb-1">Net Income</div>
                                            <div className="text-base sm:text-xl font-bold text-emerald-500">
                                                {formatCurrency(sankeyData.financials.netIncome)}
                                            </div>
                                        </div>
                                        <div className="bg-zinc-900 rounded-lg p-3 sm:p-4 border border-zinc-800">
                                            <div className="text-[10px] sm:text-xs text-zinc-400 mb-1">Period</div>
                                            <div className="text-base sm:text-xl font-medium text-zinc-200">
                                                {sankeyData.financials.period} {sankeyData.financials.fiscalYear}
                                            </div>
                                            <div className="text-[10px] sm:text-xs text-zinc-500 mt-0.5 sm:mt-1">
                                                Ended {sankeyData.financials.date}
                                            </div>
                                        </div>
                                    </div>
                                );
                            })()}

                            {/* The Sankey Chart — on mobile, remove horizontal padding so scroll area uses full width */}
                            <div className="p-0 sm:p-4 bg-black/40 rounded-xl border border-zinc-800/50 overflow-hidden">
                                <SankeyChart
                                    data={sankeyData.sankey}
                                    symbol={currentSymbol}
                                    height={chartHeight}
                                />
                            </div>
                        </div>
                    ) : (
                        <div className="h-[300px] sm:h-[400px] flex items-center justify-center">
                            <span className="text-zinc-500 text-sm">No chart data available</span>
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    )
}
