"use client"

import { useState, useMemo } from "react"
import { Link, useLocation } from "react-router-dom"
import { motion } from "framer-motion"
import {
  Home,
  LineChart,
  Newspaper,
  Star,
  Briefcase,
  Settings,
  X,
  TrendingUp,
  TrendingDown,
  Search,
} from "lucide-react"
import { cn } from "@/lib/utils"
import {
  Sidebar as ShadcnSidebar,
  SidebarContent,
  SidebarHeader,
  SidebarFooter,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
} from "@/components/ui/sidebar"
import { useWebSocket } from "@/hooks/use-websocket-context"

interface SidebarProps {
  onClose: () => void
}

export default function Sidebar({ onClose }: SidebarProps) {
  const location = useLocation()
  const { stockPrices } = useWebSocket()

  // Define top stocks with company names
  const topStocksConfig = [
    { symbol: "AAPL", name: "Apple Inc." },
    { symbol: "MSFT", name: "Microsoft Corp." },
    { symbol: "NVDA", name: "NVIDIA Corp." },
    { symbol: "AMZN", name: "Amazon.com Inc." },
    { symbol: "GOOGL", name: "Alphabet Inc." },
    { symbol: "META", name: "Meta Platforms Inc." },
    { symbol: "BRK.B", name: "Berkshire Hathaway" },
    { symbol: "TSLA", name: "Tesla Inc." },
    { symbol: "AVGO", name: "Broadcom Inc." },
    { symbol: "LLY", name: "Eli Lilly & Co." },
    { symbol: "WMT", name: "Walmart Inc." },
    { symbol: "JPM", name: "JPMorgan Chase" },
    { symbol: "V", name: "Visa Inc." },
    { symbol: "MA", name: "Mastercard Inc." },
    { symbol: "NFLX", name: "Netflix Inc." },
    { symbol: "XOM", name: "Exxon Mobil" },
    { symbol: "COST", name: "Costco Wholesale" },
    { symbol: "ORCL", name: "Oracle Corp." },
    { symbol: "PG", name: "Procter & Gamble" },
    { symbol: "JNJ", name: "Johnson & Johnson" },
    { symbol: "UNH", name: "UnitedHealth Group" },
    { symbol: "HD", name: "Home Depot Inc." },
    { symbol: "ABBV", name: "AbbVie Inc." },
    { symbol: "KO", name: "Coca-Cola Co." },
    { symbol: "CRM", name: "Salesforce Inc." },
  ]

  // Get real-time stock data with fallback to mock data
  const topStocks = useMemo(() => {
    return topStocksConfig.map(stock => {
      const realTimeData = stockPrices[stock.symbol]
      return {
        symbol: stock.symbol,
        name: stock.name,
        change: realTimeData?.changePercent || ((Math.random() - 0.5) * 10), // fallback to random
        price: realTimeData?.price || (50 + Math.random() * 450),
        isRealTime: !!realTimeData
      }
    })
  }, [stockPrices, topStocksConfig])

  const navItems = [
    { name: "Home", path: "/", icon: Home },
    { name: "Stock Analysis", path: "/stocks/AAPL", icon: LineChart },
    { name: "News", path: "/news", icon: Newspaper },
    { name: "Watchlist", path: "/watchlist", icon: Star },
  ]

  return (
    <ShadcnSidebar className="h-screen w-64 bg-black border-r border-zinc-800 overflow-hidden">
      <SidebarHeader className="p-4 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2">
          <motion.div initial={{ rotate: -10 }} animate={{ rotate: 0 }} transition={{ duration: 0.5 }}>
            <LineChart className="h-6 w-6 text-emerald-500" />
          </motion.div>
          <span className="font-bold text-xl">StockPredict AI</span>
        </Link>
        <button onClick={onClose} className="md:hidden p-1 rounded-md hover:bg-zinc-800">
          <X className="h-5 w-5" />
        </button>
      </SidebarHeader>

      <div className="px-4 py-2">
        <div className="relative">
          <Search className="absolute left-3 top-2.5 h-4 w-4 text-zinc-400" />
          <input
            type="text"
            placeholder="Search stocks..."
            className="w-full bg-zinc-900 border border-zinc-800 rounded-md py-2 pl-9 pr-4 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
          />
        </div>
      </div>

      <SidebarContent className="overflow-hidden">
        <SidebarMenu>
          {navItems.map((item) => (
            <SidebarMenuItem key={item.path}>
              <SidebarMenuButton asChild isActive={location.pathname === item.path}>
                <Link to={item.path} className="flex items-center gap-3">
                  <item.icon className="h-5 w-5" />
                  <span>{item.name}</span>
                </Link>
              </SidebarMenuButton>
            </SidebarMenuItem>
          ))}
        </SidebarMenu>

        <div className="mt-6 px-3">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-xs font-semibold text-zinc-400">TOP STOCKS</h3>
          </div>
          <div className="space-y-1 max-h-[calc(100vh-350px)] overflow-y-scroll [&::-webkit-scrollbar]:hidden" style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}>
            {topStocks.map((stock) => (
              <Link
                key={stock.symbol}
                to={`/stocks/${stock.symbol}`}
                className="flex items-center justify-between p-2 rounded-md hover:bg-zinc-900 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <div className={cn(
                    "w-1.5 h-6 rounded-sm",
                    stock.isRealTime 
                      ? "bg-gradient-to-b from-emerald-500 to-emerald-600" 
                      : "bg-gradient-to-b from-zinc-700 to-zinc-800"
                  )}></div>
                  <div className="flex flex-col">
                    <span className="font-medium text-sm">{stock.symbol}</span>
                    <span className="text-xs text-zinc-500">${stock.price.toFixed(2)}</span>
                  </div>
                </div>
                <div className="flex items-center gap-1">
                  <span className={cn("text-xs font-medium", stock.change > 0 ? "text-emerald-500" : "text-red-500")}>
                    {stock.change > 0 ? (
                      <TrendingUp className="h-3 w-3 inline mr-1" />
                    ) : (
                      <TrendingDown className="h-3 w-3 inline mr-1" />
                    )}
                    {stock.change > 0 ? "+" : ""}{stock.change.toFixed(2)}%
                  </span>
                  {stock.isRealTime && (
                    <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse"></div>
                  )}
                </div>
              </Link>
            ))}
          </div>

        </div>
      </SidebarContent>

      <SidebarFooter className="p-4 border-t border-zinc-800">
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 rounded-full bg-gradient-to-br from-emerald-500 to-blue-600 flex items-center justify-center">
            <span className="text-xs font-bold">SP</span>
          </div>
          <div>
            <p className="text-sm font-medium">StockPredict AI</p>
            <p className="text-xs text-zinc-400">Pro Plan</p>
          </div>
        </div>
      </SidebarFooter>
    </ShadcnSidebar>
  )
}
