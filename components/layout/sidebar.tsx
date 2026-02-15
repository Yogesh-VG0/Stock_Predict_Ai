"use client"

import { useMemo, useState, useCallback } from "react"
import { Link, useLocation } from "react-router-dom"
import { motion } from "framer-motion"
import {
  Home,
  LineChart,
  Newspaper,
  Star,
  X,
  TrendingUp,
  TrendingDown,
} from "lucide-react"
import { cn } from "@/lib/utils"
import {
  Sidebar as ShadcnSidebar,
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  useSidebar,
} from "@/components/ui/sidebar"
import { useWebSocket } from "@/hooks/use-websocket-context"

// Define top stocks with company names - OUTSIDE component to avoid recreation
const TOP_STOCKS_CONFIG = [
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
] as const

// Compact stock logo for sidebar - uses GitHub-hosted ticker logos with letter fallback
function StockLogo({ symbol, isLoading }: { symbol: string; isLoading: boolean }) {
  const [imgError, setImgError] = useState(false)
  const handleError = useCallback(() => setImgError(true), [])

  if (isLoading) {
    return <div className="w-6 h-6 rounded-full bg-zinc-800 animate-pulse flex-shrink-0" />
  }

  if (imgError) {
    return (
      <div className="w-6 h-6 rounded-full bg-zinc-800 flex items-center justify-center flex-shrink-0">
        <span className="text-[10px] font-bold text-zinc-400">{symbol[0]}</span>
      </div>
    )
  }

  return (
    <img
      src={`https://raw.githubusercontent.com/davidepalazzo/ticker-logos/main/ticker_icons/${symbol}.png`}
      alt={symbol}
      width={24}
      height={24}
      loading="lazy"
      decoding="async"
      className="w-6 h-6 rounded-full bg-zinc-900 object-contain flex-shrink-0"
      onError={handleError}
    />
  )
}

interface SidebarProps {
  onClose: () => void
}

export default function Sidebar({ onClose }: SidebarProps) {
  const location = useLocation()
  const { stockPrices } = useWebSocket()
  const { isMobile, setOpenMobile } = useSidebar()

  // Get real-time stock data - show loading state if no data yet
  // Only depends on stockPrices since TOP_STOCKS_CONFIG is constant
  const topStocks = useMemo(() => {
    return TOP_STOCKS_CONFIG.map(stock => {
      const realTimeData = stockPrices[stock.symbol]
      return {
        symbol: stock.symbol,
        name: stock.name,
        change: realTimeData?.changePercent ?? null, // null = loading
        price: realTimeData?.price ?? null, // null = loading
        isLoading: !realTimeData?.price
      }
    })
  }, [stockPrices])

  const navItems = [
    { name: "Home", path: "/", icon: Home },
    { name: "Stock Analysis", path: "/stocks/AAPL", icon: LineChart },
    { name: "News", path: "/news", icon: Newspaper },
    { name: "Watchlist", path: "/watchlist", icon: Star },
    { name: "Fundamentals", path: "/fundamentals", icon: LineChart },
  ]

  const handleNavigate = () => {
    if (isMobile) {
      // Close ONLY the mobile sheet sidebar.
      // Desktop sidebar should remain open when navigating.
      setOpenMobile(false)
    }
  }

  return (
    <ShadcnSidebar className="h-screen w-64 bg-black border-r border-zinc-800 overflow-hidden">
      <SidebarHeader className="p-4 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2">
          <motion.div initial={{ rotate: -10 }} animate={{ rotate: 0 }} transition={{ duration: 0.5 }}>
            <LineChart className="h-6 w-6 text-emerald-500" />
          </motion.div>
          <span className="font-bold text-xl">StockPredict AI</span>
        </Link>
      </SidebarHeader>

      <SidebarContent className="overflow-hidden">
        <SidebarMenu>
          {navItems.map((item) => (
            <SidebarMenuItem key={item.path}>
              <SidebarMenuButton asChild isActive={location.pathname === item.path}>
                <Link
                  to={item.path}
                  className="flex items-center gap-3"
                  onClick={handleNavigate}
                >
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
                onClick={handleNavigate}
              >
                <div className="flex items-center gap-2">
                  <StockLogo symbol={stock.symbol} isLoading={stock.isLoading} />
                  <div className="flex flex-col">
                    <span className="font-medium text-sm">{stock.symbol}</span>
                    {stock.isLoading ? (
                      <div className="h-3 w-12 bg-zinc-800 rounded animate-pulse"></div>
                    ) : (
                      <span className="text-xs text-zinc-500">${stock.price?.toFixed(2)}</span>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-1">
                  {stock.isLoading ? (
                    <div className="h-3 w-14 bg-zinc-800 rounded animate-pulse"></div>
                  ) : (
                    <>
                      <span className={cn("text-xs font-medium", (stock.change ?? 0) > 0 ? "text-emerald-500" : "text-red-500")}>
                        {(stock.change ?? 0) > 0 ? (
                          <TrendingUp className="h-3 w-3 inline mr-1" />
                        ) : (
                          <TrendingDown className="h-3 w-3 inline mr-1" />
                        )}
                        {(stock.change ?? 0) > 0 ? "+" : ""}{stock.change?.toFixed(2)}%
                      </span>
                      <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse"></div>
                    </>
                  )}
                </div>
              </Link>
            ))}
          </div>

        </div>
      </SidebarContent>

    </ShadcnSidebar>
  )
}
