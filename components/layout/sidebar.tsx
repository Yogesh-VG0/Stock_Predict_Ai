"use client"

import { useState } from "react"
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

interface SidebarProps {
  onClose: () => void
}

export default function Sidebar({ onClose }: SidebarProps) {
  const location = useLocation()
  const [topStocks, setTopStocks] = useState([
    { symbol: "AAPL", name: "Apple Inc.", change: 2.34 },
    { symbol: "MSFT", name: "Microsoft Corp.", change: 1.56 },
    { symbol: "GOOGL", name: "Alphabet Inc.", change: -0.78 },
    { symbol: "AMZN", name: "Amazon.com Inc.", change: 3.21 },
    { symbol: "TSLA", name: "Tesla Inc.", change: -2.45 },
    { symbol: "META", name: "Meta Platforms Inc.", change: 1.23 },
    { symbol: "NVDA", name: "NVIDIA Corp.", change: 4.56 },
    { symbol: "JPM", name: "JPMorgan Chase & Co.", change: 0.89 },
    { symbol: "V", name: "Visa Inc.", change: 0.45 },
    { symbol: "WMT", name: "Walmart Inc.", change: -0.32 },
  ])

  const navItems = [
    { name: "Home", path: "/", icon: Home },
    { name: "Predictions", path: "/predictions", icon: LineChart },
    { name: "News", path: "/news", icon: Newspaper },
    { name: "Watchlist", path: "/watchlist", icon: Star },
    { name: "Portfolio", path: "/portfolio", icon: Briefcase },
    { name: "Settings", path: "/settings", icon: Settings },
  ]

  return (
    <ShadcnSidebar className="h-screen w-64 bg-black border-r border-zinc-800">
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

      <SidebarContent>
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
          <h3 className="text-xs font-semibold text-zinc-400 mb-2">TOP STOCKS</h3>
          <div className="space-y-1 max-h-[calc(100vh-350px)] overflow-y-auto pr-2 scrollbar-thin">
            {topStocks.map((stock) => (
              <Link
                key={stock.symbol}
                to={`/stocks/${stock.symbol}`}
                className="flex items-center justify-between p-2 rounded-md hover:bg-zinc-900 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <div className="w-1.5 h-6 rounded-sm bg-gradient-to-b from-zinc-700 to-zinc-800"></div>
                  <span className="font-medium">{stock.symbol}</span>
                </div>
                <span className={cn("text-xs font-medium", stock.change > 0 ? "text-emerald-500" : "text-red-500")}>
                  {stock.change > 0 ? (
                    <TrendingUp className="h-3 w-3 inline mr-1" />
                  ) : (
                    <TrendingDown className="h-3 w-3 inline mr-1" />
                  )}
                  {Math.abs(stock.change)}%
                </span>
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
