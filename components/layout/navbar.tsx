"use client"

import { useState, useEffect } from "react"
import { Menu, Bell, Clock, Moon, CircleDot, Search, ChevronDown } from "lucide-react"
import { getMarketStatus, MarketStatus } from "@/lib/api"

// Helper functions (copy from pages/home.tsx for consistency)
function getSessionLabel(session: string | null) {
  switch (session) {
    case 'pre-market':
      return 'Pre-market';
    case 'regular':
      return 'Market Open';
    case 'after-hours':
      return 'After-hours';
    case 'closed-after-hours':
      return 'Market Closed (after hours)';
    case 'closed-before-pre-market':
      return 'Market Closed (before pre-market)';
    case null:
    default:
      return 'Market Closed';
  }
}
function getNextSessionLabel(nextSession: string | null) {
  switch (nextSession) {
    case 'pre-market':
      return 'Pre-market';
    case 'regular':
      return 'Regular';
    case 'after-hours':
      return 'After-hours';
    default:
      return '';
  }
}
function getTimeUntilClose(nextClose: string) {
  const now = new Date()
  const close = new Date(nextClose)
  const diffMs = close.getTime() - now.getTime()
  if (diffMs <= 0) return "Closed"
  const hours = Math.floor(diffMs / (1000 * 60 * 60))
  const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60))
  return `${hours}h ${minutes}m`
}
function getTimeUntilOpen(nextOpen: string) {
  const now = new Date()
  const open = new Date(nextOpen)
  const diffMs = open.getTime() - now.getTime()
  if (diffMs <= 0) return "Open"
  const hours = Math.floor(diffMs / (1000 * 60 * 60))
  const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60))
  return `${hours}h ${minutes}m`
}

interface NavbarProps {
  sidebarOpen: boolean
  toggleSidebar: () => void
}

export default function Navbar({ sidebarOpen, toggleSidebar }: NavbarProps) {
  const [currentTime, setCurrentTime] = useState(new Date())
  const [marketStatus, setMarketStatus] = useState<MarketStatus | null>(null)

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000)
    getMarketStatus().then(setMarketStatus)
    const statusInterval = setInterval(() => {
      getMarketStatus().then(setMarketStatus)
    }, 60 * 1000) // refresh every minute
    return () => {
      clearInterval(timer)
      clearInterval(statusInterval)
    }
  }, [])

  return (
    <div className="bg-black border-b border-zinc-800 py-2 px-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button onClick={toggleSidebar} className="p-2 rounded-md hover:bg-zinc-800 transition-colors">
            <Menu className="h-5 w-5" />
          </button>

          <div className="hidden md:flex items-center gap-2">
            <div className="flex items-center gap-1.5">
              <Clock className="h-4 w-4 text-zinc-400" />
              <span className="text-sm font-medium">
                {currentTime.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </span>
            </div>

            <div className="h-4 w-px bg-zinc-700 mx-1"></div>

            <div className="flex items-center gap-1.5">
              {marketStatus && marketStatus.session === "regular" && <CircleDot className="h-4 w-4 text-emerald-500" />}
              {marketStatus && marketStatus.session === "pre-market" && <Clock className="h-4 w-4 text-amber-500" />}
              {marketStatus && marketStatus.session === "after-hours" && <Moon className="h-4 w-4 text-blue-400" />}

              <span className="text-sm font-medium capitalize">
                {marketStatus ? getSessionLabel(marketStatus.session) : "Loading..."}
              </span>

              {marketStatus && marketStatus.session && marketStatus.next_close_time && (marketStatus.session === 'pre-market' || marketStatus.session === 'regular' || marketStatus.session === 'after-hours') && (
                <span className="text-xs text-zinc-400 ml-1">
                  {getTimeUntilClose(marketStatus.next_close_time)} until close
                </span>
              )}
              {marketStatus && (marketStatus.session === 'closed-after-hours' || marketStatus.session === 'closed-before-pre-market' || marketStatus.session === null) && marketStatus.next_open_time && marketStatus.next_session && marketStatus.session !== marketStatus.next_session && (
                <span className="text-xs text-zinc-400 ml-1">
                  {getTimeUntilOpen(marketStatus.next_open_time)} until {getNextSessionLabel(marketStatus.next_session)} opens
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="hidden md:block flex-1 max-w-md mx-4">
          <div className="relative">
            <Search className="absolute left-3 top-2.5 h-4 w-4 text-zinc-400" />
            <input
              type="text"
              placeholder="Search stocks, news, predictions..."
              className="w-full bg-zinc-900 border border-zinc-800 rounded-md py-2 pl-9 pr-4 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
            />
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button className="p-2 rounded-md hover:bg-zinc-800 transition-colors relative">
            <Bell className="h-5 w-5" />
            <span className="absolute top-1 right-1 h-2 w-2 rounded-full bg-emerald-500"></span>
          </button>

          <div className="hidden md:flex items-center gap-2 ml-2">
            <button className="flex items-center gap-2 px-3 py-1.5 rounded-md hover:bg-zinc-800 transition-colors text-sm">
              <span>S&P 500</span>
              <span className="text-emerald-500">+1.24%</span>
              <ChevronDown className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
