"use client"

import { useState, useEffect } from "react"
import { Menu, Clock, Moon, CircleDot } from "lucide-react"
import { getMarketStatus, MarketStatus } from "@/lib/api"
import NotificationWidget from "@/components/market/NotificationWidget"
import SearchWidget from "@/components/market/SearchWidget"
import { useSidebar } from "@/components/ui/sidebar"

// Helper functions
function getSessionLabel(session: string | null) {
  switch (session) {
    case 'pre-market':
      return 'Pre-market';
    case 'regular':
      return 'Market Open';
    case 'after-hours':
      return 'After-hours';
    case 'closed-after-hours':
      return 'Market Closed';
    case 'closed-before-pre-market':
      return 'Market Closed';
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

// Calculate market status based on current time (EST)
function calculateMarketStatus(): string {
  const now = new Date()
  const estOffset = -5 // EST is UTC-5
  const utc = now.getTime() + (now.getTimezoneOffset() * 60000)
  const est = new Date(utc + (3600000 * estOffset))
  
  const day = est.getDay()
  const hour = est.getHours()
  const minute = est.getMinutes()
  const timeInMinutes = hour * 60 + minute
  
  if (day === 0 || day === 6) return 'Market Closed'
  if (timeInMinutes >= 240 && timeInMinutes < 570) return 'Pre-market'
  if (timeInMinutes >= 570 && timeInMinutes < 960) return 'Market Open'
  if (timeInMinutes >= 960 && timeInMinutes < 1200) return 'After-hours'
  return 'Market Closed'
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
  const [currentTime, setCurrentTime] = useState<Date | null>(null)
  const [marketStatus, setMarketStatus] = useState<MarketStatus | null>(null)
  const [fallbackStatus, setFallbackStatus] = useState<string>("Market Closed")
  const { toggleSidebar: toggleSidebarContext } = useSidebar()

  useEffect(() => {
    // Set initial values on client only
    setCurrentTime(new Date())
    setFallbackStatus(calculateMarketStatus())
    
    const timer = setInterval(() => {
      setCurrentTime(new Date())
      // Update fallback status every minute
      setFallbackStatus(calculateMarketStatus())
    }, 1000)
    
    // Try to get real market status from API
    getMarketStatus().then(status => {
      if (status) setMarketStatus(status)
    })
    
    const statusInterval = setInterval(() => {
      getMarketStatus().then(status => {
        if (status) setMarketStatus(status)
      })
    }, 60 * 1000)
    
    return () => {
      clearInterval(timer)
      clearInterval(statusInterval)
    }
  }, [])

  // Display status - use API if available, otherwise use calculated fallback
  const displayStatus = marketStatus 
    ? getSessionLabel(marketStatus.session) 
    : fallbackStatus

  return (
    <div className="bg-black border-b border-zinc-800 py-2 px-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={() => {
              // Keep existing desktop behavior
              toggleSidebar()
              // Also toggle Shadcn sidebar context so mobile sheet opens/closes
              toggleSidebarContext()
            }}
            className="p-2 rounded-md hover:bg-zinc-800 transition-colors"
          >
            <Menu className="h-5 w-5" />
          </button>

          <div className="hidden md:flex items-center gap-2">
            <div className="flex items-center gap-1.5">
              <Clock className="h-4 w-4 text-zinc-400" />
              <span className="text-sm font-medium">
                {currentTime 
                  ? currentTime.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
                  : "--:--"
                }
              </span>
            </div>

            <div className="h-4 w-px bg-zinc-700 mx-1"></div>

            <div className="flex items-center gap-1.5">
              {displayStatus === "Market Open" && <CircleDot className="h-4 w-4 text-emerald-500" />}
              {displayStatus === "Pre-market" && <Clock className="h-4 w-4 text-amber-500" />}
              {displayStatus === "After-hours" && <Moon className="h-4 w-4 text-blue-400" />}

              <span className="text-sm font-medium">
                {displayStatus}
              </span>

              {/* Pre-market: show time until regular market opens */}
              {marketStatus && marketStatus.session === 'pre-market' && marketStatus.next_open_time && (
                <span className="text-xs text-zinc-400 ml-1">
                  {getTimeUntilOpen(marketStatus.next_open_time)} until open
                </span>
              )}
              {/* Regular/After-hours: show time until close */}
              {marketStatus && (marketStatus.session === 'regular' || marketStatus.session === 'after-hours') && marketStatus.next_close_time && (
                <span className="text-xs text-zinc-400 ml-1">
                  {getTimeUntilClose(marketStatus.next_close_time)} until close
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="hidden md:block flex-1 max-w-md mx-4">
          <SearchWidget />
        </div>

        <div className="flex items-center gap-2">
          <NotificationWidget />
        </div>
      </div>
    </div>
  )
}
