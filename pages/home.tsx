"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Clock, BarChart3, Activity, Zap, Newspaper } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import MarketSentimentBanner from "@/components/market/market-sentiment-banner"
import EnhancedQuickPredictionWidget from "@/components/market/EnhancedQuickPredictionWidget"
import { getMarketStatus, MarketStatus } from "@/lib/api"
import { TradingHoursBar } from "@/components/market/TradingHoursBar"
import TradingViewHotlistsWidget from "@/components/tradingview/TradingViewHotlistsWidget"
import TradingViewEconomicCalendar from "@/components/tradingview/TradingViewEconomicCalendar"
import TradingViewHeatmap from "@/components/tradingview/TradingViewHeatmap"
import TradingViewTimeline from "@/components/tradingview/TradingViewTimeline"
import FinlogixEarningsCalendar from "@/components/tradingview/FinlogixEarningsCalendar"



export default function HomePage() {
  const [marketStatus, setMarketStatus] = useState<MarketStatus | null>(null)

  useEffect(() => {
    getMarketStatus().then((status) => setMarketStatus(status))
  }, [])

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-6">
        <motion.h1 initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="text-2xl font-bold">
          Market Dashboard
        </motion.h1>
        <div className="min-w-0 flex-shrink-0" style={{ width: "400px" }}>
          <TradingHoursBar compact />
        </div>
      </div>

      <MarketSentimentBanner />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          {/* Top Gainers & Losers */}
          <Card className="p-0 overflow-hidden">
            <div className="p-4 pb-0">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <span className="inline-block"><svg width="20" height="20" fill="none" viewBox="0 0 24 24"><path fill="#3b82f6" d="M12 2a10 10 0 100 20 10 10 0 000-20zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/></svg></span>
                Top Gainers & Losers
              </h2>
            </div>
            <div className="p-0">
              <TradingViewHotlistsWidget />
            </div>
          </Card>

          {/* Market Heatmap */}
          <Card className="p-0 overflow-hidden">
            <div className="p-4 pb-0">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-purple-500" />
                Market Heatmap
              </h2>
            </div>
            <div className="p-0">
              <TradingViewHeatmap />
            </div>
          </Card>

          {/* Economic Calendar */}
          <Card className="p-0 overflow-hidden">
            <div className="p-4 pb-0">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Clock className="h-5 w-5 text-amber-500" />
                Economic Calendar
              </h2>
            </div>
            <div className="p-0">
              <TradingViewEconomicCalendar />
            </div>
          </Card>
        </div>

        <div className="space-y-6">
          <EnhancedQuickPredictionWidget />

          {/* Market Indices */}
          <Card className="hover:border-zinc-700 transition-colors">
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <Activity className="h-4 w-4 text-blue-500" />
                Major Indices
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">S&P 500</div>
                    <div className="text-xs text-zinc-400">^GSPC</div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium">Available in ticker tape</div>
                    <div className="text-xs text-zinc-400">Real-time</div>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">NASDAQ</div>
                    <div className="text-xs text-zinc-400">^IXIC</div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium">Available in ticker tape</div>
                    <div className="text-xs text-zinc-400">Real-time</div>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">Dow Jones</div>
                    <div className="text-xs text-zinc-400">^DJI</div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium">Available in ticker tape</div>
                    <div className="text-xs text-zinc-400">Real-time</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Quick Stats */}
          <Card className="hover:border-zinc-700 transition-colors">
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <Zap className="h-4 w-4 text-amber-500" />
                Market Stats
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-zinc-400">VIX</span>
                  <span className="text-sm">View in ticker tape</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-zinc-400">10Y Treasury</span>
                  <span className="text-sm">Real-time data</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-zinc-400">USD/EUR</span>
                  <span className="text-sm">Live forex</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Top Stories */}
          <Card className="p-0 overflow-hidden">
            <div className="p-4 pb-0">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Newspaper className="h-5 w-5 text-orange-500" />
                Top Stories
              </h2>
            </div>
            <div className="p-0">
              <TradingViewTimeline />
            </div>
          </Card>

          {/* Earnings Calendar */}
          <Card className="p-0 overflow-hidden">
            <div className="p-4 pb-0">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-green-500" />
                Earnings Calendar
              </h2>
            </div>
            <div className="p-0">
              <FinlogixEarningsCalendar />
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}

// Helper function
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

function formatTime(iso: string, timezone: string) {
  try {
    const date = new Date(iso)
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', timeZone: timezone })
  } catch {
    return ''
  }
}

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
