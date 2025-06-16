"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Clock, BarChart3, Activity, Zap } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import MarketSentimentBanner from "@/components/market/market-sentiment-banner"
import QuickPredictionWidget from "@/components/market/quick-prediction-widget"
import { getMarketStatus, MarketStatus } from "@/lib/api"
import { TradingHoursBar } from "@/components/market/TradingHoursBar"
import TradingViewHotlistsWidget from "@/components/tradingview/TradingViewHotlistsWidget"

export default function HomePage() {
  const [marketStatus, setMarketStatus] = useState<MarketStatus | null>(null)

  useEffect(() => {
    getMarketStatus().then((status) => setMarketStatus(status))
  }, [])

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-4">
        <motion.h1 initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="text-2xl font-bold">
          Market Dashboard
        </motion.h1>
        <div className="w-2/3 max-w-lg">
          <TradingHoursBar compact />
        </div>
      </div>

      <MarketSentimentBanner />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
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
        </div>

        <div className="space-y-6">
          <QuickPredictionWidget />

          <div>
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Activity className="h-5 w-5 text-blue-500" />
              Market Movers
            </h2>

            <div className="space-y-4">
              <Card className="hover:border-zinc-700 transition-colors">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Zap className="h-4 w-4 text-amber-500" />
                    Most Volatile
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="flex items-center justify-between py-2 border-b border-zinc-800">
                    <div className="flex items-center gap-2">
                      <div className="font-medium">GME</div>
                      <div className="text-xs text-zinc-400">GameStop</div>
                    </div>
                    <div className="text-emerald-500 text-sm font-medium">+12.4%</div>
                  </div>
                  <div className="flex items-center justify-between py-2 border-b border-zinc-800">
                    <div className="flex items-center gap-2">
                      <div className="font-medium">AMC</div>
                      <div className="text-xs text-zinc-400">AMC Entertainment</div>
                    </div>
                    <div className="text-red-500 text-sm font-medium">-8.7%</div>
                  </div>
                  <div className="flex items-center justify-between py-2">
                    <div className="flex items-center gap-2">
                      <div className="font-medium">RIVN</div>
                      <div className="text-xs text-zinc-400">Rivian Automotive</div>
                    </div>
                    <div className="text-emerald-500 text-sm font-medium">+7.2%</div>
                  </div>
                </CardContent>
              </Card>

              <Card className="hover:border-zinc-700 transition-colors">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base flex items-center gap-2">
                    <BarChart3 className="h-4 w-4 text-purple-500" />
                    Top Volume
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="flex items-center justify-between py-2 border-b border-zinc-800">
                    <div className="flex items-center gap-2">
                      <div className="font-medium">AAPL</div>
                      <div className="text-xs text-zinc-400">Apple Inc.</div>
                    </div>
                    <div className="text-xs text-zinc-400">89.4M shares</div>
                  </div>
                  <div className="flex items-center justify-between py-2 border-b border-zinc-800">
                    <div className="flex items-center gap-2">
                      <div className="font-medium">SPY</div>
                      <div className="text-xs text-zinc-400">SPDR S&P 500 ETF</div>
                    </div>
                    <div className="text-xs text-zinc-400">76.2M shares</div>
                  </div>
                  <div className="flex items-center justify-between py-2">
                    <div className="flex items-center gap-2">
                      <div className="font-medium">TSLA</div>
                      <div className="text-xs text-zinc-400">Tesla, Inc.</div>
                    </div>
                    <div className="text-xs text-zinc-400">62.8M shares</div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
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
