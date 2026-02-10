"use client"

import { useState, useEffect, memo } from "react"
import { motion } from "framer-motion"
import { Clock, BarChart3, Activity, Newspaper } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import MarketSentimentBanner from "@/components/market/market-sentiment-banner"
import EnhancedQuickPredictionWidget from "@/components/market/EnhancedQuickPredictionWidget"
import { getMarketStatus, MarketStatus } from "@/lib/api"
import { TradingHoursBar } from "@/components/market/TradingHoursBar"

// Direct imports for immediate loading - no lazy loading for visible content
import TradingViewHotlistsWidget from "@/components/tradingview/TradingViewHotlistsWidget"
import TradingViewEconomicCalendar from "@/components/tradingview/TradingViewEconomicCalendar"
import TradingViewHeatmap from "@/components/tradingview/TradingViewHeatmap"
import TradingViewTimeline from "@/components/tradingview/TradingViewTimeline"
import FinlogixEarningsCalendar from "@/components/tradingview/FinlogixEarningsCalendar"
import TradingViewSingleQuote from "@/components/tradingview/TradingViewSingleQuote"

// Memoized card wrapper to prevent re-renders
const WidgetCard = memo(({ 
  title, 
  icon: Icon, 
  iconColor, 
  children,
  minHeight 
}: { 
  title: string
  icon: React.ElementType
  iconColor: string
  children: React.ReactNode
  minHeight?: number
}) => (
  <Card className="p-0 overflow-hidden" style={minHeight ? { minHeight } : undefined}>
    <div className="p-4 pb-0">
      <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Icon className={`h-5 w-5 ${iconColor}`} />
        {title}
      </h2>
    </div>
    <div className="p-0">{children}</div>
  </Card>
))
WidgetCard.displayName = "WidgetCard"

export default function HomePage() {
  const [marketStatus, setMarketStatus] = useState<MarketStatus | null>(null)

  useEffect(() => {
    getMarketStatus().then((status) => setMarketStatus(status))
  }, [])

  return (
    <div className="space-y-6">
      {/* Page header - stack on mobile, row on desktop */}
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between md:gap-6">
        <motion.h1 
          initial={{ opacity: 0, y: -10 }} 
          animate={{ opacity: 1, y: 0 }} 
          className="text-2xl font-bold"
        >
          Market Dashboard
        </motion.h1>
        <div className="w-full md:w-[400px] min-w-0 flex-shrink-0">
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
                <span className="inline-block">
                  <svg width="20" height="20" fill="none" viewBox="0 0 24 24">
                    <path fill="#3b82f6" d="M12 2a10 10 0 100 20 10 10 0 000-20zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
                  </svg>
                </span>
                Top Gainers & Losers
              </h2>
            </div>
            <div className="p-0">
              <TradingViewHotlistsWidget />
            </div>
          </Card>

          {/* Market Heatmap */}
          <WidgetCard title="Market Heatmap" icon={BarChart3} iconColor="text-purple-500">
            <TradingViewHeatmap />
          </WidgetCard>

          {/* Economic Calendar */}
          <WidgetCard title="Economic Calendar" icon={Clock} iconColor="text-amber-500">
            <TradingViewEconomicCalendar />
          </WidgetCard>
        </div>

        <div className="space-y-6">
          <EnhancedQuickPredictionWidget />

          {/* Market Overview */}
          <Card className="hover:border-zinc-700 transition-colors" style={{ minHeight: 540 }}>
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <Activity className="h-4 w-4 text-blue-500" />
                Market Overview
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="space-y-4">
                <div style={{ height: 126, width: "100%" }}>
                  <TradingViewSingleQuote symbol="OANDA:SPX500USD" height={126} />
                </div>
                <div style={{ height: 126, width: "100%" }}>
                  <TradingViewSingleQuote symbol="OANDA:NAS100USD" height={126} />
                </div>
                <div style={{ height: 126, width: "100%" }}>
                  <TradingViewSingleQuote symbol="CAPITALCOM:VIX" height={126} />
                </div>
                <div style={{ height: 126, width: "100%" }}>
                  <TradingViewSingleQuote symbol="OANDA:EURUSD" height={126} />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Top Stories */}
          <WidgetCard title="Top Stories" icon={Newspaper} iconColor="text-orange-500">
            <TradingViewTimeline />
          </WidgetCard>

          {/* Earnings Calendar */}
          <WidgetCard title="Earnings Calendar" icon={BarChart3} iconColor="text-green-500">
            <FinlogixEarningsCalendar />
          </WidgetCard>
        </div>
      </div>
    </div>
  )
}
