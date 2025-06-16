"use client"

import { useEffect, useRef, useState } from "react"
import { motion } from "framer-motion"
import { Loader2 } from "lucide-react"

interface TradingViewAdvancedChartProps {
  symbol: string
  height?: number
  autosize?: boolean
  className?: string
}

export default function TradingViewAdvancedChart({
  symbol,
  height = 500,
  autosize = false,
  className = "",
}: TradingViewAdvancedChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [widgetKey, setWidgetKey] = useState(Date.now())
  const [isReady, setIsReady] = useState(false)

  const containerId = `tradingview-chart-${widgetKey}`

  // Determine the correct exchange prefix
  const getFullSymbol = (symbol: string) => {
    const nyseSymbols = [
      "JPM", "WMT", "DIS", "V", "PG", "JNJ", "XOM",
      "BAC", "PFE", "KO", "VZ", "T", "CVX", "MRK",
      "WFC", "C", "PEP", "HD", "MCD", "IBM",
    ]
    if (symbol.includes(":")) return symbol
    return nyseSymbols.includes(symbol) ? `NYSE:${symbol}` : `NASDAQ:${symbol}`
  }

  useEffect(() => {
    if (!window.TradingView) {
      const interval = setInterval(() => {
        if (window.TradingView) {
          clearInterval(interval)
          setIsReady(true)
        }
      }, 100)
      return () => clearInterval(interval)
    } else {
      setIsReady(true)
    }
  }, [])

  useEffect(() => {
    setWidgetKey(Date.now())
    setIsLoading(true)
  }, [symbol])

  useEffect(() => {
    if (!isReady || !containerRef.current) return

    try {
      containerRef.current.innerHTML = ""

      const fullSymbol = getFullSymbol(symbol)

      new window.TradingView.widget({
        container_id: containerId,
        symbol: fullSymbol,
        interval: "D",
        timezone: "Etc/UTC",
        theme: "dark",
        style: "1",
        locale: "en",
        toolbar_bg: "#000000",
        enable_publishing: false,
        hide_top_toolbar: true,
        hide_legend: true,
        withdateranges: true,
        save_image: true,
        show_popup_button: false,
        popup_width: "1000",
        popup_height: "650",
        autosize: autosize,
        height: height,
        width: "100%",
        allow_symbol_change: true,
        details: true,
        hotlist: false,
        calendar: true,
        overrides: {
          "paneProperties.background": "#000000",
          "paneProperties.vertGridProperties.color": "#1E1E1E",
          "paneProperties.horzGridProperties.color": "#1E1E1E",
          "symbolWatermarkProperties.transparency": 90,
          "scalesProperties.textColor": "#AAA",
          "scalesProperties.textSize": 14, // 🔍 bigger text on price/time axes
          "mainSeriesProperties.priceLineWidth": 2, // bold price line
          "paneProperties.legendProperties.showStudyArguments": true,
          "paneProperties.legendProperties.showStudyTitles": true,
          "paneProperties.legendProperties.showStudyValues": true,
          "paneProperties.legendProperties.showSeriesTitle": true,
          "paneProperties.legendProperties.showSeriesOHLC": true,
          "paneProperties.legendProperties.showBarChange": true,
          "paneProperties.legendProperties.showLegend": true,
        },
        studies_overrides: {},
        charts_storage_api_version: "1.1",
        client_id: "tradingview.com",
        user_id: "public_user",
        loading_screen: { backgroundColor: "#000000", foregroundColor: "#1E1E1E" },
      })

      const iframeCheck = setInterval(() => {
        const iframe = containerRef.current?.querySelector("iframe")
        if (iframe) {
          setIsLoading(false)
          clearInterval(iframeCheck)
        }
      }, 300) // checking every 300ms is more efficient

      return () => clearInterval(iframeCheck)
    } catch (error) {
      console.error("TradingView Chart init error:", error)
      setIsLoading(false)
    }
  }, [isReady, widgetKey, symbol, height, autosize, containerId])

  return (
    <div className={`relative w-full ${className}`}>
      {isLoading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 z-10 rounded-lg"
        >
          <div className="flex flex-col items-center gap-2">
            <Loader2 className="h-8 w-8 text-emerald-500 animate-spin" />
            <p className="text-sm text-zinc-300">Loading chart for {symbol}...</p>
          </div>
        </motion.div>
      )}
      <div
        id={containerId}
        ref={containerRef}
        className="w-full rounded-lg overflow-hidden border border-zinc-800 bg-black"
        style={{ height: autosize ? "100%" : `${height}px` }}
      />
    </div>
  )
}
