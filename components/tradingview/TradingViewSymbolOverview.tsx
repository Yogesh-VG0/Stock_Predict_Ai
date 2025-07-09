"use client"

import { useEffect, useRef } from "react"

interface TradingViewSymbolOverviewProps {
  symbol: string
  height?: number
}

export default function TradingViewSymbolOverview({ symbol, height = 500 }: TradingViewSymbolOverviewProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    // Clear any existing content
    container.innerHTML = ""

    const script = document.createElement("script")
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-symbol-overview.js"
    script.type = "text/javascript"
    script.async = true
    script.innerHTML = JSON.stringify({
      symbols: [
        [
          symbol === "AAPL" ? "Apple" : symbol === "MSFT" ? "Microsoft" : symbol === "GOOGL" ? "Google" : symbol,
          `${symbol}|1D`
        ]
      ],
      chartOnly: false,
      width: "100%",
      height: "100%",
      locale: "en",
      colorTheme: "dark",
      autosize: true,
      showVolume: false,
      showMA: false,
      hideDateRanges: false,
      hideMarketStatus: false,
      hideSymbolLogo: false,
      scalePosition: "right",
      scaleMode: "Normal",
      fontFamily: "-apple-system, BlinkMacSystemFont, Trebuchet MS, Roboto, Ubuntu, sans-serif",
      fontSize: "10",
      noTimeScale: false,
      valuesTracking: "1",
      changeMode: "price-and-percent",
      chartType: "area",
      maLineColor: "#2962FF",
      maLineWidth: 1,
      maLength: 9,
      headerFontSize: "medium",
      backgroundColor: "rgba(0, 0, 0, 1)",
      lineWidth: 2,
      lineType: 0,
      dateRanges: [
        "1d|1",
        "1m|30",
        "3m|60",
        "12m|1D",
        "60m|1W",
        "all|1M"
      ]
    })

    container.appendChild(script)

    return () => {
      if (container) container.innerHTML = ""
    }
  }, [symbol])

  return (
    <div className="rounded-lg bg-zinc-900 border border-zinc-800 shadow-lg overflow-hidden">
      <div className="tradingview-widget-container" style={{ height: `${height}px` }}>
        <div ref={containerRef} className="tradingview-widget-container__widget" style={{ height: `${height - 40}px` }} />
        <div className="tradingview-widget-copyright" style={{ height: "40px", fontSize: "12px", lineHeight: "40px", textAlign: "center" }}>
          <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
          </a>
        </div>
      </div>
    </div>
  )
} 