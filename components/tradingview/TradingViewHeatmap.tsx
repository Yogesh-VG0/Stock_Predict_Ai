"use client"

import { useEffect, useRef } from "react"

export default function TradingViewHeatmap() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    // Clear any existing content
    container.innerHTML = ""

    const script = document.createElement("script")
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-stock-heatmap.js"
    script.type = "text/javascript"
    script.async = true
    script.innerHTML = JSON.stringify({
      exchanges: [],
      dataSource: "NASDAQ100",
      grouping: "sector",
      blockSize: "market_cap_basic",
      blockColor: "change",
      locale: "en",
      symbolUrl: "",
      colorTheme: "dark",
      hasTopBar: true,
      isDataSetEnabled: true,
      isZoomEnabled: true,
      hasSymbolTooltip: true,
      isMonoSize: false,
      width: "100%",
      height: "100%"
    })

    container.appendChild(script)

    return () => {
      if (container) container.innerHTML = ""
    }
  }, [])

  return (
    <div className="rounded-lg bg-zinc-900 border border-zinc-800 shadow-lg overflow-hidden">
      <div className="tradingview-widget-container" style={{ height: "600px" }}>
        <div ref={containerRef} className="tradingview-widget-container__widget" style={{ height: "560px" }} />
        <div className="tradingview-widget-copyright" style={{ height: "40px", fontSize: "12px", lineHeight: "40px", textAlign: "center" }}>
          <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
            <span className="blue-text">Track all markets on TradingView</span>
          </a>
        </div>
      </div>
    </div>
  )
} 