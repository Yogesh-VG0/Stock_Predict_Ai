"use client"

import { useEffect, useRef, memo } from "react"

function TradingViewHeatmap() {
  const containerRef = useRef<HTMLDivElement>(null)
  const scriptLoadedRef = useRef(false)

  useEffect(() => {
    const container = containerRef.current
    if (!container || scriptLoadedRef.current) return
    
    scriptLoadedRef.current = true
    container.innerHTML = ""
    
    try {
      const script = document.createElement("script")
      script.src = "https://s3.tradingview.com/external-embedding/embed-widget-stock-heatmap.js"
      script.type = "text/javascript"
      script.async = true
      script.innerHTML = JSON.stringify({
        exchanges: [],
        dataSource: "SPX500",
        grouping: "sector",
        blockSize: "market_cap_basic",
        blockColor: "change",
        locale: "en",
        symbolUrl: "",
        colorTheme: "dark",
        hasTopBar: false,
        isDataSetEnabled: false,
        isZoomEnabled: true,
        hasSymbolTooltip: true,
        width: "100%",
        height: "400"
      })
      container.appendChild(script)
    } catch (error) {
      console.warn("Error loading TradingView Heatmap widget:", error)
    }
    
    return () => {
      if (container) container.innerHTML = ""
      scriptLoadedRef.current = false
    }
  }, [])

  return (
    <div className="rounded-lg bg-zinc-900 border border-zinc-800 shadow-lg overflow-hidden p-2">
      <div 
        ref={containerRef} 
        className="tradingview-widget-container" 
        style={{ minHeight: 400 }}
      />
    </div>
  )
}

export default memo(TradingViewHeatmap)
