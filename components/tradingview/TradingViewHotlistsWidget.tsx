"use client"

import { useEffect, useRef, memo } from "react"

function TradingViewHotlistsWidget() {
  const containerRef = useRef<HTMLDivElement>(null)
  const scriptLoadedRef = useRef(false)

  useEffect(() => {
    const container = containerRef.current
    if (!container || scriptLoadedRef.current) return
    
    scriptLoadedRef.current = true
    container.innerHTML = "" // Clear any existing content
    
    try {
      const script = document.createElement("script")
      script.src = "https://s3.tradingview.com/external-embedding/embed-widget-hotlists.js"
      script.type = "text/javascript"
      script.async = true
      script.innerHTML = JSON.stringify({
        colorTheme: "dark",
        exchange: "US",
        showChart: true,
        locale: "en",
        isTransparent: false,
        showSymbolLogo: false,
        showFloatingTooltip: false,
        width: "100%",
        height: "550",
        plotLineColorGrowing: "#3b82f6",
        plotLineColorFalling: "#ef4444",
        gridLineColor: "rgba(42, 46, 57, 0)",
        scaleFontColor: "#e5e7eb",
        belowLineFillColorGrowing: "#3b82f622",
        belowLineFillColorFalling: "#ef444422",
        symbolActiveColor: "#3b82f622"
      })
      container.appendChild(script)
    } catch (error) {
      console.warn("Error loading TradingView Hotlists widget:", error)
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
        style={{ minHeight: 550 }}
      />
    </div>
  )
}

export default memo(TradingViewHotlistsWidget)
