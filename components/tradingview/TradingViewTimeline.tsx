"use client"

import { useEffect, useRef, memo } from "react"

function TradingViewTimeline() {
  const containerRef = useRef<HTMLDivElement>(null)
  const scriptLoadedRef = useRef(false)

  useEffect(() => {
    const container = containerRef.current
    if (!container || scriptLoadedRef.current) return
    
    scriptLoadedRef.current = true
    container.innerHTML = ""
    
    try {
      const script = document.createElement("script")
      script.src = "https://s3.tradingview.com/external-embedding/embed-widget-timeline.js"
      script.type = "text/javascript"
      script.async = true
      script.innerHTML = JSON.stringify({
        feedMode: "all_symbols",
        colorTheme: "dark",
        isTransparent: false,
        displayMode: "regular",
        width: "100%",
        height: "500",
        locale: "en"
      })
      container.appendChild(script)
    } catch (error) {
      console.warn("Error loading TradingView Timeline widget:", error)
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
        style={{ minHeight: 500 }}
      />
    </div>
  )
}

export default memo(TradingViewTimeline)
