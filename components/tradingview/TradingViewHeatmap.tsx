"use client"

import { useEffect, useRef, useState } from "react"

export default function TradingViewHeatmap() {
  const containerRef = useRef<HTMLDivElement>(null)
  const [isMounted, setIsMounted] = useState(false)

  useEffect(() => {
    setIsMounted(true)
  }, [])

  useEffect(() => {
    if (!isMounted) return
    
    const container = containerRef.current
    if (!container) return
    
    container.innerHTML = ""
    
    const timeoutId = setTimeout(() => {
      try {
        const script = document.createElement("script")
        script.src = "https://s3.tradingview.com/external-embedding/embed-widget-stock-heatmap.js"
        script.type = "text/javascript"
        script.async = true
        script.onerror = () => {
          console.warn("Failed to load TradingView Heatmap widget")
        }
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
    }, 150) // Slightly different delay to stagger loading
    
    return () => {
      clearTimeout(timeoutId)
      if (container) container.innerHTML = ""
    }
  }, [isMounted])

  if (!isMounted) {
    return (
      <div className="rounded-lg bg-zinc-900 border border-zinc-800 shadow-lg overflow-hidden p-2">
        <div className="tradingview-widget-container" style={{ minHeight: 400 }}>
          <div className="flex items-center justify-center h-full">
            <div className="text-zinc-400 text-sm">Loading heatmap...</div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="rounded-lg bg-zinc-900 border border-zinc-800 shadow-lg overflow-hidden p-2">
      <div className="tradingview-widget-container" style={{ minHeight: 400 }}>
        <div ref={containerRef} className="tradingview-widget-container__widget" />
        <div className="tradingview-widget-copyright">
          <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
          </a>
        </div>
      </div>
    </div>
  )
} 