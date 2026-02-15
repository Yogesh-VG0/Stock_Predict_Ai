"use client"

import { useEffect, useRef, memo } from "react"

interface TradingViewCompanyProfileProps {
  symbol: string
  height?: number
}

function TradingViewCompanyProfile({ symbol, height = 550 }: TradingViewCompanyProfileProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const prevSymbolRef = useRef<string>("")

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    
    // Skip if same symbol already loaded
    if (prevSymbolRef.current === symbol && container.querySelector('script')) return
    
    prevSymbolRef.current = symbol
    container.innerHTML = ""

    try {
      const widgetContainer = document.createElement("div")
      widgetContainer.className = "tradingview-widget-container__widget"
      container.appendChild(widgetContainer)

      const script = document.createElement("script")
      script.src = "https://s3.tradingview.com/external-embedding/embed-widget-symbol-profile.js"
      script.type = "text/javascript"
      script.async = true
      script.innerHTML = JSON.stringify({
        symbol: symbol,
        colorTheme: "dark",
        isTransparent: true,
        width: "100%",
        height: height,
        locale: "en"
      })
      container.appendChild(script)
    } catch (error) {
      console.warn("Error loading TradingView Company Profile widget:", error)
    }

    return () => {
      if (container) container.innerHTML = ""
      prevSymbolRef.current = ""
    }
  }, [symbol, height])

  return (
    <div 
      ref={containerRef} 
      className="tradingview-widget-container" 
      style={{ minHeight: height, width: "100%" }}
    />
  )
}

export default memo(TradingViewCompanyProfile)
