"use client"

import { useEffect, useRef, memo, useMemo } from "react"
import WidgetScrollWrapper from "@/components/ui/widget-scroll-wrapper"
import { getPreferredTradingViewSymbol } from "@/lib/tradingview-symbol"

interface TradingViewCompanyProfileProps {
  symbol: string
  exchange?: string
  height?: number
}

function TradingViewCompanyProfile({ symbol, exchange, height = 550 }: TradingViewCompanyProfileProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const prevSymbolRef = useRef<string>("")
  
  const fullSymbol = useMemo(() => getPreferredTradingViewSymbol(symbol, exchange), [symbol, exchange])

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    
    // Skip if same symbol already loaded
    if (prevSymbolRef.current === fullSymbol && container.querySelector('script')) return
    
    prevSymbolRef.current = fullSymbol
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
        symbol: fullSymbol,
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
  }, [fullSymbol, height])

  return (
    <WidgetScrollWrapper>
      <div 
        ref={containerRef} 
        className="tradingview-widget-container" 
        style={{ minHeight: height, width: "100%" }}
      />
    </WidgetScrollWrapper>
  )
}

export default memo(TradingViewCompanyProfile)
