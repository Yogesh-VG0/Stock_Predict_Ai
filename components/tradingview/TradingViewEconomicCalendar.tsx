"use client"

import { useEffect, useRef, memo } from "react"

function TradingViewEconomicCalendar() {
  const containerRef = useRef<HTMLDivElement>(null)
  const scriptLoadedRef = useRef(false)

  useEffect(() => {
    const container = containerRef.current
    if (!container || scriptLoadedRef.current) return
    
    scriptLoadedRef.current = true
    container.innerHTML = ""
    
    try {
      const script = document.createElement("script")
      script.src = "https://s3.tradingview.com/external-embedding/embed-widget-events.js"
      script.type = "text/javascript"
      script.async = true
      script.innerHTML = JSON.stringify({
        colorTheme: "dark",
        isTransparent: false,
        width: "100%",
        height: "550",
        locale: "en",
        importanceFilter: "-1,0,1",
        countryFilter: "us,eu,jp,gb,ch,au,ca,nz,cn"
      })
      container.appendChild(script)
    } catch (error) {
      console.warn("Error loading TradingView Economic Calendar widget:", error)
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

export default memo(TradingViewEconomicCalendar)
