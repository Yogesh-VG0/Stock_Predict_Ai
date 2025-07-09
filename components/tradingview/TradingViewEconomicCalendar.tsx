"use client"

import { useEffect, useRef, useState } from "react"

export default function TradingViewEconomicCalendar() {
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
        script.src = "https://s3.tradingview.com/external-embedding/embed-widget-events.js"
        script.type = "text/javascript"
        script.async = true
        script.onerror = () => {
          console.warn("Failed to load TradingView Economic Calendar widget")
        }
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
    }, 250) // Different delay to stagger loading
    
    return () => {
      clearTimeout(timeoutId)
      if (container) container.innerHTML = ""
    }
  }, [isMounted])

  if (!isMounted) {
    return (
      <div className="rounded-lg bg-zinc-900 border border-zinc-800 shadow-lg overflow-hidden p-2">
        <div className="tradingview-widget-container" style={{ minHeight: 550 }}>
          <div className="flex items-center justify-center h-full">
            <div className="text-zinc-400 text-sm">Loading economic calendar...</div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="rounded-lg bg-zinc-900 border border-zinc-800 shadow-lg overflow-hidden p-2">
      <div className="tradingview-widget-container" style={{ minHeight: 550 }}>
        <div ref={containerRef} className="tradingview-widget-container__widget" />
        <div className="tradingview-widget-copyright">
          <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
          </a>
        </div>
      </div>
    </div>
  )
} 