"use client"

import { useEffect, useRef, memo, useMemo } from "react"
import WidgetScrollWrapper from "@/components/ui/widget-scroll-wrapper"

interface TradingViewCompanyProfileProps {
  symbol: string
  height?: number
}

// Shared exchange mapping for TradingView widgets
const NYSE_SYMBOLS = new Set([
  // Financials
  "JPM", "BAC", "WFC", "GS", "MS", "AXP", "C", "BRK.B", "V", "MA",
  // Healthcare
  "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "AMGN", "CVS",
  // Consumer Staples
  "PG", "KO", "PEP", "MDLZ",
  // Consumer Discretionary
  "HD", "LOW", "NKE", "MCD", "DIS", "TGT",
  // Energy
  "XOM", "CVX",
  // Industrials
  "CAT", "HON", "BA", "RTX", "LMT", "DE", "GE", "FDX", "UPS",
  // Communication
  "VZ", "T",
  // Other
  "LIN", "NEE", "AMT",
  // Tech on NYSE
  "IBM", "CRM", "ORCL"
])

const NASDAQ_SYMBOLS = new Set([
  // Technology
  "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "AMD", "INTC",
  "CSCO", "ADBE", "QCOM", "TXN", "INTU", "AMAT", "PYPL", "PLTR",
  // Consumer Discretionary
  "AMZN", "TSLA", "NFLX", "SBUX", "BKNG",
  // Healthcare
  "GILD", "ISRG",
  // Communication
  "CMCSA", "CHTR", "TMUS",
  // Consumer Staples
  "WMT", "COST"
])

function getFullSymbol(symbol: string): string {
  if (symbol.includes(":")) return symbol
  const normalized = symbol === "BRK-B" ? "BRK.B" : symbol
  if (NYSE_SYMBOLS.has(normalized)) return `NYSE:${normalized}`
  if (NASDAQ_SYMBOLS.has(normalized)) return `NASDAQ:${normalized}`
  return `NYSE:${normalized}`
}

function TradingViewCompanyProfile({ symbol, height = 550 }: TradingViewCompanyProfileProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const prevSymbolRef = useRef<string>("")
  
  const fullSymbol = useMemo(() => getFullSymbol(symbol), [symbol])

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
