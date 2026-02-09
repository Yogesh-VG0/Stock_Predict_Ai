"use client"

import { useEffect, useRef, memo } from 'react'

interface TradingViewSingleQuoteProps {
  symbol: string
  width?: string | number
  height?: string | number
  isTransparent?: boolean
  colorTheme?: 'light' | 'dark'
  locale?: string
}

function TradingViewSingleQuote({
  symbol,
  width = "100%",
  height = 80,
  isTransparent = false,
  colorTheme = "dark",
  locale = "en"
}: TradingViewSingleQuoteProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const scriptLoadedRef = useRef(false)
  const currentSymbolRef = useRef(symbol)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    
    // Only reload if symbol changed or first load
    if (scriptLoadedRef.current && currentSymbolRef.current === symbol) return
    
    scriptLoadedRef.current = true
    currentSymbolRef.current = symbol
    container.innerHTML = ""
    
    try {
      const script = document.createElement('script')
      script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-single-quote.js'
      script.type = 'text/javascript'
      script.async = true
      script.innerHTML = JSON.stringify({
        symbol,
        width,
        height,
        colorTheme,
        isTransparent,
        locale
      })
      container.appendChild(script)
    } catch (error) {
      console.warn(`Error loading TradingView Single Quote widget for ${symbol}:`, error)
    }
    
    return () => {
      if (container) container.innerHTML = ""
      scriptLoadedRef.current = false
    }
  }, [symbol, width, height, colorTheme, isTransparent, locale])

  return (
    <div 
      ref={containerRef} 
      className="tradingview-widget-container"
      style={{ height, width }}
    />
  )
}

export default memo(TradingViewSingleQuote)
