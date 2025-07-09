"use client"

import { useEffect, useRef, useState } from 'react'

interface TradingViewSingleQuoteProps {
  symbol: string
  width?: string | number
  height?: string | number
  isTransparent?: boolean
  colorTheme?: 'light' | 'dark'
  locale?: string
}

export default function TradingViewSingleQuote({
  symbol,
  width = "100%",
  height = 80,
  isTransparent = false,
  colorTheme = "dark",
  locale = "en"
}: TradingViewSingleQuoteProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [isMounted, setIsMounted] = useState(false)

  useEffect(() => {
    setIsMounted(true)
  }, [])

  useEffect(() => {
    if (!isMounted) return
    
    const container = containerRef.current
    if (!container) return
    
    container.innerHTML = ''
    
    const timeoutId = setTimeout(() => {
      try {
        const script = document.createElement('script')
        script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-single-quote.js'
        script.type = 'text/javascript'
        script.async = true
        script.onerror = () => {
          console.warn(`Failed to load TradingView Single Quote widget for ${symbol}`)
        }
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
    }, Math.random() * 100 + 300) // Random delay to stagger loading
    
    return () => {
      clearTimeout(timeoutId)
      if (container) container.innerHTML = ''
    }
  }, [isMounted, symbol, width, height, colorTheme, isTransparent, locale])

  if (!isMounted) {
    return (
      <div style={{ height: height }}>
        <div className="flex items-center justify-center h-full">
          <div className="text-zinc-400 text-xs">Loading {symbol}...</div>
        </div>
      </div>
    )
  }

  return (
    <div style={{ height: height }}>
      <div ref={containerRef} className="tradingview-widget-container__widget" />
    </div>
  )
} 