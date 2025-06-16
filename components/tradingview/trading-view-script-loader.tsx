"use client"

import { useEffect, useState } from "react"

declare global {
  interface Window {
    TradingView?: any
  }
}

export default function TradingViewScriptLoader() {
  const [isLoaded, setIsLoaded] = useState(false)
  const [hasError, setHasError] = useState(false)

  useEffect(() => {
    // Skip if already loaded or if we're in SSR
    if (isLoaded || typeof window === "undefined" || window.TradingView) {
      setIsLoaded(true)
      return
    }

    const script = document.createElement("script")
    script.src = "https://s3.tradingview.com/tv.js"
    script.async = true
    script.onload = () => {
      console.log("TradingView script loaded successfully")
      setIsLoaded(true)
    }
    script.onerror = () => {
      console.error("Failed to load TradingView script")
      setHasError(true)
    }

    document.head.appendChild(script)

    return () => {
      // We don't remove the script on unmount as it's needed globally
    }
  }, [isLoaded])

  // This component doesn't render anything visible
  return null
}
