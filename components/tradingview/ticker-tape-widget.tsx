"use client"

import { useEffect, useRef, useState } from "react"
import { motion } from "framer-motion"
import { Loader2 } from "lucide-react"

export default function TickerTapeWidget() {
  const containerRef = useRef<HTMLDivElement>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    // Always clear the container before injecting the script
    container.innerHTML = ""
    setIsLoading(true)
    const script = document.createElement("script")
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js"
    script.async = true
    script.innerHTML = JSON.stringify({
      symbols: [
        { proName: "FOREXCOM:SPXUSD", title: "S&P 500" },
        { proName: "FOREXCOM:NSXUSD", title: "Nasdaq 100" },
        { proName: "FOREXCOM:DJI", title: "Dow 30" },
        { proName: "NASDAQ:AAPL", title: "Apple" },
        { proName: "NASDAQ:MSFT", title: "Microsoft" },
        { proName: "NASDAQ:AMZN", title: "Amazon" },
        { proName: "NASDAQ:GOOGL", title: "Google" },
        { proName: "BINANCE:BTCUSDT", title: "Bitcoin" },
        { proName: "BINANCE:ETHUSDT", title: "Ethereum" },
        { proName: "FX:EURUSD", title: "EUR/USD" }
      ],
      showSymbolLogo: true,
      colorTheme: "dark",
      isTransparent: true,
      displayMode: "adaptive",
      locale: "en"
    })
    container.appendChild(script)

    // Widget is ready when script loads (not always reliable, but best effort)
    script.onload = () => setIsLoading(false)
    script.onerror = () => setIsLoading(false)

    return () => {
      // Clean up the widget
      if (container) container.innerHTML = ""
    }
  }, [])

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="w-full bg-black border-b border-zinc-800"
    >
      {isLoading && (
        <div className="flex items-center justify-center h-[46px] bg-black">
          <Loader2 className="h-5 w-5 text-zinc-400 animate-spin" />
        </div>
      )}
      <div ref={containerRef} className={isLoading ? "hidden" : "block"} />
    </motion.div>
  )
}
