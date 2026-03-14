"use client"

import { useState, useEffect, useRef } from "react"
import { motion } from "framer-motion"
import { TrendingUp, TrendingDown, Minus, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { API_BASE_URL } from "@/lib/api"
import { backendReady } from "@/lib/backend-health"

// Map API valueText to sentiment type
function mapValueTextToSentiment(valueText: string): "bullish" | "bearish" | "neutral" {
  if (!valueText) return "neutral";
  const text = valueText.toLowerCase();
  if (text.includes("greed")) return "bullish";
  if (text.includes("fear")) return "bearish";
  return "neutral";
}

interface MarketSentimentBannerProps {
  className?: string
}

export default function MarketSentimentBanner({ className }: MarketSentimentBannerProps) {
  const [sentiment, setSentiment] = useState<"bullish" | "bearish" | "neutral">("neutral")
  const [fgi, setFgi] = useState<{ value: number; valueText: string } | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    const fetchSentiment = async (retries = 3, delay = 4000) => {
      // Wait for backend to be alive before fetching
      await backendReady()
      for (let attempt = 0; attempt <= retries; attempt++) {
        try {
          const res = await fetch(`${API_BASE_URL}/api/market/sentiment`)
          if (!mountedRef.current) return
          if (!res.ok) throw new Error(`HTTP ${res.status}`)
          const data = await res.json()
          if (data && data.fgi && data.fgi.now && data.fgi.now.value !== null) {
            setFgi({ value: data.fgi.now.value, valueText: data.fgi.now.valueText })
            setSentiment(mapValueTextToSentiment(data.fgi.now.valueText))
            setError(null)
            setLoading(false)
            return
          }
          // Backend returned Unavailable data — retry after delay
          if (attempt < retries) {
            await new Promise(r => setTimeout(r, delay))
            continue
          }
          setError("No data available")
          setLoading(false)
        } catch {
          if (!mountedRef.current) return
          if (attempt < retries) {
            await new Promise(r => setTimeout(r, delay))
            continue
          }
          setError("Failed to fetch sentiment")
          setLoading(false)
        }
      }
    }
    fetchSentiment()
    return () => { mountedRef.current = false }
  }, [])

  const getSentimentColor = (type: "bullish" | "bearish" | "neutral") => {
    switch (type) {
      case "bullish":
        return "bg-emerald-500/10 text-emerald-500 border-emerald-500/20"
      case "bearish":
        return "bg-red-500/10 text-red-500 border-red-500/20"
      case "neutral":
        return "bg-amber-500/10 text-amber-500 border-amber-500/20"
    }
  }

  const getSentimentIcon = (type: "bullish" | "bearish" | "neutral") => {
    switch (type) {
      case "bullish":
        return <TrendingUp className="h-4 w-4" />
      case "bearish":
        return <TrendingDown className="h-4 w-4" />
      case "neutral":
        return <Minus className="h-4 w-4" />
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn("rounded-lg border border-zinc-800 bg-black/60 backdrop-blur-sm p-4", className)}
    >
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          {loading ? (
            <>
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-md border border-zinc-700 bg-zinc-800/50 animate-pulse">
                <Loader2 className="h-4 w-4 animate-spin text-zinc-400" />
                <span className="font-medium text-zinc-400">Loading Market Data…</span>
              </div>
              <div className="text-sm text-zinc-500">
                <span className="font-medium text-zinc-400">Fear & Greed Index:</span>{" "}
                <span className="inline-flex items-center gap-1"><Loader2 className="h-3 w-3 animate-spin" /> Fetching…</span>
              </div>
            </>
          ) : (
            <>
              <div className={cn("flex items-center gap-2 px-3 py-1.5 rounded-md border", getSentimentColor(sentiment))}>
                {getSentimentIcon(sentiment)}
                <span className="font-medium capitalize">{sentiment} Market</span>
              </div>
              <div className="text-sm text-zinc-400">
                <span className="font-medium text-white">Fear & Greed Index:</span>{" "}
                {error && <span className="text-red-500">{error}</span>}
                {!error && fgi && (
                  <span>{fgi.value} ({fgi.valueText})</span>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </motion.div>
  )
}
