"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { TrendingUp, TrendingDown, Minus } from "lucide-react"
import { cn } from "@/lib/utils"
import { API_BASE_URL } from "@/lib/api"

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

  useEffect(() => {
    setLoading(true)
    fetch(`${API_BASE_URL}/api/market/sentiment`)
      .then(res => res.json())
      .then(data => {
        if (data && data.fgi && data.fgi.now) {
          setFgi({ value: data.fgi.now.value, valueText: data.fgi.now.valueText })
          setSentiment(mapValueTextToSentiment(data.fgi.now.valueText))
        } else {
          setError("No data available")
        }
        setLoading(false)
      })
      .catch(() => {
        setError("Failed to fetch sentiment")
        setLoading(false)
      })
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
          <div className={cn("flex items-center gap-2 px-3 py-1.5 rounded-md border", getSentimentColor(sentiment))}>
            {getSentimentIcon(sentiment)}
            <span className="font-medium capitalize">{sentiment} Market</span>
          </div>
          <div className="text-sm text-zinc-400">
            <span className="font-medium text-white">Fear & Greed Index:</span>{" "}
            {loading && <span>Loading...</span>}
            {error && <span className="text-red-500">{error}</span>}
            {!loading && !error && fgi && (
              <span>{fgi.value} ({fgi.valueText})</span>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  )
}
