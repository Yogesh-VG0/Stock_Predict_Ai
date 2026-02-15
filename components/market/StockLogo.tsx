"use client"

import { useState, useCallback } from "react"

interface StockLogoProps {
  symbol: string
  size?: number
}

export default function StockLogo({ symbol, size = 24 }: StockLogoProps) {
  const [imgError, setImgError] = useState(false)
  const handleError = useCallback(() => setImgError(true), [])

  if (imgError) {
    return (
      <div
        className="rounded-full bg-zinc-800 flex items-center justify-center flex-shrink-0"
        style={{ width: size, height: size }}
      >
        <span className="font-bold text-zinc-400" style={{ fontSize: size * 0.4 }}>
          {symbol[0]}
        </span>
      </div>
    )
  }

  return (
    <img
      src={`https://raw.githubusercontent.com/davidepalazzo/ticker-logos/main/ticker_icons/${symbol}.png`}
      alt={symbol}
      width={size}
      height={size}
      loading="lazy"
      decoding="async"
      className="rounded-full bg-zinc-900 object-contain flex-shrink-0"
      style={{ width: size, height: size }}
      onError={handleError}
    />
  )
}
