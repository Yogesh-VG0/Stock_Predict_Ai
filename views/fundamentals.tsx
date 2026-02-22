"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { BarChart3, FileText, Newspaper, Layers, Search } from "lucide-react"
import StockLogo from "@/components/market/StockLogo"

const AVAILABLE_SYMBOLS = [
  "AAPL",
  "MSFT",
  "GOOGL",
  "AMZN",
  "TSLA",
  "NVDA",
  "META",
  "NFLX",
  "JPM",
  "V",
  "JNJ",
  "WMT",
  "PG",
  "UNH",
  "HD",
  "MA",
  "BAC",
  "XOM",
  "LLY",
  "ABBV",
]

export default function FundamentalsPage() {
  const [symbol, setSymbol] = useState("AAPL")
  const [customSymbol, setCustomSymbol] = useState("")

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-3">
        <div className="space-y-2">
          <motion.h1
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-xl sm:text-2xl font-bold flex items-center gap-2"
          >
            <BarChart3 className="h-5 w-5 sm:h-6 sm:w-6 text-emerald-500" />
            Fundamentals & Filings
          </motion.h1>
          <p className="text-xs sm:text-sm text-zinc-400 max-w-xl">
            Deep-dive into a symbol using fundamentals, SEC filings, and press releases widgets powered by Jika.io.
          </p>
        </div>

        {/* Symbol selector */}
        <div className="flex flex-col gap-4 bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-4 sm:p-5 shadow-2xl backdrop-blur-sm -mx-2 sm:mx-0">
          <form
            className="flex w-full items-center gap-2"
            onSubmit={(e) => {
              e.preventDefault()
              const value = customSymbol.trim().toUpperCase()
              if (!value) return
              setSymbol(value)
              setCustomSymbol("")
            }}
          >
            <div className="flex flex-col gap-2 flex-1 sm:flex-initial w-full sm:w-[400px]">
              <label
                htmlFor="fundamentals-symbol-input"
                className="text-xs sm:text-sm font-medium text-zinc-300"
              >
                Analyze Target Symbol
              </label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search className="h-4 w-4 text-zinc-500 group-focus-within:text-emerald-500 transition-colors" />
                </div>
                <input
                  id="fundamentals-symbol-input"
                  type="text"
                  value={customSymbol}
                  onChange={(e) => setCustomSymbol(e.target.value)}
                  placeholder={`Currently analyzing: ${symbol || "AAPL"}`}
                  className="w-full bg-zinc-950/50 border border-zinc-800 text-sm rounded-lg pl-10 pr-20 py-2.5 text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500 transition-all shadow-inner"
                />
                <button
                  type="submit"
                  className="absolute inset-y-1 right-1 inline-flex items-center rounded-md bg-emerald-500 px-4 text-xs font-bold text-black shadow-sm hover:bg-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 focus:ring-offset-zinc-900 transition-all"
                >
                  Analyze
                </button>
              </div>
            </div>
          </form>

          {/* Quick-pick popular symbols with Logos */}
          <div className="space-y-2">
            <span className="text-xs font-medium text-zinc-500 uppercase tracking-wider ml-1">Popular Targets</span>
            <div className="flex overflow-x-auto pb-2 -mx-1 px-1 scrollbar-hide sm:flex-wrap gap-2 items-center">
              {AVAILABLE_SYMBOLS.slice(0, 10).map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => setSymbol(s)}
                  className={`flex-shrink-0 flex items-center gap-2 rounded-lg border px-3 py-1.5 transition-all shadow-sm ${symbol === s
                    ? "border-emerald-500/50 bg-emerald-500/10 text-white ring-1 ring-emerald-500/20"
                    : "border-zinc-800/80 bg-zinc-950/50 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
                    }`}
                >
                  <StockLogo symbol={s} size={16} />
                  <span className="text-xs font-semibold">{s}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Grid of widgets */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Price history */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <BarChart3 className="h-4 w-4 text-blue-500" />
              1Y Price History ({symbol})
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-2">
            <div className="rounded-lg overflow-hidden bg-white h-[380px] sm:h-[520px] md:h-[460px]">
              <iframe
                referrerPolicy="origin"
                width="100%"
                height="100%"
                style={{
                  background: "#FFFFFF",
                  padding: "10px",
                  height: "100%",
                  border: "none",
                  borderRadius: "5px",
                  boxShadow: "0 2px 4px 0 rgba(0,0,0,.2)",
                }}
                src={`https://jika.io/embed/area-chart?symbol=${symbol}&selection=one_year&closeKey=close&boxShadow=true&graphColor=1652f0&textColor=161c2d&backgroundColor=FFFFFF&fontFamily=Nunito&`}
              />
            </div>
          </CardContent>
        </Card>

        {/* Fundamentals comparison chart */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Layers className="h-4 w-4 text-purple-500" />
              Revenue vs Net Income (Quarterly)
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-2">
            <div className="rounded-lg overflow-hidden bg-white h-[500px] sm:h-[740px] md:h-[680px]">
              <iframe
                referrerPolicy="origin"
                width="100%"
                height="100%"
                style={{
                  background: "#FFFFFF",
                  padding: "10px",
                  height: "100%",
                  border: "none",
                  borderRadius: "5px",
                  boxShadow: "0 2px 4px 0 rgba(0,0,0,.2)",
                }}
                src={`https://jika.io/embed/fundamentals-chart?symbols=${symbol}&keys=Revenue,Net Income&reportingPeriod=quarter&from=2021&to=2026&boxShadow=true&textColor=161c2d&backgroundColor=FFFFFF&fontFamily=Nunito&`}
              />
            </div>
          </CardContent>
        </Card>

        {/* Fundamentals table */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <BarChart3 className="h-4 w-4 text-emerald-500" />
              Key Financial Metrics Table
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-2">
            <div className="rounded-lg overflow-hidden bg-white h-[200px] sm:h-[260px] md:h-[230px]">
              <iframe
                referrerPolicy="origin"
                width="100%"
                height="100%"
                style={{
                  background: "#FFFFFF",
                  padding: "10px",
                  height: "100%",
                  border: "none",
                  borderRadius: "5px",
                  boxShadow: "0 2px 4px 0 rgba(0,0,0,.2)",
                }}
                src={`https://jika.io/embed/fundamentals-table?symbols=${symbol}&keys=Revenue,Net Income&reportingPeriod=quarter&from=2021&to=2026&sortMethod=companies&boxShadow=true&textColor=161c2d&backgroundColor=FFFFFF&fontFamily=Nunito&`}
              />
            </div>
          </CardContent>
        </Card>

        {/* SEC filings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <FileText className="h-4 w-4 text-amber-500" />
              Recent SEC Filings
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-2">
            <div className="rounded-lg overflow-hidden bg-white h-[240px] sm:h-[300px] md:h-[260px]">
              <iframe
                referrerPolicy="origin"
                width="100%"
                height="100%"
                style={{
                  background: "#FFFFFF",
                  padding: "10px",
                  height: "100%",
                  border: "none",
                  borderRadius: "5px",
                  boxShadow: "0 2px 4px 0 rgba(0,0,0,.2)",
                }}
                src={`https://jika.io/embed/sec-filings?symbol=${symbol}&limit=5&boxShadow=true&textColor=161c2d&backgroundColor=FFFFFF&fontFamily=Nunito&`}
              />
            </div>
          </CardContent>
        </Card>

        {/* Press releases */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Newspaper className="h-4 w-4 text-blue-400" />
              Latest Press Releases
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-2">
            <div className="rounded-lg overflow-hidden bg-white h-[380px] sm:h-[520px] md:h-[460px]">
              <iframe
                referrerPolicy="origin"
                width="100%"
                height="100%"
                style={{
                  background: "#FFFFFF",
                  padding: "10px",
                  height: "100%",
                  border: "none",
                  borderRadius: "5px",
                  boxShadow: "0 2px 4px 0 rgba(0,0,0,.2)",
                }}
                src={`https://jika.io/embed/press-releases?symbol=${symbol}&limit=3&boxShadow=true&textColor=161c2d&backgroundColor=FFFFFF&fontFamily=Nunito&`}
              />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* We keep explicit page branding minimal; widgets themselves may still show Jika.io logos inside the iframe. */}
    </div>
  )
}

