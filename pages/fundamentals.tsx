"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { BarChart3, FileText, Newspaper, Layers } from "lucide-react"

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
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div className="space-y-2">
          <motion.h1
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-2xl font-bold flex items-center gap-2"
          >
            <BarChart3 className="h-6 w-6 text-emerald-500" />
            Fundamentals & Filings
          </motion.h1>
          <p className="text-xs md:text-sm text-zinc-400 max-w-xl">
            Deep-dive into a symbol using fundamentals, SEC filings, and press releases widgets powered by Jika.io.
            This page is independent from the core AI and real-time views.
          </p>
        </div>

        {/* Symbol selector */}
        <div className="flex flex-col items-start gap-2 md:items-end">
          <form
            className="flex w-full items-center gap-2 md:w-auto"
            onSubmit={(e) => {
              e.preventDefault()
              const value = customSymbol.trim().toUpperCase()
              if (!value) return
              setSymbol(value)
              setCustomSymbol("")
            }}
          >
            <div className="flex flex-col gap-1">
              <label
                htmlFor="fundamentals-symbol-input"
                className="text-xs md:text-sm text-zinc-400"
              >
                Symbol for widgets
              </label>
              <div className="flex items-center gap-2">
                <input
                  id="fundamentals-symbol-input"
                  type="text"
                  value={customSymbol}
                  onChange={(e) => setCustomSymbol(e.target.value)}
                  placeholder={symbol || "AAPL"}
                  className="w-full md:w-32 bg-zinc-900 border border-zinc-700 text-xs md:text-sm rounded-md px-3 py-2 text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
                />
                <button
                  type="submit"
                  className="inline-flex items-center rounded-md bg-emerald-500 px-3 py-2 text-xs font-medium text-black shadow hover:bg-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 focus:ring-offset-zinc-900"
                >
                  Apply
                </button>
              </div>
            </div>
          </form>

          {/* Quick-pick popular symbols */}
          <div className="mt-1 flex w-full flex-wrap gap-1 justify-start md:justify-end">
            {AVAILABLE_SYMBOLS.slice(0, 8).map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setSymbol(s)}
                className={`rounded-full border px-2 py-0.5 text-[10px] font-medium transition-colors ${
                  symbol === s
                    ? "border-emerald-500 bg-emerald-500/10 text-emerald-300"
                    : "border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800"
                }`}
              >
                {s}
              </button>
            ))}
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
            <div className="rounded-lg overflow-hidden bg-white h-[520px] md:h-[460px]">
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
            <div className="rounded-lg overflow-hidden bg-white h-[740px] md:h-[680px]">
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
            <div className="rounded-lg overflow-hidden bg-white h-[260px] md:h-[230px]">
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
            <div className="rounded-lg overflow-hidden bg-white h-[300px] md:h-[260px]">
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
            <div className="rounded-lg overflow-hidden bg-white h-[520px] md:h-[460px]">
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

