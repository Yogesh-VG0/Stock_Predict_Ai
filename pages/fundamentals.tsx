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
        <div className="flex items-center gap-2">
          <label htmlFor="fundamentals-symbol" className="text-xs md:text-sm text-zinc-400">
            Symbol
          </label>
          <select
            id="fundamentals-symbol"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            className="bg-zinc-900 border border-zinc-700 text-xs md:text-sm rounded-md px-2 py-1 text-zinc-100 focus:outline-none focus:ring-1 focus:ring-emerald-500"
          >
            {AVAILABLE_SYMBOLS.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Grid of widgets */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Price history */}
        <Card className="bg-zinc-900 border-zinc-800">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <BarChart3 className="h-4 w-4 text-blue-500" />
              1Y Price History ({symbol})
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-2">
            <div className="rounded-lg overflow-hidden bg-[#020617]">
              <iframe
                referrerPolicy="origin"
                width="100%"
                height="470"
                style={{
                  background: "#020617",
                  padding: "10px",
                  border: "none",
                  borderRadius: "5px",
                  boxShadow: "0 2px 4px 0 rgba(0,0,0,.2)",
                }}
                src={`https://jika.io/embed/area-chart?symbol=${symbol}&selection=one_year&closeKey=close&boxShadow=true&graphColor=22c55e&textColor=e5e7eb&backgroundColor=020617&fontFamily=Nunito&`}
              />
            </div>
          </CardContent>
        </Card>

        {/* Fundamentals comparison chart */}
        <Card className="bg-zinc-900 border-zinc-800">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Layers className="h-4 w-4 text-purple-500" />
              Revenue vs Net Income (Quarterly)
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-2">
            <div className="rounded-lg overflow-hidden bg-[#020617]">
              <iframe
                referrerPolicy="origin"
                width="100%"
                height="638"
                style={{
                  background: "#020617",
                  padding: "10px",
                  border: "none",
                  borderRadius: "5px",
                  boxShadow: "0 2px 4px 0 rgba(0,0,0,.2)",
                }}
                src={`https://jika.io/embed/fundamentals-chart?symbols=${symbol}&keys=Revenue,Net Income&reportingPeriod=quarter&from=2021&to=2026&boxShadow=true&textColor=e5e7eb&backgroundColor=020617&fontFamily=Nunito&`}
              />
            </div>
          </CardContent>
        </Card>

        {/* Fundamentals table */}
        <Card className="bg-zinc-900 border-zinc-800">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <BarChart3 className="h-4 w-4 text-emerald-500" />
              Key Financial Metrics Table
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-2">
            <div className="rounded-lg overflow-hidden bg-[#020617]">
              <iframe
                referrerPolicy="origin"
                width="100%"
                height="220"
                style={{
                  background: "#020617",
                  padding: "10px",
                  border: "none",
                  borderRadius: "5px",
                  boxShadow: "0 2px 4px 0 rgba(0,0,0,.2)",
                }}
                src={`https://jika.io/embed/fundamentals-table?symbols=${symbol}&keys=Revenue,Net Income&reportingPeriod=quarter&from=2021&to=2026&sortMethod=companies&boxShadow=true&textColor=e5e7eb&backgroundColor=020617&fontFamily=Nunito&`}
              />
            </div>
          </CardContent>
        </Card>

        {/* SEC filings */}
        <Card className="bg-zinc-900 border-zinc-800">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <FileText className="h-4 w-4 text-amber-500" />
              Recent SEC Filings
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-2">
            <div className="rounded-lg overflow-hidden bg-[#020617]">
              <iframe
                referrerPolicy="origin"
                width="100%"
                height="265"
                style={{
                  background: "#020617",
                  padding: "10px",
                  border: "none",
                  borderRadius: "5px",
                  boxShadow: "0 2px 4px 0 rgba(0,0,0,.2)",
                }}
                src={`https://jika.io/embed/sec-filings?symbol=${symbol}&limit=5&boxShadow=true&textColor=e5e7eb&backgroundColor=020617&fontFamily=Nunito&`}
              />
            </div>
          </CardContent>
        </Card>

        {/* Press releases */}
        <Card className="bg-zinc-900 border-zinc-800 lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Newspaper className="h-4 w-4 text-blue-400" />
              Latest Press Releases
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-2">
            <div className="rounded-lg overflow-hidden bg-[#020617]">
              <iframe
                referrerPolicy="origin"
                width="100%"
                height="495"
                style={{
                  background: "#020617",
                  padding: "10px",
                  border: "none",
                  borderRadius: "5px",
                  boxShadow: "0 2px 4px 0 rgba(0,0,0,.2)",
                }}
                src={`https://jika.io/embed/press-releases?symbol=${symbol}&limit=3&boxShadow=true&textColor=e5e7eb&backgroundColor=020617&fontFamily=Nunito&`}
              />
            </div>
          </CardContent>
        </Card>
      </div>

      <p className="text-[11px] text-zinc-500 text-right">
        Data and widgets provided by{" "}
        <a
          href="https://www.jika.io/widgets"
          target="_blank"
          rel="noopener noreferrer"
          className="underline underline-offset-2 text-zinc-300 hover:text-white"
        >
          Jika.io
        </a>
        .
      </p>
    </div>
  )
}

