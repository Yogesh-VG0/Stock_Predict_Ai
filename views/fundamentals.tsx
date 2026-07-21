"use client"

import { useEffect, useMemo, useState } from "react"
import Link from "next/link"
import { motion } from "framer-motion"
import {
  AlertCircle,
  BarChart3,
  Building2,
  CalendarDays,
  ExternalLink,
  FileText,
  Layers,
  Loader2,
  Newspaper,
  Search,
} from "lucide-react"
import StockLogo from "@/components/market/StockLogo"
import TradingViewAdvancedChart from "@/components/tradingview/trading-view-advanced-chart"
import TradingViewCompanyProfile from "@/components/tradingview/TradingViewCompanyProfile"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import {
  getFundamentalsOverview,
  getStockDetails,
  type FundamentalsFiling,
  type FundamentalsOverview,
  type FundamentalsQuarter,
  type FundamentalsUpdate,
  type StockDetails,
} from "@/lib/api"

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

type MetricCard = {
  label: string
  value: string
  detail?: string
  accent: string
}

function sanitizeSymbol(value: string) {
  return value.trim().toUpperCase().replace(/[^A-Z0-9.\-]/g, "").slice(0, 10)
}

function formatCurrency(value?: number | null) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—"
  const sign = value < 0 ? "-" : ""
  const abs = Math.abs(value)
  if (abs >= 1e12) return `${sign}$${(abs / 1e12).toFixed(2)}T`
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(2)}B`
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(2)}M`
  return `${sign}$${abs.toLocaleString()}`
}

function isFiniteNumber(value?: number | null) {
  return typeof value === "number" && Number.isFinite(value)
}

function formatQuarterLabel(row: FundamentalsQuarter) {
  if (row.fiscalPeriod && row.fiscalYear) return `${row.fiscalPeriod} ${row.fiscalYear}`
  return row.period || formatDate(row.date)
}

function formatDate(value?: string) {
  if (!value) return "—"
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })
}

function truncate(text = "", maxLength = 140) {
  const clean = text.replace(/<[^>]*>/g, "").replace(/\s+/g, " ").trim()
  return clean.length > maxLength ? `${clean.slice(0, maxLength)}…` : clean
}

function normalizeWebsiteUrl(website?: string) {
  if (!website || website === "N/A") return null
  return website.startsWith("http") ? website : `https://${website}`
}

function FinancialBarChart({ quarters }: { quarters: FundamentalsQuarter[] }) {
  const rows = quarters.filter((item) => item.revenue !== null || item.netIncome !== null).slice(-8)
  const maxValue = Math.max(...rows.flatMap((row) => [Math.abs(row.revenue || 0), Math.abs(row.netIncome || 0)]), 1)
  const scaleLabel = maxValue >= 1e12 ? "USD trillions" : maxValue >= 1e9 ? "USD billions" : "USD millions"

  if (!rows.length) {
    return (
      <div className="flex h-64 flex-col items-center justify-center rounded-lg border border-zinc-800 bg-zinc-950/60 p-4 text-center">
        <AlertCircle className="mb-2 h-6 w-6 text-zinc-600" />
        <p className="text-sm text-zinc-400">Quarterly SEC revenue/net income facts are not available for this symbol.</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-zinc-400">
        <div className="flex flex-wrap items-center gap-4">
          <span className="inline-flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-sm bg-blue-500" /> Revenue</span>
          <span className="inline-flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-sm bg-emerald-500" /> Net income</span>
          <span className="inline-flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-sm bg-red-500" /> Loss</span>
        </div>
        <span className="rounded-full border border-zinc-800 bg-zinc-950/70 px-2 py-1 text-[11px] text-zinc-500">
          Scale: {scaleLabel} · values shown per quarter
        </span>
      </div>
      <div className="overflow-x-auto pb-2">
        <div className="flex min-w-[760px] items-end gap-3 rounded-xl border border-zinc-800 bg-zinc-950/50 p-4">
          {rows.map((row) => {
            const revenueHeight = isFiniteNumber(row.revenue) ? Math.max(6, Math.round((Math.abs(row.revenue || 0) / maxValue) * 160)) : 0
            const netIncomeHeight = isFiniteNumber(row.netIncome) ? Math.max(6, Math.round((Math.abs(row.netIncome || 0) / maxValue) * 160)) : 0
            const netIncomeIsLoss = (row.netIncome || 0) < 0
            const label = formatQuarterLabel(row)
            return (
              <div key={`${row.date}-${row.period}`} className="flex flex-1 flex-col items-center gap-2">
                <div className="flex min-h-[44px] flex-col items-center justify-end text-center text-[11px] leading-tight">
                  <span className="font-semibold text-blue-300">{formatCurrency(row.revenue)}</span>
                  <span className={netIncomeIsLoss ? "font-semibold text-red-300" : "font-semibold text-emerald-300"}>
                    {formatCurrency(row.netIncome)}
                  </span>
                </div>
                <div className="relative flex h-48 items-end gap-2 border-b border-zinc-800/80 px-1">
                  {isFiniteNumber(row.revenue) ? (
                    <div
                      className="w-7 rounded-t bg-blue-500/85 shadow-lg shadow-blue-500/10"
                      style={{ height: revenueHeight }}
                      title={`${label} revenue: ${formatCurrency(row.revenue)}${row.revenueDerived ? ` (${row.revenueDerivation || "derived"})` : ""}`}
                      aria-label={`${label} revenue ${formatCurrency(row.revenue)}`}
                    />
                  ) : (
                    <div className="flex h-10 w-7 items-center justify-center rounded border border-dashed border-zinc-800 text-[10px] text-zinc-600">N/A</div>
                  )}
                  {isFiniteNumber(row.netIncome) ? (
                    <div
                      className={`w-7 rounded-t shadow-lg ${netIncomeIsLoss ? "bg-red-500/85 shadow-red-500/10" : "bg-emerald-500/85 shadow-emerald-500/10"}`}
                      style={{ height: netIncomeHeight }}
                      title={`${label} net income: ${formatCurrency(row.netIncome)}${row.netIncomeDerived ? ` (${row.netIncomeDerivation || "derived"})` : ""}`}
                      aria-label={`${label} net income ${formatCurrency(row.netIncome)}`}
                    />
                  ) : (
                    <div className="flex h-10 w-7 items-center justify-center rounded border border-dashed border-zinc-800 text-[10px] text-zinc-600">N/A</div>
                  )}
                </div>
                <div className="text-center text-[11px] leading-tight text-zinc-500">
                  <div className="font-medium text-zinc-300">{row.fiscalPeriod || row.period}</div>
                  <div>{row.fiscalYear || formatDate(row.date)}</div>
                  {(row.revenueDerived || row.netIncomeDerived) && (
                    <div className="mt-1 rounded-full bg-amber-500/10 px-1.5 py-0.5 text-[10px] text-amber-300">derived</div>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      </div>
      <div className="overflow-x-auto rounded-lg border border-zinc-800">
        <table className="min-w-full divide-y divide-zinc-800 text-left text-xs">
          <thead className="bg-zinc-900/70 text-zinc-500">
            <tr>
              <th className="px-3 py-2 font-semibold uppercase tracking-wider">Quarter</th>
              <th className="px-3 py-2 font-semibold uppercase tracking-wider">Revenue</th>
              <th className="px-3 py-2 font-semibold uppercase tracking-wider">Net income</th>
              <th className="px-3 py-2 font-semibold uppercase tracking-wider">SEC fact</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800 bg-zinc-950/40">
            {rows.map((row) => (
              <tr key={`table-${row.date}-${row.period}`}>
                <td className="whitespace-nowrap px-3 py-2 font-medium text-zinc-200">{formatQuarterLabel(row)}</td>
                <td className="whitespace-nowrap px-3 py-2 text-blue-300">
                  {formatCurrency(row.revenue)} {row.revenueDerived && <span className="text-amber-300">*</span>}
                </td>
                <td className={`whitespace-nowrap px-3 py-2 ${(row.netIncome || 0) < 0 ? "text-red-300" : "text-emerald-300"}`}>
                  {formatCurrency(row.netIncome)} {row.netIncomeDerived && <span className="text-amber-300">*</span>}
                </td>
                <td className="max-w-[280px] truncate px-3 py-2 text-zinc-500" title={[row.revenueFactName, row.netIncomeFactName].filter(Boolean).join(" / ")}>
                  {[row.revenueFactName, row.netIncomeFactName].filter(Boolean).join(" / ") || "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.some((row) => row.revenueDerived || row.netIncomeDerived) && (
        <p className="text-xs leading-relaxed text-amber-300/80">
          * Some Q4 values are derived as full fiscal-year SEC facts minus Q1–Q3, because many companies report full-year XBRL facts instead of standalone Q4 values.
        </p>
      )}
    </div>
  )
}

function FilingsTable({ filings }: { filings: FundamentalsFiling[] }) {
  if (!filings.length) {
    return <p className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-4 text-sm text-zinc-400">No recent SEC filings found.</p>
  }

  return (
    <div className="overflow-hidden rounded-lg border border-zinc-800">
      <div className="grid grid-cols-[90px_120px_1fr_40px] gap-3 bg-zinc-900/80 px-4 py-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
        <span>Form</span>
        <span>Filed</span>
        <span>Description</span>
        <span />
      </div>
      {filings.slice(0, 8).map((filing) => (
        <a
          key={`${filing.accessionNumber}-${filing.primaryDocument}`}
          href={filing.url}
          target="_blank"
          rel="noreferrer"
          className="grid grid-cols-[90px_120px_1fr_40px] gap-3 border-t border-zinc-800 px-4 py-3 text-sm transition-colors hover:bg-emerald-500/5"
        >
          <span className="font-semibold text-amber-400">{filing.form}</span>
          <span className="text-zinc-400">{formatDate(filing.filingDate)}</span>
          <span className="truncate text-zinc-200">{filing.description}</span>
          <ExternalLink className="h-4 w-4 text-emerald-400" />
        </a>
      ))}
    </div>
  )
}

function UpdatesList({ updates }: { updates: FundamentalsUpdate[] }) {
  if (!updates.length) {
    return <p className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-4 text-sm text-zinc-400">No recent company updates found.</p>
  }

  return (
    <div className="space-y-3">
      {updates.slice(0, 5).map((item) => (
        <a
          key={`${item.title}-${item.url}`}
          href={item.url}
          target="_blank"
          rel="noreferrer"
          className="block rounded-lg border border-zinc-800 bg-zinc-950/60 p-4 transition-colors hover:border-emerald-500/40 hover:bg-emerald-500/5"
        >
          <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-zinc-500">
            <span className="inline-flex items-center gap-1"><CalendarDays className="h-3.5 w-3.5" /> {formatDate(item.published_at)}</span>
            <span>•</span>
            <span>{item.source}</span>
          </div>
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="font-semibold text-zinc-100">{item.title}</h3>
              {item.snippet && <p className="mt-1 text-sm leading-relaxed text-zinc-400">{truncate(item.snippet)}</p>}
            </div>
            <ExternalLink className="mt-1 h-4 w-4 flex-shrink-0 text-emerald-400" />
          </div>
        </a>
      ))}
    </div>
  )
}

export default function FundamentalsPage() {
  const [symbol, setSymbol] = useState("AAPL")
  const [customSymbol, setCustomSymbol] = useState("")
  const [stockDetails, setStockDetails] = useState<StockDetails | null>(null)
  const [fundamentals, setFundamentals] = useState<FundamentalsOverview | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    async function loadFundamentals() {
      setLoading(true)
      setError(null)

      const [details, overview] = await Promise.allSettled([
        getStockDetails(symbol),
        getFundamentalsOverview(symbol),
      ])

      if (cancelled) return

      setStockDetails(details.status === "fulfilled" ? details.value : null)
      setFundamentals(overview.status === "fulfilled" ? overview.value : null)

      if (overview.status === "rejected" || (overview.status === "fulfilled" && !overview.value)) {
        setError("Official SEC fundamentals are unavailable right now. Price/profile widgets and external resources are still available.")
      }

      setLoading(false)
    }

    loadFundamentals().catch(() => {
      if (!cancelled) {
        setError("Unable to load fundamentals right now. Please try again in a moment.")
        setLoading(false)
      }
    })

    return () => {
      cancelled = true
    }
  }, [symbol])

  const metricCards = useMemo<MetricCard[]>(() => {
    const metrics = fundamentals?.annualMetrics || {}
    const grossMargin = metrics.revenue?.value && metrics.grossProfit?.value
      ? (metrics.grossProfit.value / metrics.revenue.value) * 100
      : null

    return [
      { label: "Revenue", value: formatCurrency(metrics.revenue?.value), detail: metrics.revenue?.end ? `FY ending ${formatDate(metrics.revenue.end)}` : undefined, accent: "text-blue-400" },
      { label: "Gross Profit", value: formatCurrency(metrics.grossProfit?.value), detail: grossMargin ? `${grossMargin.toFixed(1)}% gross margin` : undefined, accent: "text-emerald-400" },
      { label: "Operating Income", value: formatCurrency(metrics.operatingIncome?.value), detail: metrics.operatingIncome?.factName, accent: "text-purple-400" },
      { label: "Net Income", value: formatCurrency(metrics.netIncome?.value), detail: metrics.netIncome?.end ? `Filed ${formatDate(metrics.netIncome.filed)}` : undefined, accent: "text-emerald-500" },
      { label: "Assets", value: formatCurrency(metrics.assets?.value), detail: metrics.assets?.end ? `As of ${formatDate(metrics.assets.end)}` : undefined, accent: "text-cyan-400" },
      { label: "Operating Cash Flow", value: formatCurrency(metrics.operatingCashFlow?.value), detail: metrics.operatingCashFlow?.factName, accent: "text-amber-400" },
    ]
  }, [fundamentals])

  const website = normalizeWebsiteUrl(stockDetails?.website)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const value = sanitizeSymbol(customSymbol)
    if (!value) return
    setSymbol(value)
    setCustomSymbol("")
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3">
        <div className="space-y-2">
          <motion.h1
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center gap-2 text-xl font-bold sm:text-2xl"
          >
            <BarChart3 className="h-5 w-5 text-emerald-500 sm:h-6 sm:w-6" />
            Fundamentals & Filings
          </motion.h1>
          <p className="max-w-3xl text-xs text-zinc-400 sm:text-sm">
            Durable fundamentals from official SEC APIs, company/Yahoo RSS updates, and TradingView market widgets.
            Discontinued Jika.io widgets have been removed permanently.
          </p>
        </div>

        <div className="-mx-2 flex flex-col gap-4 rounded-xl border border-zinc-800/50 bg-zinc-900/40 p-4 shadow-2xl backdrop-blur-sm sm:mx-0 sm:p-5">
          <form className="flex w-full items-center gap-2" onSubmit={handleSubmit}>
            <div className="flex w-full flex-1 flex-col gap-2 sm:w-[400px] sm:flex-initial">
              <label htmlFor="fundamentals-symbol-input" className="text-xs font-medium text-zinc-300 sm:text-sm">
                Analyze Target Symbol
              </label>
              <div className="group relative">
                <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3">
                  <Search className="h-4 w-4 text-zinc-500 transition-colors group-focus-within:text-emerald-500" />
                </div>
                <input
                  id="fundamentals-symbol-input"
                  type="text"
                  value={customSymbol}
                  onChange={(e) => setCustomSymbol(e.target.value)}
                  placeholder={`Currently analyzing: ${symbol}`}
                  className="w-full rounded-lg border border-zinc-800 bg-zinc-950/50 py-2.5 pl-10 pr-20 text-sm text-zinc-100 shadow-inner transition-all placeholder:text-zinc-600 focus:border-emerald-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
                />
                <button
                  type="submit"
                  className="absolute inset-y-1 right-1 inline-flex items-center rounded-md bg-emerald-500 px-4 text-xs font-bold text-black shadow-sm transition-all hover:bg-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 focus:ring-offset-zinc-900"
                >
                  Analyze
                </button>
              </div>
            </div>
          </form>

          <div className="space-y-2">
            <span className="ml-1 text-xs font-medium uppercase tracking-wider text-zinc-500">Popular Targets</span>
            <div className="scrollbar-hide -mx-1 flex items-center gap-2 overflow-x-auto px-1 pb-2 sm:flex-wrap">
              {AVAILABLE_SYMBOLS.slice(0, 12).map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => setSymbol(s)}
                  className={`flex flex-shrink-0 items-center gap-2 rounded-lg border px-3 py-1.5 shadow-sm transition-all ${
                    symbol === s
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

      {error && (
        <div className="flex items-start gap-3 rounded-lg border border-amber-500/20 bg-amber-500/10 p-4 text-sm text-amber-200">
          <AlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <BarChart3 className="h-4 w-4 text-blue-500" />
              1Y Price History ({symbol})
            </CardTitle>
            <CardDescription>TradingView Advanced Chart with public market data and indicators.</CardDescription>
          </CardHeader>
          <CardContent className="pt-2">
            <TradingViewAdvancedChart symbol={symbol} exchange={stockDetails?.exchange} height={420} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Building2 className="h-4 w-4 text-purple-500" />
              Company Profile
            </CardTitle>
            <CardDescription>{fundamentals?.companyName || stockDetails?.name || symbol} profile and market overview.</CardDescription>
          </CardHeader>
          <CardContent className="pt-2">
            <TradingViewCompanyProfile symbol={symbol} exchange={stockDetails?.exchange} height={420} />
          </CardContent>
        </Card>

        <Card className="xl:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Layers className="h-4 w-4 text-emerald-500" />
              Key Financial Metrics
            </CardTitle>
            <CardDescription>Official SEC CompanyFacts data cached by the backend.</CardDescription>
          </CardHeader>
          <CardContent className="pt-2">
            {loading ? (
              <div className="flex h-44 items-center justify-center gap-2 text-sm text-zinc-400">
                <Loader2 className="h-4 w-4 animate-spin text-emerald-500" />
                Loading SEC financial metrics…
              </div>
            ) : fundamentals ? (
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
                {metricCards.map((metric) => (
                  <div key={metric.label} className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-4">
                    <div className="text-xs uppercase tracking-wider text-zinc-500">{metric.label}</div>
                    <div className={`mt-2 text-xl font-bold ${metric.accent}`}>{metric.value}</div>
                    {metric.detail && <div className="mt-1 truncate text-xs text-zinc-500">{metric.detail}</div>}
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex h-44 flex-col items-center justify-center rounded-lg border border-zinc-800 bg-zinc-950/60 p-4 text-center">
                <AlertCircle className="mb-2 h-6 w-6 text-zinc-600" />
                <p className="text-sm text-zinc-400">SEC financial metrics are unavailable for {symbol} right now.</p>
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="xl:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <BarChart3 className="h-4 w-4 text-blue-400" />
              Revenue vs Net Income
            </CardTitle>
            <CardDescription>Recent quarterly values derived from SEC XBRL facts where available.</CardDescription>
          </CardHeader>
          <CardContent className="pt-2">
            {loading ? (
              <div className="flex h-64 items-center justify-center gap-2 text-sm text-zinc-400">
                <Loader2 className="h-4 w-4 animate-spin text-emerald-500" />
                Loading quarterly series…
              </div>
            ) : (
              <FinancialBarChart quarters={fundamentals?.quarterly || []} />
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <FileText className="h-4 w-4 text-amber-500" />
              Recent SEC Filings
            </CardTitle>
            <CardDescription>Official SEC submissions API; links open SEC documents directly.</CardDescription>
          </CardHeader>
          <CardContent className="pt-2">
            {loading ? (
              <div className="flex h-48 items-center justify-center gap-2 text-sm text-zinc-400">
                <Loader2 className="h-4 w-4 animate-spin text-emerald-500" />
                Loading filings…
              </div>
            ) : (
              <FilingsTable filings={fundamentals?.filings || []} />
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Newspaper className="h-4 w-4 text-blue-400" />
              Latest Company Updates
            </CardTitle>
            <CardDescription>Company newsroom RSS when available, with Yahoo Finance RSS as fallback.</CardDescription>
          </CardHeader>
          <CardContent className="pt-2">
            {loading ? (
              <div className="flex h-48 items-center justify-center gap-2 text-sm text-zinc-400">
                <Loader2 className="h-4 w-4 animate-spin text-emerald-500" />
                Loading updates…
              </div>
            ) : (
              <UpdatesList updates={fundamentals?.updates || []} />
            )}
          </CardContent>
        </Card>

        <Card className="xl:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <ExternalLink className="h-4 w-4 text-emerald-500" />
              External Resources
            </CardTitle>
            <CardDescription>Useful official and app-owned links for deeper research.</CardDescription>
          </CardHeader>
          <CardContent className="grid grid-cols-1 gap-3 pt-2 sm:grid-cols-3">
            <a
              href={`https://www.sec.gov/edgar/search/#/q=${encodeURIComponent(symbol)}&dateRange=all`}
              target="_blank"
              rel="noreferrer"
              className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-4 transition-colors hover:border-emerald-500/40 hover:bg-emerald-500/5"
            >
              <div className="flex items-center justify-between gap-3 font-semibold text-zinc-100">SEC EDGAR <ExternalLink className="h-4 w-4 text-emerald-400" /></div>
              <p className="mt-2 text-xs leading-relaxed text-zinc-500">Search official company filings on SEC.gov.</p>
            </a>
            {website ? (
              <a
                href={website}
                target="_blank"
                rel="noreferrer"
                className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-4 transition-colors hover:border-emerald-500/40 hover:bg-emerald-500/5"
              >
                <div className="flex items-center justify-between gap-3 font-semibold text-zinc-100">Company site <ExternalLink className="h-4 w-4 text-emerald-400" /></div>
                <p className="mt-2 text-xs leading-relaxed text-zinc-500">Open the official company website or investor relations area.</p>
              </a>
            ) : null}
            <Link
              href={`/sankey?symbol=${encodeURIComponent(symbol)}`}
              className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-4 transition-colors hover:border-emerald-500/40 hover:bg-emerald-500/5"
            >
              <div className="flex items-center justify-between gap-3 font-semibold text-zinc-100">Income flow <ExternalLink className="h-4 w-4 text-emerald-400" /></div>
              <p className="mt-2 text-xs leading-relaxed text-zinc-500">Open the full StockPredict Sankey financial-flow analysis.</p>
            </Link>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}