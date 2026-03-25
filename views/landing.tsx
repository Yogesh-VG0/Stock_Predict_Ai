"use client"

import { useRef, useState, useEffect } from "react"
import Link from "next/link"
import { motion, useInView } from "framer-motion"
import {
  LineChart,
  Brain,
  BarChart3,
  Newspaper,
  Shield,
  Zap,
  Database,
  GitBranch,
  ArrowRight,
  Star,
  Activity,
  Globe,
  Cpu,
  TrendingUp,
  TrendingDown,
  Clock,
} from "lucide-react"
import DisclaimerModal from "@/components/disclaimer/disclaimer-modal"
import { GradientText } from "@/components/ui/gradient-text"
import {
  SiExpress,
  SiFastapi,
  SiFramer,
  SiGithubactions,
  SiGoogle,
  SiMongodb,
  SiNextdotjs,
  SiNodedotjs,
  SiPython,
  SiReact,
  SiRedis,
  SiTailwindcss,
  SiTradingview,
  SiTypescript,
  SiVercel,
  SiKoyeb,
} from "react-icons/si"
import { TbBrain } from "react-icons/tb"

function AnimatedSection({ children, className = "", delay = 0 }: { children: React.ReactNode; className?: string; delay?: number }) {
  const ref = useRef<HTMLDivElement>(null)
  const isInView = useInView(ref, { once: true, margin: "-80px" })

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 40 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 40 }}
      transition={{ duration: 0.6, delay, ease: [0.22, 1, 0.36, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  )
}

function CountUp({ end, suffix = "" }: { end: number; suffix?: string }) {
  const [count, setCount] = useState(0)
  const nodeRef = useRef<HTMLSpanElement>(null)
  const isInView = useInView(nodeRef, { once: true })

  useEffect(() => {
    if (!isInView) return
    let frame: number
    const duration = 1800
    const start = performance.now()
    const step = (now: number) => {
      const progress = Math.min((now - start) / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3)
      setCount(Math.floor(eased * end))
      if (progress < 1) frame = requestAnimationFrame(step)
    }
    frame = requestAnimationFrame(step)
    return () => cancelAnimationFrame(frame)
  }, [isInView, end])

  return <span ref={nodeRef}>{count}{suffix}</span>
}

const FEATURES = [
  {
    icon: Brain,
    title: "LightGBM ML Predictions",
    description: "Walk-forward validated ensemble model predicting next-day, 7-day, and 30-day stock prices for S&P 100 companies.",
    color: "text-purple-400",
    bg: "from-purple-500/10 to-purple-600/5",
    border: "border-purple-500/20",
  },
  {
    icon: Zap,
    title: "Gemini AI Explanations",
    description: "Every prediction comes with a plain-English AI explanation powered by SHAP feature importance and Google Gemini.",
    color: "text-emerald-400",
    bg: "from-emerald-500/10 to-emerald-600/5",
    border: "border-emerald-500/20",
  },
  {
    icon: Activity,
    title: "Real-Time Market Data",
    description: "Live prices, Fear & Greed Index, market heatmaps, and economic calendars via TradingView and Finnhub.",
    color: "text-blue-400",
    bg: "from-blue-500/10 to-blue-600/5",
    border: "border-blue-500/20",
  },
  {
    icon: Newspaper,
    title: "Multi-Source Sentiment",
    description: "Aggregated sentiment from Reddit, SEC filings, FinViz, Marketaux, RSS feeds, and SeekingAlpha using FinBERT & VADER.",
    color: "text-amber-400",
    bg: "from-amber-500/10 to-amber-600/5",
    border: "border-amber-500/20",
  },
  {
    icon: BarChart3,
    title: "Technical Analysis",
    description: "Interactive TradingView charts with RSI, MACD, Bollinger Bands, SMA, and custom technical indicators.",
    color: "text-cyan-400",
    bg: "from-cyan-500/10 to-cyan-600/5",
    border: "border-cyan-500/20",
  },
  {
    icon: Shield,
    title: "Drift Monitoring",
    description: "Automated model health checks with PSI, directional accuracy, and Brier score tracking via daily CI/CD pipeline.",
    color: "text-rose-400",
    bg: "from-rose-500/10 to-rose-600/5",
    border: "border-rose-500/20",
  },
]

const TECH_STACK_GROUPS = [
  {
    label: "Frontend",
    items: [
      { name: "Next.js 15", icon: SiNextdotjs },
      { name: "React", icon: SiReact },
      { name: "TypeScript", icon: SiTypescript },
      { name: "Tailwind CSS", icon: SiTailwindcss },
      { name: "Framer Motion", icon: SiFramer },
      { name: "TradingView", icon: SiTradingview },
    ],
  },
  {
    label: "Backend",
    items: [
      { name: "Node.js", icon: SiNodedotjs },
      { name: "Express", icon: SiExpress },
      { name: "FastAPI", icon: SiFastapi },
      { name: "MongoDB", icon: SiMongodb },
      { name: "Redis", icon: SiRedis },
    ],
  },
  {
    label: "ML & AI",
    items: [
      { name: "Python", icon: SiPython },
      { name: "LightGBM", icon: TbBrain },
      { name: "SHAP", icon: TbBrain },
      { name: "Google Gemini", icon: SiGoogle },
      { name: "FinBERT", icon: TbBrain },
    ],
  },
  {
    label: "Infrastructure",
    items: [
      { name: "GitHub Actions", icon: SiGithubactions },
      { name: "Vercel", icon: SiVercel },
      { name: "Koyeb", icon: SiKoyeb },
    ],
  },
]

const PIPELINE_STEPS = [
  { step: "1", title: "Fetch Data", desc: "10+ sources: Finnhub, FRED, FMP, SEC, Reddit, RSS, FinViz", icon: Database },
  { step: "2", title: "Engineer Features", desc: "40+ features: technicals, sentiment, macro, insider trades", icon: Cpu },
  { step: "3", title: "Train & Predict", desc: "LightGBM walk-forward model with SHAP explainability", icon: Brain },
  { step: "4", title: "AI Explain", desc: "Gemini generates plain-English analysis from all data", icon: Zap },
  { step: "5", title: "Monitor & Evaluate", desc: "Drift detection, accuracy tracking, model health checks", icon: Shield },
]

const STOCK_LOGOS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "NFLX"]

export default function LandingPage() {
  const webApplicationData = {
    "@context": "https://schema.org",
    "@type": "WebApplication",
    name: "StockPredict AI",
    description:
      "AI-powered stock prediction platform using LightGBM ML models, SHAP explainability, and Gemini AI for S&P 100 companies.",
    url: "https://stockpredict.dev",
    applicationCategory: "FinanceApplication",
    operatingSystem: "Web",
    offers: {
      "@type": "Offer",
      price: "0",
      priceCurrency: "USD",
    },
    creator: {
      "@type": "Person",
      name: "Yogesh Vadivel",
      url: "https://yogeshv.me",
    },
    featureList: [
      "LightGBM ML Predictions",
      "SHAP Explainability",
      "Gemini AI Explanations",
      "Real-Time Market Data",
      "Multi-Source Sentiment Analysis",
      "Technical Analysis",
      "Drift Monitoring",
    ],
  }

  const websiteData = {
    "@context": "https://schema.org",
    "@type": "WebSite",
    name: "StockPredict AI",
    url: "https://stockpredict.dev",
    description:
      "AI-powered stock prediction platform using LightGBM ML models, SHAP explainability, and Gemini AI for S&P 100 companies.",
    publisher: {
      "@type": "Person",
      name: "Yogesh Vadivel",
      url: "https://yogeshv.me",
    },
  }

  // Fetch real landing page stats from backend
  const [landingStats, setLandingStats] = useState<{
    topMover: { symbol: string; change: string };
    stockCount: number;
    lastRun: string;
  } | null>(null)

  useEffect(() => {
    fetch('/api/stock/landing/stats')
      .then(res => res.json())
      .then(data => setLandingStats(data))
      .catch(() => setLandingStats(null))
  }, [])

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(webApplicationData) }}
      />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(websiteData) }}
      />
      <div className="min-h-screen bg-black text-white overflow-x-hidden">
        {/* Disclaimer Modal */}
        <DisclaimerModal />

        {/* Skip to content link for accessibility */}
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-[100] focus:px-4 focus:py-2 focus:bg-emerald-500 focus:text-black focus:rounded-lg focus:font-semibold"
        >
          Skip to main content
        </a>
        {/* Navbar */}
        <nav className="fixed top-0 left-0 right-0 z-50 border-b border-zinc-800/50 bg-black/80 backdrop-blur-xl" aria-label="Main navigation">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <Link href="/" className="flex items-center gap-2.5" aria-label="StockPredict AI Home">
              <LineChart className="h-6 w-6 text-emerald-500" aria-hidden="true" />
              <span className="font-bold text-lg">StockPredict AI</span>
            </Link>
            <div className="flex items-center gap-3">
              <Link
                href="https://github.com/Yogesh-VG0/Stock_Predict_Ai"
                target="_blank"
                rel="noopener noreferrer"
                className="hidden sm:flex items-center gap-2 text-sm text-zinc-400 hover:text-white transition-colors px-3 py-2 rounded-lg hover:bg-zinc-800/50"
              >
                <GitBranch className="h-4 w-4" />
                GitHub
              </Link>
              <Link
                href="/dashboard"
                className="bg-emerald-500 hover:bg-emerald-400 text-black font-semibold text-sm px-5 py-2.5 rounded-lg transition-all hover:shadow-lg hover:shadow-emerald-500/20"
                aria-label="Open Dashboard"
              >
                Open Dashboard
              </Link>
            </div>
          </div>
        </nav>

        {/* Red Disclaimer Banner */}
        <div className="fixed top-16 left-0 right-0 z-40 border-b border-red-500/30 bg-gradient-to-r from-red-950/90 via-red-900/90 to-red-950/90 backdrop-blur-sm">
          <div className="max-w-7xl mx-auto px-3 sm:px-4 lg:px-6 py-2 sm:py-2.5">
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              className="flex items-center justify-center gap-1.5 sm:gap-2 text-center"
            >
              <motion.div
                animate={{
                  opacity: [1, 0.5, 1],
                  scale: [1, 1.05, 1],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  ease: "easeInOut",
                }}
                className="flex-shrink-0 w-1.5 h-1.5 sm:w-2 sm:h-2 rounded-full bg-red-500"
              />
              <p className="text-[10px] xs:text-xs sm:text-sm font-semibold text-red-200 leading-tight px-1">
                <span className="inline-block animate-pulse">⚠️</span>{" "}
                <span className="font-bold text-red-100">Educational project, not financial advice.</span>{" "}
                <span className="hidden xs:inline">Predictions are probabilistic and may be wrong.</span>
                <span className="xs:hidden">Predictions may be wrong.</span>
              </p>
              <motion.div
                animate={{
                  opacity: [1, 0.5, 1],
                  scale: [1, 1.05, 1],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  ease: "easeInOut",
                  delay: 0.5,
                }}
                className="flex-shrink-0 w-1.5 h-1.5 sm:w-2 sm:h-2 rounded-full bg-red-500"
              />
            </motion.div>
          </div>
        </div>

        {/* Hero Section */}
        <section id="main-content" className="relative pt-40 pb-20 sm:pt-48 sm:pb-28 px-4 overflow-hidden">
          {/* Dot grid pattern background */}
          <div className="absolute inset-0 pointer-events-none" style={{ opacity: 0.03 }}>
            <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <pattern id="dot-grid" x="0" y="0" width="24" height="24" patternUnits="userSpaceOnUse">
                  <circle cx="1" cy="1" r="1" fill="white" />
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#dot-grid)" />
            </svg>
          </div>
          {/* Background glow */}
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[600px] bg-emerald-500/5 rounded-full blur-[120px] pointer-events-none" />
          <div className="absolute top-20 right-0 w-[400px] h-[400px] bg-purple-500/5 rounded-full blur-[100px] pointer-events-none" />

          <div className="max-w-5xl mx-auto text-center relative z-10">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 text-emerald-400 text-sm font-medium mb-8"
            >
              <Activity className="h-3.5 w-3.5" />
              Daily ML Pipeline for S&P 100
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="text-4xl sm:text-5xl md:text-7xl font-bold tracking-tight leading-[1.1] mb-6"
            >
              AI-Powered Stock
              <br />
              <GradientText
                text="Predictions & Analysis"
                gradient="linear-gradient(90deg, #10b981 0%, #34d399 25%, #22d3ee 50%, #34d399 75%, #10b981 100%)"
                transition={{ duration: 5, repeat: Infinity, ease: "linear" }}
              />
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="text-lg sm:text-xl text-zinc-400 max-w-2xl mx-auto mb-10 leading-relaxed"
            >
              LightGBM predictions with SHAP explainability, multi-source sentiment analysis,
              and Gemini AI explanations, updated daily via automated CI/CD pipeline.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
              className="flex flex-col sm:flex-row items-center justify-center gap-4"
            >
              <Link
                href="/dashboard"
                className="group flex items-center gap-2 bg-emerald-500 hover:bg-emerald-400 text-black font-semibold text-base px-8 py-3.5 rounded-xl transition-all hover:shadow-xl hover:shadow-emerald-500/20"
              >
                Explore Dashboard
                <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
              </Link>
              <Link
                href="/stocks/AAPL"
                className="group relative flex items-center gap-2 bg-emerald-500/5 hover:bg-emerald-500/10 text-white font-semibold text-base px-8 py-3.5 rounded-xl transition-all border border-emerald-500/30 hover:border-emerald-500/60 shadow-sm shadow-emerald-500/10 hover:shadow-lg hover:shadow-emerald-500/25"
                aria-label="View AAPL stock analysis"
              >
                View AAPL Analysis
                <ArrowRight className="h-4 w-4 text-zinc-500 group-hover:text-emerald-400 group-hover:translate-x-0.5 transition-all" aria-hidden="true" />
              </Link>
            </motion.div>

            {/* Stock logos — marquee on mobile, static on desktop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 1, delay: 0.6 }}
              className="mt-14 overflow-hidden"
            >
              {/* Mobile: horizontal marquee */}
              <div className="sm:hidden relative">
                <div className="absolute left-0 top-0 bottom-0 w-8 bg-gradient-to-r from-black to-transparent z-10 pointer-events-none" />
                <div className="absolute right-0 top-0 bottom-0 w-8 bg-gradient-to-l from-black to-transparent z-10 pointer-events-none" />
                <div className="flex gap-5 animate-marquee hover:[animation-play-state:paused] w-max">
                  {[...STOCK_LOGOS, ...STOCK_LOGOS].map((symbol, i) => (
                    <Link key={`${symbol}-${i}`} href={`/stocks/${symbol}`} aria-label={`View AI analysis for ${symbol}`} className="flex flex-col items-center gap-1.5 group">
                      <div className="w-11 h-11 rounded-full bg-zinc-900 border border-zinc-800 flex items-center justify-center overflow-hidden group-hover:border-emerald-500/50 transition-all">
                        <img
                          src={`https://raw.githubusercontent.com/davidepalazzo/ticker-logos/main/ticker_icons/${symbol}.png`}
                          alt={`${symbol} stock logo`}
                          width={28}
                          height={28}
                          loading="eager"
                          className="w-7 h-7 object-contain"
                        />
                      </div>
                      <span className="text-[10px] text-zinc-500 font-medium">{symbol}</span>
                    </Link>
                  ))}
                </div>
              </div>
              {/* Desktop: static row */}
              <div className="hidden sm:flex flex-wrap items-center justify-center gap-x-3 gap-y-3 px-4">
                {STOCK_LOGOS.map((symbol, i) => (
                  <motion.div
                    key={symbol}
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.4, delay: 0.7 + i * 0.08 }}
                    className="relative group"
                  >
                    <Link href={`/stocks/${symbol}`} aria-label={`View AI analysis for ${symbol}`}>
                      <div className="w-12 h-12 rounded-full bg-zinc-900 border border-zinc-800 flex items-center justify-center overflow-hidden group-hover:border-emerald-500/50 group-hover:shadow-lg group-hover:shadow-emerald-500/10 transition-all">
                        <img
                          src={`https://raw.githubusercontent.com/davidepalazzo/ticker-logos/main/ticker_icons/${symbol}.png`}
                          alt={`${symbol} stock logo`}
                          width={32}
                          height={32}
                          loading="eager"
                          className="w-8 h-8 object-contain"
                        />
                      </div>
                      <span className="absolute -bottom-5 left-1/2 -translate-x-1/2 text-[10px] text-zinc-500 font-medium group-hover:text-emerald-400 transition-colors">
                        {symbol}
                      </span>
                    </Link>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </div>
        </section>

        {/* Stats Banner */}
        <section className="max-w-5xl mx-auto px-4 py-10">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { value: 100, label: "S&P Stocks Tracked", icon: Globe, suffix: "", color: "text-emerald-400", border: "border-emerald-500/20", bg: "from-emerald-500/5" },
              { value: 50, label: "ML Features", icon: Cpu, suffix: "+", color: "text-purple-400", border: "border-purple-500/20", bg: "from-purple-500/5" },
              { value: 10, label: "Data Sources", icon: Database, suffix: "+", color: "text-blue-400", border: "border-blue-500/20", bg: "from-blue-500/5" },
              { value: -1, label: "Automated Pipeline", icon: GitBranch, suffix: "", color: "text-amber-400", border: "border-amber-500/20", bg: "from-amber-500/5" },
            ].map((stat, i) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 24, scale: 0.95 }}
                whileInView={{ opacity: 1, y: 0, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
                whileHover={{ y: -3 }}
                className={`group relative text-center p-6 rounded-xl border ${stat.border} bg-gradient-to-b ${stat.bg} to-transparent hover:shadow-lg transition-shadow duration-300 overflow-hidden`}
              >
                {/* Radial pulse shimmer */}
                <div className="absolute inset-0 flex items-start justify-center pointer-events-none">
                  <div className={`w-24 h-24 mt-2 rounded-full bg-gradient-radial ${stat.bg} to-transparent animate-radial-pulse`} />
                </div>
                <stat.icon className={`relative h-6 w-6 ${stat.color} mx-auto mb-3 group-hover:scale-110 transition-transform duration-300`} />
                <div className={`text-3xl sm:text-4xl font-bold ${stat.color} mb-1`}>
                  {stat.value >= 0 ? <CountUp end={stat.value} suffix={stat.suffix} /> : "Daily"}
                </div>
                <div className="text-xs sm:text-sm text-zinc-500">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* Features Grid */}
        <section className="py-20 px-4">
          <div className="max-w-6xl mx-auto">
            <AnimatedSection className="text-center mb-14">
              <h2 className="text-3xl sm:text-4xl font-bold mb-4">
                Full-Stack ML Platform
              </h2>
              <p className="text-zinc-400 max-w-xl mx-auto">
                From data ingestion to AI-powered explanations, every component built from scratch.
              </p>
            </AnimatedSection>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
              {FEATURES.map((feature, i) => (
                <AnimatedSection key={feature.title} delay={i * 0.08}>
                  <div className="relative h-full group/card">
                    {/* Ambient rotating border glow */}
                    <div className={`absolute -inset-[1px] rounded-xl bg-gradient-to-r ${feature.bg} opacity-[0.15] group-hover/card:opacity-40 blur-[1px] transition-opacity duration-500`} />
                    <div className={`relative h-full rounded-xl border ${feature.border} bg-gradient-to-br ${feature.bg} bg-zinc-950/80 p-6 hover:scale-[1.02] transition-all duration-300`}>
                      <feature.icon className={`h-8 w-8 ${feature.color} mb-4`} />
                      <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
                      <p className="text-sm text-zinc-400 leading-relaxed">{feature.description}</p>
                    </div>
                  </div>
                </AnimatedSection>
              ))}
            </div>
          </div>
        </section>

        {/* Pipeline Section */}
        <section className="py-20 px-4 border-t border-zinc-800/50">
          <div className="max-w-5xl mx-auto">
            <AnimatedSection className="text-center mb-14">
              <h2 className="text-3xl sm:text-4xl font-bold mb-4">
                Daily Automated Pipeline
              </h2>
              <p className="text-zinc-400 max-w-xl mx-auto">
                GitHub Actions runs the full ML pipeline every trading day with zero manual intervention.
              </p>
            </AnimatedSection>

            <div className="space-y-0">
              {PIPELINE_STEPS.map((step, i) => (
                <AnimatedSection key={step.step} delay={i * 0.1}>
                  <>
                    <div className="flex items-start gap-5 p-5 rounded-xl border border-zinc-800/80 bg-zinc-900/30 hover:bg-zinc-900/60 transition-colors">
                      {/* Glowing dot */}
                      <div className="flex-shrink-0 relative">
                        <div className="absolute -inset-1 rounded-lg bg-emerald-500/20 blur-sm" />
                        <div className="relative w-10 h-10 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
                          <step.icon className="h-5 w-5 text-emerald-400" />
                        </div>
                      </div>
                      <div className="min-w-0">
                        <div className="flex items-center gap-3 mb-1">
                          <span className="text-xs font-bold text-emerald-500 bg-emerald-500/10 px-2 py-0.5 rounded-full">
                            STEP {step.step}
                          </span>
                          <h3 className="font-semibold">{step.title}</h3>
                        </div>
                        <p className="text-sm text-zinc-400">{step.desc}</p>
                      </div>
                    </div>
                    {/* Connector arrow between cards, not after last */}
                    {i < PIPELINE_STEPS.length - 1 && (
                      <div className="flex justify-start pl-[27px] py-1">
                        <div className="w-px h-4 bg-gradient-to-b from-emerald-500/40 to-emerald-500/10" />
                      </div>
                    )}
                  </>
                </AnimatedSection>
              ))}
            </div>
          </div>
        </section>

        {/* Tech Stack */}
        <section className="py-20 px-4 border-t border-zinc-800/50">
          <div className="max-w-5xl mx-auto">
            <AnimatedSection className="text-center mb-14">
              <h2 className="text-3xl sm:text-4xl font-bold mb-4">Tech Stack</h2>
              <p className="text-zinc-400 max-w-xl mx-auto">
                Modern tools across the full stack: frontend, backend, ML, and infrastructure.
              </p>
            </AnimatedSection>

            <AnimatedSection>
              <div className="space-y-6">
                {TECH_STACK_GROUPS.map((group) => (
                  <div key={group.label}>
                    <div className="text-[11px] font-semibold uppercase tracking-widest text-zinc-600 mb-2.5 text-center">
                      {group.label}
                    </div>
                    <div className="flex flex-wrap justify-center gap-2.5">
                      {group.items.map((tech) => {
                        const Icon = tech.icon
                        return (
                          <div
                            key={tech.name}
                            className="flex items-center gap-2 px-4 py-2 rounded-lg border border-zinc-800 bg-zinc-900/50 text-sm text-zinc-300 hover:border-emerald-500/30 hover:text-white hover:shadow hover:shadow-emerald-500/10 transition-all"
                          >
                            <Icon className="h-4 w-4 text-zinc-400" />
                            {tech.name}
                          </div>
                        )
                      })}
                    </div>
                  </div>
                ))}
              </div>
            </AnimatedSection>
          </div>
        </section>



        {/* CTA */}
        <section className="py-20 px-4 border-t border-zinc-800/50">
          <AnimatedSection className="max-w-3xl mx-auto text-center">
            <div className="relative rounded-2xl border border-zinc-800 bg-gradient-to-b from-zinc-900/80 to-black p-10 sm:p-14 overflow-hidden">
              <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[500px] h-[250px] bg-emerald-500/[0.07] rounded-full blur-[100px] pointer-events-none" />
              <div className="absolute bottom-0 right-0 w-[300px] h-[200px] bg-purple-500/5 rounded-full blur-[80px] pointer-events-none" />
              <div className="relative z-10">
                <motion.div
                  animate={{ rotate: [0, 8, -8, 0] }}
                  transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
                  className="inline-block"
                >
                  <Star className="h-10 w-10 text-emerald-500 mx-auto mb-6" />
                </motion.div>
                <h2 className="text-3xl sm:text-4xl font-bold mb-4">
                  See It In Action
                </h2>
                <p className="text-zinc-400 mb-6 max-w-lg mx-auto leading-relaxed">
                  Explore real ML predictions, AI explanations, and live market data for {landingStats?.stockCount || 75} S&P stocks.
                </p>
                {/* Metric ticker pills */}
                <div className="flex flex-wrap items-center justify-center gap-3 mb-8">
                  {/* Top Mover Pill */}
                  <motion.div
                    key="mover"
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.1 }}
                    className="flex items-center gap-2 px-4 py-2 rounded-full border border-zinc-700/60 bg-zinc-900/80 backdrop-blur-sm text-sm shadow-lg shadow-black/20"
                  >
                    {landingStats ? (
                      <Link href={`/stocks/${landingStats.topMover.symbol}`} className="flex items-center gap-2 hover:opacity-80 transition-opacity">
                        <img
                          src={`https://raw.githubusercontent.com/davidepalazzo/ticker-logos/main/ticker_icons/${landingStats.topMover.symbol}.png`}
                          alt={landingStats.topMover.symbol}
                          className="w-5 h-5 rounded-full bg-zinc-800 object-contain"
                          onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
                        />
                        {landingStats.topMover.change.startsWith('-') ? (
                          <TrendingDown className="h-3.5 w-3.5 text-red-400" />
                        ) : (
                          <TrendingUp className="h-3.5 w-3.5 text-emerald-400" />
                        )}
                        <span className={landingStats.topMover.change.startsWith('-') ? "text-red-400 font-medium" : "text-emerald-400 font-medium"}>
                          {landingStats.topMover.symbol} {landingStats.topMover.change} predicted
                        </span>
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                      </Link>
                    ) : (
                      <div className="flex items-center gap-2">
                        <div className="w-5 h-5 rounded-full bg-zinc-700 animate-pulse" />
                        <div className="h-3.5 w-28 rounded-full bg-zinc-700 animate-pulse" />
                      </div>
                    )}
                  </motion.div>

                  {/* Stocks Tracked Pill */}
                  <motion.div
                    key="count"
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.25 }}
                    className="flex items-center gap-2 px-4 py-2 rounded-full border border-zinc-700/60 bg-zinc-900/80 backdrop-blur-sm text-sm shadow-lg shadow-black/20"
                  >
                    {landingStats ? (
                      <>
                        <BarChart3 className="h-3.5 w-3.5 text-blue-400" />
                        <span className="text-blue-400 font-medium">{landingStats.stockCount} stocks tracked</span>
                      </>
                    ) : (
                      <div className="flex items-center gap-2">
                        <div className="w-3.5 h-3.5 rounded bg-zinc-700 animate-pulse" />
                        <div className="h-3.5 w-24 rounded-full bg-zinc-700 animate-pulse" />
                      </div>
                    )}
                  </motion.div>

                  {/* Model Last Run Pill */}
                  <motion.div
                    key="model"
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.4 }}
                    className="flex items-center gap-2 px-4 py-2 rounded-full border border-zinc-700/60 bg-zinc-900/80 backdrop-blur-sm text-sm shadow-lg shadow-black/20"
                  >
                    {landingStats ? (
                      <>
                        <Clock className="h-3.5 w-3.5 text-purple-400" />
                        <span className="text-purple-400 font-medium">Model ran {landingStats.lastRun}</span>
                      </>
                    ) : (
                      <div className="flex items-center gap-2">
                        <div className="w-3.5 h-3.5 rounded bg-zinc-700 animate-pulse" />
                        <div className="h-3.5 w-24 rounded-full bg-zinc-700 animate-pulse" />
                      </div>
                    )}
                  </motion.div>
                </div>
                <Link
                  href="/dashboard"
                  className="group inline-flex items-center gap-2.5 bg-emerald-500 hover:bg-emerald-400 text-black font-semibold text-base px-10 py-4 rounded-xl transition-all hover:shadow-2xl hover:shadow-emerald-500/25 hover:scale-[1.02] active:scale-[0.98]"
                  aria-label="Open Dashboard"
                >
                  Open Dashboard
                  <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" aria-hidden="true" />
                </Link>
              </div>
            </div>
          </AnimatedSection>
        </section>

        {/* Footer */}
        <footer className="border-t border-zinc-800/50 py-12 px-4" role="contentinfo">
          <div className="max-w-5xl mx-auto">
            <div className="text-xs text-zinc-600 leading-relaxed mb-8 text-center max-w-2xl mx-auto">
              StockPredict AI is an educational research project and does not provide investment advice. Predictions are estimates
              based on historical data and may be inaccurate. Do your own research and consult a licensed advisor before investing.
            </div>

            <div className="border-t border-zinc-800/30 pt-8">
              <div className="flex flex-col items-center gap-6">
                <div className="flex items-center gap-2">
                  <LineChart className="h-4 w-4 text-emerald-500" />
                  <span className="font-semibold text-sm text-zinc-200">StockPredict AI</span>
                </div>

                <nav className="flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-sm text-zinc-500" aria-label="Footer navigation">
                  <Link href="/dashboard" className="hover:text-white transition-colors">Dashboard</Link>
                  <Link href="/news" className="hover:text-white transition-colors">News</Link>
                  <Link href="/how-it-works" className="hover:text-white transition-colors">How it works</Link>
                  <Link href="/methodology" className="hover:text-white transition-colors">Methodology</Link>
                  <Link href="/disclaimer" className="hover:text-white transition-colors">Disclaimer</Link>
                  <Link href="https://github.com/Yogesh-VG0/Stock_Predict_Ai" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">GitHub</Link>
                </nav>

                <div className="text-sm text-zinc-600">
                  Built by{" "}
                  <Link href="https://yogeshv.me" target="_blank" rel="noopener noreferrer" className="text-zinc-400 hover:text-emerald-400 transition-colors">
                    Yogesh Vadivel
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </>
  )
}
