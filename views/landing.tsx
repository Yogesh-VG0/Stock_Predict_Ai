"use client"

import { useRef } from "react"
import Link from "next/link"
import { motion, useInView } from "framer-motion"
import {
  LineChart,
  Brain,
  TrendingUp,
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
} from "lucide-react"

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

const TECH_STACK = [
  { name: "Next.js 15", category: "Frontend" },
  { name: "React", category: "Frontend" },
  { name: "TypeScript", category: "Frontend" },
  { name: "Tailwind CSS", category: "Frontend" },
  { name: "Framer Motion", category: "Frontend" },
  { name: "TradingView", category: "Frontend" },
  { name: "Python", category: "ML Backend" },
  { name: "FastAPI", category: "ML Backend" },
  { name: "LightGBM", category: "ML Backend" },
  { name: "SHAP", category: "ML Backend" },
  { name: "Google Gemini", category: "ML Backend" },
  { name: "FinBERT", category: "ML Backend" },
  { name: "Node.js", category: "Backend" },
  { name: "Express", category: "Backend" },
  { name: "MongoDB", category: "Backend" },
  { name: "Redis", category: "Backend" },
  { name: "GitHub Actions", category: "CI/CD" },
  { name: "Vercel", category: "Deployment" },
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
  return (
    <div className="min-h-screen bg-black text-white overflow-x-hidden">
      {/* Navbar */}
      <nav className="fixed top-0 left-0 right-0 z-50 border-b border-zinc-800/50 bg-black/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2.5">
            <LineChart className="h-6 w-6 text-emerald-500" />
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
            >
              Open Dashboard
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 sm:pt-40 sm:pb-28 px-4 overflow-hidden">
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
            <span className="bg-gradient-to-r from-emerald-400 via-emerald-300 to-cyan-400 bg-clip-text text-transparent">
              Predictions & Analysis
            </span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-lg sm:text-xl text-zinc-400 max-w-2xl mx-auto mb-10 leading-relaxed"
          >
            LightGBM predictions with SHAP explainability, multi-source sentiment analysis,
            and Gemini AI explanations — updated daily via automated CI/CD pipeline.
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
              className="flex items-center gap-2 bg-zinc-800 hover:bg-zinc-700 text-white font-medium text-base px-8 py-3.5 rounded-xl transition-all border border-zinc-700"
            >
              <TrendingUp className="h-4 w-4 text-emerald-500" />
              View AAPL Analysis
            </Link>
          </motion.div>

          {/* Floating stock logos */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 0.6 }}
            className="flex items-center justify-center gap-3 mt-14"
          >
            {STOCK_LOGOS.map((symbol, i) => (
              <motion.div
                key={symbol}
                initial={{ opacity: 0, scale: 0.5 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.4, delay: 0.7 + i * 0.08 }}
                className="relative group"
              >
                <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-full bg-zinc-900 border border-zinc-800 flex items-center justify-center overflow-hidden group-hover:border-emerald-500/50 transition-colors">
                  <img
                    src={`https://raw.githubusercontent.com/davidepalazzo/ticker-logos/main/ticker_icons/${symbol}.png`}
                    alt={symbol}
                    width={32}
                    height={32}
                    loading="eager"
                    className="w-7 h-7 sm:w-8 sm:h-8 object-contain"
                  />
                </div>
                <span className="absolute -bottom-5 left-1/2 -translate-x-1/2 text-[10px] text-zinc-500 font-medium">
                  {symbol}
                </span>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Stats Banner */}
      <AnimatedSection>
        <div className="max-w-5xl mx-auto px-4 py-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { value: "100", label: "S&P Stocks Tracked", icon: Globe },
              { value: "40+", label: "ML Features", icon: Cpu },
              { value: "10+", label: "Data Sources", icon: Database },
              { value: "Daily", label: "Automated Pipeline", icon: GitBranch },
            ].map((stat) => (
              <div
                key={stat.label}
                className="text-center p-5 rounded-xl border border-zinc-800/80 bg-zinc-900/30"
              >
                <stat.icon className="h-5 w-5 text-emerald-500 mx-auto mb-2" />
                <div className="text-2xl sm:text-3xl font-bold text-white">{stat.value}</div>
                <div className="text-xs sm:text-sm text-zinc-500 mt-1">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </AnimatedSection>

      {/* Features Grid */}
      <section className="py-20 px-4">
        <div className="max-w-6xl mx-auto">
          <AnimatedSection className="text-center mb-14">
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">
              Full-Stack ML Platform
            </h2>
            <p className="text-zinc-400 max-w-xl mx-auto">
              From data ingestion to AI-powered explanations — every component built from scratch.
            </p>
          </AnimatedSection>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {FEATURES.map((feature, i) => (
              <AnimatedSection key={feature.title} delay={i * 0.08}>
                <div className={`h-full rounded-xl border ${feature.border} bg-gradient-to-br ${feature.bg} p-6 hover:scale-[1.02] transition-transform duration-300`}>
                  <feature.icon className={`h-8 w-8 ${feature.color} mb-4`} />
                  <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
                  <p className="text-sm text-zinc-400 leading-relaxed">{feature.description}</p>
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
              GitHub Actions runs the full ML pipeline every trading day — no manual intervention.
            </p>
          </AnimatedSection>

          <div className="space-y-4">
            {PIPELINE_STEPS.map((step, i) => (
              <AnimatedSection key={step.step} delay={i * 0.1}>
                <div className="flex items-start gap-5 p-5 rounded-xl border border-zinc-800/80 bg-zinc-900/30 hover:bg-zinc-900/60 transition-colors">
                  <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
                    <step.icon className="h-5 w-5 text-emerald-400" />
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
              Modern tools across the full stack — frontend, backend, ML, and infrastructure.
            </p>
          </AnimatedSection>

          <AnimatedSection>
            <div className="flex flex-wrap justify-center gap-2.5">
              {TECH_STACK.map((tech) => (
                <span
                  key={tech.name}
                  className="px-4 py-2 rounded-lg border border-zinc-800 bg-zinc-900/50 text-sm text-zinc-300 hover:border-emerald-500/30 hover:text-white transition-colors"
                >
                  {tech.name}
                </span>
              ))}
            </div>
          </AnimatedSection>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-4 border-t border-zinc-800/50">
        <AnimatedSection className="max-w-3xl mx-auto text-center">
          <div className="rounded-2xl border border-zinc-800 bg-gradient-to-b from-zinc-900/80 to-black p-10 sm:p-14">
            <Star className="h-10 w-10 text-emerald-500 mx-auto mb-6" />
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">
              See It In Action
            </h2>
            <p className="text-zinc-400 mb-8 max-w-lg mx-auto">
              Explore real ML predictions, AI explanations, and live market data for 100 S&P stocks.
            </p>
            <Link
              href="/dashboard"
              className="group inline-flex items-center gap-2 bg-emerald-500 hover:bg-emerald-400 text-black font-semibold text-base px-8 py-3.5 rounded-xl transition-all hover:shadow-xl hover:shadow-emerald-500/20"
            >
              Open Dashboard
              <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
            </Link>
          </div>
        </AnimatedSection>
      </section>

      {/* Footer */}
      <footer className="border-t border-zinc-800/50 py-8 px-4">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-zinc-500">
          <div className="flex items-center gap-2">
            <LineChart className="h-4 w-4 text-emerald-500" />
            <span>StockPredict AI</span>
          </div>
          <div className="flex items-center gap-6">
            <Link href="/dashboard" className="hover:text-white transition-colors">Dashboard</Link>
            <Link href="/predictions" className="hover:text-white transition-colors">Predictions</Link>
            <Link href="/news" className="hover:text-white transition-colors">News</Link>
            <Link
              href="https://github.com/Yogesh-VG0/Stock_Predict_Ai"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-white transition-colors"
            >
              GitHub
            </Link>
          </div>
          <span>Built by Yogesh Vadivel</span>
        </div>
      </footer>
    </div>
  )
}
