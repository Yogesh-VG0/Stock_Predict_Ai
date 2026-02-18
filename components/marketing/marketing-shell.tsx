"use client"

import type React from "react"
import Link from "next/link"
import { LineChart, GitBranch, ArrowRight } from "lucide-react"
import { motion } from "framer-motion"

export default function MarketingShell({
  children,
  activeNav,
}: {
  children: React.ReactNode
  activeNav?: "how" | "methodology" | "disclaimer"
}) {
  return (
    <div className="min-h-screen bg-black text-white overflow-x-hidden">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-[100] focus:px-4 focus:py-2 focus:bg-emerald-500 focus:text-black focus:rounded-lg focus:font-semibold"
      >
        Skip to main content
      </a>

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
              <GitBranch className="h-4 w-4" aria-hidden="true" />
              GitHub
            </Link>
            <Link
              href="/dashboard"
              className="group inline-flex items-center gap-2 bg-emerald-500 hover:bg-emerald-400 text-black font-semibold text-sm px-4 py-2 rounded-lg transition-all hover:shadow-lg hover:shadow-emerald-500/20"
              aria-label="Open Dashboard"
            >
              Open Dashboard
              <ArrowRight className="h-4 w-4 group-hover:translate-x-0.5 transition-transform" aria-hidden="true" />
            </Link>
          </div>
        </div>
      </nav>

      {/* Subtle background */}
      <div className="pointer-events-none fixed inset-0 -z-10">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_25%_10%,rgba(16,185,129,0.12),transparent_50%),radial-gradient(circle_at_80%_25%,rgba(59,130,246,0.10),transparent_50%),radial-gradient(circle_at_50%_90%,rgba(168,85,247,0.10),transparent_55%)]" />
        <div className="absolute inset-0 bg-[linear-gradient(to_bottom,rgba(0,0,0,0.65),rgba(0,0,0,1))]" />
      </div>

      <motion.main
        id="main-content"
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
        className="pt-24 pb-14 px-4"
      >
        <div className="max-w-5xl mx-auto">{children}</div>
      </motion.main>

      <footer className="border-t border-zinc-800/50 py-8 px-4" role="contentinfo">
        <div className="max-w-5xl mx-auto">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 text-sm text-zinc-500">
            <div className="flex items-center gap-2 justify-center sm:justify-start">
              <LineChart className="h-4 w-4 text-emerald-500" aria-hidden="true" />
              <span>StockPredict AI</span>
            </div>

            <div className="flex flex-wrap items-center justify-center sm:justify-start gap-x-6 gap-y-2">
              <Link href="/dashboard" className="hover:text-white transition-colors">
                Dashboard
              </Link>
              <Link href="/predictions" className="hover:text-white transition-colors">
                Predictions
              </Link>
              <Link href="/news" className="hover:text-white transition-colors">
                News
              </Link>
              <Link
                href="/how-it-works"
                className={`hover:text-white transition-colors ${activeNav === "how" ? "text-white" : ""}`}
              >
                How it works
              </Link>
              <Link
                href="/methodology"
                className={`hover:text-white transition-colors ${activeNav === "methodology" ? "text-white" : ""}`}
              >
                Methodology
              </Link>
              <Link
                href="/disclaimer"
                className={`hover:text-white transition-colors ${activeNav === "disclaimer" ? "text-white" : ""}`}
              >
                Disclaimer
              </Link>
              <Link
                href="https://github.com/Yogesh-VG0/Stock_Predict_Ai"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-white transition-colors"
              >
                GitHub
              </Link>
            </div>

            <span className="text-center sm:text-right">Built by Yogesh Vadivel</span>
          </div>
        </div>
      </footer>
    </div>
  )
}

