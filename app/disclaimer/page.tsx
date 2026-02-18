import type { Metadata } from "next"
import React from "react"
import Link from "next/link"
import AnimatedBlock from "@/components/marketing/animated-block"
import MarketingShell from "@/components/marketing/marketing-shell"

export const metadata: Metadata = {
  title: "Disclaimer | StockPredict AI",
  description:
    "Educational disclaimer for StockPredict AI. Not financial advice. Predictions are probabilistic estimates and may be inaccurate.",
}

export default function DisclaimerPage() {
  return (
    <MarketingShell activeNav="disclaimer">
      <div className="space-y-10">
        <AnimatedBlock>
          <div className="rounded-2xl border border-zinc-800 bg-gradient-to-b from-zinc-900/70 to-black p-6 sm:p-8">
            <div className="inline-flex items-center gap-2 rounded-full border border-zinc-800 bg-zinc-950/50 px-3 py-1 text-xs text-zinc-400 mb-4">
              Legal / educational
            </div>
            <h1 className="text-3xl sm:text-4xl font-bold tracking-tight">Disclaimer</h1>
            <p className="mt-3 text-sm sm:text-base text-zinc-400 max-w-3xl leading-relaxed">
              This page explains what StockPredict AI is (and is not).
            </p>

            <div className="mt-6 flex flex-col sm:flex-row gap-3">
              <Link
                href="/how-it-works"
                className="inline-flex items-center justify-center rounded-xl border border-zinc-800 bg-zinc-900/40 hover:bg-zinc-900/60 px-4 py-2.5 text-sm text-white transition-colors"
              >
                How it works
              </Link>
              <Link
                href="/methodology"
                className="inline-flex items-center justify-center rounded-xl border border-zinc-800 bg-zinc-900/40 hover:bg-zinc-900/60 px-4 py-2.5 text-sm text-white transition-colors"
              >
                Methodology
              </Link>
            </div>
          </div>
        </AnimatedBlock>

        <AnimatedBlock delay={0.06}>
          <section className="space-y-3">
            {React.createElement("h2", { className: "text-xl sm:text-2xl font-semibold" }, "Not financial advice")}
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              StockPredict AI is an educational research project. It does not provide investment advice and is not a substitute
              for guidance from a licensed financial advisor.
            </p>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.08}>
          <section className="space-y-3">
            {React.createElement("h2", { className: "text-xl sm:text-2xl font-semibold" }, "Predictions may be wrong")}
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              All forecasts and confidence scores are probabilistic estimates based on historical data and model assumptions.
              Market conditions can change quickly and models can fail under new regimes.
            </p>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.1}>
          <section className="space-y-3">
            {React.createElement("h2", { className: "text-xl sm:text-2xl font-semibold" }, "Use at your own risk")}
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              You are responsible for your own investment decisions. Always do your own research and consider risk management,
              liquidity, costs, and your personal financial situation.
            </p>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.12}>
          <section className="space-y-3">
            {React.createElement("h2", { className: "text-xl sm:text-2xl font-semibold" }, "Where the data comes from")}
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              The platform aggregates data from third-party providers (market data, news, and macro indicators). Data may be
              delayed, incomplete, or inaccurate.
            </p>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.14}>
          <div className="rounded-2xl border border-zinc-800 bg-zinc-950/40 p-5 sm:p-6">
            <div className="text-sm text-zinc-400">
              Want details on the pipeline? Read{" "}
              <Link href="/how-it-works" className="text-emerald-400 hover:text-emerald-300 transition-colors">
                How it works
              </Link>{" "}
              and{" "}
              <Link href="/methodology" className="text-emerald-400 hover:text-emerald-300 transition-colors">
                Methodology
              </Link>
              .
            </div>
          </div>
        </AnimatedBlock>
      </div>
    </MarketingShell>
  )
}

