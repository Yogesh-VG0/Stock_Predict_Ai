import type { Metadata } from "next"
import Link from "next/link"

export const metadata: Metadata = {
  title: "Disclaimer | StockPredict AI",
  description:
    "Educational disclaimer for StockPredict AI. Not financial advice. Predictions are probabilistic estimates and may be inaccurate.",
}

export default function DisclaimerPage() {
  return (
    <div className="max-w-3xl mx-auto space-y-8">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold">Disclaimer</h1>
        <p className="text-sm text-zinc-400">
          This page explains what StockPredict AI is (and is not).
        </p>
      </header>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">Not financial advice</h2>
        <p className="text-sm text-zinc-300">
          StockPredict AI is an educational research project. It does not provide investment advice and is not a substitute for
          guidance from a licensed financial advisor.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">Predictions may be wrong</h2>
        <p className="text-sm text-zinc-300">
          All forecasts and confidence scores are probabilistic estimates based on historical data and model assumptions. Market
          conditions can change quickly and models can fail under new regimes.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">Use at your own risk</h2>
        <p className="text-sm text-zinc-300">
          You are responsible for your own investment decisions. Always do your own research and consider risk management,
          liquidity, costs, and your personal financial situation.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">Where the data comes from</h2>
        <p className="text-sm text-zinc-300">
          The platform aggregates data from third-party providers (market data, news, and macro indicators). Data may be delayed,
          incomplete, or inaccurate.
        </p>
      </section>

      <div className="pt-2 text-sm text-zinc-400">
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
  )
}

