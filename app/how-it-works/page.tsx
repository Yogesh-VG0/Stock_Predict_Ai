import type { Metadata } from "next"
import Link from "next/link"
import AnimatedBlock from "@/components/marketing/animated-block"
import MarketingShell from "@/components/marketing/marketing-shell"

export const metadata: Metadata = {
  title: "How It Works | StockPredict AI",
  description:
    "Learn how StockPredict AI fetches data, engineers features, trains LightGBM models, generates predictions, and explains them with SHAP and Gemini AI.",
}

export default function HowItWorksPage() {
  return (
    <MarketingShell activeNav="how">
      <div className="space-y-10">
        <AnimatedBlock>
          <div className="rounded-2xl border border-zinc-800 bg-gradient-to-b from-zinc-900/70 to-black p-6 sm:p-8">
            <div className="inline-flex items-center gap-2 rounded-full border border-zinc-800 bg-zinc-950/50 px-3 py-1 text-xs text-zinc-400 mb-4">
              Pipeline overview
            </div>
            <h1 className="text-3xl sm:text-4xl font-bold tracking-tight">How StockPredict AI Works</h1>
            <p className="mt-3 text-sm sm:text-base text-zinc-400 max-w-3xl leading-relaxed">
              End-to-end pipeline from raw market data to ML forecasts and plain-English explanations for the S&amp;P 100.
            </p>

            <div className="mt-6 flex flex-col sm:flex-row gap-3">
              <Link
                href="/methodology"
                className="inline-flex items-center justify-center rounded-xl border border-zinc-800 bg-zinc-900/40 hover:bg-zinc-900/60 px-4 py-2.5 text-sm text-white transition-colors"
              >
                Read Methodology
              </Link>
              <Link
                href="/dashboard"
                className="inline-flex items-center justify-center rounded-xl bg-emerald-500 hover:bg-emerald-400 px-4 py-2.5 text-sm font-semibold text-black transition-colors"
              >
                Open Dashboard
              </Link>
            </div>
          </div>
        </AnimatedBlock>

        <AnimatedBlock delay={0.05}>
          <div className="rounded-2xl border border-zinc-800 bg-zinc-950/40 p-5 sm:p-6">
            <div className="text-sm font-semibold mb-3">Contents</div>
            <div className="grid sm:grid-cols-2 gap-2 text-sm text-zinc-300">
              <a className="hover:text-white transition-colors" href="#architecture-diagram">
                Architecture diagram
              </a>
              <a className="hover:text-white transition-colors" href="#data-sources">
                1. Data sources
              </a>
              <a className="hover:text-white transition-colors" href="#feature-engineering">
                2. Feature engineering
              </a>
              <a className="hover:text-white transition-colors" href="#model-training">
                3. Model training (LightGBM)
              </a>
              <a className="hover:text-white transition-colors" href="#daily-pipeline">
                4. Daily automated pipeline
              </a>
              <a className="hover:text-white transition-colors" href="#explainability">
                5. Explainability (SHAP + Gemini)
              </a>
              <a className="hover:text-white transition-colors" href="#example-explanation">
                Example prediction explanation
              </a>
              <a className="hover:text-white transition-colors" href="#limitations">
                Limitations
              </a>
            </div>
          </div>
        </AnimatedBlock>

        <AnimatedBlock delay={0.08}>
          <section id="architecture-diagram" className="space-y-4 scroll-mt-24">
            <h2 className="text-xl sm:text-2xl font-semibold">Architecture diagram</h2>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              High-level flow from ingestion to UI. This mirrors the daily automated pipeline that runs in CI/CD.
            </p>

            <div className="rounded-2xl border border-zinc-800 bg-zinc-950/40 p-4 sm:p-6 overflow-x-auto">
              <svg
                className="min-w-[780px] w-full h-[140px]"
                viewBox="0 0 1000 180"
                role="img"
                aria-label="Architecture flow: Data Sources to Feature Engineering to LightGBM to SHAP to Gemini to API to Frontend"
              >
                <defs>
                  <linearGradient id="box" x1="0" x2="1">
                    <stop offset="0%" stopColor="rgba(24,24,27,0.95)" />
                    <stop offset="100%" stopColor="rgba(0,0,0,0.95)" />
                  </linearGradient>
                  <marker id="arrow" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto">
                    <path d="M0,0 L8,3 L0,6 Z" fill="rgba(244,244,245,0.65)" />
                  </marker>
                </defs>

                {[
                  { x: 20, label: "Data sources" },
                  { x: 175, label: "Feature engineering" },
                  { x: 375, label: "LightGBM" },
                  { x: 505, label: "SHAP" },
                  { x: 605, label: "Gemini" },
                  { x: 730, label: "API" },
                  { x: 850, label: "Frontend" },
                ].map((b) => (
                  <g key={b.x}>
                    <rect x={b.x} y="55" rx="14" ry="14" width="140" height="70" fill="url(#box)" stroke="rgba(63,63,70,0.9)" />
                    <text x={b.x + 70} y="92" textAnchor="middle" fill="rgba(244,244,245,0.92)" fontSize="16" fontFamily="ui-sans-serif, system-ui">
                      {b.label}
                    </text>
                  </g>
                ))}

                {[160, 340, 490, 590, 715, 835].map((x) => (
                  <line
                    key={x}
                    x1={x}
                    y1="90"
                    x2={x + 15}
                    y2="90"
                    stroke="rgba(244,244,245,0.65)"
                    strokeWidth="2"
                    markerEnd="url(#arrow)"
                  />
                ))}
              </svg>
            </div>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.1}>
          <section id="data-sources" className="space-y-3 scroll-mt-24">
            <h2 className="text-xl sm:text-2xl font-semibold">1. Data sources</h2>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              StockPredict AI aggregates data from multiple providers to build a rich feature set:
            </p>
            <ul className="list-disc list-inside text-sm sm:text-base text-zinc-300 space-y-2">
              <li>
                <strong className="text-zinc-100">Prices &amp; fundamentals</strong>: OHLCV, fundamentals, and analyst data from
                providers like Finnhub and FMP.
              </li>
              <li>
                <strong className="text-zinc-100">Macro &amp; economics</strong>: FRED macro indicators (rates, yields,
                unemployment, CPI, GDP) to capture the macro regime.
              </li>
              <li>
                <strong className="text-zinc-100">News &amp; filings</strong>: SEC filings, RSS feeds, and headlines from sources
                such as Reddit and FinViz.
              </li>
              <li>
                <strong className="text-zinc-100">Real-time feed</strong>: Live prices via Finnhub WebSocket and TradingView
                widgets in the UI.
              </li>
            </ul>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.12}>
          <section id="feature-engineering" className="space-y-3 scroll-mt-24">
            <h2 className="text-xl sm:text-2xl font-semibold">2. Feature engineering</h2>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              For each ticker, the ML backend builds a feature vector with 40+ signals, including:
            </p>
            <ul className="list-disc list-inside text-sm sm:text-base text-zinc-300 space-y-2">
              <li>
                <strong className="text-zinc-100">Price-based features</strong>: rolling returns, volatility, gaps, volume spikes,
                and moving averages.
              </li>
              <li>
                <strong className="text-zinc-100">Technical indicators</strong>: RSI, MACD, Bollinger Bands, trend strength, and
                overbought/oversold flags.
              </li>
              <li>
                <strong className="text-zinc-100">Sentiment features</strong>: aggregated scores from FinBERT, RoBERTa, and VADER
                across news, Reddit, and filings.
              </li>
              <li>
                <strong className="text-zinc-100">Macro &amp; cross-asset signals</strong>: yields, spreads, index levels, and
                sector ETFs.
              </li>
            </ul>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              Features are aligned on a daily timeline, normalized, and stored in MongoDB for training and analysis.
            </p>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.14}>
          <section id="model-training" className="space-y-3 scroll-mt-24">
            <h2 className="text-xl sm:text-2xl font-semibold">3. Model training (LightGBM)</h2>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              The core predictor is a LightGBM gradient-boosted tree model. It learns to forecast log-returns for three horizons:
            </p>
            <ul className="list-disc list-inside text-sm sm:text-base text-zinc-300 space-y-2">
              <li>
                <strong className="text-zinc-100">Next day</strong> (1-day / next-day horizon)
              </li>
              <li>
                <strong className="text-zinc-100">1 week</strong> (7 trading days)
              </li>
              <li>
                <strong className="text-zinc-100">1 month</strong> (30 calendar days)
              </li>
            </ul>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              Models are trained in a walk-forward fashion: each training window uses historical data only up to that point and is
              evaluated on future periods, which reduces lookahead bias and gives a more realistic picture of performance.
            </p>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.16}>
          <section id="daily-pipeline" className="space-y-3 scroll-mt-24">
            <h2 className="text-xl sm:text-2xl font-semibold">4. Daily automated pipeline</h2>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              A GitHub Actions workflow runs the full ML pipeline every trading day:
            </p>
            <ol className="list-decimal list-inside text-sm sm:text-base text-zinc-300 space-y-2">
              <li>Fetch latest prices, news, sentiment, macro, and insider data.</li>
              <li>Update features and retrain pooled LightGBM models where needed.</li>
              <li>Generate predictions for all S&amp;P 100 stocks across three horizons.</li>
              <li>Run SHAP analysis to understand which features drove each prediction.</li>
              <li>Store predictions, explanations, and monitoring metrics in MongoDB.</li>
            </ol>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              The Next.js frontend reads from the Node.js API, which serves predictions and explanations to the UI.
            </p>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.18}>
          <section id="explainability" className="space-y-3 scroll-mt-24">
            <h2 className="text-xl sm:text-2xl font-semibold">5. SHAP explainability &amp; Gemini AI</h2>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              For each prediction, SHAP decomposes the model output into feature contributions (what pushed the forecast up or
              down). These numbers, along with sentiment and technical context, are passed to Google Gemini to generate a
              plain-English narrative.
            </p>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              This is what powers the AI explanation sections in the UI: instead of raw probabilities, users see a concise
              summary of the model&apos;s reasoning.
            </p>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.2}>
          <section id="example-explanation" className="space-y-4 scroll-mt-24">
            <h2 className="text-xl sm:text-2xl font-semibold">Example prediction explanation</h2>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              The platform surfaces a concise “what changed and why” summary based on SHAP feature attributions.
            </p>

            <div className="rounded-2xl border border-zinc-800 bg-zinc-950/40 p-5 sm:p-6">
              <div className="text-sm text-zinc-400">Example (illustrative)</div>
              <div className="mt-1 text-lg font-semibold">AAPL 7-day forecast: +1.2%</div>

              <div className="mt-4 grid sm:grid-cols-2 gap-4">
                <div>
                  <div className="text-sm font-semibold text-emerald-300">Top positive drivers</div>
                  <ul className="mt-2 list-disc list-inside text-sm text-zinc-300 space-y-1">
                    <li>Improving aggregated sentiment across recent headlines</li>
                    <li>Uptrend signal from momentum + moving averages</li>
                    <li>Favorable macro trend signals</li>
                  </ul>
                </div>
                <div>
                  <div className="text-sm font-semibold text-rose-300">Top negative drivers</div>
                  <ul className="mt-2 list-disc list-inside text-sm text-zinc-300 space-y-1">
                    <li>Elevated short-term volatility</li>
                    <li>Recent drawdown pressure vs. prior highs</li>
                  </ul>
                </div>
              </div>
            </div>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.22}>
          <section id="limitations" className="space-y-3 scroll-mt-24">
            <h2 className="text-xl sm:text-2xl font-semibold">6. Limitations &amp; risk disclaimer</h2>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              StockPredict AI is an educational research project. It is not a trading signal service and does not provide
              investment advice.
            </p>
            <ul className="list-disc list-inside text-sm sm:text-base text-zinc-300 space-y-2">
              <li>Models are trained on historical data and can fail under new market regimes.</li>
              <li>Predictions are probabilistic estimates, not guarantees of future prices.</li>
              <li>Real-world execution costs, slippage, and liquidity are not modeled in detail.</li>
            </ul>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              Always do your own research and consult a licensed financial advisor before making investment decisions.
            </p>
            <div className="pt-1 text-sm text-zinc-400">
              For evaluation details, see{" "}
              <Link href="/methodology" className="text-emerald-400 hover:text-emerald-300 transition-colors">
                Methodology
              </Link>
              .
            </div>
          </section>
        </AnimatedBlock>
      </div>
    </MarketingShell>
  )
}

