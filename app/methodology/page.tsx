import type { Metadata } from "next"
import Link from "next/link"
import AnimatedBlock from "@/components/marketing/animated-block"
import MarketingShell from "@/components/marketing/marketing-shell"

export const metadata: Metadata = {
  title: "Methodology | StockPredict AI",
  description:
    "Learn about StockPredict AI's methodology: walk-forward validation, market-neutral backtesting, cross-sectional ranking, and how it avoids lookahead bias.",
}

export default function MethodologyPage() {
  return (
    <MarketingShell activeNav="methodology">
      <div className="space-y-10">
        <AnimatedBlock>
          <div className="rounded-2xl border border-zinc-800 bg-gradient-to-b from-zinc-900/70 to-black p-6 sm:p-8">
            <div className="inline-flex items-center gap-2 rounded-full border border-zinc-800 bg-zinc-950/50 px-3 py-1 text-xs text-zinc-400 mb-4">
              Evaluation &amp; monitoring
            </div>
            <h1 className="text-3xl sm:text-4xl font-bold tracking-tight">Methodology</h1>
            <p className="mt-3 text-sm sm:text-base text-zinc-400 max-w-3xl leading-relaxed">
              How StockPredict AI evaluates models, avoids lookahead bias, and monitors performance over time. Model v10.1.0 with LightGBM + LSTM hybrid.
            </p>

            <div className="mt-6 flex flex-col sm:flex-row gap-3">
              <Link
                href="/how-it-works"
                className="inline-flex items-center justify-center rounded-xl border border-zinc-800 bg-zinc-900/40 hover:bg-zinc-900/60 px-4 py-2.5 text-sm text-white transition-colors"
              >
                Read How it works
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
              <a className="hover:text-white transition-colors" href="#walk-forward">
                1. Walk-forward validation
              </a>
              <a className="hover:text-white transition-colors" href="#lookahead-bias">
                2. Avoiding lookahead bias
              </a>
              <a className="hover:text-white transition-colors" href="#metrics">
                3. Evaluation metrics
              </a>
              <a className="hover:text-white transition-colors" href="#drift">
                4. Drift monitoring
              </a>
              <a className="hover:text-white transition-colors" href="#interpretation">
                5. Interpreting results
              </a>
              <a className="hover:text-white transition-colors" href="#example">
                Example evaluation snapshot
              </a>
            </div>
          </div>
        </AnimatedBlock>

        <AnimatedBlock delay={0.08}>
          <section id="walk-forward" className="space-y-3 scroll-mt-24">
            <h2 className="text-xl sm:text-2xl font-semibold">1. Walk-forward validation</h2>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              Instead of a single static train/test split, StockPredict AI uses walk-forward validation with 4 rolling folds
              (starting at 50% of data, stepping 10% each fold). Historical data is split into sequential windows:
            </p>
            <ul className="list-disc list-inside text-sm sm:text-base text-zinc-300 space-y-2">
              <li>Train on an initial history window (~92K samples across 75 tickers).</li>
              <li>Evaluate on the following period (out-of-sample holdout: last 15%).</li>
              <li>Roll the window forward and repeat across the timeline.</li>
              <li>7-day purge gap + 3-day embargo between train/validate/holdout to prevent label leakage.</li>
            </ul>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              This matches production behavior: you always train on the past and predict the future, and it reduces overly
              optimistic backtests.
            </p>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.1}>
          <section id="lookahead-bias" className="space-y-3 scroll-mt-24">
            <h2 className="text-xl sm:text-2xl font-semibold">2. Avoiding lookahead bias</h2>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              Lookahead bias happens when a model accidentally uses future information during training. StockPredict AI avoids
              this by:
            </p>
            <ul className="list-disc list-inside text-sm sm:text-base text-zinc-300 space-y-2">
              <li>Building features only from data available up to each prediction timestamp (all features use <code className="text-zinc-200">shift(1)</code>).</li>
              <li>Separating training and evaluation periods in time (no shuffling across the timeline).</li>
              <li>Using holdout windows that simulate “live” deployment conditions.</li>
              <li>Predicting <strong className="text-zinc-100">market-neutral alpha</strong> (stock return minus SPY return) instead of absolute price, eliminating market-direction bias.</li>
            </ul>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              The goal is not to “fit the past” perfectly, but to estimate how the model might behave on unseen data.
            </p>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.12}>
          <section id="metrics" className="space-y-3 scroll-mt-24">
            <h2 className="text-xl sm:text-2xl font-semibold">3. Evaluation metrics</h2>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              The platform tracks multiple metrics to understand performance from different angles:
            </p>
            <ul className="list-disc list-inside text-sm sm:text-base text-zinc-300 space-y-2">
              <li>
                <strong className="text-zinc-100">Sharpe Ratio</strong>: return per unit of risk (annualized). Above 1.0 is good,
                above 2.0 is excellent.
              </li>
              <li>
                <strong className="text-zinc-100">Win Rate</strong>: percentage of profitable trades. Above 50% is decent,
                above 55% is good.
              </li>
              <li>
                <strong className="text-zinc-100">Max Drawdown</strong>: largest peak-to-trough decline. Below 10% for a
                cautious strategy.
              </li>
              <li>
                <strong className="text-zinc-100">Directional accuracy</strong>: how often the model predicts the sign correctly
                (up vs. down).
              </li>
              <li>
                <strong className="text-zinc-100">Rank correlation</strong>: Spearman correlation between predicted and actual
                returns &mdash; how well the model orders stocks.
              </li>
              <li>
                <strong className="text-zinc-100">Calibration / Brier score</strong>: how well predicted probabilities match
                observed frequencies.
              </li>
            </ul>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              No single metric tells the full story; a dashboard of metrics helps catch drift and overfitting.
            </p>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.14}>
          <section id="drift" className="space-y-3 scroll-mt-24">
            <h2 className="text-xl sm:text-2xl font-semibold">4. Drift monitoring</h2>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              Markets change, so the pipeline includes drift checks to detect when the relationship between features and targets
              may be breaking down:
            </p>
            <ul className="list-disc list-inside text-sm sm:text-base text-zinc-300 space-y-2">
              <li>Population Stability Index (PSI) between historical and recent feature distributions.</li>
              <li>Rolling directional accuracy and error metrics over time.</li>
              <li>Monitoring hit-rates across segments and regimes.</li>
            </ul>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              When drift is detected, models may need retraining, retuning, or in some cases, feature redesign.
            </p>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.16}>
          <section id="interpretation" className="space-y-3 scroll-mt-24">
            <h2 className="text-xl sm:text-2xl font-semibold">5. Interpreting results</h2>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              Even with careful validation, ML forecasts are noisy. StockPredict AI emphasizes:
            </p>
            <ul className="list-disc list-inside text-sm sm:text-base text-zinc-300 space-y-2">
              <li>Using predictions as probabilistic signals, not guarantees.</li>
              <li>Combining model outputs with human judgment and risk management.</li>
              <li>Being transparent about limitations and assumptions.</li>
            </ul>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              The platform is designed as an educational and research tool, not a plug-and-play trading system.
            </p>
          </section>
        </AnimatedBlock>

        <AnimatedBlock delay={0.18}>
          <section id="example" className="space-y-4 scroll-mt-24">
            <h2 className="text-xl sm:text-2xl font-semibold">Example evaluation snapshot</h2>
            <p className="text-sm sm:text-base text-zinc-300 leading-relaxed">
              The monitoring layer tracks multiple signals together so regressions are visible quickly.
            </p>

            <div className="rounded-2xl border border-zinc-800 bg-zinc-950/40 p-5 sm:p-6">
              <div className="text-sm text-zinc-400 mb-3">Latest production backtest (v10.1.0, Sep 2025 &mdash; Mar 2026, market-neutral)</div>
              <div className="grid sm:grid-cols-3 gap-4">
                <div className="rounded-xl border border-emerald-500/30 bg-emerald-950/20 p-4">
                  <div className="text-xs text-zinc-400">30-day horizon</div>
                  <div className="mt-1 text-lg font-semibold text-emerald-400">+13.71%</div>
                  <div className="text-sm text-zinc-500">Sharpe 2.68 &middot; 64.3% WR &middot; 28 trades</div>
                </div>
                <div className="rounded-xl border border-emerald-500/30 bg-emerald-950/20 p-4">
                  <div className="text-xs text-zinc-400">7-day horizon</div>
                  <div className="mt-1 text-lg font-semibold text-emerald-400">+8.05%</div>
                  <div className="text-sm text-zinc-500">Sharpe 1.61 &middot; 50.4% WR &middot; 129 trades</div>
                </div>
                <div className="rounded-xl border border-zinc-700 bg-zinc-900/30 p-4">
                  <div className="text-xs text-zinc-400">Next-day horizon</div>
                  <div className="mt-1 text-lg font-semibold">0 trades</div>
                  <div className="text-sm text-zinc-500">Model correctly abstained (no edge)</div>
                </div>
              </div>
              <div className="mt-3 text-xs text-zinc-500">SPY returned +1.90% over the same period. All returns are alpha (excess over SPY).</div>
            </div>

            <div className="pt-1 text-sm text-zinc-400">
              For end-to-end pipeline details, see{" "}
              <Link href="/how-it-works" className="text-emerald-400 hover:text-emerald-300 transition-colors">
                How it works
              </Link>
              .
            </div>
          </section>
        </AnimatedBlock>
      </div>
    </MarketingShell>
  )
}

