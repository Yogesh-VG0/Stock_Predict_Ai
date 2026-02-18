import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "Methodology | StockPredict AI",
  description:
    "Learn about StockPredict AI's methodology: walk-forward validation, metrics, and how it avoids lookahead bias when evaluating ML models.",
}

export default function MethodologyPage() {
  return (
    <div className="space-y-8 max-w-3xl mx-auto">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold">Methodology</h1>
        <p className="text-sm text-zinc-400">
          How StockPredict AI evaluates models, avoids lookahead bias, and measures performance.
        </p>
      </header>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">1. Walk-Forward Validation</h2>
        <p className="text-sm text-zinc-300">
          Instead of a single static train/test split, StockPredict AI uses walk-forward validation. Historical data is split
          into sequential windows:
        </p>
        <ul className="list-disc list-inside text-sm text-zinc-300 space-y-1">
          <li>Train on an initial history window.</li>
          <li>Evaluate on the following period (out-of-sample).</li>
          <li>Roll the window forward and repeat across the timeline.</li>
        </ul>
        <p className="text-sm text-zinc-300">
          This more closely matches how models are used in production, where you always train on the past and predict the
          future, and it helps avoid overly optimistic backtest results.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">2. Avoiding Lookahead Bias</h2>
        <p className="text-sm text-zinc-300">
          Lookahead bias happens when a model accidentally sees future information during training. StockPredict AI avoids this
          by:
        </p>
        <ul className="list-disc list-inside text-sm text-zinc-300 space-y-1">
          <li>Building features only from data available up to each prediction timestamp.</li>
          <li>Separating training and evaluation periods in time (no shuffling across the timeline).</li>
          <li>Using holdout windows that simulate &quot;live&quot; deployment conditions.</li>
        </ul>
        <p className="text-sm text-zinc-300">
          The goal is not to &quot;fit the past&quot; perfectly, but to estimate how the model might behave on unseen data.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">3. Evaluation Metrics</h2>
        <p className="text-sm text-zinc-300">
          Several metrics are used to understand model quality across different dimensions:
        </p>
        <ul className="list-disc list-inside text-sm text-zinc-300 space-y-1">
          <li>
            <strong>MAE (Mean Absolute Error)</strong>: average absolute difference between predicted and realized returns or
            prices.
          </li>
          <li>
            <strong>Directional accuracy</strong>: percentage of times the model correctly predicts the sign (up vs down).
          </li>
          <li>
            <strong>Calibration / Brier score</strong>: how well predicted probabilities match observed frequencies.
          </li>
          <li>
            <strong>Rank metrics</strong>: how well the model orders stocks (e.g., top decile vs bottom decile performance).
          </li>
        </ul>
        <p className="text-sm text-zinc-300">
          No single metric tells the full story; the system looks at a dashboard of metrics to catch issues like drift or
          overfitting.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">4. Drift Monitoring</h2>
        <p className="text-sm text-zinc-300">
          Markets change, so the pipeline includes drift checks to detect when the relationship between features and targets may
          be breaking down:
        </p>
        <ul className="list-disc list-inside text-sm text-zinc-300 space-y-1">
          <li>Population Stability Index (PSI) between historical and recent feature distributions.</li>
          <li>Rolling directional accuracy and error metrics over time.</li>
          <li>Monitoring alpha magnitude and hit rates across segments.</li>
        </ul>
        <p className="text-sm text-zinc-300">
          When drift is detected, models may need retraining, re-tuning, or in some cases, feature redesign.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">5. Interpreting Results</h2>
        <p className="text-sm text-zinc-300">
          Even with careful validation, ML forecasts are noisy. StockPredict AI emphasizes:
        </p>
        <ul className="list-disc list-inside text-sm text-zinc-300 space-y-1">
          <li>Using predictions as probabilistic signals, not guarantees.</li>
          <li>Combining model outputs with human judgment and risk management.</li>
          <li>Being transparent about model limitations and assumptions.</li>
        </ul>
        <p className="text-sm text-zinc-300">
          The platform is designed as an educational and research tool, not a plug-and-play trading system.
        </p>
      </section>
    </div>
  )
}

