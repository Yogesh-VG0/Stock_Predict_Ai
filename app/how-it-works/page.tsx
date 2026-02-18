import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "How It Works | StockPredict AI",
  description:
    "Learn how StockPredict AI fetches data, engineers features, trains LightGBM models, generates predictions, and explains them with SHAP and Gemini AI.",
}

export default function HowItWorksPage() {
  return (
    <div className="space-y-8 max-w-3xl mx-auto">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold">How StockPredict AI Works</h1>
        <p className="text-sm text-zinc-400">
          End-to-end pipeline from raw market data to AI-powered stock predictions and explanations for the S&amp;P 100.
        </p>
      </header>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">1. Data Sources</h2>
        <p className="text-sm text-zinc-300">
          StockPredict AI aggregates data from multiple providers to build a rich feature set:
        </p>
        <ul className="list-disc list-inside text-sm text-zinc-300 space-y-1">
          <li>
            <strong>Prices &amp; fundamentals</strong>: OHLCV, fundamentals, and analyst data from providers like Finnhub and
            FMP.
          </li>
          <li>
            <strong>Macro &amp; economics</strong>: FRED macro indicators (rates, yields, unemployment, CPI, GDP) to capture the
            macro regime.
          </li>
          <li>
            <strong>News &amp; filings</strong>: SEC filings, RSS feeds, SeekingAlpha-style news, and Reddit/FinViz headlines.
          </li>
          <li>
            <strong>Real-time feed</strong>: Live prices via Finnhub WebSocket and TradingView widgets in the UI.
          </li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">2. Feature Engineering</h2>
        <p className="text-sm text-zinc-300">
          For each of the 100 tickers, the ML backend builds a feature vector with 40+ signals, including:
        </p>
        <ul className="list-disc list-inside text-sm text-zinc-300 space-y-1">
          <li>
            <strong>Price-based features</strong>: rolling returns, volatility, gaps, volume spikes, and moving averages.
          </li>
          <li>
            <strong>Technical indicators</strong>: RSI, MACD, Bollinger Bands, trend strength, and overbought/oversold flags.
          </li>
          <li>
            <strong>Sentiment features</strong>: aggregated scores from FinBERT, RoBERTa, VADER on news, Reddit, and filings.
          </li>
          <li>
            <strong>Macro &amp; cross-asset signals</strong>: yields, spreads, index levels, and sector ETFs.
          </li>
        </ul>
        <p className="text-sm text-zinc-300">
          All features are aligned on a daily timeline, normalized, and stored in MongoDB for training and analysis.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">3. Model Training (LightGBM)</h2>
        <p className="text-sm text-zinc-300">
          The core predictor is a LightGBM gradient-boosted tree model. It learns to forecast log-returns for three horizons:
        </p>
        <ul className="list-disc list-inside text-sm text-zinc-300 space-y-1">
          <li>
            <strong>Next day</strong> (1-day / next-day horizon)
          </li>
          <li>
            <strong>1 week</strong> (7 trading days)
          </li>
          <li>
            <strong>1 month</strong> (30 calendar days)
          </li>
        </ul>
        <p className="text-sm text-zinc-300">
          Models are trained in a walk-forward fashion: each training window uses historical data only up to that point and is
          evaluated on future periods, which reduces lookahead bias and gives a more realistic picture of performance.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">4. Daily Automated Pipeline</h2>
        <p className="text-sm text-zinc-300">
          A GitHub Actions workflow runs the full ML pipeline every trading day:
        </p>
        <ol className="list-decimal list-inside text-sm text-zinc-300 space-y-1">
          <li>Fetch latest prices, news, sentiment, macro, and insider data.</li>
          <li>Update features and re-train pooled LightGBM models where needed.</li>
          <li>Generate predictions for all S&amp;P 100 stocks across 3 horizons.</li>
          <li>Run SHAP analysis to understand which features drove each prediction.</li>
          <li>Store predictions, explanations, and metrics in MongoDB.</li>
        </ol>
        <p className="text-sm text-zinc-300">
          The Next.js frontend then reads from the Node.js API, which serves these predictions and explanations to the UI.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">5. SHAP Explainability &amp; Gemini AI</h2>
        <p className="text-sm text-zinc-300">
          For each prediction, SHAP decomposes the model output into feature contributions (what pushed the forecast up or down).
          These numbers, along with sentiment and technical context, are passed to Google Gemini to generate a plain-English
          narrative.
        </p>
        <p className="text-sm text-zinc-300">
          This is what powers the AI explanation sections in the UI: instead of raw probabilities, users see a human-readable
          summary of the model&apos;s reasoning.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">6. Limitations &amp; Risk Disclaimer</h2>
        <p className="text-sm text-zinc-300">
          StockPredict AI is an educational research project. It is not a trading signal service and does not provide investment
          advice.
        </p>
        <ul className="list-disc list-inside text-sm text-zinc-300 space-y-1">
          <li>Models are trained on historical data and can fail under new market regimes.</li>
          <li>Predictions are probabilistic estimates, not guarantees of future prices.</li>
          <li>Real-world execution costs, slippage, and liquidity are not modeled in detail.</li>
        </ul>
        <p className="text-sm text-zinc-300">
          Always do your own research and consult a licensed financial advisor before making investment decisions.
        </p>
      </section>
    </div>
  )
}

