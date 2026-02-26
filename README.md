<p align="center">
  <img src="public/favicon.ico" alt="StockPredict AI" width="80" />
</p>

<h1 align="center">StockPredict AI</h1>

<p align="center">
  <strong>Full-stack stock analytics &amp; ML prediction platform for the S&amp;P 100</strong><br/>
  Real-time data · 10+ sentiment sources · 77 engineered features · plain-English AI explanations
</p>

<p align="center">
  <a href="https://stockpredict.dev">stockpredict.dev</a>
</p>

<p align="center">
  <a href="https://nextjs.org/"><img src="https://img.shields.io/badge/Next.js-15-black?style=flat-square&logo=next.js" alt="Next.js" /></a>
  <a href="https://reactjs.org/"><img src="https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react" alt="React" /></a>
  <a href="https://www.typescriptlang.org/"><img src="https://img.shields.io/badge/TypeScript-5-3178C6?style=flat-square&logo=typescript" alt="TypeScript" /></a>
  <a href="https://nodejs.org/"><img src="https://img.shields.io/badge/Node.js-18+-339933?style=flat-square&logo=node.js" alt="Node" /></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python" alt="Python" /></a>
  <a href="https://www.mongodb.com/"><img src="https://img.shields.io/badge/MongoDB-Atlas-47A248?style=flat-square&logo=mongodb" alt="MongoDB" /></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-ML%20API-009688?style=flat-square&logo=fastapi" alt="FastAPI" /></a>
  <a href="https://lightgbm.readthedocs.io/"><img src="https://img.shields.io/badge/LightGBM-Predictor-FFCC00?style=flat-square" alt="LightGBM" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-AGPL--3.0-blue.svg?style=flat-square" alt="License" /></a>
  <a href="https://github.com/Yogesh-VG0/Stock_Predict_Ai"><img src="https://img.shields.io/github/last-commit/Yogesh-VG0/Stock_Predict_Ai?style=flat-square" alt="GitHub last commit" /></a>
</p>

---

## Overview

StockPredict AI predicts stock prices for **all 100 S&P 100 companies** across three time horizons (1-day, 7-day, 30-day) using a **LightGBM gradient-boosted decision tree** trained on 42+ engineered features. Every prediction is explained in plain English by Google Gemini, backed by SHAP feature-importance decomposition.

A **fully automated nightly pipeline** (GitHub Actions) runs after market close — fetching data from 10+ sources, training models, generating predictions, running SHAP analysis, writing AI explanations, and evaluating accuracy — all stored in MongoDB and served through a three-tier architecture.

**Live at [stockpredict.dev](https://stockpredict.dev)**

---

## Table of Contents

- [Key Features](#key-features)
- [How It Works — The Three "Models"](#how-it-works--the-three-models)
- [System Architecture](#system-architecture)
- [Daily Automated Pipeline](#daily-automated-pipeline)
- [Machine Learning Deep Dive](#machine-learning-deep-dive)
- [Sentiment Analysis Engine](#sentiment-analysis-engine)
- [Data Sources & APIs](#data-sources--apis)
- [Technology Stack](#technology-stack)
- [Frontend Architecture](#frontend-architecture)
- [Pipeline Hardening & Reliability](#pipeline-hardening--reliability)
- [Model Validation & Backtesting](#model-validation--backtesting)
- [Full Documentation](#full-documentation)
- [License](#license)

---

## Key Features

| Feature | Description |
|---------|-------------|
| **ML Price Predictions** | LightGBM forecasts across 1-day, 7-day, and 30-day horizons with confidence scores, price ranges, and trade recommendations |
| **AI Explanations** | SHAP decomposes each prediction into bullish/bearish drivers; Gemini writes a structured plain-English report per stock |
| **Multi-Source Sentiment** | 10+ sources (Finviz, Reddit, SEC filings, Finnhub, FMP, Marketaux, Yahoo RSS, Seeking Alpha) scored with FinBERT, RoBERTa, and VADER |
| **Real-Time Market Data** | Live quotes via Finnhub WebSocket, interactive TradingView charts, and computed technical indicators (RSI, MACD, Bollinger Bands) |
| **Sankey Financial Flows** | Interactive income-statement Sankey diagrams (Apache ECharts) showing revenue sources → expenses → profit paths per company |
| **Watchlist & Alerts** | Track symbols with real-time price updates; notification system for market events |
| **Unified News Feed** | Aggregated from Yahoo Finance, Seeking Alpha, Finnhub, Marketaux, TickerTick — with inline VADER sentiment scoring |
| **Market Intelligence** | Fear & Greed Index, market open/close status with holiday detection, trading hours timeline |

---

## How It Works — The Three "Models"

> **This is the most important concept.** The project uses three very different "models" — only one predicts prices.

### 1. The Predictor — LightGBM

The **only** component that predicts stock prices. A gradient-boosted decision tree trained on 42+ numeric features (price history, sentiment scores, macro data, insider activity). Outputs a predicted log-return per stock per horizon.

### 2. The Explainer — SHAP + Gemini

SHAP mathematically decomposes _why_ LightGBM made its prediction (which features pushed the price up or down). Gemini then reads the SHAP results, sentiment data, news headlines, macro indicators, insider trades, and short interest — and writes a structured plain-English explanation. **Gemini does not predict prices.**

### 3. The Sentiment Scorers — FinBERT, RoBERTa, VADER

Read news headlines, Reddit posts, and SEC filings; produce a sentiment score (−1 to +1). These scores become **input features** for the LightGBM predictor — they are not predictions themselves.

```
News / Reddit / SEC  →  FinBERT / RoBERTa / VADER  →  sentiment score (NUMBER)
                                                              │
Price history        →  Feature engineering          →  42+ features (NUMBERS)
                                                              │
                                                              ▼
                                                       LightGBM Predictor
                                                              │
                                                      predicted return (NUMBER)
                                                              │
                                                  ┌───────────┴───────────┐
                                                  ▼                       ▼
                                            SHAP Analysis            Stored in DB
                                                  │
                                            feature contributions
                                                  │
                                                  ▼
                                            Gemini AI Explainer
                                                  │
                                            plain-English explanation (TEXT)
                                                  │
                                                  ▼
                                            Shown to user in UI
```

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         USER (Browser)                           │
│   Next.js App Router  ·  React  ·  TradingView  ·  ECharts      │
└───────────────────────────┬──────────────────────────────────────┘
                            │  HTTP / WebSocket
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    NODE.JS BACKEND (Express)                      │
│   Port 5000  ·  API Gateway  ·  News Aggregation  ·  Watchlist   │
│   Proxies ML Backend  ·  Finnhub WebSocket  ·  Notifications     │
└──────────┬──────────────────────────────────────────┬────────────┘
           │  HTTP                                     │  WebSocket
           ▼                                           ▼
┌───────────────────────┐                ┌───────────────────────┐
│  ML BACKEND (FastAPI)  │                │  Finnhub WebSocket     │
│  Port 8000             │                │  wss://ws.finnhub.io   │
│  Predictions · SHAP    │                │  Real-time trades      │
│  Sentiment · Training  │                └───────────────────────┘
│  Gemini Explanations   │
└──────────┬─────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│                        MONGODB ATLAS                              │
│  historical_data · sentiment · stock_predictions                  │
│  prediction_explanations · feature_importance                     │
│  insider_transactions · macro_data_raw · notifications            │
└──────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│                      REDIS (Optional)                             │
│  Prediction caching (60s)  ·  Rate limiting  ·  Holiday cache     │
└──────────────────────────────────────────────────────────────────┘
```

| Layer | Role |
|-------|------|
| **Frontend** (Next.js 15, port 3000) | UI, TradingView charts, Sankey diagrams, search, watchlist, stock detail pages |
| **Node Backend** (Express, port 5000) | API gateway, news aggregation, watchlist, Finnhub WebSocket, proxies ML endpoints |
| **ML Backend** (FastAPI, port 8000) | Predictions, sentiment analysis, model training, SHAP, Gemini explanations |
| **MongoDB Atlas** | All persistent data — historical prices, sentiment, predictions, explanations, insider trades, macro data |
| **Redis** (optional) | Caching (predictions, Sankey data, holidays), rate limiting (sliding window) |

---

## Daily Automated Pipeline

Runs every weeknight via **GitHub Actions** (~6:15 PM ET, after market close). Total runtime: ~60 minutes on a standard runner (7 GB RAM, 2 CPUs).

```
 ╔════════════════════════════════════════════════════════════════════╗
 ║  STEP 1: Gather Sentiment (~5 min)                                ║
 ║  Fetch news/social data for all 100 tickers from 10+ sources.     ║
 ║  Score with FinBERT, RoBERTa, VADER. Blend into composite score.  ║
 ║  Non-fatal — pipeline continues with stale sentiment if needed.   ║
 ╠════════════════════════════════════════════════════════════════════╣
 ║  STEP 2: Train Models (~15 min)                                   ║
 ║  Ingest OHLCV from Yahoo Finance. Engineer 42+ features.          ║
 ║  Train ONE pooled LightGBM model per horizon (3 models total).    ║
 ║  FATAL if fails — no predictions without trained models.          ║
 ╠════════════════════════════════════════════════════════════════════╣
 ║  STEP 3: Generate Predictions (~20 min)                           ║
 ║  Run models on all 100 tickers (10 batches × 10).                 ║
 ║  Per stock: predicted return, price, confidence, trade signal.     ║
 ║  Canary verification: checks 8 benchmark tickers for freshness.   ║
 ╠════════════════════════════════════════════════════════════════════╣
 ║  STEP 4: Explain Predictions (~15 min)                            ║
 ║  SHAP analysis decomposes each prediction into feature drivers.   ║
 ║  Gemini reads 11 data sources and writes per-stock explanations.  ║
 ╠════════════════════════════════════════════════════════════════════╣
 ║  STEP 5: Evaluate & Monitor (~5 min)                              ║
 ║  Compare last 60 days of predictions vs actuals.                  ║
 ║  Drift monitor: PSI, rolling accuracy, calibration checks.        ║
 ║  Quality gate: ≥80% tickers predicted, ≤20% data failures.       ║
 ╚════════════════════════════════════════════════════════════════════╝
```

### Failure Handling

| Step | Fatal? | Behavior |
|------|--------|----------|
| Sentiment cron | No | Logs warning, continues with stale data |
| Model training | **Yes** | Job fails immediately |
| Predictions | **Partial** | Fails if >30% of batches fail |
| Freshness verification | **Yes** | Asserts canary tickers are <3 hours old |
| SHAP analysis | **Partial** | Fails if >50% of batches fail |
| AI explanations | No | Logs warning, continues |
| Evaluation / Drift | No | Reports saved as artifacts |

---

## Machine Learning Deep Dive

### Why LightGBM?

LightGBM (gradient-boosted decision trees) was chosen over deep learning approaches like LSTMs because:

- **Tabular data advantage** — The 42+ pre-engineered numeric features are cross-sectional (many tickers × many features), which suits tree models far better than sequential RNNs
- **Speed** — Trains across 100 stocks in ~15 minutes on a free GitHub Actions runner
- **Handles missing values natively** — Critical when some API sources fail on any given day
- **Interpretable** — Built-in feature importance + fast TreeSHAP for explanations
- **Robust** — Huber loss function is resilient to outlier returns

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Objective | Huber | Robust to outlier returns |
| Learning rate | 0.05 | Slow learning for stability |
| Max depth | 4 | Prevents overfitting |
| Num leaves | 15 | Conservative tree complexity |
| N estimators | 150 | Balanced compute/accuracy |
| Min child samples | 25 | Requires sufficient evidence per leaf |
| Regularization | L1=0.1, L2=0.1 | Prevents feature over-reliance |
| Subsampling | 80% rows, 80% columns | Reduces variance |

### Market-Neutral Alpha

The model predicts **alpha** (excess return over SPY), not just absolute price direction:

```
target = stock_return − SPY_return
```

This means predictions capture how much a stock will **outperform or underperform** the S&P 500, rather than simply whether it goes up.

### Trade Recommendation Filters

A prediction generates a `trade_recommended = True` signal only when:
- Predicted alpha > 0.1%
- P(return > 0) > 52%
- Predicted return exceeds transaction costs (10 basis points)

### 77 Engineered Features

Features are organized into ten categories, all using `shift(1)` to ensure **point-in-time safety** (no future data leakage):

<details>
<summary><strong>Price & Return Features (6)</strong></summary>

| Feature | Description |
|---------|-------------|
| `log_return_1d` | Yesterday's log return |
| `log_return_5d` | 5-day log return |
| `log_return_21d` | 21-day log return |
| `volatility_20d` | 20-day rolling volatility |
| `intraday_range` | (High − Low) / Close |
| `overnight_gap` | Today's Open vs yesterday's Close |

</details>

<details>
<summary><strong>Volume Features (3)</strong></summary>

| Feature | Description |
|---------|-------------|
| `volume_ratio` | Volume / 20-day average volume |
| `volume_z60` | Volume z-score over 60 days |
| `volume_vol_ratio` | Volume volatility ratio |

</details>

<details>
<summary><strong>Technical Features (7)</strong></summary>

| Feature | Description |
|---------|-------------|
| `rsi` | 14-day Relative Strength Index |
| `rsi_divergence` | Price-RSI divergence signal |
| `bb_position` | Position within Bollinger Bands |
| `price_vs_sma20` | Price relative to 20-day SMA |
| `price_vs_sma50` | Price relative to 50-day SMA |
| `momentum_5d` | 5-day price momentum |
| `trend_20d` | 20-day linear trend slope |

</details>

<details>
<summary><strong>Market Regime Features (5)</strong></summary>

| Feature | Description |
|---------|-------------|
| `vix_level` | Current VIX (fear index) level |
| `vix_vol_20d` | VIX 20-day volatility |
| `spy_vol_20d` | S&P 500 20-day volatility |
| `spy_vol_regime` | Quantile-based volatility regime |
| `vol_regime` | Stock's own volatility regime |

</details>

<details>
<summary><strong>Sector-Relative Features (8)</strong></summary>

| Feature | Description |
|---------|-------------|
| `sector_id` | Numeric sector identifier |
| `ticker_id` | Numeric ticker identifier |
| `sector_etf_return_20d` | Sector ETF 20-day return |
| `sector_etf_return_60d` | Sector ETF 60-day return |
| `sector_etf_vol_20d` | Sector ETF 20-day volatility |
| `excess_vs_sector_5d` | Stock return minus sector return (5d) |
| `excess_vs_sector_20d` | Stock return minus sector return (20d) |
| `sector_momentum_rank` | Sector rank by recent momentum |

</details>

<details>
<summary><strong>Sentiment Features (6)</strong></summary>

| Feature | Description |
|---------|-------------|
| `sent_mean_1d` | Yesterday's composite sentiment |
| `sent_mean_7d` | 7-day rolling average sentiment |
| `sent_mean_30d` | 30-day rolling average sentiment |
| `sent_momentum` | Sentiment regime change (7d − 30d) |
| `news_count_7d` | Rolling 7-day article count |
| `news_spike_1d` | Unusual news activity detector |

</details>

<details>
<summary><strong>Macro & Insider Features (5+)</strong></summary>

| Feature | Description |
|---------|-------------|
| `macro_spread_2y10y` | Treasury yield curve spread (recession indicator) |
| `macro_fed_funds` | Federal funds rate |
| `insider_net_value_30d` | Net insider trading value (30-day) |
| `insider_buy_ratio_30d` | Insider buy/sell ratio |
| `insider_cluster_buying` | Multiple insiders buying simultaneously |

</details>

<details>
<summary><strong>Earnings Features (4) — v2.0</strong></summary>

| Feature | Description |
|---------|-------------|
| `earnings_surprise` | EPS actual − EPS estimated (latest earnings) |
| `earnings_beat` | +1 if beat, −1 if missed, 0 if met |
| `earnings_recency` | 1/(days since last earnings + 1) decay weight |
| `earnings_surprise_pct` | Surprise normalized by estimate magnitude |

</details>

<details>
<summary><strong>Fundamental Features (5) — v2.0</strong></summary>

| Feature | Description |
|---------|-------------|
| `fund_pe_ratio` | Price-to-Earnings ratio (TTM) |
| `fund_pb_ratio` | Price-to-Book ratio |
| `fund_dividend_yield` | Indicated annual dividend yield |
| `fund_roe` | Return on Equity (TTM) |
| `fund_beta` | Stock beta vs market |

</details>

<details>
<summary><strong>Short Interest Features (3) — v2.0</strong></summary>

| Feature | Description |
|---------|-------------|
| `si_short_float_pct` | Short interest as % of float |
| `si_days_to_cover` | Short interest / avg daily volume |
| `si_available` | 1 if short interest data exists, 0 otherwise |

</details>

### Confidence Scoring

A separate LightGBM binary classifier predicts **P(return > 0)** — the probability the stock will go up:

| Confidence Level | Range | Interpretation |
|-----------------|-------|----------------|
| **High** | > 65% | Strong signal — multiple features agree |
| **Medium** | 55–65% | Moderate signal — some conflicting indicators |
| **Low** | 50–55% | Near coin-flip — model is uncertain |
| **Contrarian** | < 50% | Model leans bearish |

---

## Sentiment Analysis Engine

Sentiment is collected from **10+ sources**, scored with three NLP models, and blended into a single composite score per stock per day.

### NLP Models

| Model | Type | Strength |
|-------|------|----------|
| **FinBERT** | Transformer (fine-tuned BERT) | Financial domain–specific sentiment |
| **RoBERTa** | Transformer | General-purpose sentiment robustness |
| **VADER** | Rule-based lexicon | Fast, reliable baseline for headlines |

### Source Blend Weights

| Priority | Source | Blend Weight | Rate Limited? |
|----------|--------|-------------|---------------|
| 1 | RSS News (Yahoo + Seeking Alpha) | **22%** | No (free RSS) |
| 2 | Marketaux | 15% | Yes (95/day budget) |
| 3 | SEC Filings | 10% | No (free scrape) |
| 4 | Reddit (PRAW) | 10% | Yes (90/min) |
| 5 | Finnhub (insider + recommendations) | 10% + 10% | Yes (55/min) |
| 6 | FMP (analyst estimates, ratings) | 8% | Yes (3/sec) |
| 7 | Finviz | 5% | No (free scrape) |
| 8 | Seeking Alpha Comments | 5% | No (Playwright) |

> **Resilience guarantee**: Even if all rate-limited APIs are exhausted, **42% of blend weight** comes from free, unlimited sources that never fail.

### What Gemini Reads for Explanations (11 Data Sources)

The AI explanation prompt is **stock-specific** — tailored with company name, sector, and industry context:

1. LightGBM predictions (all 3 horizons with confidence, alpha vs SPY, price ranges)
2. Technical analysis (RSI, MACD, Bollinger Bands, SMAs, EMAs, volume ratio, 52-week range)
3. News headlines (Finviz, RSS, Reddit, Marketaux — aggregated from MongoDB)
4. Sentiment scores (blended + per-source breakdown)
5. SHAP feature drivers (human-readable names with contribution values)
6. Macro economic context (Fed rate, CPI, unemployment, yield curve, GDP)
7. Insider trading activity (buy/sell ratio, recent transactions with names/prices)
8. Short interest data (short float %, days to cover)
9. Finnhub basic financials (P/E, P/B, ROE, dividend yield, market cap, beta)
10. FMP earnings data (EPS actual vs estimated, earnings surprise)
11. FMP analyst ratings and price targets

---

## Data Sources & APIs

| Provider | Purpose | Data Stored? |
|----------|---------|-------------|
| **Finnhub** | Quotes, profiles, search, news, insider trades, WebSocket prices | Insider trades + financials in MongoDB; prices in-memory |
| **Yahoo Finance** | Historical OHLCV (via yfinance), RSS news | OHLCV in MongoDB; RSS returned to frontend only |
| **FRED** | 13 macro indicators (GDP, CPI, Fed rate, Treasury yields, unemployment) | MongoDB `macro_data_raw` |
| **FMP** | Income statements, product segmentation, earnings, analyst estimates, ratings, price targets | MongoDB + Redis (Sankey data cached 14 days) |
| **Marketaux** | Financial news articles | Sentiment scores in MongoDB |
| **Reddit** | Social sentiment via PRAW (r/wallstreetbets, r/stocks, r/investing) | Sentiment scores in MongoDB |
| **SEC / Kaleidoscope** | SEC filing analysis (10-K, 10-Q, 8-K) | Filing sentiment in MongoDB |
| **Seeking Alpha** | Comment sentiment (Playwright scraping) | Sentiment scores in MongoDB |
| **Finviz** | News headlines + short interest (fallback) | Sentiment scores in MongoDB |
| **Nasdaq** | Short interest data (settlement-cycle updates) | Short interest in MongoDB |
| **Groq (Llama 3.1)** | Primary AI explanation generation | Explanations in MongoDB |
| **Google Gemini** | Fallback AI explanation generation (auto-fallback: pro → flash → flash-lite) | Explanations in MongoDB |
| **TradingView** | Chart widgets, heatmaps, economic calendar | Not stored (client-side embed) |
| **Calendarific** | US market holidays for market-status detection | Redis (1-year TTL) |

---

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | Next.js 15, React 18, TypeScript 5, Tailwind CSS, Shadcn/UI, Framer Motion, TradingView Widgets, Apache ECharts (Sankey), Recharts |
| **Backend** | Node.js 18+, Express.js, MongoDB Atlas, Redis |
| **ML / AI** | Python 3.11+, FastAPI, LightGBM, SHAP (TreeSHAP), Groq (Llama 3.1-8b), Google Gemini 2.5, yfinance, FinBERT, RoBERTa, VADER |
| **Data** | Finnhub, Yahoo Finance, FRED, FMP, Marketaux, Reddit (PRAW), Seeking Alpha (Playwright), SEC/Kaleidoscope, Nasdaq |
| **Infrastructure** | Vercel (frontend), Render (backends), GitHub Actions (daily pipeline), MongoDB Atlas, Redis Cloud |
| **Monitoring** | Drift monitor (PSI), stored prediction evaluation, quality gates, Vercel Analytics |

---

## Frontend Architecture

Built with **Next.js 15 App Router** — fully server-side rendered, SEO-friendly, with automatic code splitting per route.

### Pages

| Route | Purpose |
|-------|---------|
| `/` | Landing page with market overview |
| `/stocks/[symbol]` | Stock detail — predictions, AI explanation, TradingView chart, news, technical indicators |
| `/predictions` | Overview of all 100 stocks with stored predictions |
| `/news` | Unified multi-source news feed |
| `/sankey` | Interactive income-statement Sankey flow visualization |
| `/watchlist` | User watchlist with real-time price updates |
| `/fundamentals` | Financial fundamentals (Jika.io embeds) |
| `/how-it-works` | Educational guide on ML methodology |
| `/methodology` | Technical breakdown of the prediction pipeline |
| `/disclaimer` | Legal compliance page |

### Key Components

| Component | Purpose |
|-----------|---------|
| `AIExplanationWidget` | Displays Gemini-generated structured explanation with outlook, key drivers, and bottom line |
| `EnhancedQuickPredictionWidget` | Prediction lookup showing price targets, confidence, and trade signals |
| `SankeyChart` | Apache ECharts Sankey diagram — revenue sources → expenses → profit paths |
| `MarketSentimentBanner` | Fear & Greed Index with visual gauge |
| `TechnicalIndicators` | RSI, MACD, SMA, EMA gauges with historical context |
| `NotificationWidget` | Real-time notification bell (polls every 30s) |
| `SearchWidget` | Debounced stock search with autocomplete |
| `WebSocketProvider` | React Context for real-time Finnhub price updates |

---

## Pipeline Hardening & Reliability

The pipeline is engineered to run reliably on a **free GitHub Actions runner** (7 GB RAM, 2 CPUs) without crashing, timing out, or getting rate-limited.

### Rate Limiting

Every external API has an enforced client-side rate limiter using a sliding-window token-bucket algorithm:

| API | Plan Limit | Enforced Throttle |
|-----|-----------|-------------------|
| Finnhub | 60/min | 55/min + 25/sec burst |
| FMP | 250/day | 3/sec, 4 endpoints only |
| Marketaux | 100/day | 95/day hard budget (top-50 tickers) |
| Reddit | 100 QPM | 90/min, max 3 subreddits/ticker |
| FRED | ~120/min | 100/min (defensive) |

### Retry Logic

All external API calls use retry-with-exponential-backoff-and-jitter:
- **Finnhub**: 3 attempts, honors `Retry-After` on 429
- **FRED**: 2 attempts, re-raises config errors immediately
- **MongoDB**: 3 attempts with driver-level + application-level retry wrappers

### Safety Mechanisms

| Mechanism | Purpose |
|-----------|---------|
| **Batched predictions** (10 × 10) | Prevents CPU thermal throttling + API rate limit pressure |
| **10s cool-down between batches** | Lets APIs reset rate windows |
| **Canary verification** | Checks 8 benchmark tickers (AAPL, AMZN, JPM…) for freshness after predictions |
| **Quality gate** | Fails pipeline if <80% tickers predicted or >20% data failures |
| **Feature-name enforcement** | Skips predictions if >50% feature columns mismatch (prevents garbage output) |
| **NaN preservation** | Sentiment/insider missing values preserved as NaN instead of zero-filled |
| **MongoDB connection hardening** | Relaxed timeouts, connection pooling (10–50), retry writes/reads enabled |

---

## Model Validation & Backtesting

### Daily Out-of-Sample Evaluation

Every pipeline run compares the **last 60 days** of stored predictions against actual market outcomes. These predictions were generated _before_ outcomes were known — truly out-of-sample.

### Walk-Forward Validation

Training uses a rolling split with purge gaps to prevent data leakage:

```
|────── Train (70%) ──────|─purge─|──── Validate (15%) ────|─purge─|──── Holdout (15%) ────|
                           5 days                            5 days
```

### Tracked Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Sharpe Ratio** | Return per unit of risk (annualized) |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Win Rate** | % of profitable trades |
| **Directional Accuracy** | % of correct up/down predictions |
| **Rank Correlation** | Spearman correlation: predicted vs actual returns |
| **Brier Score** | Probability calibration quality |
| **Drift Monitor (PSI)** | Detects prediction distribution shifts over time |

---

## Full Documentation

**[DOCUMENTATION.md](DOCUMENTATION.md)** — 2,500+ lines covering architecture, data flow, every API endpoint, database schemas, field-level data mappings, MongoDB document structures, file-by-file breakdown, pipeline details, and more.

---

## License

This project is licensed under the [**GNU Affero General Public License v3.0 (AGPL-3.0)**](LICENSE).

If you deploy this software as a network service, you must make the complete source code available to users of that service under the same license.

---

<p align="center">
  Built by <a href="https://github.com/Yogesh-VG0">Yogesh Vadivel</a>
</p>
