# StockPredict AI — Complete Project Documentation

> **Last updated**: 2026-02-21

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [The Three "Models" — A Crucial Distinction](#2-the-three-models--a-crucial-distinction)
3. [Architecture Overview](#3-architecture-overview)
4. [What Happens Every Day — A Narrative Timeline](#4-what-happens-every-day--a-narrative-timeline)
5. [Where Does Each Piece Run?](#5-where-does-each-piece-run)
6. [What Users See vs What the System Does](#6-what-users-see-vs-what-the-system-does)
7. [Technology Stack](#7-technology-stack)
8. [APIs & External Services](#8-apis--external-services)
9. [Data Pipeline](#9-data-pipeline)
10. [Machine Learning Pipeline](#10-machine-learning-pipeline)
11. [Backtesting & Model Validation](#11-backtesting--model-validation)
12. [Prediction Confidence & Trustworthiness](#12-prediction-confidence--trustworthiness)
13. [GitHub Actions — Daily Predictions Workflow](#13-github-actions--daily-predictions-workflow)
14. [Frontend Architecture](#14-frontend-architecture)
15. [Backend Architecture (Node.js)](#15-backend-architecture-nodejs)
16. [ML Backend Architecture (Python/FastAPI)](#16-ml-backend-architecture-pythonfastapi)
17. [Database Schemas](#17-database-schemas)
18. [Redis Caching](#18-redis-caching)
19. [WebSocket / Real-Time Data](#19-websocket--real-time-data)
20. [Unused / Deprecated Code](#20-unused--deprecated-code)
21. [Model Reliability & Performance Tracking](#21-model-reliability--performance-tracking)
22. [Complete File-by-File Breakdown](#22-complete-file-by-file-breakdown)
23. [How to Run Locally](#23-how-to-run-locally)
24. [User Action Flows](#24-user-action-flows)
25. [Risks, Bugs & Improvements](#25-risks-bugs--improvements)
26. [Pipeline Hardening & Reliability](#26-pipeline-hardening--reliability)
27. [API Rate Limit Compliance & Data Priority](#27-api-rate-limit-compliance--data-priority)
28. [Technical Deep Dive: The AI Machine](#28-technical-deep-dive-the-ai-machine)
29. [Stateless-to-Stateful Bridge: The Role of MongoDB](#29-stateless-to-stateful-bridge-the-role-of-mongodb)

---

## 1. Project Overview

StockPredict AI is a **full-stack stock market prediction and analysis platform** that:

- **Predicts** stock prices for 100 S&P 100 stocks across 3 time horizons (1 day, 1 week, 1 month) using machine learning (LightGBM)
- **Analyzes** market sentiment from 10+ news/social sources using NLP models (FinBERT, RoBERTa, VADER)
- **Explains** predictions in plain English using Google Gemini AI
- **Displays** real-time stock prices, TradingView charts, technical indicators, and news
- **Manages** user watchlists with real-time price alerts

The system runs a **daily automated pipeline** via GitHub Actions that fetches fresh data, trains models, generates predictions, and stores everything in MongoDB.

---

## 2. The Three "Models" — A Crucial Distinction

> **This is the single most important concept to understand before reading further.**

This project uses the word "model" for three very different things. Beginners often confuse them.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   PREDICTION is math.   EXPLANATION is language.   SENTIMENT is input.  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1. The Predictor — LightGBM (the only model that predicts prices)

- **What it is**: A gradient-boosted decision tree (a math/statistics algorithm)
- **Input**: 42+ numeric features (price history, sentiment scores, macro data, etc.)
- **Output**: A number — predicted log-return (e.g., +0.002 means ~+0.2%)
- **Where it runs**: Python ML backend, during the daily GitHub Actions pipeline
- **File**: `ml_backend/models/predictor.py`

This is the **only** component that actually predicts stock prices. Everything else supports it.

### 2. The Explainer — SHAP + Google Gemini (translates numbers into English)

- **What SHAP does**: Takes the LightGBM prediction and breaks it into "which feature pushed the price up, which pushed it down" — purely mathematical decomposition
- **What Gemini does**: Takes SHAP results + sentiment + news + technicals and writes a human-readable paragraph explaining the prediction
- **Output**: Plain-English text like "Apple's short-term outlook is weak because..."
- **Where it runs**: Python ML backend (SHAP) + Google Gemini API (text generation)
- **Files**: `ml_backend/explain/shap_analysis.py`, `ml_backend/scripts/generate_explanations.py`

Gemini does NOT predict prices. It reads the LightGBM prediction and explains it in words.

### 3. The Sentiment Scorers — FinBERT, RoBERTa, VADER (produce input features)

- **What they do**: Read news headlines, Reddit posts, SEC filings, and produce a sentiment score (-1 to +1)
- **Output**: A number — composite sentiment score (e.g., 0.35 = slightly positive)
- **Where it runs**: Python ML backend, during sentiment collection
- **File**: `ml_backend/data/sentiment.py`

These models do NOT predict prices. They produce **features** (input data) that the LightGBM predictor uses.

### How They Connect

```
News/Reddit/SEC  →  FinBERT/RoBERTa/VADER  →  sentiment score (NUMBER)
                                                      │
Price history    →  Feature engineering     →  42+ features (NUMBERS)
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

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        USER (Browser)                        │
│  Next.js Shell  +  React Router (navigation)  +  TradingView │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP / Polling (every 5s)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   NODE.JS BACKEND (Express)                   │
│  Port 5000  |  Routes → Controllers → Services               │
│  News Aggregation | Watchlist | Market Status | Notifications│
│  Proxies ML Backend for predictions/explanations             │
└─────────┬──────────────────────────────────────┬────────────┘
          │ HTTP                                  │ WebSocket
          ▼                                       ▼
┌──────────────────────┐             ┌──────────────────────┐
│  ML BACKEND (FastAPI) │             │  Finnhub WebSocket    │
│  Port 8000            │             │  wss://ws.finnhub.io  │
│  Predictions          │             │  Real-time trades     │
│  Sentiment Analysis   │             └──────────────────────┘
│  Model Training       │
│  SHAP Explanations    │
│  Gemini AI            │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────┐
│                        MONGODB ATLAS                          │
│  Database: stock_predictor                                    │
│  Collections: historical_data, sentiment, stock_predictions,  │
│  prediction_explanations, feature_importance, insider_        │
│  transactions, macro_data_raw, notifications, sec_filings_raw│
└──────────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────┐
│                    REDIS (Optional)                            │
│  Prediction caching (60s TTL)                                 │
│  Rate limiting (sliding window)                               │
│  Holiday caching (1 year TTL)                                 │
└──────────────────────────────────────────────────────────────┘
```

### Request Flow (simplified):

```
User opens AAPL stock page
  → Frontend fetches /api/stock/AAPL (Node backend)
  → Node backend calls Finnhub API for stock profile
  → Frontend fetches /api/stock/AAPL/explanation (Node backend)
  → Node backend reads from MongoDB prediction_explanations collection
  → Frontend fetches /api/stock/AAPL/predictions (Node backend)
  → Node backend reads from MongoDB stock_predictions collection
  → Frontend polls /api/watchlist/updates/realtime every 5s
  → Node backend calls Finnhub WebSocket service for live prices
```

---

## 4. What Happens Every Day — A Narrative Timeline

Think of this as a story that plays out every weeknight after the US stock market closes.

```
WEEKNIGHT, ~6:15 PM ET (10:15 PM UTC)
GitHub Actions wakes up and starts the daily pipeline.

╔════════════════════════════════════════════════════════════════╗
║  CHAPTER 1: "Gather the News" (~5 min)                        ║
║                                                                ║
║  The sentiment cron fires up. It goes through each of the      ║
║  100 stocks and asks every news source: "What are people       ║
║  saying about AAPL today?" It asks Finviz, Yahoo RSS, Reddit,  ║
║  Marketaux, Finnhub, FMP, SEC filings, and Seeking Alpha.      ║
║                                                                ║
║  For each source, it runs FinBERT/RoBERTa/VADER to score the   ║
║  sentiment (positive, negative, neutral). It blends them into   ║
║  one composite score and saves it to MongoDB.                   ║
║                                                                ║
║  If this step fails, the pipeline continues — we can use        ║
║  yesterday's sentiment as a fallback.                           ║
╠════════════════════════════════════════════════════════════════╣
║  CHAPTER 2: "Crunch the Numbers" (~15 min)                    ║
║                                                                ║
║  The training pipeline starts. It fetches the latest OHLCV      ║
║  prices from Yahoo Finance, insider trades from Finnhub,        ║
║  macro indicators from FRED, and everything else from MongoDB.  ║
║                                                                ║
║  It engineers 42+ features: price returns, volatility, RSI,     ║
║  sentiment scores, insider activity, yield curve spreads...     ║
║                                                                ║
║  Then it trains ONE LightGBM model per horizon (1-day, 7-day,   ║
║  30-day) across all 100 tickers. The models are saved to disk.  ║
║                                                                ║
║  If this step fails, THE ENTIRE JOB FAILS. No predictions       ║
║  without trained models.                                        ║
╠════════════════════════════════════════════════════════════════╣
║  CHAPTER 3: "Make Predictions" (~20 min)                      ║
║                                                                ║
║  The pipeline loads the freshly trained models and runs them     ║
║  on each of the 100 stocks, in 10 batches of 10.               ║
║                                                                ║
║  For each stock: predicted return, predicted price, confidence,  ║
║  trade recommendation. Stored in MongoDB.                       ║
║                                                                ║
║  A verification step checks 8 "canary" tickers (AAPL, AMZN,    ║
║  JPM, etc.) to make sure predictions are fresh. If any are      ║
║  missing or stale, the job fails.                               ║
╠════════════════════════════════════════════════════════════════╣
║  CHAPTER 4: "Explain Why" (~15 min)                           ║
║                                                                ║
║  SHAP analysis runs: for each stock, it decomposes the          ║
║  LightGBM prediction into "which features pushed up, which      ║
║  pushed down." Stored in MongoDB.                               ║
║                                                                ║
║  Then Gemini AI reads ALL the data — predictions, SHAP,         ║
║  sentiment, news headlines, macro indicators, insider trades,   ║
║  short interest — and writes a plain-English explanation.       ║
║  Stored in MongoDB.                                             ║
╠════════════════════════════════════════════════════════════════╣
║  CHAPTER 5: "Check Our Work" (~5 min)                         ║
║                                                                ║
║  The evaluation script compares the last 60 days of predictions ║
║  to actual stock prices. How accurate were we?                  ║
║                                                                ║
║  The drift monitor checks: are our predictions drifting?        ║
║  Are features losing power? Is the model getting stale?         ║
║                                                                ║
║  Reports saved as artifacts. If things look bad, it logs a      ║
║  warning (but doesn't fail the job).                            ║
╠════════════════════════════════════════════════════════════════╣
║  CHAPTER 6: "Quality Gate" (~instant)                          ║
║                                                                ║
║  The pipeline checks its own health: were ≥80% of tickers      ║
║  predicted successfully? Did ≤20% of data fetches fail?        ║
║  If either threshold is breached, the pipeline FAILS with a    ║
║  clear error message — no silent degradation.                  ║
║  Thresholds are configurable via QG_MIN_PREDICTION_RATE and    ║
║  QG_MAX_DATA_FAILURE_RATE environment variables.               ║
╠════════════════════════════════════════════════════════════════╣
║  DONE. All artifacts uploaded. Pipeline complete.               ║
║  Next morning, users see fresh predictions on the website.      ║
╚════════════════════════════════════════════════════════════════╝
```

---

## 5. Where Does Each Piece Run?

| Part | Runs On | Language | Reads From | Writes To |
|------|---------|----------|------------|-----------|
| **Frontend (UI)** | User's browser | TypeScript/React | Node backend API | Nothing (read-only) |
| **TradingView widgets** | User's browser (iframe) | JavaScript (external) | TradingView CDN | Nothing |
| **Node.js backend** | Cloud server / Render | JavaScript | Finnhub API, MongoDB, ML backend | MongoDB (notifications), in-memory (watchlists, cache) |
| **ML backend (API)** | Cloud server / Render | Python | MongoDB, Redis | MongoDB (predictions), Redis (cache) |
| **ML pipeline (training)** | GitHub Actions runner | Python | Yahoo Finance, FRED, Finnhub, FMP, Reddit, etc. | MongoDB (all collections) |
| **Sentiment cron** | GitHub Actions runner | Python | 10+ news/social APIs | MongoDB (sentiment, insider_transactions) |
| **SHAP analysis** | GitHub Actions runner | Python | MongoDB (predictions, historical_data) | MongoDB (feature_importance) |
| **Gemini explanations** | GitHub Actions runner | Python | MongoDB (all data) + Google Gemini API | MongoDB (prediction_explanations) |
| **Evaluation/Drift** | GitHub Actions runner | Python | MongoDB (predictions, historical_data) | GitHub Actions artifacts (.txt reports) |

---

## 6. What Users See vs What the System Does

| What the User Sees | Frontend Component | Backend Endpoint | Reads From (DB/API) | External API Called |
|--------------------|--------------------|-----------------|---------------------|-------------------|
| Stock price updating live | `Sidebar`, `WebSocketProvider` | `GET /api/watchlist/updates/realtime` | Finnhub WebSocket cache | Finnhub WebSocket |
| Stock search results | `SearchWidget` | `GET /api/stock/search/:query` | Finnhub search API | `finnhub.io/api/v1/search` |
| Stock detail page (profile) | `StockDetail` page | `GET /api/stock/:symbol` | Finnhub profile + quote API | `finnhub.io/api/v1/stock/profile2`, `/quote` |
| AI explanation card | `AIExplanationWidget` | `GET /api/stock/:symbol/explanation` | MongoDB `prediction_explanations` | None (pre-generated) |
| Prediction numbers | `EnhancedQuickPredictionWidget` | `GET /api/stock/:symbol/predictions` | MongoDB `stock_predictions` | None (pre-generated) |
| Technical indicators (RSI, MACD) | `TechnicalIndicators` | `GET /api/stock/:symbol/indicators` | financialdata.net API (or calculated) | `financialdata.net/api/v1/stock-prices` |
| News feed | `NewsPage` | `GET /api/news/unified` | Marketaux, Finnhub, TickerTick APIs | Multiple (aggregated) |
| RSS news on stock page | `StockDetail` page | `GET /api/news/rss?symbol=X` | Yahoo Finance + Seeking Alpha RSS | RSS feed URLs |
| Fear & Greed gauge | `MarketSentimentBanner` | `GET /api/market/sentiment` | RapidAPI Fear & Greed | `fear-and-greed-index.p.rapidapi.com` |
| Market open/close status | `Navbar` | `GET /api/market/status` | Calendarific holidays | `calendarific.com/api/v2/holidays` |
| Trading hours bar | `TradingHoursBar` | None (client-side) | None (calculates from clock) | None |
| Watchlist + alerts | `WatchlistPage` | `GET/POST/DELETE /api/watchlist/*` | In-memory Map (Node) | Finnhub WebSocket |
| Notification bell | `NotificationWidget` | `GET /api/notifications` | MongoDB `notifications` | None |
| TradingView chart | `TradingViewAdvancedChart` | None (direct embed) | TradingView CDN | TradingView widget API |
| Fundamentals page | `FundamentalsPage` | None (iframe embed) | Jika.io CDN | Jika.io widgets |
| Predictions overview | `PredictionsPage` | `GET /api/stock/batch/available` | MongoDB `prediction_explanations` | None |

---

## 7. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Next.js 14 + React 18 | UI framework |
| **Routing** | React Router DOM (inside Next.js shell) | Page navigation (see [Why We Did This](#why-we-use-react-router-inside-nextjs)) |
| **Styling** | Tailwind CSS + Shadcn/UI | Component styling |
| **Charts** | TradingView Widgets (external) | Stock charts |
| **Animations** | Framer Motion | UI animations |
| **Node Backend** | Express.js | API gateway |
| **ML Backend** | FastAPI (Python) | ML serving |
| **Predictor** | LightGBM | Stock price prediction (the ONLY price predictor) |
| **Sentiment Scorers** | FinBERT, RoBERTa, VADER | Sentiment analysis (INPUT features, not predictors) |
| **Explainer** | SHAP + Google Gemini 2.5 (Pro/Flash auto-fallback) | SHAP decomposes predictions; Gemini writes stock-specific English text |
| **Database** | MongoDB Atlas | Primary data store |
| **Cache** | Redis (optional) | Caching & rate limiting |
| **Real-time** | Finnhub WebSocket | Live stock prices |
| **CI/CD** | GitHub Actions | Daily pipeline automation |
| **Analytics** | Vercel Analytics/Speed Insights | Performance monitoring |

---

## 8. APIs & External Services

### External API Inventory

| Provider | Purpose | Where It Runs | Endpoints | Key Env Var | Rate Limit Handling | What Gets Stored |
|----------|---------|--------------|-----------|-------------|-------------------|-----------------|
| **Finnhub** | Stock data, news, insider trades, WebSocket prices | Node backend + ML backend | `/quote`, `/stock/profile2`, `/search`, `/news`, `/stock/insider-transactions`, `/recommendation`, `/stock/insider-sentiment`, `/stock/metric`, `/stock/peers`, WebSocket | `FINNHUB_API_KEY` | 60 calls/min free tier; `_finnhub_get_with_retry()` — 3 attempts with exponential backoff + jitter on 429/5xx/timeout; honors `Retry-After` header on 429; 1s between HTTP requests | `insider_transactions`, `finnhub_basic_financials`, `finnhub_company_peers`, `finnhub_insider_sentiment` (MongoDB); prices in-memory (60s) |
| **Yahoo Finance** | Historical OHLCV data, RSS news | ML backend (yfinance) + Node backend (RSS) | yfinance library, RSS feeds | None (free) | 1.5-2.5s delay between requests | `historical_data` (MongoDB); RSS returned to frontend only |
| **FRED** | Macro economic indicators (GDP, CPI, unemployment, etc.) | ML backend | fredapi library (13 indicators) | `FRED_API_KEY` | `fetch_fred_series()` — 2 attempts with exponential backoff + jitter on transient failures; explicit warning when `FRED_API_KEY` not set | `macro_data_raw` / `macro_data` (MongoDB) |
| **FMP** | Earnings, dividends, analyst estimates, ratings, SEC filings (backup) | ML backend | `/dividend/`, `/earnings/`, `/analyst-estimates/`, `/ratings-snapshot/`, `/price-target-summary/`, `/price-target-consensus/`, `/grades-consensus/`, `/sec_filings/` | `FMP_API_KEY` | Cooldown tracking, 403/429 handling | `alpha_vantage_data` (MongoDB, multiple endpoints) |
| **Marketaux** | Financial news articles | Node backend + ML backend | `/v1/news/all` | `MARKETAUX_API_KEY` | Per-plan limits | ML: within `sentiment` collection; Node: returned to frontend only |
| **TickerTick** | News aggregation | Node backend | `/feed` | None (free) | 10 req/min (enforced in code) | Not stored (returned to frontend only) |
| **RapidAPI** | Fear & Greed Index | Node backend | `/v1/fgi` | `RAPIDAPI_KEY` | Per-plan limits | Not stored (returned to frontend only) |
| **Calendarific** | US market holidays | Node backend | `/v2/holidays` | `CALENDARIFIC_API_KEY` | Per-plan limits | Redis `us_holidays_{year}` (1yr TTL) + in-memory |
| **financialdata.net** | Stock prices for technical indicators | Node backend | `/api/v1/stock-prices` | `FINANCIALDATA_API_KEY` | Free: 300 req/day; 5 calls/min enforced | In-memory cache only (24hr indicators, 12hr prices) |
| **Google Gemini** | AI-powered stock-specific explanation generation | ML backend (GitHub Actions) | `generate_content` API (auto-fallback: pro→flash→flash-lite) | `GOOGLE_API_KEY` | Pro: 15 RPM, 1.5K RPD; Flash: 5 RPM, 20 RPD | `prediction_explanations` (MongoDB) |
| **Nasdaq** | Short interest data | ML backend | `/api/quote/{ticker}/short-interest` | None (public) | Not explicitly limited | Within `sentiment` collection (MongoDB) |
| **Finviz** | Short interest fallback + news headlines | ML backend | Web scraping | None | Sequential with delays | Within `sentiment` collection (MongoDB) |
| **Seeking Alpha** | Comment sentiment | ML backend | Web scraping (Playwright) | None | Sequential with locks; proxy rotation | `seeking_alpha_comments`, `seeking_alpha_sentiment` (MongoDB) |
| **SEC/Kaleidoscope** | SEC filing analysis | ML backend | `api.kscope.io/v2/sec/search/` + FMP fallback | `KALEIDOSCOPE_API_KEY` | 30 calls/60s (enforced) | `sec_filings`, `sec_filings_raw` (MongoDB) |
| **Reddit** | Social sentiment (PRAW) | ML backend | Reddit API via PRAW library | `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` | Handled by PRAW library | Within `sentiment` collection (MongoDB) |
| **NewsAPI** | Sector/industry news | Node backend | `/v2/everything` | `NEWSAPI_KEY` | Free: 100 req/day | Not stored (returned to frontend only) |
| **TradingView** | Chart widgets, heatmaps, economic calendar | Frontend (browser) | Widget embed scripts | None (free widget tier) | N/A (client-side) | Not stored |
| **Jika.io** | Financial fundamentals embeds | Frontend (browser) | iframe embeds | None | N/A | Not stored |

### Where API Keys Come From

All API keys are stored as **GitHub Secrets** and passed to the workflow as environment variables. For local development, they are stored in `.env` files.

**Required secrets for GitHub Actions:**
- `MONGODB_URI` — MongoDB Atlas connection string
- `FINNHUB_API_KEY` — Finnhub API key
- `FRED_API_KEY` — FRED API key
- `FMP_API_KEY` — Financial Modeling Prep key
- `GOOGLE_API_KEY` — Google Gemini API key
- `MARKETAUX_API_KEY` — Marketaux API key
- `ALPHAVANTAGE_API_KEY` — Alpha Vantage key (optional)
- `REDDIT_CLIENT_ID` — Reddit app client ID
- `REDDIT_CLIENT_SECRET` — Reddit app client secret

### Field-Level Data Mapping (Source -> Fields -> Where Stored -> Where Used)

#### Finnhub — Quote

- **Endpoint**: `GET https://finnhub.io/api/v1/quote?symbol={ticker}&token={key}`
- **Runs in**: Node backend (`websocketService.js`, `stockController.js`) + ML backend (`sentiment.py`)
- **Fields used**: `c` (current price), `d` (change), `dp` (change percent), `h` (day high), `l` (day low), `o` (open), `pc` (previous close)
- **Stored**: Node: in-memory `priceCache` Map (60s TTL). ML: `alpha_vantage_data` collection (endpoint: `quote`)
- **Used by**: Sidebar live prices, stock detail page, watchlist, prediction context

#### Finnhub — Company Profile

- **Endpoint**: `GET https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={key}`
- **Runs in**: Node backend (`stockController.js`)
- **Fields used**: `name`, `finnhubIndustry` (mapped to sector + industry), `exchange`, `marketCapitalization`, `country` (headquarters), `weburl` (website)
- **Stored**: NOT stored — returned to frontend only
- **Used by**: Stock detail page header, company info section

#### Finnhub — Symbol Search

- **Endpoint**: `GET https://finnhub.io/api/v1/search?q={query}&token={key}`
- **Runs in**: Node backend (`stockController.js`)
- **Fields used**: `result[].symbol`, `result[].description` (company name), `result[].type` (filtered for "Common Stock" or "ETP")
- **Stored**: NOT stored — returned to frontend only
- **Used by**: SearchWidget (autocomplete dropdown)

#### Finnhub — WebSocket Trades

- **Endpoint**: `wss://ws.finnhub.io?token={key}`
- **Runs in**: Node backend (`websocketService.js`)
- **Fields used**: `data[].s` (symbol), `data[].p` (price), `data[].v` (volume), `data[].t` (timestamp), `data[].c` (conditions)
- **Stored**: In-memory `priceCache` Map (60-120s TTL) + `volumeData` Map
- **Used by**: Real-time price updates across all pages, watchlist, sidebar

#### Finnhub — Company News

- **Endpoint**: `GET https://finnhub.io/api/v1/company-news?symbol={ticker}&from={date}&to={date}&token={key}`
- **Runs in**: Node backend (`finnhubNewsService.js`)
- **Fields used**: `id` (uuid), `headline` (title), `url`, `datetime` (Unix timestamp → ISO), `source`, `summary` (snippet), `related` (tickers, comma-split), `category` (industry), `image`
- **Stored**: NOT stored — returned to frontend only
- **Used by**: News page (unified feed)

#### Finnhub — General News

- **Endpoint**: `GET https://finnhub.io/api/v1/news?category={category}&minId={minId}&token={key}`
- **Runs in**: Node backend (`finnhubNewsService.js`)
- **Fields used**: Same as Company News
- **Stored**: NOT stored — returned to frontend only
- **Used by**: News page (general market news)

#### Finnhub — Insider Transactions

- **Endpoint**: `finnhub_client.stock_insider_transactions(ticker, from_date, to_date)` (Python client)
- **Runs in**: ML backend (`sentiment.py`)
- **Fields used**: `data[].filingDate`, `data[].transactionDate`, `data[].name`, `data[].share`, `data[].change`, `data[].transactionCode` (P=purchase, S=sale, M=multiple, A=award, D=disposition), `data[].transactionPrice`
- **Stored**: MongoDB `insider_transactions` collection
- **Used by**: `insider_features.py` (11 insider ML features), `generate_explanations.py` (Gemini prompt context)

#### Finnhub — Recommendation Trends

- **Endpoint**: `finnhub_client.recommendation_trends(ticker)` (Python client)
- **Runs in**: ML backend (`sentiment.py`)
- **Fields used**: `strongBuy`, `buy`, `hold`, `sell`, `strongSell`, `period`
- **Stored**: Within MongoDB `sentiment` collection (as part of blended sentiment)
- **Used by**: Sentiment scoring (weighted recommendation score)

#### Finnhub — Basic Financials

- **Endpoint**: `GET https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={key}`
- **Runs in**: ML backend (`sentiment.py`)
- **Fields used**: Full response object
- **Stored**: MongoDB `finnhub_basic_financials` collection
- **Used by**: Financial metrics analysis within sentiment pipeline

#### Finnhub — Company Peers

- **Endpoint**: `GET https://finnhub.io/api/v1/stock/peers?symbol={ticker}&token={key}`
- **Runs in**: ML backend (`sentiment.py`)
- **Fields used**: Array of peer ticker symbols
- **Stored**: MongoDB `finnhub_company_peers` collection
- **Used by**: Peer comparison analysis

#### Finnhub — Insider Sentiment (MSPR)

- **Endpoint**: `GET https://finnhub.io/api/v1/stock/insider-sentiment?symbol={ticker}&from={date}&to={date}&token={key}`
- **Runs in**: ML backend (`sentiment.py`)
- **Fields used**: MSPR sentiment data
- **Stored**: MongoDB `finnhub_insider_sentiment` collection
- **Used by**: Insider sentiment analysis

#### Yahoo Finance — Historical Data (yfinance)

- **Endpoint**: `yf.Ticker(ticker).history()` / `yf.download()` (Python library)
- **Runs in**: ML backend (`ingestion.py`, `cache_fetch.py`)
- **Fields used**: `Open`, `High`, `Low`, `Close` (or `Adj Close`), `Volume`, `Date` (index)
- **Stored**: MongoDB `historical_data` collection (OHLCV)
- **Used by**: All feature engineering, backtesting, drift monitoring, SHAP analysis

#### Yahoo Finance / Seeking Alpha — RSS News

- **Endpoint**: `https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US` and `https://seekingalpha.com/api/sa/combined/{symbol}.xml`
- **Runs in**: Node backend (`rssNewsService.js`)
- **Fields used**: `item.title`, `item.link`, `item.pubDate`, `item.contentSnippet` or `item.content`, `item.guid`
- **Stored**: NOT stored — returned to frontend only (VADER sentiment applied inline)
- **Used by**: Stock detail page (news tab), news page (RSS filter)

#### FRED — Macro Economic Indicators

- **Endpoint**: `fred.get_series(series_code, start, end)` via `fredapi` Python library
- **Runs in**: ML backend (`fred_macro.py`)
- **Series IDs fetched** (13 total):
  - `GDP` → FRED code `GDP`
  - `REAL_GDP` → `GDPC1`
  - `REAL_GDP_PER_CAPITA` → `A939RX0Q048SBEA`
  - `CPI` → `CPIAUCSL`
  - `UNEMPLOYMENT` → `UNRATE`
  - `INFLATION` → `FPCPITOTLZGUSA`
  - `FEDERAL_FUNDS_RATE` → `FEDFUNDS`
  - `TREASURY_10Y` → `GS10`
  - `TREASURY_2Y` → `GS2`
  - `TREASURY_30Y` → `GS30`
  - `RETAIL_SALES` → `RSXFSN`
  - `DURABLES` → `UMDMNO`
  - `NONFARM_PAYROLL` → `PAYEMS`
- **Stored**: MongoDB `macro_data_raw` / `macro_data` collections (date-keyed: `{"indicator": "FEDERAL_FUNDS_RATE", "2024-01-01": 5.33, ...}`)
- **Used by**: Feature engineering (`macro_spread_2y10y`, `macro_fed_funds`), Gemini explanation context

#### FMP — Dividends

- **Endpoint**: `GET https://financialmodelingprep.com/stable/dividends?symbol={ticker}&apikey={key}`
- **Runs in**: ML backend (`sentiment.py`)
- **Fields used**: Full dividend history array
- **Stored**: MongoDB `alpha_vantage_data` (endpoint: `fmp_dividends`)
- **Used by**: Dividend sentiment analysis

#### FMP — Earnings

- **Endpoint**: `GET https://financialmodelingprep.com/stable/earnings?symbol={ticker}&limit=5&apikey={key}`
- **Runs in**: ML backend (`sentiment.py`)
- **Fields used**: Earnings history (EPS actual vs estimate)
- **Stored**: MongoDB `alpha_vantage_data` (endpoint: `fmp_earnings`)
- **Used by**: Earnings sentiment analysis

#### FMP — Analyst Estimates

- **Endpoint**: `GET https://financialmodelingprep.com/stable/analyst-estimates?symbol={ticker}&period=annual&page=0&limit=10&apikey={key}`
- **Runs in**: ML backend (`sentiment.py`)
- **Fields used**: `estimatedEpsAvg`, `numberAnalystEstimatedEps`
- **Stored**: MongoDB `alpha_vantage_data` (endpoint: `fmp_analyst-estimates`)
- **Used by**: Analyst estimate sentiment scoring

#### FMP — Ratings Snapshot

- **Endpoint**: `GET https://financialmodelingprep.com/stable/ratings-snapshot?symbol={ticker}&apikey={key}`
- **Runs in**: ML backend (`sentiment.py`)
- **Fields used**: `ratingScore` (1-5 scale)
- **Stored**: MongoDB `alpha_vantage_data` (endpoint: `fmp_ratings-snapshot`)
- **Used by**: Rating sentiment scoring

#### FMP — Price Target Summary

- **Endpoint**: `GET https://financialmodelingprep.com/stable/price-target-summary?symbol={ticker}&apikey={key}`
- **Runs in**: ML backend (`sentiment.py`)
- **Fields used**: `lastYearAvgPriceTarget`, `lastYearCount`
- **Stored**: MongoDB `alpha_vantage_data` (endpoint: `fmp_price-target-summary`)
- **Used by**: Price target sentiment scoring

#### FMP — SEC Filings (Backup)

- **Endpoint**: `GET https://financialmodelingprep.com/api/v3/sec_filings/{ticker}?from={date}&to={date}&limit=30&apikey={key}`
- **Runs in**: ML backend (`sec_filings.py`)
- **Fields used**: `type`, `fillingDate`, `companyName`, `link`, `finalLink`, `cik`, `acceptedDate`
- **Stored**: MongoDB `sec_filings_raw` collection
- **Used by**: SEC filing sentiment analysis (fallback when Kaleidoscope unavailable)

#### Marketaux — News

- **Endpoint**: `GET https://api.marketaux.com/v1/news/all?symbols={ticker}&filter_entities=true&language=en&api_token={key}`
- **Runs in**: Node backend (`newsService.js`) + ML backend (`sentiment.py`)
- **Fields used**: `data[].uuid`, `data[].title`, `data[].url`, `data[].published_at`, `data[].source`, `data[].snippet`/`description`, `data[].image_url`, `data[].entities[].symbol`, `data[].entities[].sentiment_score`, `data[].entities[].industry`
- **Stored**: Node: NOT stored. ML: within `sentiment` collection (`marketaux_raw_data`)
- **Used by**: Node: News page. ML: sentiment scoring

#### TickerTick — News

- **Endpoint**: `GET https://api.tickertick.com/feed?q=tt:{symbol}&n={limit}`
- **Runs in**: Node backend (`tickertickNewsService.js`)
- **Fields used**: `id`, `title`, `url`, `time` (→ ISO), `site` (source), `description`, `favicon_url`, `tickers[]`/`tags[]`
- **Stored**: NOT stored — returned to frontend only
- **Used by**: News page (unified feed)

#### RapidAPI — Fear & Greed Index

- **Endpoint**: `GET https://fear-and-greed-index.p.rapidapi.com/v1/fgi`
- **Runs in**: Node backend (`marketService.js`)
- **Fields used**: Entire `response.data` object
- **Stored**: NOT stored — returned to frontend only
- **Used by**: Market Sentiment Banner component

#### Calendarific — US Holidays

- **Endpoint**: `GET https://calendarific.com/api/v2/holidays?api_key={key}&country=US&year={year}&type=national`
- **Runs in**: Node backend (`marketService.js`)
- **Fields used**: `response.holidays[].date.iso`, `response.holidays[].locations` (filtered for NYSE)
- **Stored**: Redis key `us_holidays_{year}` (1 year TTL) + in-memory `holidaysMemoryCache`
- **Used by**: Market status calculation (is market open today?)

#### financialdata.net — Stock Prices

- **Endpoint**: `GET https://financialdata.net/api/v1/stock-prices?identifier={symbol}&key={key}`
- **Runs in**: Node backend (`massiveService.js`)
- **Fields used**: `date`, `close` (used to calculate RSI, MACD, SMA, EMA)
- **Stored**: In-memory `historicalPriceCache` (12hr TTL), calculated indicators in `indicatorCache` (24hr TTL)
- **Used by**: Technical Indicators component on stock detail page

#### Nasdaq — Short Interest

- **Endpoint**: `GET https://api.nasdaq.com/api/quote/{ticker}/short-interest?assetClass=stocks`
- **Runs in**: ML backend (`short_interest.py`)
- **Fields used**: `data.shortInterestTable.rows[].settlementDate`, `.interest` (short interest count), `.avgDailyShareVolume`, `.daysToCover`
- **Stored**: Within MongoDB `sentiment` collection (short interest data)
- **Used by**: Short interest sentiment analysis, Gemini explanation context

#### Finviz — News Headlines + Short Interest (Fallback)

- **Endpoint**: `https://finviz.com/quote.ashx?t={ticker}` (web scraping)
- **Runs in**: ML backend (`sentiment.py`, `short_interest.py`)
- **Fields used**: News headlines from HTML table; short interest: settlement date, short interest, shares float, average daily volume, short float %, short ratio
- **Stored**: Within MongoDB `sentiment` collection (`finviz_raw_data`)
- **Used by**: Sentiment scoring, short interest analysis (NYSE fallback)

#### Reddit — Social Sentiment (PRAW)

- **Endpoint**: `reddit.subreddit(name).search(ticker, sort="new", time_filter="week", limit=20)`
- **Runs in**: ML backend (`sentiment.py`)
- **Fields used**: `post.title`, `post.selftext`, `post.score`, `comment.body`, `comment.score`
- **Stored**: Within MongoDB `sentiment` collection (`reddit_raw_data`)
- **Used by**: Sentiment scoring (FinBERT/RoBERTa applied to text)

#### Kaleidoscope — SEC Filings (Primary)

- **Endpoint**: `GET https://api.kscope.io/v2/sec/search/{ticker}?key={key}&content=sec&sd={start}&ed={end}&limit=50`
- **Runs in**: ML backend (`sec_filings.py`)
- **Fields used**: `data[].Form` (10-K, 10-Q, 8-K, etc.), `data[].Date`, `data[].html` (full filing URL), `data[].acc`, `data[].CIK`, `data[].Filer`, `data[].Form_Desc`
- **Stored**: MongoDB `sec_filings` (summary) + `sec_filings_raw` (processed with text + sentiment)
- **Used by**: SEC filing sentiment analysis

#### Seeking Alpha — Comment Sentiment

- **Endpoint**: `https://seekingalpha.com/symbol/{ticker}/comments` (Playwright web scraping)
- **Runs in**: ML backend (`seeking_alpha.py`)
- **Fields used**: Comment text (`div[data-test-id="comment-content"]`), username, timestamp, likes
- **Stored**: MongoDB `seeking_alpha_comments` (raw) + `seeking_alpha_sentiment` (scored)
- **Used by**: Sentiment scoring within `sentiment.py`

#### Google Gemini — AI Explanation

- **Endpoint**: `client.models.generate_content(model="gemini-2.5-flash", contents=prompt)`
- **Runs in**: ML backend, GitHub Actions (`generate_explanations.py`)
- **Request**: Comprehensive prompt with predictions, technicals, sentiment, SHAP, macro, insider data, short interest, news headlines (~1500 char budget)
- **Response fields used**: Full generated text (structured sections: OUTLOOK, SUMMARY, KEY_DRIVERS, etc.)
- **Stored**: MongoDB `prediction_explanations` collection (within `explanation_data.ai_explanation`)
- **Used by**: `AIExplanationWidget` on stock detail page

#### NewsAPI — Sector News

- **Endpoint**: `GET https://newsapi.org/v2/everything?q={query}&apiKey={key}`
- **Runs in**: Node backend (`newsService.js`)
- **Fields used**: `articles[].title`, `articles[].url`, `articles[].publishedAt`, `articles[].source.name`, `articles[].description`, `articles[].urlToImage`
- **Stored**: NOT stored — returned to frontend only (VADER sentiment applied inline)
- **Used by**: News page (when industry filter is active)

### Feature Mapping Table (API -> MongoDB -> ML Feature -> Predictor Input)

This table traces how external data becomes a number that LightGBM uses to predict stock prices.

| ML Feature | Source API | API Field(s) | MongoDB Collection | Transformation |
|-----------|-----------|-------------|-------------------|---------------|
| `log_return_1d` | Yahoo Finance | `Close` | `historical_data` | `ln(Close_t / Close_{t-1})` |
| `log_return_5d` | Yahoo Finance | `Close` | `historical_data` | `ln(Close_t / Close_{t-5})` |
| `log_return_21d` | Yahoo Finance | `Close` | `historical_data` | `ln(Close_t / Close_{t-21})` |
| `volatility_20d` | Yahoo Finance | `Close` | `historical_data` | Rolling 20-day std of log returns |
| `volume_ratio` | Yahoo Finance | `Volume` | `historical_data` | `Volume / SMA(Volume, 20)` |
| `volume_z60` | Yahoo Finance | `Volume` | `historical_data` | Z-score of volume over 60 days |
| `rsi` | Yahoo Finance | `Close` | `historical_data` | 14-day RSI calculated from price changes |
| `rsi_divergence` | Yahoo Finance | `Close` | `historical_data` | Price-RSI divergence signal |
| `bb_position` | Yahoo Finance | `Close` | `historical_data` | `(Close - BB_lower) / (BB_upper - BB_lower)` |
| `price_vs_sma20` | Yahoo Finance | `Close` | `historical_data` | `Close / SMA(Close, 20) - 1` |
| `price_vs_sma50` | Yahoo Finance | `Close` | `historical_data` | `Close / SMA(Close, 50) - 1` |
| `momentum_5d` | Yahoo Finance | `Close` | `historical_data` | 5-day price momentum |
| `trend_20d` | Yahoo Finance | `Close` | `historical_data` | 20-day linear trend slope |
| `intraday_range` | Yahoo Finance | `High`, `Low`, `Close` | `historical_data` | `(High - Low) / Close` |
| `overnight_gap` | Yahoo Finance | `Open`, `Close` | `historical_data` | `Open_t / Close_{t-1} - 1` |
| `vix_level` | Yahoo Finance | `Close` (^VIX) | `historical_data` (via cache_fetch) | Raw VIX close value |
| `vix_vol_20d` | Yahoo Finance | `Close` (^VIX) | `historical_data` (via cache_fetch) | 20-day VIX volatility |
| `spy_vol_20d` | Yahoo Finance | `Close` (SPY) | `historical_data` (via cache_fetch) | 20-day SPY volatility |
| `spy_vol_regime` | Yahoo Finance | `Close` (SPY) | `historical_data` (via cache_fetch) | Quantile-based regime bucket |
| `sector_etf_return_20d` | Yahoo Finance | `Close` (sector ETF) | `historical_data` (via cache_fetch) | 20-day sector ETF log return |
| `excess_vs_sector_5d` | Yahoo Finance | `Close` (stock + sector ETF) | `historical_data` | `stock_return_5d - sector_return_5d` |
| `sector_momentum_rank` | Yahoo Finance | `Close` (11 sector ETFs) | `historical_data` (via cache_fetch) | Rank of sector by recent momentum |
| `sent_mean_1d` | Finviz, RSS, Reddit, Marketaux, FMP, Finnhub | `composite_sentiment` | `sentiment` | Yesterday's composite sentiment |
| `sent_mean_7d` | (same as above) | `composite_sentiment` | `sentiment` | Rolling 7-day mean of composite |
| `sent_mean_30d` | (same as above) | `composite_sentiment` | `sentiment` | Rolling 30-day mean of composite |
| `sent_momentum` | (same as above) | `composite_sentiment` | `sentiment` | `sent_mean_7d - sent_mean_30d` |
| `news_count_7d` | (same as above) | `news_count` | `sentiment` | Rolling 7-day sum of article count |
| `news_spike_1d` | (same as above) | `news_count` | `sentiment` | `news_count_1d / SMA(news_count, 30)` |
| `insider_net_value_30d` | Finnhub | `change`, `transactionPrice`, `transactionCode` | `insider_transactions` | Sum of (change x price) for buys - sells over 30d |
| `insider_buy_ratio_30d` | Finnhub | `transactionCode` | `insider_transactions` | `buy_count / (buy_count + sell_count)` |
| `insider_cluster_buying` | Finnhub | `transactionCode`, `filingDate` | `insider_transactions` | 1 if >= 3 buys and 0 sells in 30d |
| `insider_activity_z_90d` | Finnhub | `filingDate` | `insider_transactions` | Z-score of transaction count vs 360d trailing |
| `macro_spread_2y10y` | FRED | `GS10` - `GS2` | `macro_data_raw` | `TREASURY_10Y - TREASURY_2Y` (yield curve) |
| `macro_fed_funds` | FRED | `FEDFUNDS` | `macro_data_raw` | Raw Federal Funds Rate value |
| `sector_id` | Hardcoded | N/A | N/A | Numeric sector encoding (0-10) |
| `ticker_id` | Hardcoded | N/A | N/A | Numeric ticker encoding |

### Data Freshness / Update Frequency

| Data Source | Update Frequency | When It Runs | Staleness Risk |
|------------|-----------------|-------------|---------------|
| Yahoo Finance (OHLCV) | **Daily** | GitHub Actions Step 5 (training) | Low — runs every weeknight |
| Finnhub Quote (live price) | **Real-time** (WebSocket) | Node backend (always on) | Low — WebSocket reconnects automatically |
| Finnhub Insider Transactions | **Daily** | Sentiment cron (Step 4) | Low — filings appear within 2 business days |
| Finnhub Recommendations | **Daily** | Sentiment cron (Step 4) | Low — consensus updated monthly |
| FMP Earnings/Dividends | **Daily** | Sentiment cron (Step 4) | Low — earnings quarterly |
| FRED Macro Indicators | **Monthly** (most series) | Training step (Step 5) | Medium — GDP/CPI released monthly; `shift(1)` prevents lookahead |
| Marketaux News | **Daily** | Sentiment cron (Step 4) | Low — fresh news each run |
| Finviz Headlines | **Daily** | Sentiment cron (Step 4) | Low — scraped each run |
| Reddit Sentiment | **Every 4 hours** | Sentiment cron (Step 4) | Medium — depends on post volume |
| Seeking Alpha Comments | **Daily** | Sentiment cron (Step 4) | Medium — scraping can fail silently |
| SEC Filings | **Daily** | Sentiment cron (Step 4) | Low — filings appear on EDGAR within hours |
| Short Interest (Nasdaq/Finviz) | **Bi-weekly** (settlement cycle) | Sentiment cron (Step 4) | Medium — data is 2 weeks old by nature |
| SHAP Values | **Daily** | GitHub Actions Step 8 | Low — regenerated each run |
| Gemini AI Explanations | **Daily** | GitHub Actions Step 9 | Low — regenerated each run |
| Fear & Greed Index | **On request** | Node backend (live) | Low — fetched when user visits page |
| Calendarific Holidays | **Yearly** | Node backend (cached 1 year) | None — holidays are static |
| financialdata.net Prices | **On request** | Node backend (cached 24hr) | Low — used for indicators only |

### MongoDB Document Field Mappings (API Response -> Stored Document)

This removes all ambiguity about exactly what fields are written to each collection.

**`historical_data`** (from Yahoo Finance):
```
{
  "ticker":  yf_ticker,             // e.g. "AAPL"
  "date":    row.index,             // datetime from yfinance
  "Open":    row["Open"],
  "High":    row["High"],
  "Low":     row["Low"],
  "Close":   row["Adj Close"],      // adjusted close renamed to Close
  "Volume":  row["Volume"]
}
```

**`insider_transactions`** (from Finnhub):
```
{
  "symbol":           data["symbol"],            // e.g. "AAPL"
  "name":             data["name"],              // insider name
  "share":            data["share"],             // total shares held after
  "change":           data["change"],            // shares bought(+) or sold(-)
  "filingDate":       data["filingDate"],        // SEC filing date (public)
  "transactionDate":  data["transactionDate"],   // actual transaction date
  "transactionCode":  data["transactionCode"],   // P=purchase, S=sale, M=multiple, A=award
  "transactionPrice": data["transactionPrice"]   // price per share
}
```

**`macro_data_raw`** (from FRED):
```
{
  "indicator": "FEDERAL_FUNDS_RATE",    // one of 13 FRED indicator names
  "source":    "FRED",
  "2024-01-01": 5.33,                  // date-keyed values
  "2024-02-01": 5.33,
  "2025-01-01": 4.33
}
```

**`sentiment`** (from 10+ sources, blended):
```
{
  "ticker":               "AAPL",
  "date":                 ISODate("2025-02-17"),
  "composite_sentiment":  0.35,              // blended score [-1, +1]
  "news_count":           25,                // total articles across all sources
  "blended_sentiment":    0.35,
  "finviz_sentiment":     0.4,               // per-source scores
  "finviz_volume":        10,                // per-source article count
  "finviz_confidence":    0.7,
  "finviz_raw_data":      ["headline1", ...],// raw headlines for Gemini context
  "rss_news_sentiment":   0.3,
  "rss_news_volume":      8,
  "reddit_sentiment":     0.2,
  "reddit_volume":        5,
  "reddit_raw_data":      [{title, score}],
  "fmp_sentiment":        0.5,
  "marketaux_sentiment":  0.4,
  "marketaux_raw_data":   [{title, url, sentiment_score}],
  "finnhub_sentiment":    0.3,
  "short_interest_sentiment":  -0.1,
  "short_interest_data":  [{settlementDate, short_interest, daysToCover}],
  "last_updated":         ISODate("2025-02-17T20:00:00Z")
}
```

**`short_interest_data`** (from Nasdaq API / Finviz, NEW — stored by code fix):
```
{
  "ticker":             "AAPL",
  "settlementDate":     "02/14/2025",
  "short_interest":     12345678,           // number of shares sold short
  "avgDailyShareVolume": 5000000,
  "daysToCover":        2.5,
  "fetched_at":         ISODate("2025-02-17T22:15:00Z")
}
```

**`stock_predictions`** (from LightGBM predictor):
```
{
  "ticker":             "AAPL",
  "window":             "next_day",            // or "7_day", "30_day"
  "asof_date":          ISODate("2025-02-17"),
  "timestamp":          ISODate("2025-02-17T22:30:00Z"),
  "prediction":         0.002,                 // predicted log-return (alpha)
  "predicted_price":    255.50,
  "price_change":       0.51,
  "current_price":      255.0,
  "confidence":         0.58,                  // P(return > 0)
  "prob_positive":      0.58,
  "price_range":        {"low": 248.0, "high": 263.0},
  "trade_recommended":  true,
  "alpha":              0.002,
  "alpha_pct":          0.2,
  "is_market_neutral":  true,
  "model_predictions":  {"pooled": 0.002},
  "ensemble_weights":   {"pooled": 1.0}
}
```

**`feature_importance`** (from SHAP analysis):
```
{
  "ticker":               "AAPL",
  "window":               "next_day",
  "date":                 "2025-02-17",
  "base_value":           -0.0002,
  "shap_prediction":      0.002,
  "sanity_ok":            true,
  "top_positive_contrib": [{"feature": "macro_spread_2y10y", "value": 0.3, "contrib": 0.0006}],
  "top_negative_contrib": [{"feature": "sector_etf_vol_20d", "value": 0.02, "contrib": -0.0015}],
  "global_gain_importance": [{"feature": "macro_spread_2y10y", "gain": 0.10, "gain_pct": 41.52}],
  "n_features":           42,
  "feature_list_hash":    "2468d61adac0"
}
```

**`prediction_explanations`** (from Gemini explainer):
```
{
  "ticker":           "AAPL",
  "window":           "comprehensive",
  "timestamp":        ISODate("2025-02-17T22:38:00Z"),
  "explanation_data": {
    "ticker":              "AAPL",
    "explanation_date":    "2025-02-17",
    "prediction_data":     {"next_day": {...}, "7_day": {...}, "30_day": {...}},
    "sentiment_summary":   {"blended_sentiment": 0.35, "finviz_articles": 10},
    "technical_indicators": {"RSI": 50.5, "MACD": 1.18},
    "feature_importance":  {"top_positive_contrib": [...], "top_negative_contrib": [...]},
    "macro_context":       {"FEDERAL_FUNDS_RATE": {"value": 4.33, "date": "2025-12-01"}},
    "insider_summary":     {"buys": 3, "sells": 1},
    "short_interest_summary": {"short_float_pct": 0.7},
    "ai_explanation":      "OVERALL_OUTLOOK: Slightly Bearish\n\nSUMMARY: ...",
    "data_sources_used":   ["ML Predictions", "Sentiment Analysis", "SHAP", "Macro", "Insider"],
    "explanation_quality": {"data_completeness": 0.85}
  }
}
```

---

## 9. Data Pipeline

### Data Flow Overview

```
              ┌─── Yahoo Finance (OHLCV prices)
              │
              ├─── FRED API (macro indicators)
              │
DATA SOURCES  ├─── Finnhub (insider trades, recommendations, peers, financials)
              │
              ├─── FMP (earnings, dividends, analyst estimates, ratings)
              │
              ├─── News APIs (Finviz, RSS, Marketaux, TickerTick, Finnhub)
              │
              ├─── Reddit (social sentiment via PRAW)
              │
              ├─── Seeking Alpha (comment sentiment via Playwright)
              │
              └─── SEC/Kaleidoscope (filing sentiment)
                        │
                        ▼
              ┌─────────────────────┐
              │    MongoDB Atlas     │
              │                     │
              │  historical_data    │ ← OHLCV prices
              │  sentiment          │ ← Blended sentiment scores
              │  insider_transactions│ ← Insider trading records
              │  macro_data_raw     │ ← FRED macro indicators
              │  sec_filings_raw    │ ← SEC filing sentiment
              │  alpha_vantage_data │ ← FMP financial data
              │  seeking_alpha_*    │ ← Comment sentiment
              │  finnhub_*          │ ← Financials, peers, etc.
              │                     │
              └─────────┬───────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │  Feature Engineering │
              │  (features_minimal.py)│
              │                     │
              │  42+ features:      │
              │  - Price/returns    │
              │  - Volatility       │
              │  - Technical (RSI)  │
              │  - Sector relative  │
              │  - VIX/SPY regime   │
              │  - Sentiment        │
              │  - Insider trading  │
              │  - Macro economic   │
              └─────────┬───────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │  LightGBM Predictor  │  ← The ONLY price predictor
              │  (predictor.py)      │
              │                     │
              │  3 horizons:        │
              │  - next_day (1d)    │
              │  - 7_day (5 trading)│
              │  - 30_day (21 trad.)│
              └─────────┬───────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │  stock_predictions   │ ← Stored predictions
              │  feature_importance  │ ← SHAP decompositions
              │  prediction_         │ ← Gemini AI explanations
              │   explanations       │
              └─────────────────────┘
```

### Data Storage in MongoDB

Every piece of data fetched from APIs is stored in MongoDB so it can be reused for:
1. **Feature engineering** — Historical sentiment, insider data, and macro data feed into ML features
2. **AI explanation context** — Gemini uses all stored data to generate rich explanations
3. **Historical tracking** — Enables backtesting and drift monitoring

---

## 10. Machine Learning Pipeline

### Overview

The ML pipeline predicts **stock price returns** (not absolute prices) for 100 S&P 100 stocks across 3 time horizons. It uses **market-neutral alpha** — meaning it predicts how much a stock will outperform or underperform the S&P 500 (SPY), not just whether the stock goes up or down.

> **Remember**: LightGBM is the ONLY model that predicts prices. FinBERT/RoBERTa/VADER produce input features. Gemini writes explanations.

### Step-by-Step Pipeline

#### Step 1: Data Ingestion (`data/ingestion.py`)
- Fetches OHLCV data from Yahoo Finance for each ticker
- Only downloads new data since the last MongoDB update (incremental)
- Stores in `historical_data` collection

#### Step 2: Sentiment Collection (`sentiment_cron.py` -> `data/sentiment.py`)
- Runs every 4 hours (cron) or as part of daily pipeline
- For each ticker, fetches text from 10+ sources
- Runs FinBERT/RoBERTa/VADER to score the text as positive/negative/neutral
- Blends scores with weighted averages into a single `composite_sentiment` number
- Stores in `sentiment` collection
- **This does NOT predict prices** — it produces a feature that the LightGBM predictor uses

#### Step 3: Feature Engineering (`data/features_minimal.py`)
- Reads price data, sentiment, insider trades, macro indicators from MongoDB
- Computes 42+ features organized into categories:

**Price & Return Features:**
| Feature | Description |
|---------|-------------|
| `log_return_1d` | Yesterday's log return |
| `log_return_5d` | 5-day log return |
| `log_return_21d` | 21-day log return |
| `volatility_20d` | 20-day rolling volatility |
| `intraday_range` | (High-Low)/Close |
| `overnight_gap` | Today's Open vs yesterday's Close |

**Volume Features:**
| Feature | Description |
|---------|-------------|
| `volume_ratio` | Volume / 20-day average volume |
| `volume_z60` | Volume z-score over 60 days |
| `volume_vol_ratio` | Volume volatility ratio |

**Technical Features:**
| Feature | Description |
|---------|-------------|
| `rsi` | 14-day Relative Strength Index (0-100) |
| `rsi_divergence` | Price-RSI divergence signal |
| `bb_position` | Position within Bollinger Bands (-1 to +1) |
| `price_vs_sma20` | Price relative to 20-day moving average |
| `price_vs_sma50` | Price relative to 50-day moving average |
| `momentum_5d` | 5-day price momentum |
| `trend_20d` | 20-day price trend |

**Market Regime Features:**
| Feature | Description |
|---------|-------------|
| `vix_level` | Current VIX (fear index) level |
| `vix_vol_20d` | VIX 20-day volatility |
| `spy_vol_20d` | S&P 500 20-day volatility |
| `spy_vol_regime` | Volatility regime (low/medium/high) |
| `vol_regime` | Stock's volatility regime |

**Sector Features:**
| Feature | Description |
|---------|-------------|
| `sector_id` | Numeric sector identifier |
| `ticker_id` | Numeric ticker identifier |
| `sector_etf_return_20d` | Sector ETF 20-day return |
| `sector_etf_return_60d` | Sector ETF 60-day return |
| `sector_etf_vol_20d` | Sector ETF 20-day volatility |
| `excess_vs_sector_5d` | Stock return minus sector return (5-day) |
| `excess_vs_sector_20d` | Stock return minus sector return (20-day) |
| `sector_momentum_rank` | Sector's rank by recent momentum |

**Macro Features:**
| Feature | Description |
|---------|-------------|
| `macro_spread_2y10y` | Treasury yield curve spread (recession indicator) |
| `macro_fed_funds` | Federal funds rate |

**Sentiment Features** (when enabled):
| Feature | Description |
|---------|-------------|
| `sent_mean_1d` | Yesterday's sentiment score |
| `sent_mean_7d` | 7-day rolling average sentiment |
| `sent_mean_30d` | 30-day rolling average sentiment |
| `sent_momentum` | Sentiment regime change (7d avg - 30d avg) |
| `news_count_7d` | Rolling 7-day article count |
| `news_spike_1d` | Unusual news activity detector |

**Insider Features:**
| Feature | Description |
|---------|-------------|
| `insider_net_value_30d` | Net insider trading value (30-day) |
| `insider_buy_ratio_30d` | Insider buy/sell ratio |
| `insider_cluster_buying` | Multiple insiders buying simultaneously |

**Point-in-Time Safety**: Every feature uses `shift(1)` to ensure the model never sees future data. Row t only uses data available at market close on day t-1.

#### Step 4: Target Computation

The LightGBM predictor predicts **log-returns** (percentage changes expressed as logarithms):

- `next_day`: 1 trading day ahead
- `7_day`: 5 trading days ahead
- `30_day`: 21 trading days ahead

**Market-Neutral Alpha**: When `USE_MARKET_NEUTRAL_TARGET = True` (default), the target is:
```
alpha = stock_return - SPY_return
```
This means the model predicts how much the stock will beat or lag the market, not just whether it goes up.

#### Step 5: Model Training (`models/predictor.py`)

**Algorithm**: LightGBM (gradient boosted decision trees) — the ONLY price predictor

**Why LightGBM?**
- Fast training (handles 100 stocks quickly)
- Handles missing values natively
- Built-in feature importance
- Supports Huber loss (robust to outliers)

**Why not LSTM (or other deep learning)?**
- Tabular cross-sectional features (many tickers, many numeric features) suit tree models better than sequential RNNs; LSTMs excel at long raw sequences (e.g. tick-by-tick), not pre-engineered panel data.
- LightGBM needs less data and compute, is easier to tune and deploy in CI, and gives interpretable feature importance and fast TreeSHAP for explanations.

**Hyperparameters:**
```
Objective:        huber (robust to outlier returns)
Learning rate:    0.05
Max depth:        4 (prevents overfitting)
Num leaves:       15
N estimators:     150
Min child samples: 25
Regularization:   L1=0.1, L2=0.1
Subsampling:      80% rows, 80% columns
```

**Training Strategy — Pooled Model:**
1. Combine data from all 100 tickers
2. Train ONE model per horizon (3 models total)
3. Uses walk-forward validation with purge/embargo gaps
4. Feature pruning: Keep top 30 features per horizon (protected features always kept)

**Why Pooled?**
A pooled model sees patterns across all stocks (e.g., "when VIX spikes, tech stocks drop"), making it more robust than per-ticker models which have limited data.

**Sign Classifier:**
A separate LightGBM binary classifier predicts P(return > 0), used for the confidence score shown to users.

#### Step 6: Prediction Generation

For each of the 100 tickers:
1. Fetch latest features
2. Run through the trained LightGBM model (the predictor)
3. Compute: predicted return, predicted price, confidence, price range
4. Determine if trade is recommended (alpha > 0.1%, P(up) > 52%)
5. Store in `stock_predictions` collection

#### Step 7: SHAP Explanation (`explain/shap_analysis.py`)

SHAP (SHapley Additive exPlanations) answers: **"Why did the LightGBM predictor make this prediction?"**

This is a mathematical decomposition, not a language model. For each ticker:
1. Use LightGBM's native TreeSHAP (very fast)
2. Get contribution of each feature to the prediction
3. Split into bullish drivers (+) and bearish drivers (-)
4. Also compute global feature importance (which features matter most overall)
5. Store in `feature_importance` collection

#### Step 8: AI Explanation (`scripts/generate_explanations.py`)

This step uses Google Gemini (a language model) to translate all the numeric data into plain English. Gemini does NOT predict prices — it reads the LightGBM prediction and explains it.

**Stock-Specific Prompts**: Each prompt is tailored to the individual stock using a `STOCK_META` lookup table that provides the company name, sector, and industry for all 100 S&P 100 tickers. The prompt includes sector-specific analysis guidance (e.g., interest rate sensitivity for financials, pipeline catalysts for healthcare, AI/cloud narratives for tech).

**Data fed to Gemini (11 sources):**
- LightGBM predictions (all 3 horizons with confidence, alpha vs SPY, price ranges)
- Technical analysis (RSI, MACD, Bollinger, SMAs, EMAs, volume ratio, 52-week range, performance)
- News headlines (Finviz, RSS, Reddit, Marketaux + aggregated news from MongoDB)
- Sentiment scores (blended + per-source breakdown)
- SHAP feature drivers (with human-readable names and contribution values)
- Macro economic context (Fed rate, CPI, unemployment, yield curve, GDP)
- Insider trading activity (buy/sell ratio, recent transactions with names/prices)
- Short interest data (short float %, days to cover)
- Finnhub basic financials (P/E, P/B, ROE, dividend yield, market cap, beta)
- FMP earnings data (EPS actual vs estimated, earnings surprise)
- FMP analyst ratings and price targets

**Gemini Model Fallback Chain**: The script automatically selects the best available model based on in-process RPD tracking:
1. `gemini-2.5-pro` (1.5K RPD) — preferred for quality and headroom
2. `gemini-2.5-flash` (20 RPD) — first fallback
3. `gemini-2.5-flash-lite` (20 RPD) — last resort

If `GEMINI_MODEL` env var is explicitly set, it locks to that model (no fallback). The workflow does NOT set this variable, allowing automatic fallback.

**Output format (structured sections):**
- `OVERALL_OUTLOOK`: Bullish/Bearish/Neutral/Slightly Bullish/Slightly Bearish
- `CONFIDENCE`: 1-100 integer score based on data agreement
- `SUMMARY`: 2-3 sentences referencing specific numbers and the company by name
- `WHAT_THIS_MEANS`: Plain-language actionable insight specific to the stock's business
- `KEY_DRIVERS`: Bullish (+) and bearish (-) factors, each citing a specific data value
- `NEWS_IMPACT`: References actual headlines from the data; states when unavailable
- `KEY_LEVELS`: Support and resistance prices from Bollinger Bands / prediction ranges
- `BOTTOM_LINE`: Single most important takeaway with predicted price/percentage

Stored in `prediction_explanations` collection.

---

## 11. Backtesting & Model Validation

### What is Backtesting?

Backtesting simulates how the model's trading signals would have performed on **historical data it has never seen** (out-of-sample). It answers: "If I had followed this model's advice in the past, would I have made money?"

### When Does Backtesting / Evaluation Run?

There are two separate processes. Beginners often confuse them:

**1. Daily OOS Evaluation (runs every day in GitHub Actions)**
- Command: `python -m ml_backend.scripts.evaluate_models --stored --days 60`
- What it does: Takes predictions that were **already stored** in MongoDB and compares them to what actually happened in the stock market
- Why it's trustworthy: These predictions were generated BEFORE the actual outcomes were known — they are truly out-of-sample
- Results: saved to `eval_report.txt` artifact

**2. Full Backtesting Simulation (manual, not in daily pipeline)**
- Module: `ml_backend/backtest.py`
- What it does: Simulates a full trading strategy using the model's predictions on historical data
- When to run: Manually, when evaluating model changes or new feature additions
- Why it's not daily: It re-predicts on historical data (computationally expensive) and is designed for research, not monitoring
- The daily pipeline uses **stored evaluation** instead, which is the correct approach for ongoing monitoring because it tests real predictions, not re-predictions

### How Backtesting Works (`backtest.py`)

```
Step 1: Define out-of-sample period (dates after training ended)
Step 2: Pre-compute all model predictions for all tickers
Step 3: For each trading day:
   a. Check if model recommends any trades (trade_recommended=True)
   b. Buy recommended stocks (equal weight, max 5 positions)
   c. Hold for the prediction horizon (1/5/21 trading days)
   d. Sell when holding period expires
   e. Subtract transaction costs (10 basis points round-trip)
   f. Track daily portfolio value
Step 4: Compute metrics vs SPY benchmark
```

**How to run backtest:**
- **Standalone:** `python -m ml_backend.scripts.run_backtest [--tickers AAPL MSFT NVDA] [--horizon next_day|7_day|30_day] [--no-mongo]`  
  Uses `ml_backend/backtest.py` under the hood. With `--no-mongo`, data is fetched from yfinance.
- **As part of pipeline:** `python -m ml_backend.scripts.run_pipeline --tickers AAPL MSFT NVDA [--no-mongo]`  
  After training, the pipeline runs the same backtest automatically (OOS period from training cutoff to latest data) and prints the summary.

### Metrics Computed

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| **Total Return** | Overall portfolio gain/loss | Positive, > SPY return |
| **Sharpe Ratio** | Return per unit of risk (annualized) | > 1.0 is good, > 2.0 is excellent |
| **Max Drawdown** | Largest peak-to-trough decline | < 20% for a cautious strategy |
| **Win Rate** | % of trades that were profitable | > 50% is decent, > 55% is good |
| **Directional Accuracy** | % of times the model correctly predicted up/down | > 52% is useful, > 55% is very good |
| **Rank Correlation** | Spearman correlation between predicted and actual returns | > 0.05 is meaningful |
| **Brier Score** | How well-calibrated the probability estimates are | Lower is better; < 0.25 is good |

### Walk-Forward Validation

The training process uses **walk-forward validation** to prevent overfitting:

```
|────── Train ──────|─purge─|──── Validate ──── |─purge─|──── Holdout ────|
|       70%         |  gap  |      15%          |  gap  |     15%         |
```

- **Purge gap**: 5 days between train and validation to prevent label leakage
- **Embargo**: 2 additional days after purge
- **Walk-forward folds**: 3 rolling folds, report median metrics

---

## 12. Prediction Confidence & Trustworthiness

### How Confidence Scores Are Calculated

The confidence score displayed to users is **P(return > 0)** — the probability that the stock will go up. This comes from a separate LightGBM binary classifier trained alongside the regression predictor.

**How it works:**
1. A LightGBM binary classifier is trained on the same features
2. Label = 1 if actual return > 0, else 0
3. Output = calibrated probability (0 to 1)
4. Displayed as percentage: 51% means "barely more likely up than down", 70% means "strong bullish signal"

### What Makes a Prediction Confident vs Uncertain

| Confidence Level | Range | What It Means |
|-----------------|-------|---------------|
| **High** | > 65% | Strong signal in one direction. Multiple features agree. |
| **Medium** | 55-65% | Moderate signal. Some conflicting indicators. |
| **Low** | 50-55% | Essentially a coin flip. Model is uncertain. |
| **Contrarian** | < 50% | Model thinks the stock is more likely to go down. |

### When to Trust vs Not Trust Predictions

**Trust MORE when:**
- Confidence > 60%
- Multiple SHAP drivers agree (all bullish or all bearish)
- Strong recent data (high news count, fresh sentiment)
- The model's historical directional accuracy for this horizon is > 53%

**Trust LESS when:**
- Confidence is 48-52% (coin-flip zone)
- Bullish and bearish drivers are balanced (conflicting signals)
- `data_completeness` quality score < 0.5
- No recent news (sentiment score = 0)
- During high-volatility events (earnings, FOMC meetings)
- Drift monitor shows WARNING status

### Trade Recommendation Filters

A prediction only generates a `trade_recommended = True` signal when:
- Predicted alpha > 0.1% (`TRADE_MIN_ALPHA`)
- P(return > 0) > 52% (`TRADE_MIN_PROB_POSITIVE`)
- Predicted return exceeds transaction costs (10 basis points)

---

## 13. GitHub Actions — Daily Predictions Workflow

**File**: `.github/workflows/daily-predictions.yml`

### Schedule

- **Cron**: `15 22 * * 1-5` = 10:15 PM UTC, Monday through Friday
- **Why this time?** US markets close at 4:00 PM ET (9:00 PM UTC in winter / 8:00 PM UTC in summer). The 10:15 PM UTC schedule gives ~1-2 hours buffer after close.
- **Manual trigger**: Can also be run manually via `workflow_dispatch`
- **Concurrency**: Only one run at a time (`cancel-in-progress: false`)
- **Timeout**: 120 minutes max

### Step-by-Step Execution

```
Step 1: CHECKOUT REPOSITORY
  → Clone the latest code

Step 2: SETUP PYTHON 3.11
  → Install Python, cache pip dependencies

Step 3: INSTALL DEPENDENCIES
  → pip install -r ml_backend/requirements-prod.txt

Step 4: RUN SENTIMENT CRON (non-fatal if fails)
  → python -m ml_backend.sentiment_cron
  → Fetches sentiment for all 100 tickers from all sources
  → Stores in MongoDB sentiment collection
  → Env vars: MONGODB_URI, FINNHUB_API_KEY, FMP_API_KEY, MARKETAUX_API_KEY,
              GOOGLE_API_KEY, REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET

Step 5: TRAIN POOLED MODEL
  → python -m ml_backend.scripts.run_pipeline --all-tickers --no-predict
  → Fetches historical data for all 100 tickers
  → Engineers features
  → Trains ONE pooled LightGBM model per horizon (3 models)
  → Models saved to disk (models/v1/_pooled/)
  → Does NOT generate predictions yet (--no-predict flag)
  → Env vars: All API keys + FRED_API_KEY, ALPHAVANTAGE_API_KEY

Step 6: GENERATE PREDICTIONS (10 batches x 10 tickers)
  → python -m ml_backend.scripts.run_pipeline --predict-only --tickers [batch]
  → Loads trained models from disk
  → Generates predictions for each batch
  → Stores in MongoDB stock_predictions collection
  → 3 retry attempts per batch with exponential backoff (20s, 40s, 60s)
  → 5s pause between batches
  → Fails job if > 3 batches fail (> 30%)

Step 7: VERIFY PREDICTIONS ARE FRESH
  → Python script checks MongoDB for canary tickers (AAPL, AMZN, JPM, etc.)
  → Verifies predictions exist and are < 3 hours old
  → Asserts >= 200 fresh prediction documents
  → Fails if any canary ticker is missing or stale

Step 8: GENERATE SHAP FEATURE IMPORTANCE (10 batches x 10 tickers)
  → python -m ml_backend.explain.shap_analysis --tickers [batch]
  → Computes SHAP values for each ticker (math, not AI)
  → Stores in MongoDB feature_importance collection
  → Non-fatal: fails if > 5 batches fail

Step 9: GENERATE AI EXPLANATIONS (Gemini)
  → python -m ml_backend.scripts.generate_explanations
  → Uses all MongoDB data (predictions, sentiment, technicals, SHAP, macro, insider, short interest)
  → Calls Gemini API (auto-fallback: pro→flash→flash-lite) to EXPLAIN (not predict) each ticker with stock-specific prompts
  → Stores in MongoDB prediction_explanations collection
  → Non-fatal if fails

Step 10: EVALUATE STORED PREDICTIONS (last 60 days)
  → python -m ml_backend.scripts.evaluate_models --stored --days 60
  → Compares past predictions to actual price outcomes
  → Outputs metrics: directional accuracy, rank correlation, Brier score
  → Saved to eval_report.txt artifact
  → Non-fatal

Step 11: RUN DRIFT MONITOR
  → python -m ml_backend.scripts.drift_monitor
  → Checks for prediction distribution shifts (PSI)
  → Checks rolling directional accuracy
  → Checks calibration degradation
  → Saved to drift_report.txt artifact
  → Non-fatal

Step 12: UPLOAD ARTIFACTS (always runs)
  → Uploads all .log files, eval_report.txt, drift_report.txt
  → Available for download from GitHub Actions UI
```

### Failure Handling

| Step | Fatal? | Behavior on Failure |
|------|--------|-------------------|
| Sentiment cron | No | Logs "non-fatal" and continues |
| Train model | **Yes** | Job fails (set -e) |
| Predictions | **Partially** | Fails if > 30% of batches fail |
| Verify freshness | **Yes** | Asserts fail the job |
| SHAP analysis | **Partially** | Fails if > 50% of batches fail |
| AI explanations | No | Logs "non-fatal" and continues |
| Evaluation | No | Logs "non-fatal" and continues |
| Drift monitor | No | Logs "non-fatal" and continues |

---

## 14. Frontend Architecture

### Routing (Next.js App Router — Consolidated)

> **History**: This project started as a React SPA using React Router DOM for client-side navigation. It was later wrapped in a Next.js shell for Vercel deployment. In Feb 2026, the routing was **fully consolidated to Next.js App Router**, removing `react-router-dom` entirely.
>
> **How it works now**:
> - `app/layout.tsx` provides the HTML shell, metadata, `WebSocketProvider`, and the `Layout` component (sidebar + navbar)
> - Each route has its own `page.tsx` in the `app/` directory (standard Next.js App Router)
> - Page view components live in `views/` and are imported by the thin `app/*/page.tsx` wrappers
> - Navigation uses `next/link` (`Link` with `href`) and `next/navigation` (`useRouter`, `usePathname`, `useParams`)
> - No Pages Router (`pages/` directory removed completely)
>
> **Benefits of the migration**:
> - Direct URL access to any route works (e.g., `/stocks/AAPL` no longer 404s)
> - SEO-friendly: Next.js can SSR/SSG pages
> - Smaller bundle: `react-router-dom` removed (~45KB gzipped)
> - Native Next.js features: prefetching, code splitting per route

#### Performance and other improvements from full Next.js

Removing React Router and using the App Router as the sole routing system yields measurable and qualitative gains:

| Area | Improvement |
|------|-------------|
| **Bundle size** | ~45KB less client JS (react-router-dom removed); less to download, parse, and execute on first load. |
| **Code splitting** | Each route gets its own chunk; users load JS mainly for the page they’re on. Lighter initial load and faster subsequent navigations. |
| **Navigation** | Next.js `<Link>` prefetches routes in the viewport by default; client-side transitions avoid full reloads and keep the shared layout without re-mounting. |
| **Direct URLs** | No redirect chain (e.g. `/stocks/AAPL` is a real App Router route); single response, no double load. |
| **SEO** | Routes can be server-rendered or statically generated; crawlers get full HTML instead of a client-only shell. |
| **Reliability** | Single routing model (App Router only) reduces edge cases, 404s from the hybrid setup, and hydration issues from mixing two routers. |
| **Future use** | Enables server components, streaming, and Next.js caching where applicable, without conflicting with React Router. |

### Pages

| Route | App Router File | View Component | Purpose |
|-------|----------------|----------------|---------|
| `/` | `app/page.tsx` | `views/home.tsx` | Market dashboard with TradingView widgets |
| `/stocks/[symbol]` | `app/stocks/[symbol]/page.tsx` | `views/stock-detail.tsx` | Stock chart, predictions, AI analysis, news |
| `/predictions` | `app/predictions/page.tsx` | `views/predictions.tsx` | AI predictions overview for all stocks |
| `/news` | `app/news/page.tsx` | `views/news.tsx` | Aggregated market news with filters |
| `/watchlist` | `app/watchlist/page.tsx` | `views/watchlist.tsx` | User's watchlist with real-time prices |
| `/fundamentals` | `app/fundamentals/page.tsx` | `views/fundamentals.tsx` | Financial fundamentals (Jika.io embeds) |

### Key Components

| Component | Purpose |
|-----------|---------|
| `AIExplanationWidget` | Displays Gemini-generated AI market intelligence (the EXPLAINER output) |
| `EnhancedQuickPredictionWidget` | Quick stock price prediction lookup (the PREDICTOR output) |
| `MarketSentimentBanner` | Fear & Greed Index banner |
| `TechnicalIndicators` | RSI, MACD, SMA, EMA gauges |
| `NotificationWidget` | Notification bell (polls every 30s) |
| `SearchWidget` | Stock search with debounced results |
| `TradingHoursBar` | Visual trading hours timeline |
| TradingView widgets | Charts, heatmaps, economic calendar, etc. |

### Data Flow Pattern

```
1. usePrefetch() hook prefetches top 5 stocks on app load
2. getCachedData() checks in-memory cache (5 min TTL)
3. If cache miss → fetch from Node backend API
4. Node backend reads from MongoDB or proxies to ML backend
5. Response cached in memory for next request
6. Polling updates prices every 5 seconds
```

### State Management

- **No global store** (no Redux/Zustand)
- React Context: `WebSocketProvider` (real-time prices)
- Local component state for everything else
- In-memory cache via `use-prefetch.tsx` hook

---

## 15. Backend Architecture (Node.js)

### Express Routes

```
/api/market/status          GET  → Market open/close status
/api/market/fear-greed      GET  → Fear & Greed Index
/api/market/sentiment       GET  → Market sentiment composite

/api/news/aggregate         GET  → Aggregated news from multiple sources
/api/news/unified           GET  → Unified news feed (paginated)
/api/news/rss               GET  → RSS news for specific ticker

/api/stock/search/:query    GET  → Search stocks by name/symbol
/api/stock/batch/status     GET  → Batch prediction coverage stats
/api/stock/batch/available  GET  → Stocks with stored explanations
/api/stock/:symbol          GET  → Stock details (profile, quote)
/api/stock/:symbol/predictions   GET  → ML predictions (from LightGBM predictor)
/api/stock/:symbol/explanation   GET  → Stored AI explanation (from Gemini explainer)
/api/stock/:symbol/indicators    GET  → Technical indicators

/api/watchlist/:userId          GET     → Get user's watchlist
/api/watchlist/:userId/add      POST    → Add stock to watchlist
/api/watchlist/:userId/:symbol  DELETE  → Remove stock from watchlist
/api/watchlist/updates/realtime GET     → Real-time price updates

/api/notifications              GET     → Recent notifications
/api/notifications/unread-count GET     → Unread count

/health                         GET     → Health check
```

### Services

| Service | Purpose |
|---------|---------|
| `websocketService` | Finnhub WebSocket for real-time prices |
| `aggregateNewsService` | Merges news from Finnhub, Marketaux, TickerTick, RSS |
| `massiveService` | Technical indicators (RSI, MACD, SMA, EMA) with fallback calculation |
| `marketService` | Market status, holidays, Fear & Greed Index |
| `notificationService` | Market session alerts, price alerts, cleanup |
| `redisClient` | Redis connection (optional, with mock fallback) |

### Watchlist Storage

**Important**: Watchlists are stored **in-memory** (a JavaScript `Map`), NOT in MongoDB. They are lost on server restart. This is a known limitation.

---

## 16. ML Backend Architecture (Python/FastAPI)

### API Endpoints

```
GET  /health                          → Health check (MongoDB ping)
GET  /api/v1/predictions/{ticker}     → Get stored predictions (from LightGBM predictor)
POST /api/v1/predictions/batch        → Batch predictions (1-10 tickers)
GET  /api/v1/sentiment/{ticker}       → Get sentiment analysis (from FinBERT/RoBERTa/VADER scorers)
POST /api/v1/train                    → Start training (background)
POST /api/v1/ingest                   → Ingest historical data
GET  /api/v1/explain/{ticker}/{date}  → Get AI explanation (from Gemini explainer)
POST /api/v1/explain/batch            → Batch generate explanations
```

### Rate Limiting
- 100 requests per hour per IP (sliding window)
- Redis-backed with in-memory fallback
- Response headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`

### Error Handling
- Structured error codes: `TICKER_INVALID`, `PREDICTION_ERROR`, `DATABASE_ERROR`, etc.
- Request tracking with UUID per request
- Pydantic validation error handling

---

## 17. Database Schemas

### MongoDB Collections

#### `historical_data`
```json
{
  "ticker": "AAPL",
  "date": "2025-02-17T00:00:00Z",
  "Open": 150.0,
  "High": 152.0,
  "Low": 149.0,
  "Close": 151.0,
  "Volume": 75000000
}
```
- **Index**: `(ticker, date)` unique
- **Written by**: `ingestion.py` (daily)
- **Read by**: `features_minimal.py`, `predictor.py`, `generate_explanations.py`, `backtest.py`, `drift_monitor.py`

#### `sentiment`
```json
{
  "ticker": "AAPL",
  "date": "2025-02-17T00:00:00Z",
  "composite_sentiment": 0.35,
  "news_count": 25,
  "blended_sentiment": 0.35,
  "finviz_sentiment": 0.4, "finviz_volume": 10,
  "rss_news_sentiment": 0.3, "rss_news_volume": 8,
  "reddit_sentiment": 0.2, "reddit_volume": 5,
  "fmp_sentiment": 0.5, "fmp_volume": 3,
  "finviz_raw_data": ["headline1", "headline2"],
  "rss_news_raw_data": [{"title": "...", "url": "..."}],
  "reddit_raw_data": [{"title": "...", "score": 100}],
  "marketaux_raw_data": [{"title": "...", "sentiment_score": 0.5}],
  "last_updated": "2025-02-17T20:00:00Z"
}
```
- **Index**: `(ticker, date)`, `(ticker, last_updated)`
- **Written by**: `sentiment_cron.py` (every 4h)
- **Read by**: `sentiment_features.py` (ML features), `generate_explanations.py` (Gemini prompt)

#### `stock_predictions`
```json
{
  "ticker": "AAPL",
  "window": "next_day",
  "asof_date": "2025-02-17T00:00:00Z",
  "timestamp": "2025-02-17T22:30:00Z",
  "prediction": 0.002,
  "predicted_price": 255.50,
  "price_change": 0.51,
  "current_price": 255.0,
  "confidence": 0.58,
  "prob_positive": 0.58,
  "prob_above_threshold": 0.55,
  "price_range": {"low": 248.0, "high": 263.0},
  "trade_recommended": true,
  "trade_threshold": 0.001,
  "alpha": 0.002,
  "alpha_pct": 0.2,
  "is_market_neutral": true,
  "model_predictions": {"pooled": 0.002, "per_ticker": null},
  "ensemble_weights": {"pooled": 1.0}
}
```
- **Index**: `(ticker, window, asof_date)`, `(ticker, timestamp)`
- **Written by**: `predictor.py` (daily via GitHub Actions)
- **Read by**: Node backend `stockController.js`, frontend via API, `evaluate_models.py`, `drift_monitor.py`

#### `feature_importance`
```json
{
  "ticker": "AAPL",
  "window": "next_day",
  "date": "2025-02-17",
  "timestamp": "2025-02-17T22:30:00Z",
  "predicted_value": 0.002,
  "predicted_price": 255.50,
  "prob_up": 0.58,
  "current_price": 255.0,
  "base_value": -0.0002,
  "shap_prediction": 0.002,
  "sanity_ok": true,
  "top_positive_contrib": [
    {"feature": "macro_spread_2y10y", "value": 0.3, "contrib": 0.0006},
    {"feature": "vix_vol_20d", "value": 0.1, "contrib": 0.0003}
  ],
  "top_negative_contrib": [
    {"feature": "sector_etf_vol_20d", "value": 0.02, "contrib": -0.0015}
  ],
  "global_gain_importance": [
    {"feature": "macro_spread_2y10y", "gain": 0.10, "gain_pct": 41.52}
  ],
  "n_features": 42,
  "feature_list_hash": "2468d61adac0"
}
```
- **Written by**: `shap_analysis.py` (daily via GitHub Actions)
- **Read by**: `generate_explanations.py` (Gemini prompt context)

#### `prediction_explanations`
```json
{
  "ticker": "AAPL",
  "window": "comprehensive",
  "timestamp": "2025-02-17T22:38:00Z",
  "explanation_data": {
    "ticker": "AAPL",
    "explanation_date": "2025-02-17",
    "prediction_data": { "next_day": {...}, "7_day": {...}, "30_day": {...} },
    "sentiment_summary": { "blended_sentiment": 0.35, "finviz_articles": 10 },
    "technical_indicators": { "RSI": 50.5, "MACD": 1.18 },
    "feature_importance": { "top_positive_contrib": [...], "top_negative_contrib": [...] },
    "macro_context": { "FEDERAL_FUNDS_RATE": {"value": 4.33, "date": "2025-12-01"} },
    "insider_summary": { "buys": 3, "sells": 1 },
    "short_interest_summary": { "short_float_pct": 0.7 },
    "ai_explanation": "OVERALL_OUTLOOK: Slightly Bearish\n\nSUMMARY: ...",
    "data_sources_used": ["ML Predictions", "Sentiment Analysis"],
    "explanation_quality": { "data_completeness": 0.85 }
  }
}
```
- **Written by**: `generate_explanations.py` (daily via GitHub Actions)
- **Read by**: Node backend -> frontend `AIExplanationWidget`

#### `insider_transactions`
```json
{
  "symbol": "AAPL",
  "name": "Tim Cook",
  "share": 50000,
  "change": -50000,
  "filingDate": "2025-02-15",
  "transactionDate": "2025-02-10",
  "transactionCode": "S",
  "transactionPrice": 255.0
}
```
- **Written by**: `sentiment.py` (via Finnhub insider API)
- **Read by**: `insider_features.py` (ML features), `generate_explanations.py` (Gemini prompt)

#### `macro_data_raw`
```json
{
  "indicator": "FEDERAL_FUNDS_RATE",
  "source": "FRED",
  "2024-01-01": 5.33,
  "2024-02-01": 5.33,
  "2024-03-01": 5.33,
  "2025-01-01": 4.33
}
```
- **Written by**: `fred_macro.py`, `macro.py`
- **Read by**: `features_minimal.py` (ML features), `generate_explanations.py` (Gemini prompt)

#### `alpha_vantage_data`
```json
{
  "ticker": "AAPL",
  "endpoint": "fmp_earnings",
  "data": [...],
  "timestamp": "2025-02-17T20:00:00Z"
}
```
- **Written by**: `sentiment.py` (FMP financial data, Finnhub quote)
- **Read by**: `sentiment.py` (financial sentiment analysis)
- **Note**: Despite the name, this collection primarily stores FMP data, not Alpha Vantage data (legacy naming)

#### `notifications`
```json
{
  "type": "market",
  "title": "Market Open",
  "message": "US markets are now open for trading",
  "symbol": null,
  "priority": "medium",
  "createdAt": "2025-02-17T14:30:00Z"
}
```
- **Written by**: `notificationService.js`
- **Read by**: Frontend `NotificationWidget`

#### `sec_filings_raw`
```json
{
  "ticker": "AAPL",
  "form_type": "10-K",
  "filing_date": "2025-01-15",
  "text_content": "...",
  "sentiment": 0.2,
  "source": "kaleidoscope"
}
```
- **Written by**: `sec_filings.py` (Kaleidoscope API + FMP fallback)
- **Read by**: Sentiment pipeline within `sentiment.py`

#### `seeking_alpha_sentiment`
```json
{
  "ticker": "AAPL",
  "date": "2025-02-17",
  "sentiment_score": 0.3,
  "comment_count": 15,
  "avg_likes": 5.2
}
```
- **Written by**: `seeking_alpha.py`
- **Read by**: Sentiment pipeline within `sentiment.py`

#### `short_interest_data` (NEW — added by code fix)
```json
{
  "ticker": "AAPL",
  "settlementDate": "02/14/2025",
  "short_interest": 12345678,
  "avgDailyShareVolume": 5000000,
  "daysToCover": 2.5,
  "fetched_at": "2025-02-17T22:15:00Z"
}
```
- **Written by**: `short_interest.py` `_store_short_interest_raw()` (added in this review)
- **Read by**: `generate_explanations.py` (Gemini prompt context)

#### `finnhub_basic_financials`, `finnhub_company_peers`, `finnhub_insider_sentiment`
- **Written by**: `sentiment.py` (now called during sentiment pipeline — fixed in this review)
- **Read by**: Sentiment pipeline (internal use), available for future Gemini context

---

## 18. Redis Caching

Redis is **optional**. If `REDIS_URL` is not set, a mock client returns `null` for all operations.

| Key Pattern | TTL | Purpose | Written By | Read By |
|------------|-----|---------|-----------|--------|
| `predictions:v1:{ticker}` | 60s | Cache prediction responses | ML backend | ML backend |
| `us_holidays_{year}` | 1 year | Cache US market holidays | `marketService.js` | `marketService.js` |
| `ratelimit:{ip}` | 3600s | Rate limit counters | ML backend middleware | ML backend middleware |

---

## 19. WebSocket / Real-Time Data

### Finnhub WebSocket (Server-Side)

The Node backend maintains a WebSocket connection to Finnhub for real-time trade data.

- **URL**: `wss://ws.finnhub.io?token=${FINNHUB_API_KEY}`
- **Keepalive**: Ping every 25 seconds
- **Reconnection**: Exponential backoff (5s -> 10s -> 20s -> 40s -> 60s, max 5 attempts)

**Message format received:**
```json
{
  "type": "trade",
  "data": [
    {"s": "AAPL", "p": 255.50, "v": 100, "t": 1708185600000, "c": ["1"]}
  ]
}
```

### Frontend "WebSocket" (Actually HTTP Polling)

The frontend does NOT use a true WebSocket. Instead, it **polls** the backend every 5 seconds:

```
GET /api/watchlist/updates/realtime?symbols=AAPL,MSFT,GOOGL
```

The `WebSocketProvider` context manages:
- Subscribed symbols
- Polling interval (5s)
- Price cache (1 min TTL)
- Rate limiting (2s minimum between fetches)

Hooks:
- `useWebSocket()` — full context access
- `useStockPrice(symbol)` — single stock subscription
- `useStockPrices(symbols[])` — multiple stock subscription

---

## 20. Unused / Deprecated Code

| File/Code | Status | Evidence |
|-----------|--------|---------|
| `hooks/use-sidebar-context.tsx` | **Unused** | Shadcn/UI sidebar used instead; no imports found in active components |
| `components/theme-provider.tsx` | **Unused** | Theme provider exists but no theme toggle in the UI |
| `pages/_app.tsx` | **Duplicate** | Both `pages/_app.tsx` and `app/layout.tsx` wrap with WebSocketProvider; dual routing |
| Polygon API code in `massiveService.js` | **Disabled** | Lines commented out with note: "Polygon has rate limit issues" |
| `alpha_vantage_data` collection name | **Legacy naming** | Collection stores FMP data, not Alpha Vantage data; name was never updated |
| `explain/renderers/__init__.py` | **Placeholder** | Empty file with TODO comment; renderers never implemented |
| Mock data fallbacks | **Legacy** | Multiple components have hardcoded mock data; should be removed when backend is stable |
| `scripts/generate_smoke_data.py` | **Test-only** | Generates fake AAPL data for smoke tests; not used in production |

---

## 21. Model Reliability & Performance Tracking

### Drift Monitoring (`scripts/drift_monitor.py`)

Runs daily after predictions. Checks for model degradation:

1. **Population Stability Index (PSI)**
   - Compares prediction distribution between "baseline" and "recent" periods
   - PSI < 0.10 -> Stable
   - PSI 0.10-0.25 -> Moderate shift (watch closely)
   - PSI > 0.25 -> Significant shift (model may need retraining)

2. **Rolling Directional Accuracy**
   - Calculates accuracy in 14-day rolling windows
   - Compares first half vs second half of the period
   - Alerts if accuracy drops below 48%

3. **Calibration Degradation (Brier Score)**
   - Measures how well probability estimates match reality
   - Alerts if recent Brier score > 1.3x baseline

4. **Alpha Magnitude Decay**
   - Tracks whether predictions are getting weaker/noisier
   - Could indicate features losing predictive power

5. **Sentiment Coverage**
   - Canary metric: checks if sentiment data is still flowing
   - Low coverage could mean API outages or scraping failures

### Model Evaluation (`scripts/evaluate_models.py`)

Two modes:

**Stored evaluation** (daily): Compares stored predictions to actual outcomes
**A/B evaluation** (manual): Trains two models (with/without sentiment) and compares

### When Models Need Retraining

Models are **retrained daily** in GitHub Actions. However, the drift monitor provides early warnings of degradation. If you see:
- PSI > 0.25 for multiple horizons -> Consider feature engineering changes
- Directional accuracy trending down -> May need more training data or new features
- Brier score increasing -> Probability calibration is off; may need classifier tuning

---

## 22. Complete File-by-File Breakdown

### Frontend Files

| File | Purpose | Key Exports |
|------|---------|-------------|
| `app/globals.css` | Global styles (Tailwind + custom) | N/A |
| `app/layout.tsx` | Root layout (WebSocketProvider, analytics) | `RootLayout`, `metadata` |
| `app/page.tsx` | Root page — imports `views/home.tsx` | `Page` |
| `app/stocks/[symbol]/page.tsx` | Dynamic stock detail route | `Page` |
| `app/news/page.tsx` | News route | `Page` |
| `app/watchlist/page.tsx` | Watchlist route | `Page` |
| `app/predictions/page.tsx` | Predictions route | `Page` |
| `app/fundamentals/page.tsx` | Fundamentals route | `Page` |
| `pages/home.tsx` | Market dashboard | `HomePage` |
| `pages/stock-detail.tsx` | Stock detail page | `StockDetail` |
| `pages/predictions.tsx` | Predictions overview | `Predictions` |
| `pages/news.tsx` | News aggregation | `NewsPage` |
| `pages/fundamentals.tsx` | Financial fundamentals | `FundamentalsPage` |
| `pages/watchlist.tsx` | User watchlist | `WatchlistPage` |
| `components/layout/layout.tsx` | Main layout (sidebar + navbar) | `Layout` |
| `components/layout/navbar.tsx` | Top navigation bar | `Navbar` |
| `components/layout/sidebar.tsx` | Side navigation + top stocks | `Sidebar` |
| `components/market/AIExplanationWidget.tsx` | Gemini explanation display | `AIExplanationWidget` |
| `components/market/EnhancedQuickPredictionWidget.tsx` | LightGBM prediction display | `EnhancedQuickPredictionWidget` |
| `components/market/market-sentiment-banner.tsx` | Fear & Greed banner | `MarketSentimentBanner` |
| `components/market/NotificationWidget.tsx` | Notification bell | `NotificationWidget` |
| `components/market/SearchWidget.tsx` | Stock search | `SearchWidget` |
| `components/market/StockLogo.tsx` | Stock logo with fallback | `StockLogo` |
| `components/market/TechnicalIndicators.tsx` | RSI/MACD/SMA gauges | `TechnicalIndicators` |
| `components/market/TradingHoursBar.tsx` | Trading hours timeline | `TradingHoursBar` |
| `components/tradingview/*.tsx` | TradingView widget wrappers | Various |
| `components/theme-provider.tsx` | Theme provider (unused) | `ThemeProvider` |
| `hooks/use-mobile.tsx` | Mobile detection | `useIsMobile` |
| `hooks/use-prefetch.tsx` | Data prefetching/caching | `usePrefetch`, `getCachedData`, `setCachedData` |
| `hooks/use-sidebar-context.tsx` | Sidebar context (unused) | `SidebarProvider`, `useSidebarContext` |
| `hooks/use-toast.ts` | Toast notifications | `useToast`, `toast` |
| `hooks/use-websocket-context.tsx` | Real-time price context | `WebSocketProvider`, `useWebSocket`, `useStockPrice` |
| `lib/api.ts` | Centralized API client | All API functions, TypeScript interfaces |
| `lib/utils.ts` | Utility (className merge) | `cn` |

### Node Backend Files

| File | Purpose | Key Exports |
|------|---------|-------------|
| `backend/src/server.js` | Server entry point | N/A |
| `backend/src/app.js` | Express configuration | Express app |
| `backend/healthcheck.js` | Container health check | N/A |
| `backend/src/config/mongodb.js` | MongoDB connection | `mongoConnection` |
| `backend/src/controllers/marketController.js` | Market endpoints | `getMarketStatus`, `getFearGreedIndex`, `getMarketSentiment` |
| `backend/src/controllers/newsController.js` | News endpoints | `getNews` |
| `backend/src/controllers/stockController.js` | Stock/prediction/explanation endpoints | `getStockDetails`, `getPredictions`, `getStoredExplanation`, etc. |
| `backend/src/controllers/watchlistController.js` | Watchlist endpoints | `getWatchlist`, `addToWatchlist`, `removeFromWatchlist`, etc. |
| `backend/src/routes/*.js` | Route definitions | Express routers |
| `backend/src/services/websocketService.js` | Finnhub WebSocket | `WebSocketService` class |
| `backend/src/services/aggregateNewsService.js` | News aggregation | `getUnifiedNews` |
| `backend/src/services/massiveService.js` | Technical indicators | `getAllIndicators` |
| `backend/src/services/marketService.js` | Market status/FGI | `fetchMarketStatus`, `fetchFearGreedIndex` |
| `backend/src/services/newsService.js` | Marketaux/NewsAPI | `getAggregateNews` |
| `backend/src/services/notificationService.js` | Notification management | `checkMarketSessionNotifications`, `getNotifications` |
| `backend/src/services/redisClient.js` | Redis client | Redis client or mock |
| `backend/src/services/rssNewsService.js` | RSS feeds | `fetchRssNews` |
| `backend/src/services/finnhubNewsService.js` | Finnhub news | `getFinnhubGeneralNews`, `getFinnhubCompanyNews` |
| `backend/src/services/tickertickNewsService.js` | TickerTick news | `getTickerTickNews` |

### ML Backend Files

| File | Purpose | Key Exports |
|------|---------|-------------|
| `ml_backend/api/main.py` | FastAPI application | FastAPI app with all endpoints |
| `ml_backend/api/routes/batch_predictions.py` | Batch prediction router | Router with batch endpoints |
| `ml_backend/api/cache.py` | Redis caching | `get_predictions_cached`, `set_predictions_cache` |
| `ml_backend/api/errors.py` | Error handling | Custom exceptions, middleware |
| `ml_backend/api/rate_limiter.py` | Rate limiting | `RateLimitMiddleware` |
| `ml_backend/api/utils.py` | Utilities | `normalize_prediction_dict`, `validate_ticker` |
| `ml_backend/config/constants.py` | Configuration | `TOP_100_TICKERS`, `MONGO_COLLECTIONS`, etc. |
| `ml_backend/config/feature_config_v1.py` | Feature/model config | All hyperparameters and feature settings |
| `ml_backend/data/ingestion.py` | Data ingestion (Yahoo Finance) | `DataIngestion` class |
| `ml_backend/data/features_minimal.py` | Feature engineering | `MinimalFeatureEngineer` class |
| `ml_backend/data/sentiment.py` | Sentiment scoring (FinBERT/RoBERTa/VADER) | `SentimentAnalyzer` class |
| `ml_backend/data/sentiment_features.py` | Sentiment -> ML features | `make_sentiment_features` |
| `ml_backend/data/insider_features.py` | Insider data -> ML features | `make_insider_features` |
| `ml_backend/data/macro.py` | Macro data fetching | `fetch_macro_data` |
| `ml_backend/data/fred_macro.py` | FRED API integration | `fetch_and_store_all_fred_indicators` |
| `ml_backend/data/economic_calendar.py` | Economic events | `EconomicCalendar` class |
| `ml_backend/data/sec_filings.py` | SEC filing analysis | `SECFilingsAnalyzer` class |
| `ml_backend/data/seeking_alpha.py` | Seeking Alpha scraper | `SeekingAlphaAnalyzer` class |
| `ml_backend/data/short_interest.py` | Short interest data | `ShortInterestAnalyzer` class |
| `ml_backend/data/cache_fetch.py` | Price data caching | `FrameCache`, `fetch_price_df_mongo_first` |
| `ml_backend/models/predictor.py` | LightGBM predictor (THE price predictor) | `StockPredictor` class |
| `ml_backend/explain/shap_analysis.py` | SHAP decomposition (math, not AI) | `run_shap_analysis`, `compute_shap_for_prediction` |
| `ml_backend/explain/budget.py` | Prompt character budget | `truncate_section`, `take_top_n` |
| `ml_backend/scripts/generate_explanations.py` | Gemini AI text generation (the explainer) | `generate_explanations` |
| `ml_backend/scripts/evaluate_models.py` | Model evaluation | `run_ab_evaluation`, `evaluate_stored_predictions` |
| `ml_backend/scripts/drift_monitor.py` | Drift detection | `run_drift_monitor` |
| `ml_backend/scripts/diagnose_sentiment.py` | Sentiment diagnostics | `diagnose_*` functions |
| `ml_backend/scripts/generate_smoke_data.py` | Smoke test data | `main` |
| `ml_backend/backtest.py` | Backtesting | `run_backtest` |
| `ml_backend/sentiment_config.py` | Sentiment config | Priority tickers, source selection |
| `ml_backend/sentiment_cron.py` | Sentiment cron job | `run_sentiment_pipeline` |
| `ml_backend/utils/mongodb.py` | MongoDB client | `MongoDBClient` class |

---

## 23. How to Run Locally

### Prerequisites
- Node.js 18+
- Python 3.11+
- MongoDB Atlas account (or local MongoDB)
- Redis (optional)

### Environment Variables

Create a `.env` file in the project root:

```bash
# MongoDB
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/stock_predictor

# API Keys (Node Backend)
FINNHUB_API_KEY=your_finnhub_key
CALENDARIFIC_API_KEY=your_calendarific_key
RAPIDAPI_KEY=your_rapidapi_key
MARKETAUX_API_KEY=your_marketaux_key
NEWSAPI_KEY=your_newsapi_key
FINANCIALDATA_API_KEY=your_financialdata_key

# API Keys (ML Backend)
FRED_API_KEY=your_fred_key
FMP_API_KEY=your_fmp_key
GOOGLE_API_KEY=your_google_gemini_key
KALEIDOSCOPE_API_KEY=your_kaleidoscope_key
REDDIT_CLIENT_ID=your_reddit_id
REDDIT_CLIENT_SECRET=your_reddit_secret

# Optional
REDIS_URL=redis://localhost:6379
ML_BACKEND_URL=http://127.0.0.1:8000
```

### Start the Frontend
```bash
cd stockpredict-ai
npm install
npm run dev
# -> http://localhost:3000
```

### Start the Node Backend
```bash
cd stockpredict-ai/backend
npm install
node src/server.js
# -> http://localhost:5000
```

### Start the ML Backend
```bash
cd stockpredict-ai/ml_backend
pip install -r requirements-prod.txt
uvicorn ml_backend.api.main:app --host 0.0.0.0 --port 8000
# -> http://localhost:8000
```

### Run the ML Pipeline Manually
```bash
# Ingest data
python -m ml_backend.data.ingestion

# Run sentiment
python -m ml_backend.sentiment_cron

# Train and predict
python -m ml_backend.scripts.run_pipeline --all-tickers

# Generate SHAP
python -m ml_backend.explain.shap_analysis --tickers AAPL MSFT

# Generate explanations
python -m ml_backend.scripts.generate_explanations
```

---

## 24. User Action Flows

### Search for a Stock
```
User types "AAPL" in SearchWidget
  -> Debounced 300ms
  -> GET /api/stock/search/AAPL (Node backend)
  -> Node backend calls Finnhub /search?q=AAPL
  -> Returns matches with symbol, name
  -> User clicks result
  -> React Router navigates to /stocks/AAPL
```

### Open Stock Detail Page
```
StockDetail page loads for AAPL
  -> Check prefetch cache for stock-details-AAPL
  -> If miss: GET /api/stock/AAPL (Node backend -> Finnhub profile + quote)
  -> GET /api/stock/AAPL/explanation (Node backend -> MongoDB prediction_explanations)
  -> GET /api/stock/AAPL/predictions (Node backend -> MongoDB stock_predictions)
  -> GET /api/news/rss?symbol=AAPL (Node backend -> Yahoo/SA RSS feeds)
  -> TradingView Advanced Chart widget loads
  -> TechnicalIndicators fetches /api/stock/AAPL/indicators
  -> Polling starts for real-time price updates
```

### Load Predictions
```
Predictions page loads
  -> GET /api/stock/batch/available (Node backend -> MongoDB distinct tickers)
  -> For each available ticker:
    -> Check prefetch cache for explanation-{ticker}
    -> If miss: GET /api/stock/{ticker}/explanation (Node backend -> MongoDB)
  -> Display prediction cards with confidence, direction, price targets
```

### Add to Watchlist
```
User clicks "Follow" on stock detail page
  -> POST /api/watchlist/default/add { symbol: "AAPL" }
  -> Node backend adds to in-memory Map
  -> POST /api/watchlist/subscribe { symbols: ["AAPL"] }
  -> WebSocket service subscribes to AAPL trade data
  -> Real-time prices start flowing for AAPL
```

### Notifications
```
NotificationWidget polls every 30 seconds
  -> GET /api/notifications?limit=20&since={lastFetch}
  -> Node backend queries MongoDB notifications collection
  -> Returns new notifications since last check
  -> Frontend displays unread count badge
  -> Types: market open/close, price alerts, high gain/loss
```

---

## 25. Risks, Bugs & Improvements

### <a id="routing-risks"></a>Routing — RESOLVED (Feb 2026)

React Router DOM has been fully removed and routing consolidated to Next.js App Router. The following risks from the previous hybrid approach **no longer apply**:

| Former Risk | Status | Resolution |
|------------|--------|------------|
| 404 errors on direct URL access | FIXED | Each route now has its own `app/*/page.tsx` |
| Bundle bloat (~45KB) | FIXED | `react-router-dom` package removed |
| SSR/hydration mismatches | FIXED | No more `BrowserRouter` or `isClient` checks |
| No code splitting per route | FIXED | Next.js automatically code-splits per route |
| SEO limitations | FIXED | Pages can be SSR'd/SSG'd by Next.js |
| Duplicate WebSocketProvider | FIXED | Single provider in `app/layout.tsx` |

**Files removed**: `app_v0modified.tsx`, `pages/_app.tsx`, `pages/stocks/[symbol].tsx`, entire `pages/` directory.
**Files added**: `views/` directory (page components), `app/*/page.tsx` route files.

**Performance and other improvements** from full Next.js (smaller bundle, per-route code splitting, prefetching, no redirect chain, SEO, single routing model) are documented in [§14 Frontend Architecture — Routing: Performance and other improvements from full Next.js](#performance-and-other-improvements-from-full-nextjs).

### Critical Issues

| Issue | Severity | Description | Suggested Fix |
|-------|----------|-------------|---------------|
| **Watchlist in-memory** | High | Watchlists stored in JS Map, lost on restart | Persist to MongoDB |
| **No authentication** | High | All endpoints are public, userId is "default" | Add auth (NextAuth, Clerk, etc.) |

### Performance Issues

| Issue | Impact | Suggested Fix |
|-------|--------|---------------|
| Frontend polling every 5s | Battery/bandwidth on mobile | Use true WebSocket or Server-Sent Events |
| TradingView widgets not code-split | Heavy load on homepage | Load only visible widgets |

### Gemini API Rate Limits (Free Tier)

**Problem**: Google Gemini API free tier has strict rate limits that can block explanation generation for all 100 tickers:

| Model | Free Tier Limits | Impact |
|-------|------------------|--------|
| **gemini-2.5-flash** | 5 RPM, 250K TPM, **20 RPD** | ❌ Only 20 requests/day — insufficient for 100 tickers |
| **gemini-2.5-pro** | 15 RPM, Unlimited TPM, **1.5K RPD** | ✅ 1,500 requests/day — sufficient for daily batch (100 tickers) |
| **gemini-2.5-flash-lite** | 10 RPM, 250K TPM, **20 RPD** | ❌ Only 20 requests/day — last-resort fallback |

**Solutions implemented** (Feb 2026):

1. **Automatic model fallback chain** (`_pick_model()`) — Cycles through `gemini-2.5-pro` → `gemini-2.5-flash` → `gemini-2.5-flash-lite` based on in-process RPD usage tracking. Each model has a headroom buffer (e.g., 1,400 of 1,500 for pro) to avoid hitting hard limits.
2. **RPD tracking** (`_model_rpd_count`) — Tracks per-model request counts within a single pipeline run. When a model approaches its limit, the script automatically falls back to the next model.
3. **Check for existing explanations** — Script now checks MongoDB for existing explanations for today's date before calling Gemini API, avoiding redundant API calls
4. **Quota exceeded handling** — After 3 consecutive quota failures, script stops processing remaining tickers gracefully (no wasted API calls)
5. **GitHub Actions workflow** — Does NOT force `GEMINI_MODEL`, allowing the automatic fallback chain to work. Previously was locked to `gemini-2.5-flash` (20 RPD), now uses the full chain starting with pro.

**For production**: Consider upgrading to Gemini API paid tier for:
- Higher rate limits (unlimited RPD on paid tier)
- Better reliability
- Priority support

**Alternative solutions** (if staying on free tier):
- Use OpenRouter free models (e.g., Gemma 3, DeepSeek R1) — see `openrouter.ai/collections/free-models`
- Self-host lightweight LLMs (e.g., Llama 3.2, Gemma 2B) for explanation generation
- Implement template-based fallback explanations when API quota is exceeded

### Data Quality Issues

| Issue | Impact | Suggested Fix |
|-------|--------|---------------|
| Sentiment often 0 for many stocks | Weaker predictions | Add more news sources; lower thresholds |
| Seeking Alpha scraping fragile | Data gaps when Playwright fails | Add retry logic; monitor scraping health |
| Short interest only from Nasdaq/Finviz | Limited coverage, scraping-dependent | Add ORTEX or S3 Partners as source |
| `alpha_vantage_data` collection misnamed | Confusion for developers | Rename collection or add clear comments |

### Suggested Improvements (Prioritized)

1. **Persist watchlists to MongoDB** — Prevent data loss on restart
2. **Add authentication** — Protect user data and prevent abuse
3. ~~**Consolidate routing**~~ — DONE (Feb 2026): React Router removed, Next.js App Router is now the sole routing system
4. **Switch to true WebSocket on frontend** — Reduce polling overhead
5. **Add model version tracking** — Track which model version generated each prediction
6. **Add prediction accuracy display** — Show users historical accuracy for transparency
7. **Add database migrations** — Track schema changes over time
8. **Add health monitoring dashboard** — Track API health, model drift, data freshness
9. **Add integration tests** — Test the full pipeline end-to-end
10. **Rename `alpha_vantage_data` collection** — Align naming with actual contents (FMP data)
11. ~~**Add data retention policy**~~ — DONE (Feb 2026): `ml_backend/scripts/data_retention.py` runs daily in GitHub Actions, removes data older than 12 months

### Pipeline Health Summary (`run_pipeline.py`)

The `PipelineHealthSummary` class tracks the status of every pipeline stage and prints a clear summary at the end of each run. This turns "it didn't crash" into "we know exactly what happened."

```
========================================================
  PIPELINE HEALTH SUMMARY
========================================================
  MongoDB connected:        YES
  Historical data fetched:  100 tickers
  Training status:          success (100 tickers)
  Backtest status:          skipped
  Predictions stored:       100 tickers x 3 horizons
  SeekingAlpha scraped:     SKIPPED (not in pipeline)
  Gemini explanations:      not run (separate script)
  Evaluation samples found: 0 (expected early)
========================================================
```

Tracked fields: `mongo_connected`, `data_fetched`, `data_failed`, `training_status`, `training_tickers`, `backtest_status`, `predictions_stored`, `predictions_failed`, `predictions_skipped`, `horizons_used`, `seeking_alpha_status`, `gemini_explanations`, `evaluation_samples`.

### Hardened Prediction History Methods (`utils/mongodb.py`)

- **`get_prediction_history()`** — Logs the effective query range (`start`/`end`) and result count at `INFO` level on every call. Used for drift detection and accuracy tracking.
- **`get_prediction_history_simple()`** — Raises `TypeError` if caller passes `start_date` or `end_date` via `**kwargs`. This guards against accidentally using the simple helper where the full method is needed.

### Code Fixes Applied in This Review

The following code changes were made to address data storage gaps and pipeline issues:

| Fix | File | What Changed |
|-----|------|-------------|
| **Store Finnhub basic financials & peers** | `ml_backend/data/sentiment.py` | `analyze_finnhub_sentiment()` now calls `get_finnhub_basic_financials()` and `get_finnhub_company_peers()` during the sentiment pipeline, storing results in MongoDB |
| **Store short interest raw data** | `ml_backend/data/short_interest.py` | Added `_store_short_interest_raw()` method; `fetch_short_interest()` now persists raw records to new `short_interest_data` MongoDB collection |
| **Decouple insider features** | `ml_backend/config/feature_config_v1.py` + `ml_backend/data/features_minimal.py` | Added `USE_INSIDER_FEATURES` flag separate from `USE_SENTIMENT_FEATURES`, allowing independent A/B testing of insider vs sentiment features |
| **Gemini reads Finnhub financials** | `ml_backend/scripts/generate_explanations.py` | Added `_get_financials_context()` — reads P/E, P/B, dividend yield, ROE, market cap, beta from `finnhub_basic_financials` collection |
| **Gemini reads FMP earnings/ratings** | `ml_backend/scripts/generate_explanations.py` | Added `_get_fmp_context()` — reads latest earnings (EPS actual vs estimated, surprise), analyst ratings, and price targets from `alpha_vantage_data` collection |
| **Gemini reads short interest directly** | `ml_backend/scripts/generate_explanations.py` | Updated `_get_short_interest_context()` to read from `short_interest_data` collection first (more granular), with fallback to `sentiment` collection |
| **Gemini API rate limit handling** | `ml_backend/scripts/generate_explanations.py` + `ml_backend/api/main.py` | Default model switched to `gemini-2.5-pro` (1.5K RPD vs 20 for flash); added check for existing explanations to avoid redundant API calls; added quota exceeded detection and graceful stopping |
| **Gemini API error handling** | `ml_backend/scripts/generate_explanations.py` + `ml_backend/api/main.py` | `_call_gemini()` now returns error type (`quota_exceeded`, `rate_limit`, `api_error`); batch script stops processing when quota exceeded; API route returns clear error messages |
| **Stock-specific Gemini prompts** | `ml_backend/scripts/generate_explanations.py` | Added `STOCK_META` dictionary (100 tickers → company name, sector, industry). Prompt now addresses each stock by name, includes sector-specific guidance, and requires every KEY_DRIVER to cite a specific data value |
| **Gemini model auto-fallback** | `ml_backend/scripts/generate_explanations.py` + `.github/workflows/daily-predictions.yml` | `_pick_model()` auto-selects best model via in-process RPD tracking (pro→flash→flash-lite). Workflow no longer forces `GEMINI_MODEL=gemini-2.5-flash` |
| **Pipeline health summary** | `ml_backend/scripts/run_pipeline.py` | `PipelineHealthSummary` class tracks every pipeline stage and prints clear summary at end of run |
| **Hardened prediction history** | `ml_backend/utils/mongodb.py` | `get_prediction_history()` logs effective query range; `get_prediction_history_simple()` rejects `start_date`/`end_date` kwargs with `TypeError` |
| **CONFIDENCE in AI output** | `components/market/AIExplanationWidget.tsx` | Parser extracts the new `CONFIDENCE:` section from Gemini output |
| **News in AI Insights** | `views/stock-detail.tsx` + `components/market/AIExplanationWidget.tsx` | Stock detail view now passes `recentNews` prop to AI widget; widget displays news with sentiment indicators even when AI text lacks specific headlines |

---

## 26. Complete Data Pipeline Audit (Feb 2026)

### What Gets Fetched and Stored in MongoDB Every Daily Pipeline Run?

This audit traces the complete data flow: **API fetch → MongoDB storage → ML feature engineering → LightGBM prediction** and separately **MongoDB → Gemini explanation prompt**.

#### A. Sentiment Cron (Step 1: `ml_backend.sentiment_cron`)

Calls `get_combined_sentiment()` for each of the 100 S&P tickers, which fetches from **10 sources**:

| # | Source | Method | Stored In (MongoDB) | Fields Stored |
|---|--------|--------|-------------------|---------------|
| 1 | FinViz | `get_finviz_sentiment()` | `sentiment.finviz_raw_data` | Headlines, sentiment scores |
| 2 | SEC EDGAR | `get_sec_sentiment()` | `sentiment.sec_raw_data` + `sec_filings` | Filing type, text, sentiment |
| 3 | Marketaux | `get_marketaux_sentiment()` | `sentiment.marketaux_raw_data` | Headlines, entities, sentiment |
| 4 | RSS (Yahoo+SeekingAlpha) | `get_rss_news_sentiment()` | `sentiment.rss_news_raw_data` | Headlines, dates, sentiment |
| 5 | Reddit | `get_reddit_sentiment()` | `sentiment.reddit_raw_data` | Post titles, scores, sentiment |
| 6 | FMP | `get_fmp_sentiment()` | `sentiment.fmp_raw_data` + `alpha_vantage_data` | Earnings, dividends, ratings, price targets, analyst estimates |
| 7 | Finnhub | `get_finnhub_sentiment()` | `sentiment.finnhub_raw_data` + `insider_transactions` + `finnhub_recommendation_trends` + `finnhub_basic_financials` + `finnhub_company_peers` | Insider trades, analyst recs, P/E, market cap, beta, peers |
| 8 | SeekingAlpha Comments | `get_seekingalpha_comments_sentiment()` | `sentiment.seekingalpha_comments_raw_data` | Comments, NLP scores |
| 9 | Economic Calendar | `integrate_economic_events_sentiment()` | `sentiment.economic_event_sentiment` + `economic_events` | Event names, dates, impact |
| 10 | Short Interest | `analyze_short_interest_sentiment()` | `sentiment.short_interest_data` + `short_interest_data` | Short float %, days to cover, settlement dates |

Final stored document in `sentiment` collection also includes:
- `composite_sentiment` — blended score across all sources
- `news_count` — total article/post count
- `date` — aligned to NYSE trading day
- `api_status` — per-source health tracking

#### B. Training (Step 2: `run_pipeline --all-tickers --no-predict`)

Feature engineering in `features_minimal.py` reads these data for each ticker:

| Feature Group | Source | MongoDB Collection | Fields Actually Used | Unused Fields |
|--------------|--------|-------------------|---------------------|---------------|
| Price technicals | yfinance/MongoDB | `historical_data` | OHLCV → RSI, MACD, Bollinger, SMA, log returns, volume ratios | None |
| Cross-asset | yfinance/MongoDB | `historical_data` (for SPY, VIX, sector ETFs) | Returns, volatility, relative strength | None |
| FRED Macro | FRED API (on-demand) | `macro_data` / `macro_data_raw` | `TREASURY_10Y`, `TREASURY_2Y`, `FEDERAL_FUNDS_RATE` | **10 unused: GDP, CPI, UNEMPLOYMENT, INFLATION, NONFARM_PAYROLL, RETAIL_SALES, DURABLES, TREASURY_30Y, REAL_GDP** |
| Sentiment | MongoDB | `sentiment` | `composite_sentiment`, `news_count` | **All per-source scores (finviz, sec, reddit, etc.), all raw data fields** |
| Insider | MongoDB | `insider_transactions` | `filingDate`, `transactionCode`, `change`, `share`, `transactionPrice` → 11 rolling aggregate features | None |
| Earnings proximity | MongoDB | `fundamentals_events` / `sentiment` | `earnings` dates → `days_since_earnings` | **EPS actual/estimated, revenue, earnings surprise** |

#### C. Data That IS Stored but NOT Used as ML Features (Improvement Opportunities)

| Data Source | MongoDB Collection | Stored? | Used as ML Feature? | Could Improve Predictions? | Priority |
|------------|-------------------|---------|--------------------|-----------------------------|----------|
| FMP earnings surprise (beat/miss) | `alpha_vantage_data` | YES | NO | YES — post-earnings drift is well-documented | High |
| FMP analyst estimate revisions | `alpha_vantage_data` | YES | NO | YES — estimate momentum is predictive | High |
| FMP price targets (vs current) | `alpha_vantage_data` | YES | NO | YES — target gap signals analyst consensus | Medium |
| Finnhub P/E, ROE, margins | `finnhub_basic_financials` | YES | NO | YES — valuation features are fundamental | Medium |
| Finnhub recommendation trends | `finnhub_recommendation_trends` | YES | NO | YES — strongBuy/sell counts signal analyst sentiment | Medium |
| Short interest ratio | `short_interest_data` | YES | NO | YES — high short float signals crowded trades | Medium |
| Economic calendar proximity | `economic_events` | YES | NO | YES — days to/from FOMC, CPI releases | Medium |
| Per-source sentiment scores | `sentiment` | YES | NO (only composite used) | MAYBE — per-source may add noise | Low |
| SEC filing sentiment | `sec_filings` | YES | NO | MAYBE — filing-specific signal | Low |
| FRED GDP/CPI/Unemployment | `macro_data_raw` | YES (if fetched) | NO | YES — regime indicators | Medium |

**Key takeaway**: The pipeline stores far more data than it uses for features. The LightGBM model currently uses **~40 features** from price, cross-asset, 3 FRED indicators, 2 sentiment aggregates, 11 insider aggregates, and 1 earnings proximity. There are **~15-20 additional features** that could be engineered from already-stored data.

#### D. What the Gemini Explanation Now Reads from MongoDB

After the fixes applied in this review, `generate_explanations.py` reads:

| # | Data Source | MongoDB Collection | Status |
|---|-----------|-------------------|--------|
| 1 | ML Predictions (all 3 windows) | `stock_predictions` | Was reading, still reading |
| 2 | Blended sentiment + per-source scores + raw headlines | `sentiment` | Was reading, still reading |
| 3 | Technical indicators (RSI, MACD, Bollinger, SMAs, performance) | Calculated from `historical_data` | Was reading, still reading |
| 4 | SHAP feature importance (top positive/negative/global) | `feature_importance` | Was reading, still reading |
| 5 | Macro economic indicators (fed funds, CPI, unemployment, yields) | `macro_data_raw` / `macro_data` | Was reading, still reading |
| 6 | Insider trading activity (recent buys/sells) | `insider_transactions` | Was reading, still reading |
| 7 | Short interest (short float %, days to cover) | `short_interest_data` (direct) → `sentiment` (fallback) | **IMPROVED** — now reads dedicated collection first |
| 8 | Finnhub basic financials (P/E, P/B, ROE, dividend yield, market cap, beta) | `finnhub_basic_financials` | **NEW** — added `_get_financials_context()` |
| 9 | FMP earnings (EPS actual vs estimated, surprise, beat/miss) | `alpha_vantage_data` (endpoint: `fmp_earnings`) | **NEW** — added `_get_fmp_context()` |
| 10 | FMP analyst ratings (score, recommendation) | `alpha_vantage_data` (endpoint: `fmp_ratings-snapshot`) | **NEW** — added `_get_fmp_context()` |
| 11 | FMP analyst price targets (avg target, count) | `alpha_vantage_data` (endpoint: `fmp_price-target-summary`) | **NEW** — added `_get_fmp_context()` |

**Still not read** (lower priority — could be added later):
- SEC filing text/sentiment (from `sec_filings`) — would add regulatory context
- Finnhub company peers (from `finnhub_company_peers`) — would add peer comparison
- SeekingAlpha raw comments (already partially included via `sentiment.seekingalpha_comments_raw_data`)

---

## 26. Pipeline Hardening & Reliability

This section documents the reliability mechanisms built into the pipeline to prevent silent failures and data degradation.

### API Retry Logic

All external API calls use retry-with-backoff to handle transient errors (rate limits, server errors, timeouts).

| Component | File | Retry Strategy | Retryable Errors | Max Attempts |
|-----------|------|---------------|-----------------|-------------|
| Finnhub API (Basic Financials, Peers, Insider Sentiment) | `sentiment.py` | `_finnhub_get_with_retry()` — async, exponential backoff with jitter, honors `Retry-After` on 429 | HTTP 429, 500, 502, 503, 504, `asyncio.TimeoutError`, `aiohttp.ClientError` | 3 |
| FRED API (all 13 macro indicators) | `fred_macro.py` | `fetch_fred_series()` — exponential backoff with jitter, re-raises config errors immediately | All exceptions except `ValueError` | 2 |
| MongoDB (reads + writes) | `mongodb.py` | `_retry_operation()` — exponential backoff with jitter for all CRUD | `AutoReconnect`, `ConnectionFailure`, `ServerSelectionTimeoutError`, `NetworkTimeout` | 3 (via `RETRY_CONFIG`) |

**Log format convention**: All retry logs use a `[RETRY]`/`[FINNHUB-RETRY]`/`[FRED-RETRY]` prefix for easy CI log scanning. Example:
```
[RETRY] op=bulk_write predictions for AAPL | attempt=2/3 | sleep=2.1s | err=AutoReconnect(...)
[FINNHUB-RETRY] basic_financials AAPL | status=429 | attempt=1/3 | body=...
```

**Error logging**: All exception handlers use `repr(e)` instead of `str(e)` to prevent blank error messages when exceptions have no message (e.g., bare `aiohttp.ClientError()`).

### Feature-Name Enforcement

The LightGBM predictor requires exact feature-column alignment between training and inference.

**File**: `predictor.py` — `_select_features()` method

| Scenario | Behavior |
|----------|----------|
| **Columns match** | Returns DataFrame unchanged |
| **Extra columns in current data** | Ignored (debug-logged) |
| **Missing columns** (<= 50%) | Zero-filled with missing-data marker; warning logged listing which columns are missing |
| **Missing columns** (> 50%) | Prediction **skipped** for that ticker/window with `reason: "feature_mismatch"` — prevents garbage predictions |

### Quality Gate

The pipeline asserts minimum success thresholds before exiting successfully.

**File**: `run_pipeline.py` — `PipelineHealthSummary.check_quality_gate()`

| Gate | Default Threshold | Env Var | Description |
|------|-------------------|---------|-------------|
| **Prediction rate** | ≥ 80% | `QG_MIN_PREDICTION_RATE` | At least 80% of tickers must have predictions stored |
| **Data failure rate** | ≤ 20% | `QG_MAX_DATA_FAILURE_RATE` | At most 20% of tickers can fail data fetch |

When the quality gate fails, the pipeline exits with `sys.exit(1)` and prints a `QUALITY GATE: FAILED` banner listing which thresholds were breached. This ensures GitHub Actions marks the run as failed rather than silently degrading.

The quality gate only applies when predictions are expected (i.e., not when running with `--no-predict`).

### MongoDB Connection Hardening

**File**: `mongodb.py`

- **Connection timeouts** relaxed for Atlas under CI load: `serverSelectionTimeoutMS=15000`, `connectTimeoutMS=20000`, `socketTimeoutMS=45000`
- **Connection pool**: `maxPoolSize=50`, `minPoolSize=10`
- **Retry**: `retryWrites=True`, `retryReads=True` (driver-level) + `_retry_operation()` wrapper (application-level)

---

## 27. API Rate Limit Compliance & Data Priority

This section documents the centralized rate-limiting layer added to protect every external API from throttling, bans, or silent data loss — **while guaranteeing that the most prediction-critical data is always fetched first**.

### 27.1 Per-API Rate Limits (Enforced)

| API | Plan | Hard Limit | Enforced Throttle | Mechanism | File |
|-----|------|-----------|-------------------|-----------|------|
| **Finnhub** | Free | 60/min overall, 30/sec | 55/min + 25/sec burst | `AsyncRateLimiter` token-bucket | `rate_limiter.py` |
| **FMP** | Free | 250 total calls, 4/sec | 3/sec + 4 endpoints only | `AsyncRateLimiter` + endpoint reduction | `rate_limiter.py`, `sentiment.py` |
| **Marketaux** | Free | 100/day | 95/day hard budget | `DailyBudgetLimiter` + top-50 tickers | `rate_limiter.py`, `sentiment.py` |
| **Reddit** | Free (OAuth) | 100 QPM | 90/min + max 3 subreddits/ticker | `AsyncRateLimiter` | `rate_limiter.py`, `sentiment.py` |
| **FRED** | Free | ~120/min | 100/min (defensive) | `AsyncRateLimiter` sync mode | `rate_limiter.py`, `fred_macro.py` |
| **Alpha Vantage** | Free | 5/min, 25/day | N/A (disabled) | Methods gutted in code | `sentiment.py` |

### 27.2 Rate Limiter Architecture

**File**: `ml_backend/utils/rate_limiter.py`

Two classes:

1. **`AsyncRateLimiter`** — Sliding-window token-bucket with optional burst cap.
   - `acquire()` — async; sleeps until a token is available (no request dropped).
   - `acquire_sync()` — blocking version for synchronous callers (e.g., FRED).
   - Configurable: `max_calls`, `period_seconds`, `burst_limit`.

2. **`DailyBudgetLimiter`** — Hard daily cap resetting at UTC midnight.
   - `acquire()` — async; raises `BudgetExhausted` when daily cap is hit.
   - Logs `[MARKETAUX-BUDGET]` warnings when ≤10 calls remain.

Global singletons are instantiated at module level and imported by calling modules:
```python
finnhub_limiter    = AsyncRateLimiter(55, 60, burst_limit=25, name="Finnhub")
fmp_limiter        = AsyncRateLimiter(3,  1,  name="FMP")
marketaux_limiter  = DailyBudgetLimiter(95, name="Marketaux")
reddit_limiter     = AsyncRateLimiter(90, 60, name="Reddit")
fred_limiter       = AsyncRateLimiter(100, 60, name="FRED")
```

### 27.3 Data Priority — Ensuring Best Prediction Accuracy Under Rate Limits

The `get_combined_sentiment()` method in `sentiment.py` calls sources **sequentially** in a deliberate priority order. The blend weights determine each source's contribution to prediction accuracy:

| Priority | Source | Blend Weight | API Type | Rate-Limited? | Notes |
|----------|--------|-------------|----------|---------------|-------|
| 1 | **FinViz** | 5% | Free scrape | No | Always succeeds — no API key needed |
| 2 | **SEC Filings** | 10% | Free scrape | No | Always succeeds — no API key needed |
| 3 | **Marketaux** | 15% | API (95/day) | Yes (daily budget) | Top 50 tickers only; skipped for others |
| 4 | **RSS News** (Yahoo+SA) | **22%** | Free RSS | No | **Highest weight — always runs first among API sources** |
| 5 | **Reddit** | 10% | OAuth API | Yes (90/min) | Capped at 3 subreddits per ticker |
| 6 | **FMP** | 8% | API (3/sec) | Yes | Reduced to 4 high-value endpoints |
| 7 | **Finnhub** | 10% + 10% insider | API (55/min) | Yes | Core calls (insider+recommendations) run before non-critical (basic_financials, peers) |
| 8 | **SA Comments** | 5% | Free scrape | No | Always succeeds |

**Key guarantee**: Even if all rate-limited APIs are exhausted, **42% of blend weight** (FinViz 5% + SEC 10% + RSS News 22% + SA Comments 5%) comes from **free, unlimited sources** that never fail due to rate limits. Adding Marketaux's top-50-ticker budget, **57% of blend weight** is available before any rate-limited source could be cut off.

**Within Finnhub** (5 calls per ticker), the call order is:
1. `insider_transactions` — contributes to `finnhub_insider_sentiment` (10% weight) ✅
2. `recommendation_trends` — contributes to `finnhub_sentiment` (10% weight) ✅
3. `basic_financials` — supplementary context only (0% blend weight, used in explanations)
4. `company_peers` — supplementary context only (0% blend weight, used in explanations)
5. `insider_sentiment` (MSPR) — supplementary context only

Calls 3-5 are wrapped in try/except and marked as "non-critical" in code — failures are logged but never block the sentiment result.

**Within FMP** (reduced from 8 to 4 endpoints), only the high-value endpoints remain:
- `analyst_estimates` — analyst earnings estimates
- `ratings_snapshot` — overall rating scores
- `price_target_summary` — consensus price targets
- `price_target_consensus` — target consensus

Dropped endpoints (lower sentiment value): `dividends`, `dividends_calendar`, `earnings`, `earnings_calendar`.

### 27.4 Pipeline Pacing

| Component | Pacing | Purpose |
|-----------|--------|---------|
| `sentiment_cron.py` | `asyncio.sleep(1.0)` after each ticker (in `finally`) | Spreads 100 tickers over ~100s minimum |
| `sentiment_cron.py` | `CONCURRENCY=3` (Semaphore) | Max 3 tickers processed in parallel |
| `daily-predictions.yml` | `sleep 10` between prediction batches | Paces Finnhub calls across 10 batches |
| `run-daily-pipeline-local.ps1` | `Start-Sleep -Seconds 10` between batches | Mirrors CI pacing locally |

### 27.5 Post-Throttling Call Budget

| API | Calls/Run | Limit | Headroom |
|-----|-----------|-------|---------|
| Finnhub | 500 (paced at 55/min) | 60/min | ~9% margin |
| FMP | ~240 (4 endpoints × 60 tickers) | 250 total | ~4% margin |
| Marketaux | 50 (top-50 tickers) | 100/day | 50% margin |
| Reddit | ~300 (3 subs × 100 tickers, paced 90/min) | 100 QPM | 10% margin |
| FRED | 13 indicators (one-time) | 120/min | ~89% margin |

### 27.6 Env Vars for Rate Limiting

No new env vars are required. The rate limits are hardcoded in `rate_limiter.py` as conservative defaults. The Marketaux top-50 ticker list is defined as `_MARKETAUX_TICKERS` in the `SentimentAnalyzer` class.

To adjust limits, modify the singleton instantiation in `rate_limiter.py`.

---

## 28. Technical Deep Dive: The AI Machine

This section explains the "Black Magic" of how a free GitHub Actions runner can handle a massive 100-stock AI pipeline daily without crashing, timing out, or getting banned.

### 28.1 The Orchestrator: Disposable Virtual Machines
GitHub Actions provides a **standard Ubuntu VM** (approx. 7GB RAM, 2 CPUs) for every run. 
- **Disposable**: Every VM starts fresh. Any files written to disk are deleted at the end of the run.
- **Stateless**: The system doesn't "remember" yesterday's run unless it reads from MongoDB.
- **Life Limit**: A single run can last up to 6 hours. Our pipeline optimizes this down to ~40 minutes.

### 28.2 Parallel Data Fetching via Concurrency
Fetching news, sentiment, and prices for 100 stocks sequentially would take hours.
- **Asyncio Semaphores**: `sentiment_cron.py` uses `asyncio.Semaphore(3)`. This means the machine works on **3 stocks at the exact same time**.
- **Bounded Parallelism**: We use 3 (not 100) to avoid overwhelming the CPU and getting IP-banned by news sources.

### 28.3 Hybrid Model Strategy (Pooled vs. Per-Ticker)
Training 100 deep-learning models would exceed the 7GB RAM limit. We use a more efficient approach:
- **The Pooled Model**: We train one large "Base Model" using the historical data of *all* 100 stocks combined. This lets the AI learn general market patterns (e.g., "When inflation rises, Tech stocks usually drop").
- **LightGBM**: This algorithm is used because it is an order of magnitude faster and lighter than Neural Networks, while maintaining top-tier accuracy for tabular data.
- **Personalization**: The model uses "Ticker Labels" as a feature, allowing it to learn the unique "personality" of AAPL vs. JPM without needing 100 separate files.

### 28.4 Batched Prediction & The "Cool Down" Cycle
Calculating predictions and **SHAP explanations** is the most CPU-intensive part. 
- **Batching**: We split the 100 stocks into **10 batches of 10**.
- **The Loop**: The workflow pick 10 tickers, runs the AI, saves data, then **sleeps for 10 seconds**.
- **The Wait**: This 10-second pause is critical. It prevents the GitHub runner's CPU from thermal throttling and ensures external APIs (like Finnhub) have time to reset their rate limits.

---

## 29. Stateless-to-Stateful Bridge: The Role of MongoDB

GitHub Actions is **stateless** (it forgets everything), but a Stock platform needs to be **stateful** (it must remember prices and trends).

### 29.1 MongoDB as the "Persistent Brain"
The MongoDB Atlas cluster acts as the bridge between the **Worker VM** (GitHub) and the **Frontend** (Your browser).
1. **GitHub** fetches data → Saves it to MongoDB.
2. **GitHub** trains model → Saves weights/metrics to MongoDB.
3. **Frontend** loads → Reads directly from MongoDB.

The Frontend never talks to GitHub; it just reads the "Knowledge" that the worker left behind in the database.

### 29.2 Resiliency via Persistence
If the GitHub VM crashes halfway through Batch #5:
- All data from Batches #1 to #4 is **already safe** in MongoDB. 
- The Frontend will continue showing the latest successful data.
- The next day's run will simply overwrite the old data with fresh numbers.

---

*End of documentation. Last updated 2026-02-21.*
