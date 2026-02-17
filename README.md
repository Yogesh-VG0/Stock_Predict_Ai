# StockPredict AI

> Full-stack stock analytics and ML predictions for S&P 100 companies — real-time data, 10+ sentiment sources, and plain-English AI explanations.

[![Next.js](https://img.shields.io/badge/Next.js-15-black?style=flat-square&logo=next.js)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?style=flat-square&logo=typescript)](https://www.typescriptlang.org/)
[![Node](https://img.shields.io/badge/Node.js-18+-339933?style=flat-square&logo=node.js)](https://nodejs.org/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python)](https://www.python.org/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=flat-square&logo=mongodb)](https://www.mongodb.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-ML%20API-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Predictor-FFCC00?style=flat-square)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/Yogesh-VG0/stockpredict-ai?style=flat-square)](https://github.com/Yogesh-VG0/stockpredict-ai)

---

## Overview

StockPredict AI predicts stock prices for **100 S&P 100 stocks** across three horizons (1 day, 7 days, 30 days), explains predictions in plain English, and surfaces real-time market data. A **daily automated pipeline** (GitHub Actions) fetches data from 10+ sources, trains models, generates predictions, runs SHAP analysis, and uses Google Gemini to write human-readable explanations — all stored in MongoDB and served by a Node.js API.

---

## Features

| Feature | Description |
|--------|-------------|
| **Price predictions** | LightGBM forecasts with confidence scores and trade recommendations |
| **AI explanations** | SHAP decomposes the model; Gemini turns it into short, readable summaries |
| **Sentiment** | 10+ sources (Finviz, Reddit, SEC, Finnhub, FMP, Marketaux, RSS, etc.) scored with FinBERT, RoBERTa, VADER |
| **Live data** | Real-time quotes via Finnhub WebSocket; TradingView charts; technical indicators |
| **Watchlist & alerts** | Track symbols; get real-time price updates |
| **News** | Unified feed from Yahoo, Seeking Alpha, Finnhub, Marketaux, and more |

---

## How it works

Three distinct pieces work together:

1. **Predictor (LightGBM)** — The only component that predicts prices. Uses 42+ features (price history, sentiment, macro data, insider activity) and outputs predicted returns.
2. **Explainer (SHAP + Gemini)** — SHAP breaks down *why* the model predicted what it did; Gemini writes a plain-English summary for the UI.
3. **Sentiment scorers (FinBERT, RoBERTa, VADER)** — Score news and social content; these scores become **input features** for the predictor, not predictions themselves.

```
News/Reddit/SEC → Sentiment models → scores
Price + macro + sentiment → LightGBM → predicted return → SHAP → Gemini → explanation text
```

---

## Architecture

| Layer | Role |
|-------|------|
| **Frontend** (Next.js, port 3000) | UI, TradingView charts, search, watchlist, stock pages |
| **Node backend** (Express, port 5000) | API gateway, news aggregation, watchlist, proxies ML endpoints, Finnhub WebSocket |
| **ML backend** (FastAPI, port 8000) | Predictions, sentiment, model training, SHAP, Gemini |
| **MongoDB** | Historical data, sentiment, predictions, explanations, insider trades, macro data |
| **Redis** (optional) | Caching, rate limiting, holiday cache |

---

## Daily pipeline

Runs nightly via GitHub Actions (~6:15 PM ET):

1. **Gather sentiment** — Fetch from 10+ sources; score with NLP; store in MongoDB (~5 min)
2. **Train** — Ingest OHLCV, engineer 42+ features; train LightGBM per horizon (~15 min)
3. **Predict** — Run models on all 100 tickers; store predictions (~20 min)
4. **Explain** — SHAP analysis; Gemini writes explanations from predictions + sentiment + news (~15 min)
5. **Evaluate** — Compare predictions to actuals; drift monitoring (~5 min)

---

## Tech stack

| Layer | Technologies |
|-------|--------------|
| **Frontend** | Next.js 15, React 18, TypeScript, Tailwind, Framer Motion, TradingView, Recharts |
| **Backend** | Node.js, Express, MongoDB, Redis |
| **ML** | Python 3.11+, FastAPI, LightGBM, SHAP, Google Gemini, yfinance, FinBERT, RoBERTa, VADER |

---

## Data & APIs

- **Finnhub** — Quotes, profiles, search, news, insider trades, WebSocket prices
- **Yahoo Finance** — Historical OHLCV, RSS news
- **FRED** — Macro indicators (rates, yields)
- **FMP** — Earnings, analyst estimates, ratings, price targets
- **Marketaux, Reddit, SEC, Seeking Alpha** — Sentiment and news
- **Google Gemini** — Explanation generation

---

## Quick start

**Prerequisites:** Node.js 18+, Python 3.11+, MongoDB Atlas, API keys (Finnhub, FMP, FRED, Google Gemini, etc.)

```bash
git clone https://github.com/Yogesh-VG0/stockpredict-ai.git
cd stockpredict-ai
npm install
```

Create `.env` with `MONGODB_URI`, `FINNHUB_API_KEY`, `FMP_API_KEY`, `FRED_API_KEY`, `GOOGLE_API_KEY`, and others (see [DOCUMENTATION.md § How to Run Locally](DOCUMENTATION.md#23-how-to-run-locally)).

```bash
# Terminal 1 — Frontend
npm run dev                    # → localhost:3000

# Terminal 2 — Node API
cd backend && npm install && node src/server.js   # → localhost:5000

# Terminal 3 — ML API
cd ml_backend && pip install -r requirements-prod.txt
uvicorn ml_backend.api.main:app --host 0.0.0.0 --port 8000   # → localhost:8000
```

---

## Full documentation

**[DOCUMENTATION.md](DOCUMENTATION.md)** — Architecture, data flow, APIs, database schemas, run instructions, pipeline details, and file-by-file breakdown.

---

[MIT](LICENSE) · *Built by [Yogesh Vadivel](https://github.com/Yogesh-VG0)*
