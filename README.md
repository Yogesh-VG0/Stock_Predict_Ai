<p align="center">
  <img src="public/favicon.ico" alt="StockPredict AI" width="80" />
</p>

<h1 align="center">StockPredict AI</h1>

<p align="center">
  <strong>AI-powered stock prediction platform for S&P 75 companies</strong><br/>
  LightGBM + LSTM · 113 engineered features · SHAP explanations · Groq AI insights
</p>

<p align="center">
  <a href="https://stockpredict.dev"><strong>🚀 Live Demo →</strong></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Next.js-15-black?style=flat-square&logo=next.js" alt="Next.js" />
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/LightGBM-Predictor-FFCC00?style=flat-square" alt="LightGBM" />
  <img src="https://img.shields.io/badge/License-AGPL--3.0-blue.svg?style=flat-square" alt="License" />
</p>

---

## 📸 Screenshots

<p align="center">
  <img src="public/screenshots/dashboard.png" alt="Dashboard - Stock predictions overview" width="100%" />
  <br/><em>Dashboard with real-time stock predictions and market sentiment</em>
</p>

<p align="center">
  <img src="public/screenshots/stock-detail.png" alt="Stock Detail Page" width="100%" />
  <br/><em>Detailed stock analysis with AI explanations and TradingView charts</em>
</p>

<p align="center">
  <img src="public/screenshots/predictions.png" alt="Predictions Grid" width="100%" />
  <br/><em>All 75 stock predictions with confidence scores and price targets</em>
</p>

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **ML Predictions** | 1-day, 7-day, 30-day forecasts with confidence scores and price ranges |
| 💬 **AI Explanations** | Plain-English insights powered by Groq AI (Llama 3.1) + SHAP analysis |
| 📊 **Real-Time Data** | Live quotes via Finnhub WebSocket + TradingView charts |
| 📰 **News Sentiment** | 10+ sources scored with FinBERT, RoBERTa, and VADER |
| 📈 **Technical Analysis** | RSI, MACD, Bollinger Bands, moving averages |
| 💰 **Sankey Diagrams** | Interactive revenue → expense → profit flow visualizations |
| 👀 **Watchlist** | Track stocks with real-time updates and alerts |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│              Frontend (Next.js 15)                   │
│         Vercel · TradingView · ECharts              │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│            Node.js Backend (Express)                 │
│     Koyeb · API Gateway · Finnhub WebSocket         │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│            ML Backend (FastAPI + Python)             │
│   LightGBM · LSTM · SHAP · Groq AI · Sentiment      │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│              MongoDB Atlas + Redis                   │
│      Predictions · Sentiment · Historical Data       │
└─────────────────────────────────────────────────────┘
```

---

## 🧠 How Predictions Work

1. **Data Collection** — Fetch OHLCV, sentiment from 10+ sources, macro indicators, insider trades
2. **Feature Engineering** — Build 113 numeric features (price, volume, sentiment, technicals, fundamentals)
3. **Model Training** — LightGBM with LSTM temporal embeddings, walk-forward validation
4. **Prediction** — Cross-sectional ranking selects top quintile stocks
5. **Explanation** — SHAP decomposes prediction; Groq AI writes human-readable insight

### Latest Backtest Results (Sep 2025 — Mar 2026)

| Horizon | Return | Sharpe | Win Rate |
|---------|--------|--------|----------|
| **30-day** | +13.71% | 2.68 | 64.3% |
| **7-day** | +8.05% | 1.61 | 50.4% |

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | Next.js 15, React 18, TypeScript, Tailwind, Shadcn/UI, TradingView |
| **Backend** | Node.js, Express, FastAPI, MongoDB, Redis |
| **ML/AI** | LightGBM, PyTorch (LSTM), SHAP, Groq (Llama 3.1), FinBERT |
| **Infrastructure** | Vercel, Koyeb, GitHub Actions, MongoDB Atlas |

---

## ⚡ Daily Pipeline

Runs automatically via GitHub Actions after market close (~2 hours):

1. **Sentiment** — Fetch & score news from 10+ sources
2. **Training** — Engineer features, train LightGBM + LSTM
3. **Predictions** — Generate forecasts for all 75 tickers
4. **Explanations** — SHAP analysis + Groq AI writing
5. **Evaluation** — Compare predictions vs actual outcomes

---

## 💸 Cost: $0/month

Runs entirely on free tiers:
- **Vercel** (frontend)
- **Koyeb** (backends, scale-to-zero)
- **MongoDB Atlas** (M0 free)
- **Redis Cloud** (free)
- **Groq** (14.4K requests/day free)
- **GitHub Actions** (2,000 min/month free)

---

## 📚 Documentation

For detailed technical documentation, see **[DOCUMENTATION.md](DOCUMENTATION.md)** (2,600+ lines covering architecture, APIs, schemas, and pipeline details).

---

## License

This project is licensed under the [**GNU Affero General Public License v3.0 (AGPL-3.0)**](LICENSE).

If you deploy this software as a network service, you must make the complete source code available to users of that service under the same license.

---

<p align="center">
  Built by <a href="https://github.com/Yogesh-VG0">Yogesh Vadivel</a>
</p>
