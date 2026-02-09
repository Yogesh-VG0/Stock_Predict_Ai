# StockPredict AI

A full-stack stock analytics and prediction platform for S&P 100 companies with interactive charts and ML-powered forecasting.

![Next.js](https://img.shields.io/badge/Next.js-15-black?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)

## ✨ Features

- **Interactive TradingView Charts** - Real-time market visualization
- **ML Predictions** - LSTM, XGBoost, LightGBM ensemble models
- **Multi-timeframe Forecasts** - Next day, 7-day, and 30-day predictions
- **Sentiment Analysis** - Reddit, news, and social media sentiment
- **Real-time Data** - Live market data via WebSockets
- **Clean Dashboard UI** - Modern interface with animations

## 🏗️ Architecture

```
stockpredict-ai/
├── app/                # Next.js App Directory
├── pages/              # Page components
├── components/         # React components
│   ├── market/        # Market widgets
│   ├── tradingview/   # TradingView integrations
│   └── ui/            # UI components
├── hooks/              # Custom React hooks
├── lib/                # Utilities & API client
├── backend/            # Node.js backend
└── ml_backend/         # Python ML pipeline
```

## 🛠️ Tech Stack

**Frontend:**
- Next.js 15, React, TypeScript
- Tailwind CSS, Framer Motion
- TradingView Widgets, Recharts

**Backend:**
- Node.js, Express
- MongoDB, Redis

**ML Pipeline:**
- Python, FastAPI
- TensorFlow, XGBoost, LightGBM
- SHAP for explainability

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- Python 3.8+
- MongoDB

### Installation

```bash
# Clone the repo
git clone https://github.com/Yogesh-VG0/StockPredict-AI.git
cd StockPredict-AI

# Install frontend dependencies
npm install

# Install ML backend dependencies
cd ml_backend
pip install -r requirements.txt

# Install Node backend dependencies
cd ../backend
npm install
```

### Environment Setup

Create `.env` in root:
```env
NEXT_PUBLIC_API_URL=http://localhost:3001
NEXT_PUBLIC_ML_API_URL=http://localhost:8000
```

### Run Development Servers

```bash
# Terminal 1 - ML Backend
cd ml_backend
uvicorn api.main:app --port 8000

# Terminal 2 - Node Backend  
cd backend
npm run dev

# Terminal 3 - Frontend
npm run dev
```

## 📊 ML Models

The platform uses an ensemble of models:
- **LSTM** - Long Short-Term Memory networks
- **XGBoost** - Gradient boosting
- **LightGBM** - Light gradient boosting

Models are pre-trained on S&P 100 stocks with 100+ features including:
- Technical indicators (RSI, MACD, Bollinger Bands)
- Sentiment scores (Reddit, News)
- Economic indicators

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Built by Yogesh Vadivel**
