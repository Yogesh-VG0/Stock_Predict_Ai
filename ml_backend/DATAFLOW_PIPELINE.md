# Stock Prediction AI - Complete Data Pipeline Documentation

## Overview
This document provides a comprehensive overview of the data pipeline from ingestion to sentiment analysis, model training, predictions, and AI explanations. All data is stored in MongoDB with proper source attribution.

## Data Pipeline Architecture

### 1. Data Ingestion Pipeline

#### Historical Stock Data (OHLCV)
- **Source**: yfinance API (Yahoo Finance)
- **Process**: `DataIngestion` class in `ml_backend/data/ingestion.py`
- **Data Collected**:
  - Open, High, Low, Close, Volume (OHLCV)
  - Adjusted Close
  - Daily data for S&P 100 stocks
- **Storage**: MongoDB collection `historical_data`
- **Schema**:
  ```json
  {
    "ticker": "AAPL",
    "date": "2024-01-01",
    "Open": 150.25,
    "High": 152.50,
    "Low": 149.75,
    "Close": 151.20,
    "Volume": 50000000
  }
  ```

#### Macro Economic Data
- **Source**: FRED (Federal Reserve Economic Data) API
- **Process**: `fred_macro.py` / `macro.py`
- **Indicators Collected**:
  - GDP Growth Rate (GDPC1)
  - CPI (Consumer Price Index) - CPIAUCSL
  - Unemployment Rate - UNRATE
  - Federal Funds Rate - DFF
  - 10-Year Treasury Yield - DGS10
  - 2-Year Treasury Yield - DGS2
  - VIX (Volatility Index) - VIXCLS
  - Dollar Index - DTWEXBGS
  - Oil Prices (WTI) - WTISPLC
  - Gold Prices - GOLDAMGBD228NLBM
  - S&P 500 Index - SP500
  - Housing Starts - HOUST
  - Consumer Confidence - UMCSENT
- **Storage**: MongoDB collection `macro_data`
- **Schema**:
  ```json
  {
    "indicator": "GDPC1",
    "date": "2024-01-01",
    "value": 2.5,
    "source": "FRED"
  }
  ```

### 2. Sentiment Analysis Pipeline

#### Stock-Specific News Sources

##### Yahoo Finance RSS
- **URL**: `https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US`
- **Data**: Stock-specific news headlines and summaries
- **NLP**: FinBERT/VADER sentiment analysis

##### SeekingAlpha RSS
- **URL**: `https://seekingalpha.com/api/sa/combined/{ticker}.xml`
- **Data**: Stock-specific analysis articles
- **NLP**: FinBERT/VADER sentiment analysis

##### FinViz Scraping
- **URL**: Web scraping from finviz.com
- **Data**: News headlines, analyst ratings
- **NLP**: VADER sentiment analysis

##### Reddit API (PRAW)
- **Subreddits**: r/wallstreetbets, r/stocks, r/investing, r/StockMarket
- **Data**: Post titles, comments, scores
- **NLP**: VADER sentiment analysis

##### Marketaux News API
- **API**: Marketaux financial news
- **Data**: Professional news articles
- **Rate Limit**: 100 requests/day (free tier)

##### SEC Filings (FMP)
- **API**: Financial Modeling Prep
- **Data**: 10-K, 10-Q, 8-K filings
- **Processing**: Extract key financial metrics

#### Market Indicators

##### Finnhub APIs
- **Quote API**: Real-time stock quotes
- **Insider Transactions**: Insider buying/selling data
- **Insider Sentiment**: Aggregated insider sentiment
- **Recommendation Trends**: Analyst recommendations
- **Rate Limit**: 60 calls/minute

##### Financial Modeling Prep (FMP) APIs
- **Earnings Report**: Quarterly earnings data
- **Dividends**: Dividend history and calendar
- **Earnings Calendar**: Upcoming earnings dates
- **Analyst Ratings**: Buy/Hold/Sell recommendations
- **Rate Limit**: 250 calls/day

#### NLP Processing
- **FinBERT**: Financial domain-specific BERT model
  - Labels: positive, negative, neutral
  - Confidence scores
- **VADER**: Valence Aware Dictionary for Sentiment Reasoning
  - Compound score: -1 (negative) to +1 (positive)

#### MongoDB Storage
- **Collection**: `sentiment_data`
- **Schema Example**:
  ```json
  {
    "ticker": "AAPL",
    "date": "2024-01-01",
    "last_updated": "2024-01-01T12:00:00Z",
    
    // Stock quotes
    "current_price": 150.25,
    "percent_change": 1.5,
    
    // News sentiment
    "rss_sentiment": 0.75,
    "rss_volume": 15,
    "rss_confidence": 0.85,
    "yahoo_news_raw_data": [...],
    "seekingalpha_raw_data": [...],
    
    // Social sentiment
    "reddit_sentiment": 0.45,
    "reddit_volume": 120,
    
    // Professional analysis
    "finviz_sentiment": 0.60,
    "marketaux_sentiment": 0.55,
    "sec_sentiment": 0.0,
    
    // Market indicators
    "insider_sentiment": -0.20,
    "analyst_sentiment": 0.80,
    "recommendation_trends": {...},
    
    // Technical indicators
    "earnings_surprise": 0.05,
    "dividend_yield": 2.5
  }
  ```

### 3. Feature Engineering Pipeline

#### Technical Indicators Calculated
- **Price-based**:
  - Returns (1d, 5d, 20d)
  - Moving Averages (SMA 20, 50, 200)
  - Exponential Moving Averages (EMA 12, 26)
  - Bollinger Bands
  - Price ratios
  
- **Momentum**:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Stochastic Oscillator
  - Williams %R
  
- **Volume**:
  - Volume Moving Average
  - On-Balance Volume (OBV)
  - Volume Rate of Change
  
- **Volatility**:
  - ATR (Average True Range)
  - Historical Volatility
  - Garman-Klass Volatility

#### Feature Arrays
- Combined features from:
  - Technical indicators
  - Sentiment scores
  - Macro economic data
- Normalization and scaling
- Sequence creation for time series models

### 4. Model Training & Prediction Pipeline

#### Models Used
1. **LSTM (Long Short-Term Memory)**
   - Architecture: 2 LSTM layers + Dense layers
   - Sequence length: 30 days
   - Predictions: 1, 7, 30 days

2. **GRU (Gated Recurrent Unit)**
   - Architecture: 2 GRU layers + Dense layers
   - Faster training than LSTM
   - Similar performance

3. **Transformer**
   - Self-attention mechanism
   - Better for long-range dependencies
   - Parallel processing

4. **XGBoost**
   - Gradient boosting
   - Feature importance
   - Fast predictions

#### Ensemble Method
- Weighted average of all models
- Confidence scores based on:
  - Model agreement
  - Historical accuracy
  - Data quality

#### MongoDB Storage
- **Collection**: `predictions`
- **Schema**:
  ```json
  {
    "ticker": "AAPL",
    "window": "next_day",
    "prediction": 152.50,
    "confidence": 0.85,
    "range": "151.00-154.00",
    "timestamp": "2024-01-01T12:00:00Z"
  }
  ```

### 5. AI Explainability Pipeline

#### SHAP Analysis
- **Purpose**: Feature attribution
- **Process**: Calculate SHAP values for each prediction
- **Output**: Top 3 most influential features

#### Natural Language Generation
- **LLM**: Google Gemini Pro
- **Input Context**:
  - Current prediction and confidence
  - Technical indicators
  - Sentiment scores from all sources
  - SHAP feature importance
  - Recent news headlines
  - Macro economic context

#### Explanation Structure
```json
{
  "main_reason": "Primary driver of the prediction",
  "supporting_factors": [
    "Factor 1 with evidence",
    "Factor 2 with evidence"
  ],
  "caveats": [
    "Data limitations",
    "Market uncertainties"
  ],
  "top_factors": [
    {"feature": "RSI", "value": 0.75},
    {"feature": "insider_sentiment", "value": -0.30}
  ]
}
```

## API Endpoints

### Prediction Endpoints
- `GET /api/v1/predictions/{ticker}` - Get predictions for a ticker
- `GET /api/v1/metadata/{ticker}` - Get comprehensive ticker data

### Sentiment Endpoints
- `GET /api/v1/sentiment/{ticker}` - Get sentiment analysis
- `POST /api/v1/sentiment` - Trigger sentiment update

### Training Endpoints
- `POST /api/v1/train` - Train models on existing data
- `POST /api/v1/ingest` - Ingest new historical data

### Explainability Endpoints
- `GET /api/v1/explain/{ticker}/{date}` - Get AI explanation

### Historical Data Endpoints
- `GET /api/v1/historical/{ticker}` - Get historical OHLCV data

## Data Update Schedule

### Real-time/Near Real-time
- Stock quotes (Finnhub) - On demand
- News sentiment - Every prediction request

### Daily Updates
- Historical OHLCV data
- SEC filings
- Insider transactions
- Analyst ratings

### Weekly/Monthly
- Macro economic indicators (varies by indicator)
- Earnings data (quarterly)
- Dividend data

## Rate Limits & API Constraints

### Free Tier Limits
- **Finnhub**: 60 calls/minute
- **FMP**: 250 calls/day
- **Reddit**: 60 requests/minute
- **Marketaux**: 100 requests/day
- **FRED**: 120 requests/minute

### Optimization Strategies
- Batch requests where possible
- Cache frequently accessed data
- Prioritize S&P 100 stocks
- Stagger API calls to avoid limits

## Error Handling & Resilience

### Retry Logic
- Exponential backoff for failed requests
- Maximum 3 retries per API call
- Fallback to cached data

### Data Quality Checks
- Validate schema before MongoDB storage
- Handle missing data gracefully
- Log all errors with context

### Monitoring
- API call success rates
- Data freshness metrics
- Model performance tracking
- System resource usage

## Security & Compliance

### API Key Management
- Environment variables for all API keys
- No hardcoded credentials
- Secure key rotation

### Data Privacy
- No personal user data stored
- Public financial data only
- Compliance with API terms of service

### Access Control
- MongoDB authentication
- API rate limiting
- CORS configuration

## Future Enhancements

### Planned Features
1. Real-time WebSocket updates
2. Additional alternative data sources
3. More sophisticated NLP models
4. Automated model retraining
5. Advanced portfolio optimization

### Scalability Considerations
- Horizontal scaling for API servers
- MongoDB sharding for large datasets
- Redis caching for frequently accessed data
- Async processing for heavy computations 