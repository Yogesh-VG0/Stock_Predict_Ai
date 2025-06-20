"""
Configuration constants for the stock prediction system.
"""

import os

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "stockpredict_ai")

# S&P 100 Tickers (Top 25 from your list, with company names)
TOP_100_TICKERS = [
    "AAPL",   # Apple Inc.
    "MSFT",   # Microsoft Corporation
    "NVDA",   # Nvidia Corporation
    "AMZN",   # Amazon.com, Inc.
    "GOOGL",  # Alphabet Inc. (Class A)
    "META",   # Meta Platforms, Inc.
    "BRK-B",  # Berkshire Hathaway Inc. (Class B)
    "TSLA",   # Tesla, Inc.
    "AVGO",   # Broadcom Inc.
    "LLY",    # Eli Lilly and Company
    "WMT",    # Walmart Inc.
    "JPM",    # JPMorgan Chase & Co.
    "V",      # Visa Inc.
    "MA",     # Mastercard Incorporated
    "NFLX",   # Netflix, Inc.
    "XOM",    # Exxon Mobil Corporation
    "COST",   # Costco Wholesale Corporation
    "ORCL",   # Oracle Corporation
    "PG",     # The Procter & Gamble Company
    "JNJ",    # Johnson & Johnson
    "UNH",    # UnitedHealth Group Incorporated
    "HD",     # The Home Depot, Inc.
    "ABBV",   # AbbVie Inc.
    "KO",     # The Coca-Cola Company
    "CRM"     # Salesforce, Inc.
]

# API Configuration
API_PREFIX = "/api"
API_VERSION = "v1"
RATE_LIMIT = 100  # requests per hour
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# API Configuration
API_CONFIG = {
    "title": "Stock Prediction API",
    "description": "API for S&P 100 stock predictions and analysis",
    "version": API_VERSION,
    "docs_url": "/docs",
    "redoc_url": "/redoc"
}

# Data Collection
HISTORICAL_DATA_YEARS = 10
REDDIT_SUBREDDITS = [
    "stocks", "wallstreetbets", "investing", "finance", "economics",
    "StockMarket", "SecurityAnalysis", "OptionTrading"
]

# RSS Feeds - Removed general market feeds since we use stock-specific RSS feeds
# Stock-specific RSS feeds are generated dynamically in sentiment.py:
# - Yahoo Finance: https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US
# - SeekingAlpha: https://seekingalpha.com/api/sa/combined/{ticker}.xml
RSS_FEEDS = {}

# Model Configuration
MODEL_CONFIG = {
    "lstm_units": 64,
    "dense_units": 32,
    "dropout_rate": 0.3,
    "batch_size": 64,
    "epochs": 15,
    "early_stopping_patience": 5,
    "learning_rate": 0.001,
    "n_trials": 15,
    "default_hyperparameters": {
        "lstm_units": 64,
        "dense_units": 32,
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "l2_reg": 1e-3,
        "batch_size": 64
    }
}

# Prediction Windows
PREDICTION_WINDOWS = {
    "next_day": 1,
    "30_day": 30,
    "90_day": 90
}

# Technical Indicators
TECHNICAL_INDICATORS = {
    "sma_periods": [5, 10, 20, 50, 200],
    "ema_periods": [5, 10, 20, 50, 200],
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bollinger_period": 20,
    "bollinger_std": 2,
    "atr_period": 14
}

# MongoDB Collections
MONGO_COLLECTIONS = {
    "predictions": "stock_predictions",
    "historical_data": "historical_data",
    "sentiment_data": "sentiment_data",
    "model_versions": "model_versions",
    "prediction_metrics": "prediction_metrics",
    "economic_events": "economic_events",
    "sec_filings": "sec_filings", 
    "seeking_alpha_sentiment": "seeking_alpha_sentiment",
    "seeking_alpha_comments": "seeking_alpha_comments",
    "short_interest_data": "short_interest_data",
    "macro_data_raw": "macro_data_raw",
    "llm_explanations": "llm_explanations",
    "feature_importance": "feature_importance",
    "api_cache": "api_cache"
}

# Retry Configuration
RETRY_CONFIG = {
    "max_retries": 3,
    "base_delay": 1,
    "max_delay": 10
}

# Feature Engineering
FEATURE_CONFIG = {
    "lookback_days": 60,
    "sequence_length": 30,
    "train_test_split": 0.8,
    "validation_split": 0.1
}

# Ticker mapping for yfinance special cases
TICKER_YFINANCE_MAP = {
    "BRK.B": "BRK-B"
}

# Canary tickers for data freshness checks
CANARY_TICKERS = ["AAPL", "MSFT", "SPY"]

# If you have hyperparameter search space defined here, update as follows:
HYPERPARAM_SEARCH_SPACE = {
    "lstm_units": (32, 64),
    "dense_units": (16, 32),
    "dropout_rate": (0.1, 0.3),
    "batch_size": [64, 128],
    # ... other params ...
} 