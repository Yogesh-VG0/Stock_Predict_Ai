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
    "BRK.B",  # Berkshire Hathaway Inc. (Class B)
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
# Reddit Configuration
REDDIT_SUBREDDITS = [
    "stocks", "wallstreetbets", "investing", "finance", "economics",
    "StockMarket", "SecurityAnalysis", "OptionTrading"
]

# Ticker-specific subreddit mapping for enhanced Reddit sentiment analysis
TICKER_SUBREDDITS = {
    "AAPL": ["AAPL", "applestocks", "stocks", "wallstreetbets"],
    "MSFT": ["Microsoft", "stocks", "investing"],
    "NVDA": ["NVDA_Stock", "NVDA", "stocks", "wallstreetbets"],
    "AMZN": ["AmazonStock", "Amazon", "stocks", "investing"],
    "GOOG": ["stocks", "investing", "wallstreetbets"],
    "GOOGL": ["stocks", "investing", "wallstreetbets"],
    "META": ["MetaStock", "Facebook", "stocks", "wallstreetbets"],
    "BRK.B": ["brkb", "stocks", "investing"],
    "TSLA": ["teslainvestorsclub", "TSLA", "stocks", "wallstreetbets"],
    "AVGO": ["stocks", "investing", "wallstreetbets"],
    "LLY": ["stocks", "investing"],
    "WMT": ["walmart", "stocks", "investing"],
    "JPM": ["stocks", "investing"],
    "V": ["stocks", "investing"],
    "MA": ["stocks", "investing"],
    "NFLX": ["NetflixStock", "Netflix", "stocks", "wallstreetbets"],
    "XOM": ["stocks", "investing"],
    "COST": ["Costco", "stocks", "investing"],
    "ORCL": ["stocks", "investing"],
    "PG": ["stocks", "investing"],
    "JNJ": ["stocks", "investing"],
    "UNH": ["stocks", "investing"],
    "HD": ["HomeDepot", "stocks", "investing"],
    "ABBV": ["stocks", "investing"],
    "KO": ["CocaCola", "stocks", "investing"],
    "BAC": ["stocks", "investing"],
    "TMUS": ["tmobile", "stocks", "investing"],
    "PLTR": ["PLTR", "wallstreetbets", "stocks"],
    "PM": ["stocks", "investing"],
    "CRM": ["salesforce", "stocks", "investing"],
    "CVX": ["stocks", "investing"],
    "WFC": ["stocks", "investing"],
    "CSCO": ["cisco", "stocks", "investing"],
    "MCD": ["McDonalds", "stocks", "investing"],
    "ABT": ["stocks", "investing"],
    "IBM": ["IBM", "stocks", "investing"],
    "GE": ["GeneralElectric", "stocks", "investing"],
    "MRK": ["stocks", "investing"],
    "LIN": ["stocks", "investing"],
    "T": ["ATT", "stocks", "investing"],
    "NOW": ["servicenow", "stocks", "investing"],
    "ACN": ["Accenture", "stocks", "investing"],
    "AXP": ["stocks", "investing"],
    "MS": ["stocks", "investing"],
    "PEP": ["Pepsi", "stocks", "investing"],
    "VZ": ["verizon", "stocks", "investing"],
    "ISRG": ["IntuitiveSurgical", "stocks", "investing"],
    "INTU": ["Intuit", "stocks", "investing"],
    "GS": ["GoldmanSachs", "stocks", "investing"],
    "RTX": ["Raytheon", "stocks", "investing"],
    "BKNG": ["Booking", "stocks", "investing"],
    "DIS": ["Disney", "stocks", "investing"],
    "QCOM": ["Qualcomm", "stocks", "investing"],
    "TMO": ["ThermoFisher", "stocks", "investing"],
    "ADBE": ["Adobe", "stocks", "investing"],
    "AMD": ["AMD_Stock", "AMD", "stocks", "wallstreetbets"],
    "AMGN": ["stocks", "investing"],
    "SCHW": ["CharlesSchwab", "stocks", "investing"],
    "CAT": ["Caterpillar", "stocks", "investing"],
    "TXN": ["TexasInstruments", "stocks", "investing"],
    "DHR": ["stocks", "investing"],
    "BLK": ["stocks", "investing"],
    "PFE": ["Pfizer", "stocks", "investing"],
    "BA": ["boeing", "stocks", "investing"],
    "NEE": ["stocks", "investing"],
    "HON": ["Honeywell", "stocks", "investing"],
    "GILD": ["stocks", "investing"],
    "UNP": ["UnionPacific", "stocks", "investing"],
    "C": ["Citigroup", "stocks", "investing"],
    "CMCSA": ["Comcast", "stocks", "investing"],
    "DE": ["JohnDeere", "stocks", "investing"],
    "LOW": ["Lowes", "stocks", "investing"],
    "COP": ["stocks", "investing"],
    "LMT": ["LockheedMartin", "stocks", "investing"],
    "CHTR": ["Charter", "stocks", "investing"],
    "MDT": ["Medtronic", "stocks", "investing"],
    "AMT": ["stocks", "investing"],
    "BMY": ["BMS", "stocks", "investing"],
    "SO": ["SouthernCompany", "stocks", "investing"],
    "MO": ["Altria", "stocks", "investing"],
    "DUK": ["DukeEnergy", "stocks", "investing"],
    "SBUX": ["starbucks", "stocks", "investing"],
    "MDLZ": ["stocks", "investing"],
    "INTC": ["Intel", "stocks", "investing"],
    "CVS": ["CVS", "stocks", "investing"],
    "NKE": ["Nike", "stocks", "investing"],
    "UPS": ["UPS", "stocks", "investing"],
    "MMM": ["3M", "stocks", "investing"],
    "CL": ["Colgate", "stocks", "investing"],
    "GD": ["GeneralDynamics", "stocks", "investing"],
    "COF": ["CapitalOne", "stocks", "investing"],
    "PYPL": ["PayPal", "stocks", "investing"],
    "USB": ["USBank", "stocks", "investing"],
    "EMR": ["EmersonElectric", "stocks", "investing"],
    "BK": ["BankofNewYork", "stocks", "investing"],
    "SPG": ["SimonProperty", "stocks", "investing"],
    "MET": ["MetLife", "stocks", "investing"],
    "FDX": ["FedEx", "stocks", "investing"],
    "AIG": ["AIG", "stocks", "investing"],
    "TGT": ["Target", "stocks", "investing"]
}

# RSS Feeds - Removed general market feeds since we use stock-specific RSS feeds
# Stock-specific RSS feeds are generated dynamically in sentiment.py:
# - Yahoo Finance: https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US
# - SeekingAlpha: https://seekingalpha.com/api/sa/combined/{ticker}.xml
RSS_FEEDS = {}

# Model Configuration
MODEL_CONFIG = {
    "lstm_units": 128,  # Increased from 64 for better capacity
    "dense_units": 64,  # Increased from 32
    "dropout_rate": 0.2,  # Reduced from 0.3 for better learning
    "batch_size": 32,  # Reduced from 64 for better gradient updates
    "epochs": 25,  # Increased from 15 for better convergence
    "early_stopping_patience": 7,  # Increased from 5
    "learning_rate": 0.0005,  # Reduced from 0.001 for more stable training
    "n_trials": 25,  # Increased from 15 for better hyperparameter search
    "attention_dim": 64,  # New parameter for attention mechanism
    "default_hyperparameters": {
        "lstm_units": 128,
        "dense_units": 64,
        "dropout_rate": 0.2,
        "learning_rate": 0.0005,
        "l2_reg": 5e-4,  # Reduced regularization
        "batch_size": 32,
        "attention_dim": 64,
        # New parameters for enhanced model
        "recurrent_dropout": 0.1,
        "gradient_clip_norm": 1.0,
        "use_attention": True,
        "use_bidirectional": True
    }
}

# Prediction Windows
PREDICTION_WINDOWS = {
    "next_day": 1,
    "7_day": 7,
    "30_day": 30
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

# Hyperparameter Search Space - Optimized for Financial Time Series
HYPERPARAM_SEARCH_SPACE = {
    "lstm_units": (64, 256),  # Wider range, higher max
    "dense_units": (32, 128),  # Wider range
    "dropout_rate": (0.1, 0.3),  # Lower minimum
    "learning_rate": (0.0001, 0.002),  # Better range for financial data
    "l2_reg": (1e-5, 1e-3),  # Wider regularization range
    "batch_size": [16, 32, 64],  # More options
    "attention_dim": (32, 128),  # New parameter
    "recurrent_dropout": (0.0, 0.2),  # New parameter
    # Model architecture choices
    "use_attention": [True, False],
    "use_bidirectional": [True, False],
    "activation": ['relu', 'swish', 'gelu'],  # Modern activation functions
    "optimizer": ['adam', 'adamw', 'rmsprop']  # Optimizer choices
} 