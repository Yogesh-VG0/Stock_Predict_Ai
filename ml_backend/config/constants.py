"""
Configuration constants for the stock prediction system.
"""

import hashlib
import os

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "stockpredict_ai")

# S&P 100 Tickers (full list — 100 large-cap US equities)
TOP_100_TICKERS = [
    # Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CRM",
    "AMD", "INTC", "CSCO", "ADBE", "QCOM", "TXN", "NOW", "INTU", "AMAT",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "NFLX", "LOW", "SBUX", "NKE", "MCD",
    "DIS", "BKNG", "TGT",
    # Financials
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP",
    "BLK", "SCHW", "C", "COF", "BK", "MET", "AIG", "USB",
    # Energy
    "XOM", "CVX", "COP",
    # Healthcare
    "JNJ", "UNH", "LLY", "PFE", "ABBV", "ABT", "TMO", "DHR",
    "MRK", "AMGN", "GILD", "ISRG", "MDT", "BMY", "CVS",
    # Consumer Staples
    "WMT", "COST", "PG", "KO", "PEP", "MDLZ", "CL", "MO",
    # Industrials
    "CAT", "HON", "UNP", "BA", "RTX", "LMT", "DE", "GE",
    "GD", "EMR", "FDX", "UPS", "MMM",
    # Communication
    "CMCSA", "VZ", "T", "CHTR",
    # Conglomerates / Other
    "BRK-B", "ACN", "IBM", "PYPL", "LIN", "NEE", "SO",
    "DUK", "AMT", "SPG", "PLTR", "TMUS", "PM",
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
    "BRK-B": ["brkb", "stocks", "investing"],
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

# Deterministic ticker_id for pooled models (stable across runs; Python hash() is salted)
_ALL_TICKERS = sorted(set(TOP_100_TICKERS) | set(TICKER_SUBREDDITS.keys()) | {"SPY"})
TICKER_TO_ID = {ticker: i for i, ticker in enumerate(_ALL_TICKERS)}


# Prediction Windows
PREDICTION_WINDOWS = {
    "next_day": 1,
    "7_day": 7,
    "30_day": 30
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

# Sentiment volume keys that represent actual article/headline counts.
# Single source of truth — used in:
#   sentiment.py  → news_count computation
#   mongodb.py    → get_sentiment_timeseries() fallback
#   cron          → per-source health logging
# Everything NOT in this set (finnhub_volume = shares, short_interest_volume,
# economic_event_volume, sentiment_volume = aggregate, any future key) is
# IGNORED when computing news_count.
#
# alphavantage_volume REMOVED: source explicitly disabled (free tier
# insufficient — 25 req/day for 100 tickers).  Always contributed 0.
ARTICLE_COUNT_VOLUME_KEYS = frozenset({
    "rss_news_volume",
    "marketaux_volume",
    "finviz_volume",
})

# Retry Configuration
RETRY_CONFIG = {
    "max_retries": 3,
    "base_delay": 1,
    "max_delay": 10
}

# Ticker mapping for yfinance special cases
# Pipeline uses BRK-B everywhere; yfinance also accepts BRK-B.
TICKER_YFINANCE_MAP = {
    "BRK-B": "BRK-B",
}

# Canary tickers for data freshness checks
CANARY_TICKERS = ["AAPL", "MSFT", "SPY"]