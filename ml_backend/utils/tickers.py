import hashlib

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

# Ticker-specific subreddit mapping
TICKER_SUBREDDITS = {
    "AAPL": ["AAPL", "applestocks", "stocks", "wallstreetbets"],
    "MSFT": ["Microsoft", "stocks", "investing"],
    "NVDA": ["NVDA_Stock", "NVDA", "stocks", "wallstreetbets"],
}

# Deterministic ticker_id for pooled models
_ALL_TICKERS = sorted(set(TOP_100_TICKERS) | set(TICKER_SUBREDDITS.keys()) | {"SPY"})
TICKER_TO_ID = {ticker: i for i, ticker in enumerate(_ALL_TICKERS)}

def get_ticker_id(ticker: str) -> int:
    """Return deterministic ticker_id. Unknown tickers use md5 fallback."""
    key = (ticker or "UNK").upper()
    if key in TICKER_TO_ID:
        return TICKER_TO_ID[key]
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 100000
