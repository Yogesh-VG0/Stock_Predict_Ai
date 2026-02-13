import hashlib

# S&P 100 Tickers (Top 25 from your list, with company names)
TOP_100_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "TSLA",
    "AVGO", "LLY", "WMT", "JPM", "V", "MA", "NFLX", "XOM", "COST",
    "ORCL", "PG", "JNJ", "UNH", "HD", "ABBV", "KO", "CRM"
]

# Ticker-specific subreddit mapping
TICKER_SUBREDDITS = {
    "AAPL": ["AAPL", "applestocks", "stocks", "wallstreetbets"],
    "MSFT": ["Microsoft", "stocks", "investing"],
    "NVDA": ["NVDA_Stock", "NVDA", "stocks", "wallstreetbets"],
    # ... (other mappings can be imported or re-defined if needed, but for ID generation we just need keys)
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
