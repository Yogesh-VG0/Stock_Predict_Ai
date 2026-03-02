import hashlib

# S&P 75 Tickers — trimmed from 100 to improve pipeline reliability.
# Removed 25 tickers with worst API coverage. Canonical source: config/constants.py.
TOP_100_TICKERS = [
    # Technology (19)
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CRM",
    "AMD", "INTC", "CSCO", "ADBE", "QCOM", "TXN", "INTU", "AMAT",
    "IBM", "PYPL", "PLTR",
    # Consumer Discretionary (11)
    "AMZN", "TSLA", "HD", "NFLX", "LOW", "SBUX", "NKE", "MCD",
    "DIS", "BKNG", "TGT",
    # Financials (10)
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "C", "BRK-B",
    # Energy (2)
    "XOM", "CVX",
    # Healthcare (10)
    "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "AMGN", "GILD",
    "ISRG", "CVS",
    # Consumer Staples (6)
    "WMT", "COST", "PG", "KO", "PEP", "MDLZ",
    # Industrials (9)
    "CAT", "HON", "BA", "RTX", "LMT", "DE", "GE", "FDX", "UPS",
    # Communication (5)
    "CMCSA", "VZ", "T", "CHTR", "TMUS",
    # Other (3)
    "LIN", "NEE", "AMT",
]

# Ticker-specific subreddit mapping
TICKER_SUBREDDITS = {
    "AAPL": ["AAPL", "applestocks", "stocks", "wallstreetbets"],
    "MSFT": ["Microsoft", "stocks", "investing"],
    "NVDA": ["NVDA_Stock", "NVDA", "stocks", "wallstreetbets"],
}

# Deterministic ticker_id for pooled models — HASH-BASED for stability.
# Sequential IDs (enumerate) shift when tickers are added/removed, silently
# corrupting the pooled model's learned ticker associations.  Hash-based IDs
# are stable across ticker list changes.
_ALL_TICKERS = sorted(set(TOP_100_TICKERS) | set(TICKER_SUBREDDITS.keys()) | {"SPY"})


def _stable_ticker_hash(ticker: str) -> int:
    """Compute a stable, deterministic integer ID from a ticker string.

    Uses SHA-256 (truncated to 6 hex chars → 0..16_777_215) so that IDs
    never change when the ticker list is modified.
    """
    return int(hashlib.sha256(ticker.encode()).hexdigest()[:6], 16)


TICKER_TO_ID = {ticker: _stable_ticker_hash(ticker) for ticker in _ALL_TICKERS}


def get_ticker_id(ticker: str) -> int:
    """Return deterministic, stable ticker_id.  Safe across ticker list changes."""
    key = (ticker or "UNK").upper()
    if key in TICKER_TO_ID:
        return TICKER_TO_ID[key]
    return _stable_ticker_hash(key)
