"""
Batch AI Explanation Generator for CI/CD

Generates AI explanations for all tickers with stored predictions using Groq (preferred)
or Gemini (fallback) APIs and stores them in the ``prediction_explanations`` MongoDB collection.

API Provider Selection:
- Groq (preferred): Uses GROQ_API_KEY if available
  - llama-3.3-70b-versatile: 1K RPD (best quality)
  - llama-3.1-8b-instant: 14.4K RPD (fast fallback)
- Gemini (fallback): Uses GOOGLE_API_KEY if Groq not available
  - gemini-2.5-flash: 20 RPD (free tier limit)
  - gemini-2.5-flash-lite: 20 RPD

Groq free tier is MUCH better than Gemini (1K-14K RPD vs 20 RPD).

Usage (CI):
    python -m ml_backend.scripts.generate_explanations
    python -m ml_backend.scripts.generate_explanations --tickers AAPL MSFT
    python -m ml_backend.scripts.generate_explanations --max-tickers 20
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── S&P 100 tickers (same list as run_pipeline) ──────────────────────────
TOP_100_TICKERS: List[str] = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CRM", "AMD", "INTC",
    "CSCO", "ADBE", "QCOM", "TXN", "NOW", "INTU", "AMZN", "TSLA", "HD", "NFLX",
    "LOW", "SBUX", "NKE", "MCD", "DIS", "BKNG", "TGT", "JPM", "V", "MA",
    "BAC", "WFC", "GS", "MS", "AXP", "BLK", "SCHW", "C", "COF", "BK",
    "MET", "AIG", "USB", "XOM", "CVX", "COP", "JNJ", "UNH", "LLY", "PFE",
    "ABBV", "ABT", "TMO", "DHR", "MRK", "AMGN", "GILD", "ISRG", "MDT", "BMY",
    "CVS", "WMT", "COST", "PG", "KO", "PEP", "MDLZ", "CL", "MO", "CAT",
    "HON", "UNP", "BA", "RTX", "LMT", "DE", "GE", "GD", "EMR", "FDX",
    "UPS", "MMM", "CMCSA", "VZ", "T", "CHTR", "BRK-B", "ACN", "IBM", "PYPL",
    "LIN", "NEE", "SO", "DUK", "AMT", "SPG", "PLTR", "TMUS", "PM", "AMAT",
]

# ── Human-readable feature name mapping ──
FEATURE_DISPLAY_NAMES = {
    "macro_spread_2y10y": "Treasury yield curve (2Y-10Y spread)",
    "macro_fed_funds": "Federal funds rate",
    "spy_vol_20d": "S&P 500 volatility (20-day)",
    "spy_vol_regime": "S&P 500 volatility regime",
    "sector_etf_vol_20d": "Sector ETF volatility (20-day)",
    "sector_etf_return_20d": "Sector ETF return (20-day)",
    "sector_etf_return_60d": "Sector ETF return (60-day)",
    "sector_momentum_rank": "Sector momentum ranking",
    "vix_level": "VIX fear index level",
    "vix_vol_20d": "VIX volatility (20-day)",
    "vol_regime": "Volatility regime",
    "volatility_20d": "Stock volatility (20-day)",
    "price_vs_sma20": "Price vs 20-day moving average",
    "price_vs_sma50": "Price vs 50-day moving average",
    "rsi": "RSI (relative strength index)",
    "rsi_divergence": "RSI divergence signal",
    "bb_position": "Bollinger Band position",
    "log_return_1d": "1-day return",
    "log_return_5d": "5-day return",
    "log_return_21d": "21-day return",
    "trend_20d": "20-day price trend",
    "momentum_5d": "5-day momentum",
    "intraday_range": "Intraday price range",
    "overnight_gap": "Overnight gap",
    "volume_ratio": "Volume vs average ratio",
    "volume_z60": "Volume z-score (60-day)",
    "volume_vol_ratio": "Volume volatility ratio",
    "excess_vs_sector_5d": "Excess return vs sector (5-day)",
    "excess_vs_sector_20d": "Excess return vs sector (20-day)",
    "ticker_id": "Stock-specific factor",
    "sector_id": "Sector factor",
    "sent_mean_1d": "News sentiment (1-day)",
    "sent_mean_7d": "News sentiment (7-day avg)",
    "sent_mean_30d": "News sentiment (30-day avg)",
    "sent_momentum": "Sentiment momentum shift",
    "news_count_7d": "News volume (7-day)",
    "news_spike_1d": "Unusual news activity",
    "insider_net_value_30d": "Insider net trading (30-day)",
    "insider_buy_ratio_30d": "Insider buy ratio (30-day)",
    "insider_cluster_buying": "Insider cluster buying signal",
}


def _friendly_feature_name(raw: str) -> str:
    """Convert raw feature name to human-readable name."""
    return FEATURE_DISPLAY_NAMES.get(raw, raw.replace("_", " ").title())


# ── Per-stock metadata for stock-specific prompts ──
STOCK_META: Dict[str, Dict[str, str]] = {
    "AAPL": {"name": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics"},
    "MSFT": {"name": "Microsoft Corp.", "sector": "Technology", "industry": "Software"},
    "NVDA": {"name": "NVIDIA Corp.", "sector": "Technology", "industry": "Semiconductors"},
    "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology", "industry": "Internet Services"},
    "META": {"name": "Meta Platforms", "sector": "Technology", "industry": "Social Media"},
    "AVGO": {"name": "Broadcom Inc.", "sector": "Technology", "industry": "Semiconductors"},
    "ORCL": {"name": "Oracle Corp.", "sector": "Technology", "industry": "Enterprise Software"},
    "CRM": {"name": "Salesforce Inc.", "sector": "Technology", "industry": "Cloud Software"},
    "AMD": {"name": "Advanced Micro Devices", "sector": "Technology", "industry": "Semiconductors"},
    "INTC": {"name": "Intel Corp.", "sector": "Technology", "industry": "Semiconductors"},
    "CSCO": {"name": "Cisco Systems", "sector": "Technology", "industry": "Networking Equipment"},
    "ADBE": {"name": "Adobe Inc.", "sector": "Technology", "industry": "Software"},
    "QCOM": {"name": "Qualcomm Inc.", "sector": "Technology", "industry": "Semiconductors"},
    "TXN": {"name": "Texas Instruments", "sector": "Technology", "industry": "Semiconductors"},
    "NOW": {"name": "ServiceNow Inc.", "sector": "Technology", "industry": "Cloud Software"},
    "INTU": {"name": "Intuit Inc.", "sector": "Technology", "industry": "Financial Software"},
    "AMZN": {"name": "Amazon.com Inc.", "sector": "Consumer Cyclical", "industry": "E-Commerce / Cloud"},
    "TSLA": {"name": "Tesla Inc.", "sector": "Consumer Cyclical", "industry": "Electric Vehicles"},
    "HD": {"name": "Home Depot Inc.", "sector": "Consumer Cyclical", "industry": "Home Improvement Retail"},
    "NFLX": {"name": "Netflix Inc.", "sector": "Communication Services", "industry": "Streaming"},
    "LOW": {"name": "Lowe's Companies", "sector": "Consumer Cyclical", "industry": "Home Improvement Retail"},
    "SBUX": {"name": "Starbucks Corp.", "sector": "Consumer Cyclical", "industry": "Restaurants"},
    "NKE": {"name": "Nike Inc.", "sector": "Consumer Cyclical", "industry": "Footwear / Apparel"},
    "MCD": {"name": "McDonald's Corp.", "sector": "Consumer Cyclical", "industry": "Restaurants"},
    "DIS": {"name": "Walt Disney Co.", "sector": "Communication Services", "industry": "Entertainment"},
    "BKNG": {"name": "Booking Holdings", "sector": "Consumer Cyclical", "industry": "Online Travel"},
    "TGT": {"name": "Target Corp.", "sector": "Consumer Defensive", "industry": "Discount Retail"},
    "JPM": {"name": "JPMorgan Chase", "sector": "Financial Services", "industry": "Diversified Banking"},
    "V": {"name": "Visa Inc.", "sector": "Financial Services", "industry": "Payment Processing"},
    "MA": {"name": "Mastercard Inc.", "sector": "Financial Services", "industry": "Payment Processing"},
    "BAC": {"name": "Bank of America", "sector": "Financial Services", "industry": "Diversified Banking"},
    "WFC": {"name": "Wells Fargo", "sector": "Financial Services", "industry": "Diversified Banking"},
    "GS": {"name": "Goldman Sachs", "sector": "Financial Services", "industry": "Investment Banking"},
    "MS": {"name": "Morgan Stanley", "sector": "Financial Services", "industry": "Investment Banking"},
    "AXP": {"name": "American Express", "sector": "Financial Services", "industry": "Credit Services"},
    "BLK": {"name": "BlackRock Inc.", "sector": "Financial Services", "industry": "Asset Management"},
    "SCHW": {"name": "Charles Schwab", "sector": "Financial Services", "industry": "Brokerage"},
    "C": {"name": "Citigroup Inc.", "sector": "Financial Services", "industry": "Diversified Banking"},
    "COF": {"name": "Capital One Financial", "sector": "Financial Services", "industry": "Credit Services"},
    "BK": {"name": "Bank of New York Mellon", "sector": "Financial Services", "industry": "Custody Banking"},
    "MET": {"name": "MetLife Inc.", "sector": "Financial Services", "industry": "Insurance"},
    "AIG": {"name": "American International Group", "sector": "Financial Services", "industry": "Insurance"},
    "USB": {"name": "U.S. Bancorp", "sector": "Financial Services", "industry": "Regional Banking"},
    "XOM": {"name": "Exxon Mobil", "sector": "Energy", "industry": "Oil & Gas Integrated"},
    "CVX": {"name": "Chevron Corp.", "sector": "Energy", "industry": "Oil & Gas Integrated"},
    "COP": {"name": "ConocoPhillips", "sector": "Energy", "industry": "Oil & Gas E&P"},
    "JNJ": {"name": "Johnson & Johnson", "sector": "Healthcare", "industry": "Pharmaceuticals / MedTech"},
    "UNH": {"name": "UnitedHealth Group", "sector": "Healthcare", "industry": "Health Insurance"},
    "LLY": {"name": "Eli Lilly", "sector": "Healthcare", "industry": "Pharmaceuticals"},
    "PFE": {"name": "Pfizer Inc.", "sector": "Healthcare", "industry": "Pharmaceuticals"},
    "ABBV": {"name": "AbbVie Inc.", "sector": "Healthcare", "industry": "Pharmaceuticals"},
    "ABT": {"name": "Abbott Laboratories", "sector": "Healthcare", "industry": "Medical Devices"},
    "TMO": {"name": "Thermo Fisher Scientific", "sector": "Healthcare", "industry": "Life Sciences Tools"},
    "DHR": {"name": "Danaher Corp.", "sector": "Healthcare", "industry": "Life Sciences Tools"},
    "MRK": {"name": "Merck & Co.", "sector": "Healthcare", "industry": "Pharmaceuticals"},
    "AMGN": {"name": "Amgen Inc.", "sector": "Healthcare", "industry": "Biotechnology"},
    "GILD": {"name": "Gilead Sciences", "sector": "Healthcare", "industry": "Biotechnology"},
    "ISRG": {"name": "Intuitive Surgical", "sector": "Healthcare", "industry": "Medical Devices"},
    "MDT": {"name": "Medtronic plc", "sector": "Healthcare", "industry": "Medical Devices"},
    "BMY": {"name": "Bristol-Myers Squibb", "sector": "Healthcare", "industry": "Pharmaceuticals"},
    "CVS": {"name": "CVS Health Corp.", "sector": "Healthcare", "industry": "Healthcare Plans / Pharmacy"},
    "WMT": {"name": "Walmart Inc.", "sector": "Consumer Defensive", "industry": "Discount Retail"},
    "COST": {"name": "Costco Wholesale", "sector": "Consumer Defensive", "industry": "Warehouse Clubs"},
    "PG": {"name": "Procter & Gamble", "sector": "Consumer Defensive", "industry": "Household Products"},
    "KO": {"name": "Coca-Cola Co.", "sector": "Consumer Defensive", "industry": "Beverages"},
    "PEP": {"name": "PepsiCo Inc.", "sector": "Consumer Defensive", "industry": "Beverages / Snacks"},
    "MDLZ": {"name": "Mondelez International", "sector": "Consumer Defensive", "industry": "Packaged Foods"},
    "CL": {"name": "Colgate-Palmolive", "sector": "Consumer Defensive", "industry": "Household Products"},
    "MO": {"name": "Altria Group", "sector": "Consumer Defensive", "industry": "Tobacco"},
    "CAT": {"name": "Caterpillar Inc.", "sector": "Industrials", "industry": "Construction Equipment"},
    "HON": {"name": "Honeywell International", "sector": "Industrials", "industry": "Industrial Conglomerate"},
    "UNP": {"name": "Union Pacific Corp.", "sector": "Industrials", "industry": "Railroads"},
    "BA": {"name": "Boeing Co.", "sector": "Industrials", "industry": "Aerospace & Defense"},
    "RTX": {"name": "RTX Corp.", "sector": "Industrials", "industry": "Aerospace & Defense"},
    "LMT": {"name": "Lockheed Martin", "sector": "Industrials", "industry": "Aerospace & Defense"},
    "DE": {"name": "Deere & Co.", "sector": "Industrials", "industry": "Farm Machinery"},
    "GE": {"name": "GE Aerospace", "sector": "Industrials", "industry": "Aerospace & Defense"},
    "GD": {"name": "General Dynamics", "sector": "Industrials", "industry": "Aerospace & Defense"},
    "EMR": {"name": "Emerson Electric", "sector": "Industrials", "industry": "Industrial Automation"},
    "FDX": {"name": "FedEx Corp.", "sector": "Industrials", "industry": "Logistics / Freight"},
    "UPS": {"name": "United Parcel Service", "sector": "Industrials", "industry": "Logistics / Freight"},
    "MMM": {"name": "3M Company", "sector": "Industrials", "industry": "Industrial Conglomerate"},
    "CMCSA": {"name": "Comcast Corp.", "sector": "Communication Services", "industry": "Cable / Telecom"},
    "VZ": {"name": "Verizon Communications", "sector": "Communication Services", "industry": "Telecom"},
    "T": {"name": "AT&T Inc.", "sector": "Communication Services", "industry": "Telecom"},
    "CHTR": {"name": "Charter Communications", "sector": "Communication Services", "industry": "Cable"},
    "BRK-B": {"name": "Berkshire Hathaway", "sector": "Financial Services", "industry": "Diversified Holding"},
    "ACN": {"name": "Accenture plc", "sector": "Technology", "industry": "IT Consulting"},
    "IBM": {"name": "IBM Corp.", "sector": "Technology", "industry": "IT Services / Cloud"},
    "PYPL": {"name": "PayPal Holdings", "sector": "Financial Services", "industry": "Digital Payments"},
    "LIN": {"name": "Linde plc", "sector": "Basic Materials", "industry": "Industrial Gases"},
    "NEE": {"name": "NextEra Energy", "sector": "Utilities", "industry": "Electric Utilities / Renewables"},
    "SO": {"name": "Southern Company", "sector": "Utilities", "industry": "Electric Utilities"},
    "DUK": {"name": "Duke Energy", "sector": "Utilities", "industry": "Electric Utilities"},
    "AMT": {"name": "American Tower Corp.", "sector": "Real Estate", "industry": "Cell Tower REITs"},
    "SPG": {"name": "Simon Property Group", "sector": "Real Estate", "industry": "Retail REITs"},
    "PLTR": {"name": "Palantir Technologies", "sector": "Technology", "industry": "Data Analytics / AI"},
    "TMUS": {"name": "T-Mobile US", "sector": "Communication Services", "industry": "Wireless Telecom"},
    "PM": {"name": "Philip Morris International", "sector": "Consumer Defensive", "industry": "Tobacco"},
    "AMAT": {"name": "Applied Materials", "sector": "Technology", "industry": "Semiconductor Equipment"},
}


def _get_stock_meta(ticker: str) -> Dict[str, str]:
    """Return company name, sector, and industry for a ticker."""
    return STOCK_META.get(ticker, {"name": ticker, "sector": "Unknown", "industry": "Unknown"})


# ── Technical indicator calculator (standalone, no FastAPI dependency) ──
def calculate_technicals(df: pd.DataFrame) -> Dict:
    """Calculate key technicals from OHLCV DataFrame."""
    if df is None or len(df) < 20:
        return {}
    try:
        close = df["close"] if "close" in df.columns else df["Close"]
        high = df["high"] if "high" in df.columns else df["High"]
        low = df["low"] if "low" in df.columns else df["Low"]
        volume = df["volume"] if "volume" in df.columns else df["Volume"]

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()

        # Bollinger
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()

        # SMAs
        sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
        sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
        vol_sma = volume.rolling(20).mean().iloc[-1]

        # Price performance
        close_now = float(close.iloc[-1])
        perf_1w = float((close_now / close.iloc[-5] - 1) * 100) if len(close) >= 5 else None
        perf_1m = float((close_now / close.iloc[-21] - 1) * 100) if len(close) >= 21 else None
        perf_3m = float((close_now / close.iloc[-63] - 1) * 100) if len(close) >= 63 else None

        # 52-week high/low
        high_52w = float(high.iloc[-252:].max()) if len(high) >= 252 else float(high.max())
        low_52w = float(low.iloc[-252:].min()) if len(low) >= 252 else float(low.min())

        return {
            "RSI": float(rsi),
            "MACD": float(macd.iloc[-1]),
            "MACD_Signal": float(signal.iloc[-1]),
            "Bollinger_Upper": float((sma20 + 2 * std20).iloc[-1]),
            "Bollinger_Lower": float((sma20 - 2 * std20).iloc[-1]),
            "SMA_20": float(sma20.iloc[-1]),
            "SMA_50": float(sma50) if sma50 is not None else None,
            "SMA_200": float(sma200) if sma200 is not None else None,
            "EMA_12": float(ema12.iloc[-1]),
            "EMA_26": float(ema26.iloc[-1]),
            "Close": close_now,
            "Volume": float(volume.iloc[-1]),
            "Volume_SMA": float(vol_sma),
            "Perf_1W": perf_1w,
            "Perf_1M": perf_1m,
            "Perf_3M": perf_3m,
            "High_52W": high_52w,
            "Low_52W": low_52w,
        }
    except Exception as e:
        logger.warning("Technicals failed: %s", e)
        return {}


# API Provider Selection: Use GROQ_API_KEY if available (better free tier), else fall back to Gemini
USE_GROQ = os.getenv("GROQ_API_KEY") is not None
USE_GEMINI = os.getenv("GOOGLE_API_KEY") is not None

# Groq models (much better free tier: 1K-14K RPD vs Gemini's 20 RPD)
GROQ_MODEL_FALLBACK_CHAIN = [
    "llama-3.3-70b-versatile",  # Best quality: 1K RPD (vs Gemini's 20)
    "llama-3.1-8b-instant",      # Fast fallback: 14.4K RPD
]

# Gemini models (free tier: 20 RPD max)
GEMINI_MODEL_FALLBACK_CHAIN = [
    "gemini-2.5-pro",      # Free tier: 0 RPD (not available)
    "gemini-2.5-flash",    # Free tier: 20 RPD
    "gemini-2.5-flash-lite",  # Free tier: 20 RPD
]

# Select provider and model (Groq preferred if both are available)
if USE_GROQ:
    _MODEL_FALLBACK_CHAIN = GROQ_MODEL_FALLBACK_CHAIN
    API_PROVIDER = "groq"
    DEFAULT_MODEL = GROQ_MODEL_FALLBACK_CHAIN[0]
elif USE_GEMINI:
    _MODEL_FALLBACK_CHAIN = GEMINI_MODEL_FALLBACK_CHAIN
    API_PROVIDER = "gemini"
    DEFAULT_MODEL = os.getenv("GEMINI_MODEL", GEMINI_MODEL_FALLBACK_CHAIN[1])  # Skip pro (0 RPD)
else:
    _MODEL_FALLBACK_CHAIN = []
    API_PROVIDER = None
    DEFAULT_MODEL = None

MAX_RETRIES = 5
INITIAL_BACKOFF = 60
MAX_BACKOFF = 300  # Max 5 minutes between retries
BASE_BACKOFF = 2   # Exponential base

# Global rate limiter: track last API call time per model to prevent hammering
_last_api_call_time: Dict[str, float] = {}
_MIN_CALL_INTERVAL = 2.5  # Minimum seconds between API calls per model

# Per-model RPD tracking within this process run
_model_rpd_count: Dict[str, int] = {}
_MODEL_RPD_LIMITS: Dict[str, int] = {
    # Groq models (much higher limits)
    "llama-3.3-70b-versatile": 900,    # leave headroom vs 1K limit
    "llama-3.1-8b-instant": 14000,     # leave headroom vs 14.4K limit
    # Gemini models (very low free tier limits)
    "gemini-2.5-pro": 0,               # Free tier: 0 RPD (not available)
    "gemini-2.5-flash": 18,             # leave headroom vs 20 limit
    "gemini-2.5-flash-lite": 18,
}


def _pick_model() -> str:
    """Pick the best available model based on RPD usage tracking.

    If model env var is explicitly set by the user, lock to that model (no fallback).
    Otherwise cycle through the fallback chain for the selected provider.
    """
    if not _MODEL_FALLBACK_CHAIN:
        return None
    
    # Check for forced model selection
    if API_PROVIDER == "groq":
        forced = os.getenv("GROQ_MODEL")
    else:
        forced = os.getenv("GEMINI_MODEL")
    
    if forced:
        return forced

    for model in _MODEL_FALLBACK_CHAIN:
        used = _model_rpd_count.get(model, 0)
        limit = _MODEL_RPD_LIMITS.get(model, 900)
        if used < limit:
            return model
    return _MODEL_FALLBACK_CHAIN[0] if _MODEL_FALLBACK_CHAIN else None


def _parse_retry_after(error_str: str) -> int:
    """Try to extract a Retry-After hint (in seconds) from the error message."""
    import re
    m = re.search(r"retry.*?(\d+)\s*s", error_str, re.IGNORECASE)
    if m:
        return min(int(m.group(1)), MAX_BACKOFF)
    return 0


def _exponential_backoff(attempt: int) -> float:
    """Calculate exponential backoff delay in seconds."""
    delay = INITIAL_BACKOFF * (BASE_BACKOFF ** (attempt - 1))
    return min(delay, MAX_BACKOFF)


def _rate_limit_wait(model: str):
    """Wait if needed to respect minimum call interval per model."""
    import time
    last_call = _last_api_call_time.get(model, 0)
    elapsed = time.time() - last_call
    if elapsed < _MIN_CALL_INTERVAL:
        sleep_time = _MIN_CALL_INTERVAL - elapsed
        time.sleep(sleep_time)
    _last_api_call_time[model] = time.time()


def _call_groq(prompt: str, ticker: str, model: str) -> tuple[str, Optional[str]]:
    """
    Call Groq API (much better free tier limits).
    
    Returns (explanation_text, error_type) where error_type is None on success,
    or "quota_exceeded", "rate_limit", or "api_error" on failure.
    """
    try:
        from groq import Groq
    except ImportError:
        return ("AI explanation unavailable: groq package not installed", "api_error")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return ("AI explanation unavailable: GROQ_API_KEY not set", "api_error")
    
    client = Groq(api_key=api_key)
    
    try:
        # Rate limit: wait if needed before making API call
        _rate_limit_wait(model)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial analyst providing clear, concise stock market explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048,
        )
        
        if response and response.choices and len(response.choices) > 0:
            explanation_text = response.choices[0].message.content
            if explanation_text:
                _model_rpd_count[model] = _model_rpd_count.get(model, 0) + 1
                logger.info(
                    "Groq %s response for %s: %d chars (RPD usage: %d)",
                    model, ticker, len(explanation_text), _model_rpd_count[model],
                )
                return (explanation_text, None)
        
        return ("AI explanation unavailable: empty response from Groq", "api_error")
        
    except Exception as e:
        error_str = str(e).lower()
        is_rate_limit = "429" in error_str or "rate limit" in error_str or "resource_exhausted" in error_str
        is_quota = "quota" in error_str or "limit" in error_str
        
        if is_quota:
            return (f"AI explanation unavailable: Groq API quota exceeded ({model})", "quota_exceeded")
        elif is_rate_limit:
            return (f"AI explanation unavailable: Groq API rate limited ({model})", "rate_limit")
        else:
            logger.error("Groq API error for %s (%s): %s", ticker, model, e)
            return (f"AI explanation unavailable: Groq API error: {str(e)[:100]}", "api_error")


def _call_gemini(prompt: str, ticker: str, model: str) -> tuple[str, Optional[str]]:
    """
    Synchronous Gemini API call with automatic model fallback and RPD tracking.

    Returns (explanation_text, error_type) where error_type is None on success,
    or "quota_exceeded" (daily limit hit), "rate_limit" (transient 429),
    or "api_error" on permanent failure.
    """
    from google import genai
    from google.genai import types

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return ("AI explanation unavailable: GOOGLE_API_KEY not set", "api_error")

    client = genai.Client(api_key=api_key)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Rate limit: wait if needed before making API call
            _rate_limit_wait(model)
            
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=2000)
                ),
            )
            if response and response.text:
                _model_rpd_count[model] = _model_rpd_count.get(model, 0) + 1
                logger.info(
                    "%s response for %s: %d chars (RPD usage: %d)",
                    model, ticker, len(response.text), _model_rpd_count[model],
                )
                return (response.text, None)
            return ("AI explanation unavailable: empty response", "api_error")

        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "429" in error_str or "rate limit" in error_str or "resource_exhausted" in error_str
            is_quota = "quota" in error_str or "limit: 0" in error_str

            if is_quota or (is_rate_limit and attempt >= MAX_RETRIES):
                # Mark this model as exhausted and try the next one in the chain
                _model_rpd_count[model] = _MODEL_RPD_LIMITS.get(model, 9999)
                next_model = _pick_model()
                if next_model != model:
                    logger.warning(
                        "%s quota exhausted for %s — falling back to %s",
                        model, ticker, next_model,
                    )
                    model = next_model
                    # Reset attempt counter when switching models
                    attempt = 0
                    continue

                logger.error(
                    "All Gemini models exhausted for %s after %d attempts: %s",
                    ticker, attempt, e,
                )
                return (
                    f"AI explanation unavailable: Gemini API quota exceeded (all models at limit)",
                    "quota_exceeded",
                )

            if is_rate_limit and attempt < MAX_RETRIES:
                # Use exponential backoff with retry-after hint if available
                retry_after = _parse_retry_after(error_str)
                if retry_after > 0:
                    retry_secs = retry_after
                else:
                    retry_secs = _exponential_backoff(attempt)
                
                logger.warning(
                    "%s rate-limited for %s (attempt %d/%d) — sleeping %ds",
                    model, ticker, attempt, MAX_RETRIES, retry_secs,
                )
                time.sleep(retry_secs)
                continue

            logger.error("%s API error for %s: %s", model, ticker, e)
            return (f"AI explanation unavailable: {e}", "api_error")

    return ("AI explanation unavailable: max retries exceeded", "quota_exceeded")


def _call_llm_api(prompt: str, ticker: str) -> tuple[str, Optional[str]]:
    """
    Unified LLM API caller that uses Groq (preferred) or Gemini (fallback).
    
    Returns (explanation_text, error_type) where error_type is None on success,
    or "quota_exceeded", "rate_limit", or "api_error" on failure.
    """
    global API_PROVIDER, _MODEL_FALLBACK_CHAIN
    
    if not API_PROVIDER:
        return ("AI explanation unavailable: No API provider configured (set GROQ_API_KEY or GOOGLE_API_KEY)", "api_error")
    
    model = _pick_model()
    if not model:
        return ("AI explanation unavailable: No available models", "quota_exceeded")
    
    # Try Groq first if available
    if API_PROVIDER == "groq":
        current_model = model
        for attempt in range(1, MAX_RETRIES + 1):
            explanation_text, error_type = _call_groq(prompt, ticker, current_model)
            
            if error_type is None:
                return (explanation_text, None)
            
            if error_type == "quota_exceeded":
                # Try next model in fallback chain
                _model_rpd_count[current_model] = _MODEL_RPD_LIMITS.get(current_model, 9999)
                next_model = _pick_model()
                if next_model != current_model and next_model:
                    logger.warning(
                        "Groq %s quota exhausted for %s — falling back to %s",
                        current_model, ticker, next_model,
                    )
                    current_model = next_model
                    continue
                else:
                    # All Groq models exhausted, try Gemini if available
                    if USE_GEMINI:
                        logger.warning("All Groq models exhausted, falling back to Gemini")
                        # Temporarily switch to Gemini fallback chain
                        original_chain = _MODEL_FALLBACK_CHAIN
                        original_provider = API_PROVIDER
                        _MODEL_FALLBACK_CHAIN = GEMINI_MODEL_FALLBACK_CHAIN
                        API_PROVIDER = "gemini"
                        gemini_model = _pick_model()
                        if gemini_model and gemini_model != "gemini-2.5-pro":  # Skip pro (0 RPD)
                            result = _call_gemini(prompt, ticker, gemini_model)
                            _MODEL_FALLBACK_CHAIN = original_chain  # Restore original chain
                            API_PROVIDER = original_provider
                            return result
                        _MODEL_FALLBACK_CHAIN = original_chain  # Restore original chain
                        API_PROVIDER = original_provider
                    return (explanation_text, error_type)
            
            elif error_type == "rate_limit" and attempt < MAX_RETRIES:
                retry_after = _parse_retry_after(explanation_text)
                retry_secs = retry_after if retry_after > 0 else _exponential_backoff(attempt)
                logger.warning(
                    "Groq %s rate-limited for %s (attempt %d/%d) — sleeping %ds",
                    current_model, ticker, attempt, MAX_RETRIES, retry_secs,
                )
                time.sleep(retry_secs)
                continue
            
            return (explanation_text, error_type)
        
        return ("AI explanation unavailable: max retries exceeded", "quota_exceeded")
    
    # Fall back to Gemini
    elif API_PROVIDER == "gemini":
        return _call_gemini(prompt, ticker, model)
    
    return ("AI explanation unavailable: Unknown API provider", "api_error")


def _get_macro_context(mongo) -> Dict:
    """Fetch the latest macro economic indicators from MongoDB.

    The macro_data_raw collection stores documents in a flat format where
    date strings are keys: {"indicator": "RETAIL_SALES", "source": "FRED",
    "2024-01-01": 539834, "2024-02-01": 544169, ...}.
    We also check the macro_data collection which may use a similar schema.
    """
    macro_context = {}
    try:
        # Try both collections
        for coll_name in ("macro_data_raw", "macro_data"):
            try:
                coll = mongo.db[coll_name]
            except Exception:
                continue

            indicators = ["FEDERAL_FUNDS_RATE", "CPI", "UNEMPLOYMENT", "NONFARM_PAYROLL",
                          "RETAIL_SALES", "TREASURY_10Y", "TREASURY_2Y", "GDP"]
            for indicator in indicators:
                if indicator in macro_context:
                    continue
                try:
                    doc = coll.find_one({"indicator": indicator}, sort=[("_id", -1)])
                    if not doc:
                        continue
                    # Extract date-keyed numeric fields
                    date_fields = {}
                    for k, v in doc.items():
                        if k in ("_id", "indicator", "source", "date", "processed"):
                            continue
                        if isinstance(v, (int, float)):
                            date_fields[k] = v
                    if date_fields:
                        latest_date = max(date_fields.keys())
                        macro_context[indicator] = {"value": date_fields[latest_date], "date": latest_date}
                except Exception:
                    pass
    except Exception as e:
        logger.warning("Macro context fetch failed: %s", e)
    return macro_context


def _get_insider_context(mongo, ticker: str) -> Dict:
    """Fetch recent insider trading activity from MongoDB."""
    insider_ctx = {}
    try:
        coll = mongo.db["insider_transactions"]
        recent = list(coll.find(
            {"symbol": ticker},
            {"filingDate": 1, "transactionCode": 1, "change": 1, "name": 1,
             "transactionPrice": 1, "share": 1, "_id": 0}
        ).sort("filingDate", -1).limit(10))
        if recent:
            buys = [t for t in recent if t.get("transactionCode", "").upper() in ("P", "M", "A")]
            sells = [t for t in recent if t.get("transactionCode", "").upper() in ("S", "D")]
            insider_ctx = {
                "recent_transactions": len(recent),
                "buys": len(buys),
                "sells": len(sells),
                "latest_transactions": recent[:5],
            }
    except Exception as e:
        logger.warning("Insider context fetch failed for %s: %s", ticker, e)
    return insider_ctx


def _get_short_interest_context(mongo, ticker: str) -> Dict:
    """Fetch short interest data from the dedicated collection first, then sentiment."""
    short_ctx = {}
    try:
        # Try direct short_interest_data collection first (more granular)
        si_col = mongo.db.get_collection("short_interest_data")
        if si_col is not None:
            latest = si_col.find_one({"ticker": ticker.upper()}, sort=[("fetched_at", -1)])
            if latest:
                short_ctx = {
                    "short_float_pct": latest.get("short_float_pct", latest.get("shortFloatPct", 0)),
                    "days_to_cover": latest.get("daysToCover", latest.get("days_to_cover", 0)),
                    "short_interest": latest.get("short_interest", latest.get("interest", 0)),
                    "settlement_date": latest.get("settlementDate", ""),
                }
        # Fallback to sentiment collection
        if not short_ctx:
            sent = mongo.get_latest_sentiment(ticker) or {}
            sources = sent.get("sources", {})
            si_data = sources.get("short_interest", {})
            if isinstance(si_data, dict) and si_data:
                short_ctx = {
                    "short_float_pct": si_data.get("short_float_percentage", 0),
                    "days_to_cover": si_data.get("days_to_cover", 0),
                    "sentiment_score": si_data.get("score", 0),
                }
    except Exception as e:
        logger.warning("Short interest context failed for %s: %s", ticker, e)
    return short_ctx


def _get_financials_context(mongo, ticker: str) -> Dict:
    """Fetch Finnhub basic financials (P/E, margins, etc.) from MongoDB."""
    fin_ctx = {}
    try:
        col = mongo.db.get_collection("finnhub_basic_financials")
        if col is not None:
            doc = col.find_one({"ticker": ticker.upper()}, sort=[("fetched_at", -1)])
            if not doc:
                doc = col.find_one({"ticker": ticker.upper()})
            if doc and isinstance(doc.get("data"), dict):
                metric = doc["data"].get("metric", doc["data"])
                fin_ctx = {
                    "pe_ratio": metric.get("peBasicExclExtraTTM") or metric.get("peTTM"),
                    "pb_ratio": metric.get("pbAnnual") or metric.get("pbQuarterly"),
                    "dividend_yield": metric.get("dividendYieldIndicatedAnnual"),
                    "roe": metric.get("roeTTM"),
                    "market_cap": metric.get("marketCapitalization"),
                    "52w_high": metric.get("52WeekHigh"),
                    "52w_low": metric.get("52WeekLow"),
                    "beta": metric.get("beta"),
                }
                fin_ctx = {k: v for k, v in fin_ctx.items() if v is not None}
    except Exception as e:
        logger.warning("Financials context failed for %s: %s", ticker, e)
    return fin_ctx


def _get_recent_news(mongo, ticker: str) -> List[Dict]:
    """Fetch recent news headlines from the aggregated_news collection.

    This supplements sentiment-based headlines with actual stored news articles,
    solving the problem where sentiment docs lack raw headline data but the
    news aggregation pipeline has fresh articles.
    """
    headlines: List[Dict] = []
    try:
        for coll_name in ("aggregated_news", "news_articles", "rss_news"):
            try:
                coll = mongo.db[coll_name]
            except Exception:
                continue
            cutoff = datetime.utcnow() - timedelta(days=7)
            query = {
                "$or": [
                    {"tickers": ticker.upper()},
                    {"symbols": ticker.upper()},
                    {"ticker": ticker.upper()},
                ],
                "published_at": {"$gte": cutoff},
            }
            docs = list(coll.find(query).sort("published_at", -1).limit(8))
            for doc in docs:
                title = doc.get("title", "")
                if title and title not in [h.get("title") for h in headlines]:
                    headlines.append({
                        "title": title,
                        "source": doc.get("source", doc.get("provider", "")),
                        "published_at": str(doc.get("published_at", "")),
                        "sentiment": doc.get("sentiment", "neutral"),
                    })
            if headlines:
                break
    except Exception as e:
        logger.warning("Recent news fetch failed for %s: %s", ticker, e)
    return headlines[:8]


def _get_fmp_context(mongo, ticker: str) -> Dict:
    """Fetch FMP earnings, ratings, and price target data from MongoDB."""
    fmp_ctx = {}
    try:
        col = mongo.db.get_collection("alpha_vantage_data")
        if col is None:
            return fmp_ctx

        # Earnings
        earnings_doc = col.find_one({"ticker": ticker.upper(), "endpoint": "fmp_earnings"}, sort=[("timestamp", -1)])
        if earnings_doc and isinstance(earnings_doc.get("data"), list) and earnings_doc["data"]:
            latest_e = earnings_doc["data"][0]
            fmp_ctx["latest_earnings"] = {
                "eps_actual": latest_e.get("eps"),
                "eps_estimated": latest_e.get("epsEstimated"),
                "revenue": latest_e.get("revenue"),
                "date": latest_e.get("date"),
            }
            if latest_e.get("eps") and latest_e.get("epsEstimated"):
                fmp_ctx["earnings_surprise"] = latest_e["eps"] - latest_e["epsEstimated"]

        # Ratings
        rating_doc = col.find_one({"ticker": ticker.upper(), "endpoint": "fmp_ratings-snapshot"}, sort=[("timestamp", -1)])
        if rating_doc and isinstance(rating_doc.get("data"), list) and rating_doc["data"]:
            rd = rating_doc["data"][0]
            fmp_ctx["rating_score"] = rd.get("ratingScore")
            fmp_ctx["rating_recommendation"] = rd.get("ratingRecommendation")

        # Price target
        pt_doc = col.find_one({"ticker": ticker.upper(), "endpoint": "fmp_price-target-summary"}, sort=[("timestamp", -1)])
        if pt_doc and isinstance(pt_doc.get("data"), list) and pt_doc["data"]:
            pt = pt_doc["data"][0]
            fmp_ctx["analyst_avg_target"] = pt.get("lastYearAvgPriceTarget")
            fmp_ctx["analyst_count"] = pt.get("lastYearCount")
    except Exception as e:
        logger.warning("FMP context failed for %s: %s", ticker, e)
    return fmp_ctx


def _build_prompt(
    ticker: str,
    date: str,
    predictions: Dict,
    sentiment: Dict,
    technicals: Dict,
    shap_data: Optional[Dict],
    macro_context: Optional[Dict] = None,
    insider_context: Optional[Dict] = None,
    short_interest: Optional[Dict] = None,
    financials_context: Optional[Dict] = None,
    fmp_context: Optional[Dict] = None,
    recent_news: Optional[List[Dict]] = None,
) -> str:
    """Build a comprehensive explanation prompt using all available data."""
    sections = []

    # ── 1. PRICE & PREDICTION OVERVIEW ──
    current_price = None
    pred_lines = []
    for window, label in [("next_day", "Next Day"), ("7_day", "1 Week"), ("30_day", "1 Month")]:
        pd_ = predictions.get(window, {})
        if isinstance(pd_, dict) and pd_:
            cp = pd_.get("current_price", 0)
            if cp and not current_price:
                current_price = cp
            pp = pd_.get("predicted_price", 0)
            pc = pd_.get("price_change", 0)
            conf = pd_.get("confidence", 0)
            prob_up = pd_.get("prob_positive", pd_.get("prob_up", 0))
            alpha = pd_.get("alpha_pct", 0)
            pr = pd_.get("price_range", {})
            trade_rec = pd_.get("trade_recommended", 0)

            line = f"  {label}: predicted ${pp:.2f}"
            if pc:
                line += f" (change ${pc:+.2f})"
            if alpha:
                line += f", alpha vs SPY: {alpha:+.2f}%"
            line += f", confidence: {conf:.1%}, prob up: {prob_up:.1%}"
            if pr:
                line += f", range: ${pr.get('low', 0):.2f}-${pr.get('high', 0):.2f}"
            if trade_rec:
                line += " [TRADE RECOMMENDED]"
            pred_lines.append(line)

    meta = _get_stock_meta(ticker)
    header = f"STOCK: {ticker} ({meta['name']}) | Sector: {meta['sector']} | Industry: {meta['industry']}"
    if current_price:
        header += f" | Current Price: ${current_price:.2f}"
    header += f" | Date: {date}"
    sections.append(header + "\n\nML PRICE PREDICTIONS:\n" + "\n".join(pred_lines))

    # ── 2. TECHNICAL ANALYSIS ──
    if technicals:
        rsi = technicals.get("RSI")
        macd = technicals.get("MACD")
        macd_sig = technicals.get("MACD_Signal")
        bb_upper = technicals.get("Bollinger_Upper")
        bb_lower = technicals.get("Bollinger_Lower")
        sma20 = technicals.get("SMA_20")
        sma50 = technicals.get("SMA_50")
        sma200 = technicals.get("SMA_200")
        close = technicals.get("Close", current_price or 0)
        vol = technicals.get("Volume", 0)
        vol_sma = technicals.get("Volume_SMA", 0)
        perf_1w = technicals.get("Perf_1W")
        perf_1m = technicals.get("Perf_1M")
        perf_3m = technicals.get("Perf_3M")
        high_52w = technicals.get("High_52W")
        low_52w = technicals.get("Low_52W")

        tech_lines = ["TECHNICAL ANALYSIS:"]
        if rsi is not None:
            rsi_label = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
            tech_lines.append(f"  RSI: {rsi:.1f} ({rsi_label})")
        if macd is not None and macd_sig is not None:
            macd_cross = "bullish crossover" if macd > macd_sig else "bearish crossunder"
            tech_lines.append(f"  MACD: {macd:.2f} vs Signal {macd_sig:.2f} ({macd_cross})")
        if sma20 and close:
            pct_from_sma20 = ((close / sma20) - 1) * 100
            tech_lines.append(f"  Price vs SMA-20: {pct_from_sma20:+.1f}% ({'above' if pct_from_sma20 > 0 else 'below'})")
        if sma50 and close:
            pct_from_sma50 = ((close / sma50) - 1) * 100
            tech_lines.append(f"  Price vs SMA-50: {pct_from_sma50:+.1f}% ({'above' if pct_from_sma50 > 0 else 'below'})")
        if sma200 and close:
            pct_from_sma200 = ((close / sma200) - 1) * 100
            tech_lines.append(f"  Price vs SMA-200: {pct_from_sma200:+.1f}% ({'above' if pct_from_sma200 > 0 else 'below'})")
        if bb_upper and bb_lower and close:
            bb_width_pct = ((bb_upper - bb_lower) / close) * 100
            bb_pos = "near upper band (potential resistance)" if close > (bb_upper - (bb_upper - bb_lower) * 0.2) else \
                     "near lower band (potential support)" if close < (bb_lower + (bb_upper - bb_lower) * 0.2) else "mid-range"
            tech_lines.append(f"  Bollinger Bands: ${bb_lower:.2f} - ${bb_upper:.2f} ({bb_pos}, width {bb_width_pct:.1f}%)")
        if vol and vol_sma:
            vol_ratio = vol / vol_sma if vol_sma > 0 else 1
            vol_label = "above average" if vol_ratio > 1.2 else "below average" if vol_ratio < 0.8 else "normal"
            tech_lines.append(f"  Volume: {vol:,.0f} vs avg {vol_sma:,.0f} ({vol_label}, {vol_ratio:.1f}x)")

        perf_parts = []
        if perf_1w is not None:
            perf_parts.append(f"1W: {perf_1w:+.1f}%")
        if perf_1m is not None:
            perf_parts.append(f"1M: {perf_1m:+.1f}%")
        if perf_3m is not None:
            perf_parts.append(f"3M: {perf_3m:+.1f}%")
        if perf_parts:
            tech_lines.append(f"  Recent Performance: {', '.join(perf_parts)}")
        if high_52w and low_52w and close:
            pct_from_high = ((close / high_52w) - 1) * 100
            pct_from_low = ((close / low_52w) - 1) * 100
            tech_lines.append(f"  52-Week Range: ${low_52w:.2f} - ${high_52w:.2f} ({pct_from_high:+.1f}% from high, {pct_from_low:+.1f}% from low)")

        sections.append("\n".join(tech_lines))

    # ── 3. SENTIMENT & NEWS ──
    blended = sentiment.get("blended_sentiment", 0)
    sources = sentiment.get("sources", {})
    sent_lines = ["NEWS & SENTIMENT:"]
    sent_lines.append(f"  Overall sentiment score: {blended:.3f} ({'bullish' if blended > 0.1 else 'bearish' if blended < -0.1 else 'neutral'})")

    for src, data in sources.items():
        if isinstance(data, dict) and data.get("volume", 0) > 0:
            sent_lines.append(f"  {src}: score={data.get('score', 0):.3f}, articles/posts={data.get('volume', 0)}")

    # News headlines
    news_headlines = []
    finviz_headlines = sentiment.get("finviz_raw_data", [])
    if isinstance(finviz_headlines, list):
        for h in finviz_headlines[:5]:
            if isinstance(h, str) and h.strip():
                news_headlines.append(h.strip())
            elif isinstance(h, dict) and h.get("title"):
                news_headlines.append(h["title"].strip())
    rss_news = sentiment.get("rss_news_raw_data", [])
    if isinstance(rss_news, list):
        for item in rss_news[:5]:
            title = item.get("title", "") if isinstance(item, dict) else str(item)
            if title.strip() and title.strip() not in news_headlines:
                news_headlines.append(title.strip())
    reddit_posts = sentiment.get("reddit_raw_data", [])
    if isinstance(reddit_posts, list):
        for item in reddit_posts[:3]:
            title = item.get("title", "") if isinstance(item, dict) else str(item)
            if title.strip():
                news_headlines.append(f"[Reddit] {title.strip()}")
    marketaux = sentiment.get("marketaux_raw_data", [])
    if isinstance(marketaux, list):
        for item in marketaux[:3]:
            title = item.get("title", "") if isinstance(item, dict) else str(item)
            if title.strip() and title.strip() not in news_headlines:
                news_headlines.append(title.strip())

    # Supplement with recent_news from aggregated news collection
    if recent_news:
        for item in recent_news:
            title = item.get("title", "")
            if title and title not in news_headlines:
                source = item.get("source", "")
                suffix = f" ({source})" if source else ""
                news_headlines.append(f"{title}{suffix}")

    if news_headlines:
        sent_lines.append("  Recent headlines:")
        for h in news_headlines[:8]:
            sent_lines.append(f"    - {h}")
    else:
        sent_lines.append("  No recent news headlines available.")

    sections.append("\n".join(sent_lines))

    # ── 4. ML MODEL DRIVERS (SHAP) ──
    # Provide SHAP data but instruct AI to translate it into plain English
    if shap_data:
        pos = shap_data.get("top_positive_contrib", [])
        neg = shap_data.get("top_negative_contrib", [])
        global_imp = shap_data.get("global_gain_importance", [])
        shap_lines = ["ML MODEL KEY DRIVERS (translate these into plain English for users):"]

        if pos:
            shap_lines.append("  Bullish factors:")
            for f in pos[:6]:
                fname = _friendly_feature_name(f.get("feature", "?"))
                contrib = f.get("contrib", 0)
                value = f.get("value")
                # Provide data but don't expose raw technical values in final output
                val_str = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
                shap_lines.append(f"    {fname}: contribution={contrib:+.4f}, current_value={val_str}")
        if neg:
            shap_lines.append("  Bearish factors:")
            for f in neg[:6]:
                fname = _friendly_feature_name(f.get("feature", "?"))
                contrib = f.get("contrib", 0)
                value = f.get("value")
                val_str = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
                shap_lines.append(f"    {fname}: contribution={contrib:+.4f}, current_value={val_str}")

        if global_imp:
            top_global = [g for g in global_imp[:5] if g.get("gain_pct", 0) > 0]
            if top_global:
                shap_lines.append("  Most influential factors (for context only - translate to plain English):")
                for g in top_global:
                    fname = _friendly_feature_name(g.get("feature", "?"))
                    shap_lines.append(f"    {fname}: relative_importance={g.get('gain_pct', 0):.1f}%")

        sections.append("\n".join(shap_lines))

    # ── 5. MACRO ECONOMIC CONTEXT ──
    if macro_context:
        macro_lines = ["MACRO ECONOMIC CONTEXT:"]
        display_map = {
            "FEDERAL_FUNDS_RATE": "Fed Funds Rate",
            "CPI": "CPI (Consumer Price Index)",
            "UNEMPLOYMENT": "Unemployment Rate",
            "NONFARM_PAYROLL": "Nonfarm Payrolls",
            "RETAIL_SALES": "Retail Sales",
            "TREASURY_10Y": "10-Year Treasury Yield",
            "TREASURY_2Y": "2-Year Treasury Yield",
            "GDP": "GDP",
        }
        for key, info in macro_context.items():
            name = display_map.get(key, key)
            value = info.get('value')
            # Handle non-numeric values (None, string, etc.)
            if isinstance(value, (int, float)):
                val_str = f"{value:,.2f}"
            else:
                val_str = str(value) if value is not None else "N/A"
            macro_lines.append(f"  {name}: {val_str} (as of {info.get('date', 'N/A')})")
        if "TREASURY_10Y" in macro_context and "TREASURY_2Y" in macro_context:
            val_10y = macro_context["TREASURY_10Y"].get("value")
            val_2y = macro_context["TREASURY_2Y"].get("value")
            if isinstance(val_10y, (int, float)) and isinstance(val_2y, (int, float)):
                spread = val_10y - val_2y
                inversion = " (INVERTED - recession signal)" if spread < 0 else ""
                macro_lines.append(f"  Yield Curve Spread (10Y-2Y): {spread:+.2f}%{inversion}")
        sections.append("\n".join(macro_lines))

    # ── 6. INSIDER TRADING ──
    if insider_context and insider_context.get("recent_transactions", 0) > 0:
        ins_lines = ["INSIDER TRADING (recent 90 days):"]
        ins_lines.append(f"  Total transactions: {insider_context['recent_transactions']}")
        ins_lines.append(f"  Buys: {insider_context.get('buys', 0)}, Sells: {insider_context.get('sells', 0)}")
        ratio_text = "net buying" if insider_context.get("buys", 0) > insider_context.get("sells", 0) else \
                     "net selling" if insider_context.get("sells", 0) > insider_context.get("buys", 0) else "balanced"
        ins_lines.append(f"  Pattern: {ratio_text}")
        for txn in insider_context.get("latest_transactions", [])[:3]:
            code = txn.get("transactionCode", "?")
            action = "BUY" if code.upper() in ("P", "M", "A") else "SELL" if code.upper() in ("S", "D") else code
            name = txn.get("name", "Unknown")
            change = txn.get("change", 0)
            price = txn.get("transactionPrice", 0)
            date_str = txn.get("filingDate", "")
            ins_lines.append(f"    {action} by {name}: {change:+,.0f} shares @ ${price:.2f} ({date_str})")
        sections.append("\n".join(ins_lines))

    # ── 7. SHORT INTEREST ──
    if short_interest and short_interest.get("short_float_pct", 0) > 0:
        si_lines = ["SHORT INTEREST:"]
        si_lines.append(f"  Short float: {short_interest['short_float_pct']:.2f}%")
        if short_interest.get("days_to_cover"):
            si_lines.append(f"  Days to cover: {short_interest['days_to_cover']:.1f}")
        if short_interest["short_float_pct"] > 10:
            si_lines.append("  Note: High short interest - potential for short squeeze or continued bearish pressure")
        sections.append("\n".join(si_lines))

    # ── 8. FUNDAMENTAL FINANCIALS (Finnhub) ──
    if financials_context:
        fin_lines = ["FUNDAMENTAL FINANCIALS:"]
        if financials_context.get("pe_ratio"):
            fin_lines.append(f"  P/E Ratio: {financials_context['pe_ratio']:.2f}")
        if financials_context.get("pb_ratio"):
            fin_lines.append(f"  P/B Ratio: {financials_context['pb_ratio']:.2f}")
        if financials_context.get("dividend_yield"):
            fin_lines.append(f"  Dividend Yield: {financials_context['dividend_yield']:.2f}%")
        if financials_context.get("roe"):
            fin_lines.append(f"  Return on Equity (ROE): {financials_context['roe']:.2f}%")
        if financials_context.get("market_cap"):
            mc = financials_context["market_cap"]
            if mc > 1e6:
                fin_lines.append(f"  Market Cap: ${mc / 1e3:.1f}T")
            elif mc > 1e3:
                fin_lines.append(f"  Market Cap: ${mc:.1f}B")
            else:
                fin_lines.append(f"  Market Cap: ${mc:.1f}M")
        if financials_context.get("beta"):
            fin_lines.append(f"  Beta: {financials_context['beta']:.2f}")
        if len(fin_lines) > 1:
            sections.append("\n".join(fin_lines))

    # ── 9. EARNINGS & ANALYST DATA (FMP) ──
    if fmp_context:
        fmp_lines = ["EARNINGS & ANALYST DATA:"]
        le = fmp_context.get("latest_earnings", {})
        if le:
            fmp_lines.append(f"  Latest Earnings ({le.get('date', 'N/A')}): EPS actual={le.get('eps_actual', 'N/A')}, estimated={le.get('eps_estimated', 'N/A')}")
            if fmp_context.get("earnings_surprise") is not None:
                surprise = fmp_context["earnings_surprise"]
                beat_miss = "beat" if surprise > 0 else "missed" if surprise < 0 else "met"
                fmp_lines.append(f"  Earnings surprise: ${surprise:+.2f} ({beat_miss} estimates)")
        if fmp_context.get("rating_recommendation"):
            fmp_lines.append(f"  Analyst Rating: {fmp_context['rating_recommendation']} (score {fmp_context.get('rating_score', 'N/A')})")
        if fmp_context.get("analyst_avg_target"):
            fmp_lines.append(f"  Avg Analyst Price Target: ${fmp_context['analyst_avg_target']:.2f} (from {fmp_context.get('analyst_count', '?')} analysts)")
        if len(fmp_lines) > 1:
            sections.append("\n".join(fmp_lines))

    # ── 10. INSTRUCTIONS (the actual prompt to Gemini) ──
    meta = _get_stock_meta(ticker)
    company_name = meta["name"]
    sector = meta["sector"]
    industry = meta["industry"]

    # Build sector-specific analysis guidance
    sector_guidance = ""
    if sector == "Technology":
        sector_guidance = "For this tech stock, pay special attention to AI/cloud growth narratives, semiconductor cycle positioning, valuation multiples relative to growth, and sector ETF rotation signals."
    elif sector == "Financial Services":
        sector_guidance = "For this financial stock, factor in interest rate sensitivity (yield curve shape, Fed funds rate), credit risk signals, capital ratios, and how bank earnings link to the macro cycle."
    elif sector == "Healthcare":
        sector_guidance = "For this healthcare stock, consider pipeline catalysts, FDA approval cycles, patent cliffs, drug pricing risks, and whether recent earnings beat/miss reflects one-time items or trend."
    elif sector == "Energy":
        sector_guidance = "For this energy stock, link analysis to oil/gas price trends, OPEC dynamics, capital discipline, and how macro data (CPI, GDP) affects energy demand expectations."
    elif sector == "Consumer Cyclical":
        sector_guidance = "For this consumer cyclical stock, focus on consumer spending trends (retail sales data), discretionary vs staple rotation, and how the macro cycle affects demand."
    elif sector == "Consumer Defensive":
        sector_guidance = "For this consumer staple, note its defensive characteristics — how it performs in risk-off environments, dividend stability, and pricing power vs inflation."
    elif sector == "Industrials":
        sector_guidance = "For this industrial stock, connect analysis to manufacturing PMI, capex cycles, infrastructure spending, and order backlog trends."
    elif sector == "Utilities":
        sector_guidance = "For this utility stock, highlight its interest-rate sensitivity (bond proxy), dividend yield relative to treasuries, and any renewable energy transition exposure."
    elif sector == "Communication Services":
        sector_guidance = "For this communication services stock, consider ad revenue trends, subscriber growth, content spending ROI, and regulatory risk."
    else:
        sector_guidance = "Relate the analysis to the company's specific industry dynamics and competitive position."

    # Extract key prediction numbers for consistency
    pred_30d = predictions.get("30_day", {})
    pred_7d = predictions.get("7_day", {})
    pred_next = predictions.get("next_day", {})
    
    # Use 30-day prediction as primary (most reliable)
    primary_pred = pred_30d if pred_30d else (pred_7d if pred_7d else pred_next)
    predicted_price = primary_pred.get("predicted_price", 0) if isinstance(primary_pred, dict) else 0
    price_change = primary_pred.get("price_change", 0) if isinstance(primary_pred, dict) else 0
    price_change_pct = (price_change / current_price * 100) if current_price and price_change else 0
    
    # Determine outlook from predictions
    if price_change_pct > 2:
        outlook_hint = "Bullish"
    elif price_change_pct > 0.5:
        outlook_hint = "Slightly Bullish"
    elif price_change_pct < -2:
        outlook_hint = "Bearish"
    elif price_change_pct < -0.5:
        outlook_hint = "Slightly Bearish"
    else:
        outlook_hint = "Neutral"
    
    # Calculate confidence from prediction confidence scores
    conf_scores = []
    for w in [pred_next, pred_7d, pred_30d]:
        if isinstance(w, dict) and w.get("confidence"):
            conf_scores.append(w["confidence"])
    avg_confidence = int(sum(conf_scores) / len(conf_scores) * 100) if conf_scores else 50
    
    # Format numeric values before f-string to avoid format specifier errors
    current_price_str = f"{current_price:.2f}" if isinstance(current_price, (int, float)) and current_price else "N/A"
    predicted_price_str = f"{predicted_price:.2f}" if isinstance(predicted_price, (int, float)) else "0.00"
    price_change_pct_str = f"{price_change_pct:+.2f}" if isinstance(price_change_pct, (int, float)) else "0.00"
    price_change_abs_str = f"{abs(price_change):.2f}" if isinstance(price_change, (int, float)) else "0.00"
    
    # Format prediction prices safely
    pred_next_price = pred_next.get("predicted_price", 0) if isinstance(pred_next, dict) else 0
    pred_next_str = f"{pred_next_price:.2f}" if isinstance(pred_next_price, (int, float)) else "0.00"
    pred_7d_price = pred_7d.get("predicted_price", 0) if isinstance(pred_7d, dict) else 0
    pred_7d_str = f"{pred_7d_price:.2f}" if isinstance(pred_7d_price, (int, float)) else "0.00"
    
    # Determine movement direction text
    movement_text = "slight declines" if price_change_pct < 0 else "modest gains" if price_change_pct > 0 else "minimal movement"
    pressure_text = "modest downward pressure" if price_change_pct < 0 else "moderate upward momentum"
    change_type = "decline" if price_change_pct < 0 else "gain"
    conditions_text = "potential headwinds" if price_change_pct < 0 else "supportive conditions"
    source_text = "sector volatility" if price_change_pct < 0 else "favorable market conditions"
    
    # Pre-compute Bollinger and prediction values to avoid invalid f-string format specifiers
    bb_lower_val = technicals.get("Bollinger_Lower", 0) if technicals else 0
    bb_upper_val = technicals.get("Bollinger_Upper", 0) if technicals else 0
    bb_lower_str = f"{bb_lower_val:.2f}" if isinstance(bb_lower_val, (int, float)) else "0.00"
    bb_upper_str = f"{bb_upper_val:.2f}" if isinstance(bb_upper_val, (int, float)) else "0.00"
    direction_text = "decline" if price_change_pct < 0 else "rise"
    abs_pct_str = f"{abs(price_change_pct):.1f}" if isinstance(price_change_pct, (int, float)) else "0.0"
    driver_text = "elevated sector volatility" if price_change_pct < 0 else "supportive market conditions"
    
    sections.append(f"""
INSTRUCTIONS: You are a professional equity analyst writing a clear, concise market intelligence briefing for {company_name} ({ticker}) — a {industry} company in the {sector} sector.

{sector_guidance}

Your audience is retail investors who want clear, actionable insights. Write like Bloomberg or professional investment platforms — professional but accessible. Be specific to {ticker}, not generic market commentary.

CRITICAL: Translate ALL technical ML data into plain English. NEVER mention:
- SHAP values, feature importance percentages, or model coefficients
- Raw numeric impacts like "+0.0009" or "impact: -0.0015"
- Technical ML jargon like "model driver context" or "feature importance"
- Internal model mechanics

Instead, convert technical signals into human meaning:
- "Treasury yield curve (2Y-10Y spread) at 41.5% importance, value 0.0000" → "Yield curve conditions are currently neutral"
- "Sector ETF volatility: 0.0179, impact -0.0015" → "Rising technology sector volatility is creating headwinds"
- "Feature importance 41.5%" → Just describe what the factor means, don't mention the percentage

OUTPUT FORMAT — use this EXACT structure. Plain text only, no markdown.

OVERALL_OUTLOOK: {outlook_hint}

CONFIDENCE: {avg_confidence}

PREDICTION: {price_change_pct_str}% avg. predicted move

EXPECTED_PRICE: ${predicted_price_str}

CURRENT_PRICE: ${current_price_str}

SUMMARY: [{company_name} ({ticker}) is currently trading at ${current_price_str}. The ML model predicts {movement_text} for {ticker} across all time horizons, with next day at ${pred_next_str}, 1 week at ${pred_7d_str}, and 1 month at ${predicted_price_str}. Reference ONE key driver from the data — a specific news headline, technical indicator, or macro factor — that explains the prediction.]

WHAT_THIS_MEANS: [2-3 sentences translating the prediction into investment implications. Be specific to {ticker}'s business model and sector. Example: "The model anticipates {ticker} may experience {pressure_text} in the near term, with a projected {change_type} of ${price_change_abs_str} over the next month. This suggests {conditions_text} from {source_text}."]

BULLISH_FACTORS: [2-4 bullet points, each starting with "+". Translate technical factors into plain English. Examples:
+ Treasury yield curve conditions: A positive yield curve spread supports {ticker}'s outlook
+ Price momentum: Currently trading above its 20-day moving average, indicating short-term strength
+ Insider activity: Recent buying by executives signals internal confidence]

BEARISH_FACTORS: [2-4 bullet points, each starting with "-". Translate technical factors into plain English. Examples:
- Sector volatility: Elevated volatility in the {sector.lower()} sector is creating headwinds
- Market sentiment: Rising fear levels (VIX elevated) suggest broader market caution
- Technical indicators: MACD shows weakening momentum signals]

NEWS_IMPACT: [1-2 sentences referencing SPECIFIC headlines from above. If news is available, quote or paraphrase actual headlines and explain relevance. If no news: "No significant recent news detected for {company_name}. The analysis relies on technical, quantitative, and macro factors."]

KEY_LEVELS: [Support around ${bb_lower_str} | Resistance around ${bb_upper_str}]

BOTTOM_LINE: [1-2 sentences. The single most important takeaway. Use the 30-day prediction: "{company_name} ({ticker}) is predicted to {direction_text} {abs_pct_str}% over the next month to ${predicted_price_str}, primarily driven by {driver_text} — watch the ${bb_lower_str} support level."]

STRICT RULES:
- Maximum 2000 characters total
- Use ONLY data provided above. NEVER invent numbers
- Translate ALL technical ML jargon into plain English
- NEVER mention SHAP %, feature importance %, or model coefficients
- Use consistent numbers: primary prediction is 30-day (${predicted_price_str}, {price_change_pct_str}%)
- Reference {company_name} or {ticker} by name, not "the stock"
- Write for regular investors, not quants
- No markdown formatting (no **, ##, bold, italics)
- Each section header on its own line
""")
    return "\n\n".join(sections)


def generate_explanations(
    tickers: Optional[List[str]] = None,
    max_tickers: int = 100,
    date: Optional[str] = None,
) -> Dict:
    """Generate and store AI explanations for tickers with predictions.
    
    Uses Groq API (preferred) or Gemini API (fallback) based on available API keys.
    Groq free tier: 1K-14K RPD vs Gemini's 20 RPD.
    """
    from ml_backend.utils.mongodb import MongoDBClient

    mongo = MongoDBClient()
    target_date = date or datetime.utcnow().strftime("%Y-%m-%d")
    ticker_list = tickers or TOP_100_TICKERS
    ticker_list = ticker_list[:max_tickers]

    results = {"success": 0, "skipped": 0, "failed": 0, "details": []}

    # Log API provider selection
    if API_PROVIDER == "groq":
        logger.info("Using Groq API (better free tier limits: 1K-14K RPD)")
    elif API_PROVIDER == "gemini":
        logger.info("Using Gemini API (free tier: 20 RPD max)")
    else:
        logger.warning("No API provider configured! Set GROQ_API_KEY or GOOGLE_API_KEY")
        results["failed"] = len(ticker_list)
        for ticker in ticker_list:
            results["details"].append({"ticker": ticker, "status": "failed", "reason": "no_api_key"})
        return results

    # Fetch macro context once for all tickers
    macro_context = _get_macro_context(mongo)
    logger.info("Fetched macro context: %d indicators", len(macro_context))

    quota_failures = 0
    MAX_QUOTA_FAILURES = 3

    for i, ticker in enumerate(ticker_list, 1):
        logger.info("[%d/%d] Processing %s…", i, len(ticker_list), ticker)

        if quota_failures >= MAX_QUOTA_FAILURES:
            remaining = len(ticker_list) - i + 1
            logger.warning("  %d consecutive quota failures — skipping remaining %d tickers", quota_failures, remaining)
            results["skipped"] += remaining
            for remaining_ticker in ticker_list[i-1:]:
                results["details"].append({"ticker": remaining_ticker, "status": "skipped", "reason": "quota_exceeded"})
            break

        # 0. Check for existing explanation for today (avoid redundant API calls)
        # Query both nested (explanation_data.explanation_date) and top-level
        # (explanation_date) patterns so it works regardless of schema.
        existing_explanation = mongo.db["prediction_explanations"].find_one(
            {
                "ticker": ticker,
                "window": "comprehensive",
                "$or": [
                    {"explanation_data.explanation_date": target_date},
                    {"explanation_date": target_date},
                ],
            },
            sort=[("timestamp", -1)]
        )
        if existing_explanation:
            logger.info("  Explanation already exists for %s on %s — skipping", ticker, target_date)
            results["skipped"] += 1
            results["details"].append({"ticker": ticker, "status": "skipped", "reason": "already_exists"})
            continue

        # 1. Get predictions (required)
        predictions = mongo.get_latest_predictions(ticker)
        if not predictions:
            logger.warning("  No predictions for %s — skipping", ticker)
            results["skipped"] += 1
            results["details"].append({"ticker": ticker, "status": "skipped", "reason": "no predictions"})
            continue

        # 2. Get sentiment (with full raw data for news headlines)
        sentiment = mongo.get_latest_sentiment(ticker) or {"blended_sentiment": 0, "sources": {}}

        # 3. Get technicals from historical data
        try:
            end_dt = datetime.strptime(target_date, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=365)
            hist = mongo.get_historical_data(ticker, start_dt, end_dt)
            technicals = calculate_technicals(hist) if hist is not None and not hist.empty else {}
            if not technicals:
                try:
                    import yfinance as yf
                    yf_data = yf.download(ticker, start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), progress=False)
                    if yf_data is not None and not yf_data.empty:
                        if hasattr(yf_data.columns, 'levels'):
                            yf_data.columns = yf_data.columns.get_level_values(0)
                        technicals = calculate_technicals(yf_data)
                        if technicals:
                            logger.info("  Got technicals from yfinance fallback for %s", ticker)
                except Exception as yf_err:
                    logger.warning("  yfinance fallback failed for %s: %s", ticker, yf_err)
        except Exception as e:
            logger.warning("  Technicals failed for %s: %s", ticker, e)
            technicals = {}

        # 4. Get SHAP / feature importance data (all windows, merge)
        shap_data = None
        try:
            fi_docs = list(mongo.db["feature_importance"].find(
                {"ticker": ticker}, sort=[("timestamp", -1)]
            ).limit(3))
            if fi_docs:
                merged_pos = []
                merged_neg = []
                merged_global = []
                for fi_doc in fi_docs:
                    merged_pos.extend(fi_doc.get("top_positive_contrib", []))
                    merged_neg.extend(fi_doc.get("top_negative_contrib", []))
                    if not merged_global:
                        merged_global = fi_doc.get("global_gain_importance", [])

                # Deduplicate by feature name, keeping highest abs contrib
                def dedup_by_feature(items, top_n=8):
                    seen = {}
                    for f in items:
                        feat = f.get("feature", "")
                        if feat not in seen or abs(f.get("contrib", 0)) > abs(seen[feat].get("contrib", 0)):
                            seen[feat] = f
                    return sorted(seen.values(), key=lambda x: abs(x.get("contrib", 0)), reverse=True)[:top_n]

                shap_data = {
                    "top_positive_contrib": dedup_by_feature(merged_pos),
                    "top_negative_contrib": dedup_by_feature(merged_neg),
                    "global_gain_importance": merged_global[:10],
                    "prob_up": fi_docs[0].get("prob_up"),
                    "predicted_value": fi_docs[0].get("predicted_value"),
                }
        except Exception as e:
            logger.warning("  SHAP lookup failed for %s: %s", ticker, e)

        # 5. Get insider trading context
        insider_context = _get_insider_context(mongo, ticker)

        # 6. Get short interest context
        short_interest = _get_short_interest_context(mongo, ticker)

        # 7. Get Finnhub basic financials (P/E, margins, etc.)
        financials_context = _get_financials_context(mongo, ticker)

        # 8. Get FMP earnings, ratings, price targets
        fmp_context = _get_fmp_context(mongo, ticker)

        # 9. Get recent news from aggregated news collections
        recent_news = _get_recent_news(mongo, ticker)
        if recent_news:
            logger.info("  Found %d recent news articles for %s", len(recent_news), ticker)

        # 10. Build prompt & call Gemini
        prompt = _build_prompt(
            ticker, target_date, predictions, sentiment, technicals, shap_data,
            macro_context=macro_context,
            insider_context=insider_context,
            short_interest=short_interest,
            financials_context=financials_context,
            fmp_context=fmp_context,
            recent_news=recent_news,
        )
        explanation_text, error_type = _call_llm_api(prompt, ticker)

        if error_type == "quota_exceeded":
            quota_failures += 1
            provider_name = "Groq" if API_PROVIDER == "groq" else "Gemini"
            logger.error("  %s quota exceeded (%d/%d consecutive failures)", provider_name, quota_failures, MAX_QUOTA_FAILURES)
            results["failed"] += 1
            results["details"].append({"ticker": ticker, "status": "failed", "reason": "quota_exceeded"})
            
            # Early exit: if quota is exhausted, stop processing remaining tickers
            if quota_failures >= MAX_QUOTA_FAILURES:
                remaining = len(ticker_list) - i
                provider_name = "Groq" if API_PROVIDER == "groq" else "Gemini"
                logger.error(
                    "  ⚠️  %s QUOTA EXHAUSTED: Stopping explanation generation after %d consecutive failures. "
                    "Remaining %d tickers will be skipped. Consider enabling billing, switching API provider, or reducing max_tickers.",
                    provider_name, quota_failures, remaining
                )
                results["quota_exhausted"] = True
                # Mark remaining tickers as skipped
                if remaining > 0:
                    results["skipped"] += remaining
                    for remaining_ticker in ticker_list[i:]:
                        results["details"].append({"ticker": remaining_ticker, "status": "skipped", "reason": "quota_exceeded"})
                break  # Exit the loop early instead of continuing
            
            # Exponential backoff before trying next ticker
            backoff_time = min(60 * (2 ** (quota_failures - 1)), 300)  # Max 5 minutes
            logger.info("  Sleeping %ds before retrying next ticker…", backoff_time)
            time.sleep(backoff_time)
            continue

        if "unavailable" in explanation_text.lower() or error_type:
            logger.error("  Gemini failed for %s: %s", ticker, explanation_text[:80])
            results["failed"] += 1
            results["details"].append({"ticker": ticker, "status": "failed", "reason": explanation_text[:200]})
            continue

        # 10. Build explanation doc & store
        data_sources = ["ML Predictions"]
        if sentiment.get("blended_sentiment") or sentiment.get("sources"):
            data_sources.append("Sentiment Analysis")
        if technicals:
            data_sources.append("Technical Indicators")
        if shap_data:
            data_sources.append("SHAP Feature Importance")
        if macro_context:
            data_sources.append("Macro Economic Data (FRED)")
        if insider_context and insider_context.get("recent_transactions", 0) > 0:
            data_sources.append("Insider Trading Data")
        if short_interest and short_interest.get("short_float_pct", 0) > 0:
            data_sources.append("Short Interest Data")
        if financials_context:
            data_sources.append("Fundamental Financials (Finnhub)")
        if fmp_context:
            data_sources.append("Earnings & Analyst Data (FMP)")
        news_count = (len(sentiment.get("finviz_raw_data", [])) +
                      len(sentiment.get("rss_news_raw_data", [])) +
                      len(sentiment.get("reddit_raw_data", [])) +
                      len(sentiment.get("marketaux_raw_data", [])) +
                      len(recent_news))
        if news_count > 0:
            data_sources.append(f"News Headlines ({news_count} sources)")
        active_model = _pick_model()
        data_sources.append(f"Google {active_model}")

        explanation_data = {
            "ticker": ticker,
            "explanation_date": target_date,
            "prediction_data": predictions,
            "sentiment_summary": {
                "blended_sentiment": sentiment.get("blended_sentiment", 0),
                "total_data_points": sum(
                    s.get("volume", 0) if isinstance(s, dict) else 0
                    for s in sentiment.get("sources", {}).values()
                ),
                "finviz_articles": len(sentiment.get("finviz_raw_data", [])),
                "reddit_posts": len(sentiment.get("reddit_raw_data", [])),
                "rss_articles": len(sentiment.get("rss_news_raw_data", [])),
                "marketaux_articles": len(sentiment.get("marketaux_raw_data", [])),
            },
            "technical_indicators": technicals,
            "feature_importance": shap_data or {},
            "macro_context": macro_context,
            "insider_summary": insider_context,
            "short_interest_summary": short_interest,
            "ai_explanation": explanation_text,
            "data_sources_used": data_sources,
            "financials_context": financials_context,
            "fmp_context": fmp_context,
            "explanation_quality": {
                "data_completeness": min(1.0, (
                    (0.20 if predictions else 0) +
                    (0.12 if sentiment.get("blended_sentiment") else 0) +
                    (0.12 if technicals else 0) +
                    (0.12 if shap_data else 0) +
                    (0.08 if macro_context else 0) +
                    (0.05 if insider_context and insider_context.get("recent_transactions") else 0) +
                    (0.05 if short_interest and short_interest.get("short_float_pct") else 0) +
                    (0.08 if news_count > 0 else 0) +
                    (0.08 if financials_context else 0) +
                    (0.10 if fmp_context else 0)
                )),
            },
            "timestamp": datetime.utcnow().isoformat(),
            "prompt_length": len(prompt),
            "explanation_length": len(explanation_text),
        }

        stored = mongo.store_prediction_explanation(ticker, "comprehensive", explanation_data)
        if stored:
            quota_failures = 0
            results["success"] += 1
            results["details"].append({"ticker": ticker, "status": "success", "chars": len(explanation_text)})
            logger.info("  Stored for %s (%d chars, %d sources)", ticker, len(explanation_text), len(data_sources))
        else:
            results["failed"] += 1
            results["details"].append({"ticker": ticker, "status": "failed", "reason": "mongo store failed"})
            logger.error("  MongoDB store failed for %s", ticker)

        # Rate limit: gemini-2.5-pro free tier ~15 RPM, 1.5K RPD; flash ~5 RPM, 20 RPD
        current_model = _pick_model()
        sleep_secs = 5 if "pro" in current_model else 3
        time.sleep(sleep_secs)

    # Log RPD usage summary
    if _model_rpd_count:
        logger.info("Gemini RPD usage this run: %s",
                     ", ".join(f"{m}={c}" for m, c in _model_rpd_count.items()))

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch Gemini AI Explanation Generator")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Specific tickers (default: all with predictions)")
    parser.add_argument("--max-tickers", type=int, default=100,
                        help="Max tickers to process (default: 100)")
    parser.add_argument("--date", type=str, default=None,
                        help="Target date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    logger.info("Starting batch AI explanation generation")
    results = generate_explanations(
        tickers=args.tickers,
        max_tickers=args.max_tickers,
        date=args.date,
    )

    print(f"\n{'='*50}")
    print(f"Batch Explanation Generation Complete")
    print(f"  Success: {results['success']}")
    print(f"  Skipped: {results['skipped']}")
    print(f"  Failed:  {results['failed']}")
    print(f"{'='*50}")

    # Fail CI if more than half failed
    total_attempted = results["success"] + results["failed"]
    if total_attempted > 0 and results["failed"] > total_attempted * 0.5:
        print("Too many failures — exiting with error")
        sys.exit(1)


if __name__ == "__main__":
    main()
