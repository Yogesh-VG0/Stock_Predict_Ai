"""
Mongo-first fetch for SPY/macro with in-process cache.
Reduces external API calls during training and prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def _key(ticker: str, start: datetime, end: datetime) -> Tuple[str, str, str]:
    """Normalize to date strings so keys are stable."""
    return (
        ticker.upper(),
        pd.Timestamp(start).strftime("%Y-%m-%d"),
        pd.Timestamp(end).strftime("%Y-%m-%d"),
    )


@dataclass
class FrameCache:
    """Process-local cache. Not shared across workers. Great for training loops."""

    max_items: int = 64

    def __post_init__(self):
        self._cache: Dict[Tuple[str, str, str], pd.DataFrame] = {}

    def get(self, ticker: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        return self._cache.get(_key(ticker, start, end))

    def set(self, ticker: str, start: datetime, end: datetime, df: pd.DataFrame) -> None:
        k = _key(ticker, start, end)
        if k in self._cache:
            self._cache[k] = df
            return
        if len(self._cache) >= self.max_items:
            self._cache.pop(next(iter(self._cache)))
        self._cache[k] = df


def fetch_price_df_mongo_first(
    mongo_client,
    ticker: str,
    start: datetime,
    end: datetime,
    *,
    cache: Optional[FrameCache] = None,
    allow_fallback_yfinance: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV from Mongo first. Optional process cache. Optional yfinance fallback.
    """
    if cache is not None:
        hit = cache.get(ticker, start, end)
        if hit is not None and not hit.empty:
            return hit.copy()

    df = None
    try:
        if mongo_client is not None and hasattr(mongo_client, "get_historical_data"):
            df = mongo_client.get_historical_data(ticker, start, end)
    except Exception as e:
        logger.warning("Mongo fetch failed for %s: %s", ticker, e)

    if df is None or getattr(df, "empty", True):
        if not allow_fallback_yfinance:
            return None
        try:
            import yfinance as yf

            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df is None or df.empty:
                return None
            
            # Handle yfinance v0.2+ MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            df = df.reset_index()
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "date"})
        except Exception as e:
            logger.warning("yfinance fallback failed for %s: %s", ticker, e)
            return None

    if cache is not None and df is not None and not df.empty:
        cache.set(ticker, start, end, df)
    return df
