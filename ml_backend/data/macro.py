# macro.py
# NOTE: This is a legacy module. The canonical FRED integration is fred_macro.py.
# This file is retained for backward compatibility but uses lazy FRED init.
import os
import pandas as pd
from fredapi import Fred
import yfinance as yf
import logging
from ml_backend.utils.mongodb import MongoDBClient

logger = logging.getLogger(__name__)

FRED_API_KEY = os.getenv("FRED_API_KEY")

# Comprehensive macro indicators from FRED
FRED_INDICATORS = {
    "FEDFUNDS": "interest_rate",
    "CPIAUCSL": "cpi",
    "UNRATE": "unemployment",
    "GDP": "gdp",
    "GDPC1": "real_gdp",
    "A939RX0Q048SBEA": "real_gdp_per_capita",
    "FPCPITOTLZGUSA": "inflation",
    "GS10": "treasury_10y",
    "GS2": "treasury_2y",
    "GS30": "treasury_30y",
    "RSXFSN": "retail_sales",
    "UMDMNO": "durables",
    "PAYEMS": "nonfarm_payroll"
}

# Example sector ETFs
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE"]

# Lazy FRED client: created on first use to avoid import-time crashes
_fred_client = None


def _get_fred_client() -> "Fred | None":
    """Return a cached FRED client, initializing lazily on first call."""
    global _fred_client
    if _fred_client is None:
        key = os.getenv("FRED_API_KEY")
        if not key:
            logger.warning("FRED_API_KEY not set — FRED client unavailable.")
            return None
        try:
            _fred_client = Fred(api_key=key)
        except Exception as e:
            logger.error("Failed to initialize FRED client: %s", e)
            return None
    return _fred_client


def fetch_macro_data(start_date, end_date, mongo_client=None):
    """Fetch macro data from FRED and optionally store in MongoDB."""
    fred = _get_fred_client()
    if not fred:
        logger.warning("FRED client unavailable. Skipping macro data fetch.")
        return pd.DataFrame()
    macro_df = pd.DataFrame()
    for code, name in FRED_INDICATORS.items():
        try:
            series = fred.get_series(code, observation_start=start_date, observation_end=end_date)
            macro_df[name] = series

            # Store in MongoDB if client provided
            if mongo_client is not None:
                data_dict = {d.strftime('%Y-%m-%d'): float(v) for d, v in series.items() if pd.notna(v)}
                mongo_client.store_macro_data(name, data_dict, source='FRED')
                logger.info("Stored %d data points for %s in MongoDB", len(data_dict), name)

        except Exception as e:
            logger.error("Error fetching %s from FRED: %s", code, e)

    macro_df.index = pd.to_datetime(macro_df.index)
    macro_df = macro_df.resample('D').ffill()
    return macro_df


def fetch_sector_etf_data(start_date, end_date):
    sector_df = pd.DataFrame()
    for etf in SECTOR_ETFS:
        try:
            data = yf.download(etf, start=start_date, end=end_date)
            if not data.empty:
                sector_df[f"{etf}_close"] = data["Close"]
        except Exception as e:
            logger.error("Error fetching %s from yfinance: %s", etf, e)
    sector_df.index = pd.to_datetime(sector_df.index)
    sector_df = sector_df.resample('D').ffill()
    return sector_df


def merge_macro_sector_features(stock_df, macro_df, sector_df):
    # Assumes stock_df has a DatetimeIndex or 'date' column
    if 'date' in stock_df.columns:
        stock_df = stock_df.set_index('date')
    merged = stock_df.join(macro_df, how='left').join(sector_df, how='left')
    merged = merged.ffill().bfill()
    return merged.reset_index()
