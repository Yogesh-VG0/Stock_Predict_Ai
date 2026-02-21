import os
import logging
import random
import time
import pandas as pd
from datetime import datetime
from ml_backend.utils.mongodb import MongoDBClient
from ml_backend.utils.rate_limiter import fred_limiter
from fredapi import Fred

logger = logging.getLogger(__name__)

_fred_client = None

# Retry configuration for transient FRED API failures
_FRED_MAX_RETRIES = 2
_FRED_BASE_DELAY = 1.5  # seconds


def _get_fred():
    """Lazy init so env var is read at call time, not import time."""
    global _fred_client
    if _fred_client is None:
        key = os.getenv('FRED_API_KEY')
        if key:
            _fred_client = Fred(api_key=key)
        else:
            logger.warning("FRED_API_KEY not set — macro features will be zeros")
    return _fred_client


# Supported FRED indicators (FRED code: friendly name)
FRED_INDICATORS = {
    'GDP': 'GDP',
    'REAL_GDP': 'GDPC1',
    'REAL_GDP_PER_CAPITA': 'A939RX0Q048SBEA',
    'CPI': 'CPIAUCSL',
    'UNEMPLOYMENT': 'UNRATE',
    'INFLATION': 'FPCPITOTLZGUSA',
    'FEDERAL_FUNDS_RATE': 'FEDFUNDS',
    'TREASURY_10Y': 'GS10',
    'TREASURY_2Y': 'GS2',
    'TREASURY_30Y': 'GS30',
    'RETAIL_SALES': 'RSXFSN',
    'DURABLES': 'UMDMNO',
    'NONFARM_PAYROLL': 'PAYEMS',
}


def fetch_fred_series(series_code, start_date, end_date):
    """Fetch a FRED series with retry on transient failures."""
    fred = _get_fred()
    if not fred:
        raise ValueError('FRED_API_KEY not set.')
    last_exc = None
    for attempt in range(1, _FRED_MAX_RETRIES + 1):
        try:
            fred_limiter.acquire_sync()
            data = fred.get_series(series_code, observation_start=start_date, observation_end=end_date)
            data = data.dropna()
            return data
        except ValueError:
            raise  # re-raise config errors immediately
        except Exception as exc:
            last_exc = exc
            if attempt < _FRED_MAX_RETRIES:
                delay = _FRED_BASE_DELAY * (2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
                logger.warning(
                    "[FRED-RETRY] %s | attempt=%d/%d | err=%r | sleep=%.1fs",
                    series_code, attempt, _FRED_MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
    raise last_exc


def fetch_and_store_fred_indicator(indicator, start_date, end_date, mongo_client=None):
    if indicator not in FRED_INDICATORS:
        raise ValueError(f'Unsupported FRED indicator: {indicator}')
    series_code = FRED_INDICATORS[indicator]
    data = fetch_fred_series(series_code, start_date, end_date)
    data_dict = {d.strftime('%Y-%m-%d'): float(v) for d, v in data.items()}
    if mongo_client is not None:
        mongo_client.store_macro_data(indicator, data_dict, source='FRED')
    return data_dict


def fetch_and_store_all_fred_indicators(start_date, end_date, mongo_client=None):
    """
    Fetch and store all supported FRED indicators for the given date range.
    Returns a dict: {indicator: data_dict}
    """
    results = {}
    for indicator in FRED_INDICATORS:
        try:
            data_dict = fetch_and_store_fred_indicator(indicator, start_date, end_date, mongo_client=mongo_client)
            results[indicator] = data_dict
        except Exception as e:
            # Log but continue; fallback should be handled in feature engineering
            logger.warning("Error fetching/storing %s from FRED: %r", indicator, e)
    return results
