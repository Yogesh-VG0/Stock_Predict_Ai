import os
import pandas as pd
from datetime import datetime
from ml_backend.utils.mongodb import MongoDBClient
from fredapi import Fred

_fred_client = None

def _get_fred():
    """Lazy init so env var is read at call time, not import time."""
    global _fred_client
    if _fred_client is None:
        key = os.getenv('FRED_API_KEY')
        if key:
            _fred_client = Fred(api_key=key)
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
    fred = _get_fred()
    if not fred:
        raise ValueError('FRED_API_KEY not set.')
    data = fred.get_series(series_code, observation_start=start_date, observation_end=end_date)
    data = data.dropna()
    return data

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
            print(f"Error fetching/storing {indicator} from FRED: {e}")
    return results 