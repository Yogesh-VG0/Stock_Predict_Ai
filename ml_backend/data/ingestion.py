"""
Data ingestion module for fetching and processing market data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
import logging
from ..config.constants import (
    HISTORICAL_DATA_YEARS,
    RETRY_CONFIG,
    TOP_100_TICKERS,
    TICKER_YFINANCE_MAP,
    CANARY_TICKERS
)
from ..utils.mongodb import MongoDBClient
import pytz
import pandas_market_calendars as mcal
import argparse
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self, mongo_client: MongoDBClient):
        """
        Initialize the DataIngestion class.
        
        Args:
            mongo_client: MongoDB client instance
        """
        self.max_retries = RETRY_CONFIG["max_retries"]
        self.base_delay = RETRY_CONFIG["base_delay"]
        self.max_delay = RETRY_CONFIG["max_delay"]
        self.mongo_client = mongo_client

    def _exponential_backoff(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay.
        
        Args:
            attempt: Current retry attempt number
            
        Returns:
            Delay in seconds
        """
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        return delay

    def _fetch_yahoo_data(self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Fetch data from Yahoo Finance with retry logic.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame containing historical data or None if fetch failed
        """
        for attempt in range(self.max_retries):
            try:
                yf_ticker = TICKER_YFINANCE_MAP.get(ticker, ticker)
                stock = yf.Ticker(yf_ticker)
                data = stock.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    auto_adjust=True
                )
                if not data.empty and not data['Close'].isna().all():
                    # If 'Adj Close' exists, rename to 'Close' for consistency
                    if 'Adj Close' in data.columns:
                        data = data.rename(columns={'Adj Close': 'Close'})
                    return data
                logger.warning(f"Empty or all-NaN data for {ticker}, attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {str(e)}")
            
            if attempt < self.max_retries - 1:
                delay = self._exponential_backoff(attempt)
                time.sleep(delay)
        
        return None

    def get_last_available_trading_day(self) -> datetime.date:
        """
        Returns the last available trading day by downloading a wide window and using the last index.
        If the DataFrame is empty, tries a fallback window ending at a known recent date.
        """
        today = datetime.utcnow().date()
        start = today - timedelta(days=60)
        data = yf.download("AAPL", start=start.strftime("%Y-%m-%d"), end=(today + timedelta(days=1)).strftime("%Y-%m-%d"))
        if not data.empty:
            return data.index[-1].date()

        # Fallback: try a fixed end date (e.g., 2025-05-03)
        fallback_end = datetime(2025, 5, 3).date()
        fallback_start = fallback_end - timedelta(days=60)
        data = yf.download("AAPL", start=fallback_start.strftime("%Y-%m-%d"), end=(fallback_end + timedelta(days=1)).strftime("%Y-%m-%d"))
        if not data.empty:
            return data.index[-1].date()

        raise RuntimeError("Failed to determine the last available trading day after multiple attempts.")

    def fetch_historical_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data for a ticker, only fetching new data if available.
        """
        try:
            latest_date = self.mongo_client.get_latest_date(ticker)
            end_date = self.get_last_available_trading_day()
            if latest_date is None or pd.isna(latest_date):
                logger.warning(f"NaT/null latest_date for {ticker}, treating as no data exists. Fetching full 10 years from Yahoo Finance.")
                start_date = (datetime.utcnow() - pd.DateOffset(years=10)).date()
                data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
                if data is None or data.empty:
                    logger.error(f"No data fetched for {ticker} from Yahoo Finance.")
                    return None
                # Always reset index
                data = data.reset_index()
                # If 'Adj Close' exists, rename to 'Close' for consistency
                if 'Adj Close' in data.columns:
                    data = data.rename(columns={'Adj Close': 'Close'})
                # Rename 'Date' or 'index' to 'date'
                if 'Date' in data.columns:
                    data = data.rename(columns={'Date': 'date'})
                elif 'index' in data.columns:
                    data = data.rename(columns={'index': 'date'})
                if 'date' not in data.columns:
                    logger.error(f"'date' column missing after resetting index for {ticker}. Skipping.")
                    return None
                data['date'] = pd.to_datetime(data['date'])
                data = self.rename_ohlcv_columns(data, ticker)
                required = ["Close", "Open", "High", "Low", "Volume"]
                missing = [col for col in required if col not in data.columns]
                if missing:
                    logger.error(f"After renaming, missing columns for {ticker}: {missing}. Data columns: {list(data.columns)}")
                    return None
                if 'date' not in data.columns or data['date'].isna().any():
                    logger.error(f"After renaming, 'date' column missing or contains NaT/null for {ticker}. Skipping.")
                    return None
                self.mongo_client.store_historical_data(ticker, data)
                return data
            if pd.to_datetime(latest_date).date() >= end_date:
                logger.info(f"MongoDB already has the latest data for {ticker} (up to {latest_date}). Skipping fetch.")
                return None
            start_date = pd.to_datetime(latest_date).date() + pd.Timedelta(days=1)
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
            if data is None or data.empty:
                logger.error(f"No new data to fetch for {ticker}")
                return None
            data = data.reset_index()
            # If 'Adj Close' exists, rename to 'Close' for consistency
            if 'Adj Close' in data.columns:
                data = data.rename(columns={'Adj Close': 'Close'})
            if 'Date' in data.columns:
                data = data.rename(columns={'Date': 'date'})
            elif 'index' in data.columns:
                data = data.rename(columns={'index': 'date'})
            if 'date' not in data.columns:
                logger.error(f"'date' column missing after resetting index for {ticker}. Skipping.")
                return None
            data['date'] = pd.to_datetime(data['date'])
            data = self.rename_ohlcv_columns(data, ticker)
            required = ["Close", "Open", "High", "Low", "Volume"]
            missing = [col for col in required if col not in data.columns]
            if missing:
                logger.error(f"After renaming, missing columns for {ticker}: {missing}. Data columns: {list(data.columns)}")
                return None
            if 'date' not in data.columns or data['date'].isna().any():
                logger.error(f"After renaming, 'date' column missing or contains NaT/null for {ticker}. Skipping.")
                return None
            self.mongo_client.store_historical_data(ticker, data)
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None

    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean the fetched data.
        
        Args:
            data: Raw historical data DataFrame
            
        Returns:
            Processed DataFrame with additional features
        """
        try:
            # Handle missing values
            data = data.ffill().bfill()
            
            # Add additional features
            data['returns'] = data['Close'].pct_change()
            data['log_returns'] = np.log1p(data['returns'])
            data['volatility'] = data['returns'].rolling(window=20).std()
            
            # Add date features
            data['day_of_week'] = data.index.dayofweek
            data['month'] = data.index.month
            data['quarter'] = data.index.quarter
            
            return data
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return data

    def fetch_all_tickers(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all S&P 100 tickers, but only if a new trading day is available (using configurable canary tickers and market calendar check).
        Returns:
            Dictionary mapping tickers to their historical data
        """
        results = {}
        nyse = mcal.get_calendar('NYSE')
        today = datetime.utcnow().date()
        canary_found = False
        for canary in CANARY_TICKERS:
            latest_date = self.mongo_client.get_latest_date(canary)
            if latest_date is not None and not pd.isna(latest_date):
                end_date = self.get_last_available_trading_day()
                # Ensure both are datetime.date
                if hasattr(latest_date, 'date'):
                    latest_date = latest_date.date()
                if hasattr(end_date, 'date'):
                    end_date = end_date
                # Use NYSE market calendar to check if today is a trading day
                schedule = nyse.schedule(start_date=latest_date, end_date=today)
                trading_days = schedule.index.date if not schedule.empty else []
                logger.info(f"Canary ticker: {canary}, Last update: {latest_date}, Market latest: {end_date}, Trading days in range: {trading_days}")
                if latest_date is not None and (latest_date >= end_date or (len(trading_days) == 0 or trading_days[-1] == latest_date)):
                    logger.info(f"No new trading day since {latest_date} (latest close) for canary {canary}. Market closed on {end_date}.")
                    canary_found = True
                    break
                elif latest_date is not None:
                    logger.info(f"New trading day detected for canary {canary}. Proceeding to fetch for all tickers.")
                    canary_found = True
                    break
        if not canary_found:
            logger.warning("All canary tickers failed to provide a valid last update. Proceeding to fetch for all tickers as fallback.")
        # Proceed to fetch for all tickers (fallback or after canary logic)
        for ticker in TOP_100_TICKERS:
            try:
                logger.info(f"Fetching data for {ticker}")
                data = self.fetch_historical_data(ticker)
                if data is not None and not data.empty:
                    results[ticker] = data
                elif self.mongo_client.get_latest_date(ticker) is not None:
                    logger.info(f"MongoDB already has the latest data for {ticker}. Continuing to next step.")
                    continue
                else:
                    logger.error(f"No data fetched for {ticker}, and no data in MongoDB. Stopping pipeline.")
                    raise RuntimeError(f"No data fetched for {ticker} from MongoDB or Yahoo Finance. Stopping pipeline.")
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {str(e)}")
                raise  # Stop the pipeline on any error
        return results

    def update_database(self, data: Dict[str, pd.DataFrame]) -> bool:
        """
        Update MongoDB with new data.
        """
        try:
            for ticker, df in data.items():
                # Always rename columns before storing
                df = self.rename_ohlcv_columns(df, ticker)
                required = ["Close", "Open", "High", "Low", "Volume"]
                missing = [col for col in required if col not in df.columns]
                if missing:
                    logger.error(f"After renaming, missing columns for {ticker}: {missing}. Data columns: {list(df.columns)}")
                    return False
                success = self.mongo_client.store_historical_data(ticker, df)
                if not success:
                    logger.error(f"Failed to update database for {ticker}")
                    return False
            logger.info("Successfully updated database with new data")
            return True
        except Exception as e:
            logger.error(f"Error updating database: {str(e)}")
            return False

    def rename_ohlcv_columns(self, df, ticker):
        # Return immediately if DataFrame is empty
        if df is None or df.empty:
            logging.error(f"DataFrame is empty for {ticker} in rename_ohlcv_columns.")
            return pd.DataFrame()
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join([str(i) for i in col if i]) for col in df.columns.values]
        rename_map = {
            f"Close_{ticker}": "Close",
            f"Open_{ticker}": "Open",
            f"High_{ticker}": "High",
            f"Low_{ticker}": "Low",
            f"Volume_{ticker}": "Volume"
        }
        logging.info(f"Columns before renaming: {list(df.columns)}")
        df = df.rename(columns=rename_map)
        # Ensure 'date' is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            logging.error(f"No 'date' column found after resetting index for {ticker}.")
            return pd.DataFrame()
        # Remove any 'Date' (capital D) column
        if 'Date' in df.columns:
            df = df.drop(columns=['Date'])
        # Remove rows with date == 1970-01-01
        bad_dates = df['date'] == pd.Timestamp('1970-01-01')
        if bad_dates.any():
            logging.error(f"Found {bad_dates.sum()} rows with date 1970-01-01. These will be dropped.")
            df = df[~bad_dates]
        # Log min/max/unique dates and sample types
        logging.info(f"Date range before storing: min={df['date'].min()}, max={df['date'].max()}, unique_dates={df['date'].nunique()}")
        logging.info(f"Sample dates and types: {df['date'].head().tolist()} / {[type(x) for x in df['date'].head().tolist()]}")
        # Reorder columns to put 'date' first if present
        if 'date' in df.columns:
            cols = list(df.columns)
            cols.insert(0, cols.pop(cols.index('date')))
            df = df[cols]
        logging.info(f"Columns after renaming: {list(df.columns)}")
        missing = [col for col in ["Close", "Open", "High", "Low", "Volume"] if col not in df.columns]
        if missing:
            logging.warning(f"Missing expected columns after renaming: {missing}")
        return df

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ingest historical market data into MongoDB.")
    parser.add_argument('--ticker', type=str, default=None, help='Single ticker to ingest (default: all)')
    args = parser.parse_args()
    from ml_backend.utils.mongodb import MongoDBClient
    mongo_uri = os.getenv("MONGODB_URI")
    mongo_client = MongoDBClient(mongo_uri)
    ingestion = DataIngestion(mongo_client)
    if args.ticker:
        print(f"Ingesting data for {args.ticker}...")
        df = ingestion.fetch_historical_data(args.ticker)
        if df is not None:
            print(f"Ingested {len(df)} rows for {args.ticker}.")
        else:
            print(f"Failed to ingest data for {args.ticker}.")
    else:
        print("Ingesting data for all tickers...")
        all_data = ingestion.fetch_all_tickers()
        print(f"Ingested data for {len(all_data)} tickers.")
    mongo_client.close() 