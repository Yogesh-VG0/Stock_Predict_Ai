"""
Feature engineering module for generating technical indicators and other features.
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from ..config.constants import (
    TECHNICAL_INDICATORS,
    FEATURE_CONFIG
)
from fredapi import Fred
import os
import yfinance as yf
from sklearn.impute import SimpleImputer
import requests
import shap
import csv
import io
from ml_backend.data.sentiment import SentimentAnalyzer
from ml_backend.utils.mongodb import MongoDBClient
from ml_backend.data.economic_calendar import EconomicCalendar
from ml_backend.data.fred_macro import fetch_and_store_all_fred_indicators, FRED_INDICATORS
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MacroDataFetcher:
    def __init__(self):
        # Use centralized FRED fetcher from fred_macro.py
        self.fred_indicators = FRED_INDICATORS
        self.cache = {}
        self.cache_expiry = 3600  # 1 hour

    def fetch_all(self, start_date, end_date, mongo_client=None):
        """Fetch all macro indicators with caching using centralized fred_macro."""
        try:
            # Check cache first
            cache_key = f"{start_date}_{end_date}"
            if cache_key in self.cache:
                cache_time, cache_data = self.cache[cache_key]
                if (datetime.now() - cache_time).seconds < self.cache_expiry:
                    return cache_data

            # Use centralized FRED fetcher
            results = fetch_and_store_all_fred_indicators(start_date, end_date, mongo_client)
            
            # Convert to DataFrame format expected by features
            dates = pd.date_range(start=start_date, end=end_date)
            df = pd.DataFrame({'date': dates})
            
            for indicator, data_dict in results.items():
                if data_dict:
                    # Convert data_dict to series
                    data_series = pd.Series(data_dict, name=indicator)
                    data_series.index = pd.to_datetime(data_series.index)
                    data_series = data_series.reindex(dates, method='ffill')
                    df[indicator] = data_series.values
                    
                    # Calculate derived features
                    if len(data_series) > 1:
                        df[f'{indicator}_change'] = data_series.pct_change().values
                        df[f'{indicator}_ma5'] = data_series.rolling(5).mean().values
                        df[f'{indicator}_ma20'] = data_series.rolling(20).mean().values

            # Fill missing values
            df = df.ffill().bfill()

            # Update cache
            self.cache[cache_key] = (datetime.now(), df)

            return df

        except Exception as e:
            logger.error(f"Error fetching macro data: {e}")
            return pd.DataFrame({'date': pd.date_range(start=start_date, end=end_date)})

class SectorDataFetcher:
    def __init__(self):
        # Use sector ETFs from macro.py
        self.etfs = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE"]
        self.cache = {}
        self.cache_expiry = 3600  # 1 hour

    def fetch_all(self, start_date, end_date, mongo_client=None):
        """Fetch sector ETF data with caching."""
        try:
            # Check cache first
            cache_key = f"{start_date}_{end_date}"
            if cache_key in self.cache:
                cache_time, cache_data = self.cache[cache_key]
                if (datetime.now() - cache_time).seconds < self.cache_expiry:
                    return cache_data

            dates = pd.date_range(start=start_date, end=end_date)
            df = pd.DataFrame({'date': dates})

            for etf in self.etfs:
                try:
                    # Try to get from MongoDB first
                    if mongo_client:
                        cached_data = mongo_client.get_sector_data(etf, start_date, end_date)
                        if cached_data:
                            data = pd.Series(cached_data)
                        else:
                            data = yf.download(etf, start=start_date, end=end_date, progress=False, auto_adjust=True)
                            # Store in MongoDB
                            data_dict = {d.strftime('%Y-%m-%d'): float(v) for d, v in data['Close'].items()}
                            mongo_client.store_sector_data(etf, data_dict)
                    else:
                        data = yf.download(etf, start=start_date, end=end_date, progress=False, auto_adjust=True)

                    data = data.reindex(dates, method='ffill')
                    df[f"{etf}_close"] = data['Close'].values
                    df[f"{etf}_volume"] = data['Volume'].values

                    # Calculate derived features
                    if len(data) > 1:
                        df[f"{etf}_return"] = data['Close'].pct_change().values
                        df[f"{etf}_volatility"] = data['Close'].rolling(20).std().values
                        df[f"{etf}_ma5"] = data['Close'].rolling(5).mean().values
                        df[f"{etf}_ma20"] = data['Close'].rolling(20).mean().values

                except Exception as e:
                    logger.error(f"yfinance fetch failed for {etf}: {e}")
                    df[f"{etf}_close"] = np.nan
                    df[f"{etf}_volume"] = np.nan

            # Fill missing values
            df = df.ffill().bfill()

            # Update cache
            self.cache[cache_key] = (datetime.now(), df)

            return df

        except Exception as e:
            logger.error(f"Error fetching sector data: {e}")
            return pd.DataFrame({'date': dates})

EVENT_WINDOW_DAYS = 2  # Number of days before/after event to mark as event window

def _mark_event_window(df, event_dates, col_name, window=EVENT_WINDOW_DAYS):
    df[col_name] = 0
    for d in event_dates:
        d = pd.to_datetime(d)
        mask = (df['date'] >= d - pd.Timedelta(days=window)) & (df['date'] <= d + pd.Timedelta(days=window))
        df.loc[mask, col_name] = 1
    return df

class FeatureEngineer:
    """
    Feature engineering class that supports macro/sector data from multiple sources (FRED, AlphaVantage, FMP).
    Always prefers FRED for macro indicators, but will automatically fallback to FMP, then AlphaVantage if FRED data is missing for any date/indicator.
    Only one source per indicator is used per date to avoid duplication. Always uses the latest available macro data.
    """
    def __init__(self, macro_sources: list = None, sentiment_window_days: int = None, sentiment_analyzer: SentimentAnalyzer = None, mongo_client: MongoDBClient = None, calendar_fetcher=None):
        """
        macro_sources: ordered list of sources to try, e.g. ['FRED', 'FMP', 'AlphaVantage']
        sentiment_window_days: number of days to aggregate sentiment/news features (default from FEATURE_CONFIG or 7)
        """
        self.lookback_days = FEATURE_CONFIG["lookback_days"]
        self.sequence_length = FEATURE_CONFIG["sequence_length"]
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        self.macro_fetcher = MacroDataFetcher()
        self.sector_fetcher = SectorDataFetcher()
        self.macro_sources = macro_sources or ['FRED', 'AlphaVantage', 'FMP']
        self.sentiment_window_days = sentiment_window_days or FEATURE_CONFIG.get("sentiment_window_days", 7)
        self.mongo_client = mongo_client or MongoDBClient(os.getenv("MONGODB_URI"))
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer(self.mongo_client)
        self.calendar_fetcher = calendar_fetcher or EconomicCalendar(self.mongo_client)
        self.short_interest_analyzer = None  # Will be initialized when needed

    def _init_short_interest_analyzer(self, mongo_client):
        """Initialize short interest analyzer if not already initialized."""
        if not self.short_interest_analyzer:
            from .short_interest import ShortInterestAnalyzer
            self.short_interest_analyzer = ShortInterestAnalyzer(mongo_client)
            
    def add_short_interest_features(self, df: pd.DataFrame, ticker: str, mongo_client=None) -> pd.DataFrame:
        """Add short interest features to the DataFrame."""
        try:
            self._init_short_interest_analyzer(mongo_client)
            df = df.copy()
            
            # Get short interest data for each date
            short_interest_data = []
            for date in df['date']:
                data = self.short_interest_analyzer.get_short_interest_data(ticker, date)
                if data:
                    short_interest_data.append({
                        'short_interest': data.get('short_interest', 0),
                        'days_to_cover': data.get('days_to_cover', 0),
                        'avg_daily_volume': data.get('avg_daily_volume', 0)
                    })
                else:
                    short_interest_data.append({
                        'short_interest': 0,
                        'days_to_cover': 0,
                        'avg_daily_volume': 0
                    })
            
            # Add features to DataFrame
            short_interest_df = pd.DataFrame(short_interest_data)
            short_interest_df.index = df.index
            
            # Calculate derived features
            short_interest_df['short_interest_change'] = short_interest_df['short_interest'].pct_change()
            short_interest_df['days_to_cover_change'] = short_interest_df['days_to_cover'].pct_change()
            short_interest_df['volume_ratio'] = short_interest_df['short_interest'] / short_interest_df['avg_daily_volume']
            
            df = pd.concat([df, short_interest_df], axis=1)
            
            logger.info(f"Added short interest features for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error adding short interest features for {ticker}: {e}")
            return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe."""
        try:
            logger.info(f"Before technical indicators: columns={list(df.columns)}, shape={df.shape}")
            # Moving Averages
            for period in TECHNICAL_INDICATORS["sma_periods"]:
                df[f'SMA_{period}'] = ta.trend.sma_indicator(df['Close'], window=period)
            
            for period in TECHNICAL_INDICATORS["ema_periods"]:
                df[f'EMA_{period}'] = ta.trend.ema_indicator(df['Close'], window=period)
            
            # RSI
            df['RSI'] = ta.momentum.rsi(df['Close'], window=TECHNICAL_INDICATORS["rsi_period"])
            
            # MACD
            macd = ta.trend.MACD(
                df['Close'],
                window_slow=TECHNICAL_INDICATORS["macd_slow"],
                window_fast=TECHNICAL_INDICATORS["macd_fast"],
                window_sign=TECHNICAL_INDICATORS["macd_signal"]
            )
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Hist'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(
                df['Close'],
                window=TECHNICAL_INDICATORS["bollinger_period"],
                window_dev=TECHNICAL_INDICATORS["bollinger_std"]
            )
            df['BB_Upper'] = bollinger.bollinger_hband()
            df['BB_Lower'] = bollinger.bollinger_lband()
            df['BB_Middle'] = bollinger.bollinger_mavg()
            
            # ATR
            df['ATR'] = ta.volatility.average_true_range(
                df['High'],
                df['Low'],
                df['Close'],
                window=TECHNICAL_INDICATORS["atr_period"]
            )
            
            # Volume Indicators
            df['Volume_SMA'] = ta.volume.volume_weighted_average_price(
                df['High'],
                df['Low'],
                df['Close'],
                df['Volume']
            )
            
            # Additional Features
            df['Daily_Return'] = df['Close'].pct_change(fill_method=None)
            df['Log_Return'] = np.log1p(df['Daily_Return'])
            df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
            
            # Support/Resistance Levels
            df = self._add_support_resistance(df)
            
            # Handle NaN values
            df = self._handle_nan_values(df)
            
            # Outlier handling
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].astype(float)  # Ensure float dtype for outlier replacement
                col_zscore = (df[col] - df[col].mean()) / (df[col].std() if df[col].std() != 0 else 1)
                outliers = col_zscore.abs() > 4.0
                if outliers.any():
                    replacement = (np.sign(col_zscore[outliers]) * 4.0 * df[col].std() + df[col].mean()).astype(float)
                    df.loc[outliers, col] = replacement
            
            logger.info(f"After technical indicators: columns={list(df.columns)}, shape={df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df

    def _handle_nan_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle NaN values in the dataframe."""
        try:
            # Forward fill for technical indicators
            df = df.ffill()
            # Backward fill for any remaining NaNs
            df = df.bfill()
            # Fill any remaining NaNs with 0
            df = df.fillna(0)
            return df
        except Exception as e:
            logger.error(f"Error handling NaN values: {str(e)}")
            return df

    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add support and resistance levels."""
        window = 20
        df['Support'] = df['Low'].rolling(window=window).min()
        df['Resistance'] = df['High'].rolling(window=window).max()
        return df

    def handle_outliers(self, df: pd.DataFrame, z_thresh: float = 4.0) -> pd.DataFrame:
        """Clip outliers in numeric columns using z-score threshold."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].astype(float)  # Ensure float dtype for outlier replacement
            col_zscore = (df[col] - df[col].mean()) / (df[col].std() if df[col].std() != 0 else 1)
            outliers = col_zscore.abs() > z_thresh
            if outliers.any():
                df.loc[outliers, col] = np.sign(col_zscore[outliers]) * z_thresh * df[col].std() + df[col].mean()
        return df

    def merge_external_features(self, df: pd.DataFrame, ticker: str, alpha_vantage_dict: Dict = None) -> pd.DataFrame:
        """
        Merge external features with stock data.
        
        Args:
            df (pd.DataFrame): Main stock data DataFrame
            ticker (str): Stock ticker symbol
            alpha_vantage_dict (Dict, optional): Dictionary containing Alpha Vantage data with keys:
                - technical_indicators: Technical analysis indicators
                - fundamental_data: Company fundamentals
                - earnings: Earnings data
                - dividends: Dividend history
                - insider_transactions: Insider trading data
                - news_sentiment: News sentiment analysis
                
        Returns:
            pd.DataFrame: DataFrame with merged features
        """
        try:
            # Get date range from dataframe
            if 'date' in df.columns:
                start_date = df['date'].min().strftime('%Y-%m-%d')
                end_date = df['date'].max().strftime('%Y-%m-%d')
            else:
                # Use index if no date column
                start_date = df.index.min().strftime('%Y-%m-%d') if hasattr(df.index.min(), 'strftime') else '2023-01-01'
                end_date = df.index.max().strftime('%Y-%m-%d') if hasattr(df.index.max(), 'strftime') else '2024-01-01'
            
            # Get macro data from MongoDBClient
            try:
                macro_data_dict = {}
                # Get key macro indicators
                macro_indicators = ['FEDFUNDS', 'CPIAUCSL', 'UNRATE', 'GDP', 'GS10', 'GS2']
                for indicator in macro_indicators:
                    data = self.mongo_client.get_macro_data(indicator, start_date, end_date)
                    if data:
                        # Convert to series and add to macro_data_dict
                        macro_data_dict[indicator] = data
                
                # Convert to DataFrame if we have data
                if macro_data_dict:
                    macro_df = pd.DataFrame(macro_data_dict)
                    macro_df.index = pd.to_datetime(macro_df.index)
                    
                    # Merge with main dataframe
                    df = pd.merge(
                        df,
                        macro_df,
                        left_index=True,
                        right_index=True,
                        how='left'
                    )
            except Exception as e:
                logger.warning(f"Error fetching macro data: {e}")
            
            # Get sector data (sector ETF performance)
            try:
                # Get sector mapping for the ticker
                sector_etfs = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE']
                sector_data = self.sector_fetcher.fetch_all(start_date, end_date, self.mongo_client)
                
                if sector_data is not None and not sector_data.empty:
                    # Merge sector data
                    df = pd.merge(
                        df,
                        sector_data,
                        left_index=True,
                        right_index=True,
                        how='left'
                    )
            except Exception as e:
                logger.warning(f"Error fetching sector data: {e}")
            
            # Merge Alpha Vantage data if available
            if alpha_vantage_dict is not None:
                # Process technical indicators
                if 'technical_indicators' in alpha_vantage_dict:
                    tech_indicators = pd.DataFrame(alpha_vantage_dict['technical_indicators'])
                    if not tech_indicators.empty:
                        # Ensure index is datetime
                        if not isinstance(tech_indicators.index, pd.DatetimeIndex):
                            tech_indicators.index = pd.to_datetime(tech_indicators.index)
                        df = pd.merge(
                            df,
                            tech_indicators,
                            left_index=True,
                            right_index=True,
                            how='left'
                        )
                
                # Process fundamental data
                if 'fundamental_data' in alpha_vantage_dict:
                    fundamental = pd.DataFrame(alpha_vantage_dict['fundamental_data'])
                    if not fundamental.empty:
                        # Forward fill fundamental data (it's less frequent)
                        fundamental = fundamental.ffill()
                        df = pd.merge(
                            df,
                            fundamental,
                            left_index=True,
                            right_index=True,
                            how='left'
                        )
                
                # Process earnings data
                if 'earnings' in alpha_vantage_dict:
                    earnings = pd.DataFrame(alpha_vantage_dict['earnings'])
                    if not earnings.empty:
                        # Ensure earnings dates are datetime
                        if not isinstance(earnings.index, pd.DatetimeIndex):
                            earnings.index = pd.to_datetime(earnings.index)
                        df = pd.merge(
                            df,
                            earnings,
                            left_index=True,
                            right_index=True,
                            how='left'
                        )
            
            # Add event features
            df = self.add_event_features(df, ticker)
            
            # Forward fill missing values (except for technical indicators)
            non_tech_cols = [col for col in df.columns if not any(x in col.lower() for x in ['rsi', 'macd', 'bollinger'])]
            df[non_tech_cols] = df[non_tech_cols].ffill()
            
            # Backward fill any remaining missing values
            df = df.bfill()
            
            return df
            
        except Exception as e:
            logger.error(f"Error merging external features: {e}")
            return df

    def add_event_features(self, df: pd.DataFrame, ticker=None, apikey=None) -> pd.DataFrame:
        """Add event-based binary features for earnings, dividends, FOMC, and options expiry windows using real data and MongoDB caching."""
        earnings_dates = []
        dividend_dates = []
        fomc_dates = []
        # Use SentimentAnalyzer's cached fetchers
        if ticker and apikey:
            try:
                earnings_obj = self.sentiment_analyzer.fetch_alpha_vantage_earnings_and_store(ticker)
                earnings_dates = [q.get('reportedDate') for q in earnings_obj.get('data', {}).get('quarterlyEarnings', []) if q.get('reportedDate')]
            except Exception as e:
                logger.warning(f"Earnings fetch failed: {e}")
            try:
                div_obj = self.sentiment_analyzer.fetch_alpha_vantage_dividends_and_store(ticker)
                dividend_dates = [d.get('date') for d in div_obj.get('dividends', {}).get('historical', []) if d.get('date')]
            except Exception as e:
                logger.warning(f"Dividends fetch failed: {e}")
        try:
            fomc_dates = self.sentiment_analyzer.fetch_fomc_meetings_and_store()
        except Exception as e:
            logger.warning(f"FOMC fetch failed: {e}")
        df = _mark_event_window(df, earnings_dates, 'is_earnings_event')
        df = _mark_event_window(df, dividend_dates, 'is_dividend_event')
        df = _mark_event_window(df, fomc_dates, 'is_fomc_event')
        return df

    def add_economic_event_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Add economic event features from Investing.com scraper."""
        try:
            df = df.copy()
            event_features_list = []
            
            for d in df['date']:
                try:
                    # Get event features for each date
                    event_features = self.calendar_fetcher.get_event_features(ticker, d)
                    if event_features:
                        event_features_list.append(event_features)
                    else:
                        # Add default values if no features found
                        default_features = {
                            'has_high_impact_event_today': 0,
                            'days_to_next_high_impact': 999,
                            'days_since_last_high_impact': 999,
                            'event_density_7d': 0,
                            'event_importance_sum_7d': 0.0,
                            'event_volatility_score': 0.0
                        }
                        event_features_list.append(default_features)
                except Exception as e:
                    logger.warning(f"Error getting event features for {ticker} on {d}: {e}")
                    # Add default values on error
                    default_features = {
                        'has_high_impact_event_today': 0,
                        'days_to_next_high_impact': 999,
                        'days_since_last_high_impact': 999,
                        'event_density_7d': 0,
                        'event_importance_sum_7d': 0.0,
                        'event_volatility_score': 0.0
                    }
                    event_features_list.append(default_features)
            
            if event_features_list:
                event_df = pd.DataFrame(event_features_list)
                event_df.index = df.index
                df = pd.concat([df, event_df], axis=1)
                logger.info(f"Added economic event features for {ticker} with {len(event_features_list)} events.")
            else:
                logger.warning(f"No economic event features found for {ticker}")
            
            return df
        except Exception as e:
            logger.error(f"Error adding economic event features for {ticker}: {e}")
        return df

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling volatility, lagged returns, and rolling means for 'Close' price.
        """
        df['rolling_volatility_20'] = df['Close'].rolling(window=20).std()
        df['lagged_return_1'] = df['Close'].pct_change(1)
        df['lagged_return_5'] = df['Close'].pct_change(5)
        df['rolling_mean_20'] = df['Close'].rolling(window=20).mean()
        logging.info("Added rolling features: ['rolling_volatility_20', 'lagged_return_1', 'lagged_return_5', 'rolling_mean_20']")
        return df

    def aggregate_sentiment(self, sentiment_dict: Dict) -> Tuple[float, Dict[str, float]]:
        """
        Aggregate all available nonzero sentiment sources into a blended score.
        Weighted by confidence and volume. Also returns a dict of used sources.
        """
        sources = [
            'finviz', 'seekingalpha', 'yahoo_news', 'marketaux', 'reddit',
            'sec', 'news', 'alphavantage'
        ]
        total_score = 0.0
        total_weight = 0.0
        used_sources = {}
        for src in sources:
            s = sentiment_dict.get(f'{src}_sentiment', 0)
            v = sentiment_dict.get(f'{src}_volume', 0)
            c = sentiment_dict.get(f'{src}_confidence', 0)
            if s != 0 and v > 0 and c > 0:
                weight = v * c
                total_score += s * weight
                total_weight += weight
                used_sources[src] = s
        blended = total_score / total_weight if total_weight > 0 else 0.0
        logging.info(f"Blended sentiment: {blended:.3f} from sources: {used_sources}")
        return blended, used_sources

    def add_lagged_and_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add multiple lagged returns and volatility features."""
        # Lagged returns
        for lag in [1, 2, 5, 10, 21]:
            df[f'lagged_return_{lag}'] = df['Close'].pct_change(lag)
        # Rolling volatility (std)
        for window in [5, 10, 20, 21]:
            df[f'rolling_volatility_{window}'] = df['Close'].pct_change().rolling(window=window).std()
        # Median absolute deviation
        for window in [5, 10, 21]:
            df[f'rolling_mad_{window}'] = df['Close'].rolling(window=window).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
        # Downside volatility
        for window in [5, 10, 21]:
            df[f'downside_vol_{window}'] = df['Close'].pct_change().rolling(window=window).apply(lambda x: np.std(x[x < 0]) if np.any(x < 0) else 0, raw=True)
        return df

    def fetch_options_news_features_from_db(self, mongo_client, ticker, date):
        """Fetch options and news sentiment features from MongoDB, aggregating news over the rolling window."""
        try:
            # Aggregate over the rolling window
            date_obj = pd.to_datetime(date)
            start_window = date_obj - pd.Timedelta(days=self.sentiment_window_days - 1)
            # Query for all docs in the window
            docs = list(mongo_client.db['options_sentiment'].find({
                'ticker': ticker,
                'date': {'$gte': start_window, '$lte': date_obj}
            }))
            # Aggregate options features (use most recent available in window)
            doc = docs[-1] if docs else None
            options_data = doc.get('options_data', {}) if doc else {}
            # Aggregate news sentiment features over the window
            all_articles = []
            for d in docs:
                news_data = d.get('news_data', {})
                articles = news_data.get('feed', [])
                all_articles.extend(articles)
            greeks = ['implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho']
            features = {}
            contracts = options_data.get('option_chain', [])
            for g in greeks:
                values = [float(c.get(g, 0)) for c in contracts if c.get(g) is not None]
                if values:
                    features[f'options_{g}_mean'] = np.mean(values)
                    features[f'options_{g}_std'] = np.std(values)
                    features[f'options_{g}_min'] = np.min(values)
                    features[f'options_{g}_max'] = np.max(values)
                else:
                    features[f'options_{g}_mean'] = np.nan
                    features[f'options_{g}_std'] = np.nan
                    features[f'options_{g}_min'] = np.nan
                    features[f'options_{g}_max'] = np.nan
            puts = [c for c in contracts if c.get('type') == 'put']
            calls = [c for c in contracts if c.get('type') == 'call']
            put_oi = sum(float(c.get('open_interest', 0)) for c in puts)
            call_oi = sum(float(c.get('open_interest', 0)) for c in calls)
            features['put_call_oi_ratio'] = put_oi / call_oi if call_oi > 0 else np.nan
            # Aggregate news sentiment features
            sentiments = [float(a.get('overall_sentiment_score', 0)) for a in all_articles if a.get('overall_sentiment_score') is not None]
            features['news_sentiment_mean'] = np.mean(sentiments) if sentiments else np.nan
            features['news_sentiment_std'] = np.std(sentiments) if sentiments else np.nan
            features['news_article_count'] = len(all_articles)
            topics = ['earnings', 'economy_macro', 'mergers_and_acquisitions']
            for topic in topics:
                topic_scores = [float(a.get('overall_sentiment_score', 0)) for a in all_articles if topic in a.get('topics', [])]
                features[f'news_sentiment_{topic}'] = np.mean(topic_scores) if topic_scores else np.nan
            return features
        except Exception as e:
            logger.warning(f"MongoDB options/news sentiment fetch error for {ticker} {date}: {e}")
            return {}

    def add_external_features(self, df, ticker, mongo_client=None):
        """Add options and news sentiment features from MongoDB to the DataFrame by date, using rolling window aggregation."""
        df = df.copy()
        if 'date' not in df.columns:
            logger.warning("No 'date' column in DataFrame for external features.")
            return df
        ext_features = []
        for d in df['date'].dt.strftime('%Y-%m-%d'):
            features = self.fetch_options_news_features_from_db(mongo_client, ticker, d)
            ext_features.append(features)
        ext_df = pd.DataFrame(ext_features)
        ext_df.index = df.index
        df = pd.concat([df, ext_df], axis=1)
        return df

    def shap_feature_selection(self, features_array, targets_array, feature_names, top_n=30):
        """Select top N features by mean absolute SHAP value using XGBoost."""
        import xgboost as xgb
        X_flat = features_array.reshape(features_array.shape[0], -1)
        model = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
        model.fit(X_flat, targets_array)
        explainer = shap.Explainer(model, X_flat)
        shap_values = explainer(X_flat)
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[-top_n:]
        selected_features = [feature_names[i] for i in top_idx]
        logger.info(f"Top {top_n} features by SHAP: {selected_features}")
        X_selected = X_flat[:, top_idx]
        seq_len = self.sequence_length
        num_features = X_selected.shape[1] // seq_len
        features_array_shap = X_selected.reshape(-1, seq_len, num_features)
        return features_array_shap, selected_features

    def add_sentiment_features(self, df: pd.DataFrame, sentiment_dict: Dict = None) -> pd.DataFrame:
        """
        Add sentiment features to the DataFrame, avoiding double-counting of sentiment sources.
        
        This method ONLY uses the blended sentiment score and metadata to avoid duplicating
        individual sentiment sources (FinViz, Yahoo, RSS, etc.) that are already analyzed
        in the sentiment pipeline.
        
        Args:
            df: DataFrame to add features to
            sentiment_dict: Dictionary containing blended sentiment results from sentiment.py
            
        Returns:
            DataFrame with sentiment features added
            
        Note:
            Individual sentiment sources (finviz_sentiment, yahoo_news_sentiment, etc.) 
            are NOT used here to prevent duplication with the sentiment analysis pipeline.
        """
        if sentiment_dict is None:
            logger.warning("No sentiment data provided for feature engineering")
            return df

        df = df.copy()
        
        # Use only the blended sentiment score and its metadata
        # These are aggregated scores, not individual source duplicates
        sentiment_features = {
            'blended_sentiment_score': sentiment_dict.get('blended_sentiment', 0.0),
            'sentiment_confidence': sentiment_dict.get('sentiment_confidence', 0.5),
            'sentiment_volume': sentiment_dict.get('sentiment_volume', 0),
            
            # Economic events are separate from news sentiment sources
            'economic_event_sentiment': sentiment_dict.get('economic_event_sentiment', 0.0),
            'economic_event_volatility': sentiment_dict.get('economic_event_volatility', 0.0),
            'economic_event_volume': sentiment_dict.get('economic_event_volume', 0),
            
            # Insider trading sentiment (separate from news)
            'insider_sentiment': sentiment_dict.get('finnhub_insider_sentiment', 0.0),
            'recommendation_sentiment': sentiment_dict.get('finnhub_recommendation_sentiment', 0.0),
        }
        
        # Add economic calendar features separately (event-driven, not sentiment)
        if 'economic_event_features' in sentiment_dict:
            for key, value in sentiment_dict['economic_event_features'].items():
                # Prefix to avoid conflicts and make source clear
                if isinstance(value, (int, float)):  # Only add numeric features
                    sentiment_features[f'event_{key}'] = value
        
        # Add features to DataFrame
        for feature, value in sentiment_features.items():
            df[feature] = value
            
        logger.info(f"Added {len(sentiment_features)} sentiment features (avoiding source duplication)")
        return df

    def add_alpha_vantage_features(self, df: pd.DataFrame, alpha_vantage_dict: Dict = None) -> pd.DataFrame:
        """Add features from Alpha Vantage data."""
        if not alpha_vantage_dict:
            return df
            
        df = df.copy()
        
        # Earnings call sentiment
        if 'alpha_earnings_call' in alpha_vantage_dict:
            transcript = alpha_vantage_dict['alpha_earnings_call'].get('transcript', '')
            if transcript:
                # Use existing sentiment analyzer
                sentiment = self.sentiment_analyzer._analyze_sentiment(transcript)
                df['earnings_call_sentiment'] = sentiment
        
        # Insider transactions
        if 'alpha_insider_transactions' in alpha_vantage_dict:
            insider_data = alpha_vantage_dict['alpha_insider_transactions']
            if insider_data and 'insiderTransactions' in insider_data:
                transactions = insider_data['insiderTransactions']
                df['insider_buy_volume'] = df['date'].apply(
                    lambda d: sum(t['transactionShares'] for t in transactions 
                                if t['transactionType'] == 'Buy' and 
                                pd.to_datetime(t['transactionDate']) <= d)
                )
                df['insider_sell_volume'] = df['date'].apply(
                    lambda d: sum(t['transactionShares'] for t in transactions 
                                if t['transactionType'] == 'Sell' and 
                                pd.to_datetime(t['transactionDate']) <= d)
                )
        
        # Options sentiment
        if 'options_sentiment' in alpha_vantage_dict:
            options_data = alpha_vantage_dict['options_sentiment']
            if options_data and 'data' in options_data:
                df['put_call_ratio'] = df['date'].apply(
                    lambda d: self._calculate_put_call_ratio(options_data, d)
                )
        
        return df
        
    def _calculate_put_call_ratio(self, options_data: Dict, date: datetime) -> float:
        """Calculate put/call ratio from options data."""
        try:
            date_str = date.strftime('%Y-%m-%d')
            if date_str in options_data['data']:
                day_data = options_data['data'][date_str]
                put_volume = sum(opt['volume'] for opt in day_data if opt['type'] == 'put')
                call_volume = sum(opt['volume'] for opt in day_data if opt['type'] == 'call')
                return put_volume / call_volume if call_volume > 0 else 1.0
        except Exception as e:
            logger.warning(f"Error calculating put/call ratio: {e}")
        return 1.0

    def add_sec_filings_features(self, df: pd.DataFrame, ticker: str, mongo_client=None) -> pd.DataFrame:
        """Add comprehensive SEC filings features beyond basic sentiment."""
        try:
            from .sec_filings import SECFilingsAnalyzer
            
            df = df.copy()
            sec_analyzer = SECFilingsAnalyzer(mongo_client)
            
            # Get SEC filings data for each date
            sec_features_list = []
            for date in df['date']:
                try:
                    # Get filings within 30 days of the date
                    filings_data = sec_analyzer.analyze_filings_sentiment(ticker, lookback_days=30)
                    
                    # Extract comprehensive features
                    features = {
                        # Basic sentiment
                        'sec_sentiment': filings_data.get('sec_filings_sentiment', 0.0),
                        'sec_volume': filings_data.get('sec_filings_volume', 0),
                        'sec_confidence': filings_data.get('sec_filings_confidence', 0.0),
                        
                        # Form-specific features
                        'sec_10k_count': len(filings_data.get('categorized_filings', {}).get('10-K', [])),
                        'sec_10q_count': len(filings_data.get('categorized_filings', {}).get('10-Q', [])),
                        'sec_8k_count': len(filings_data.get('categorized_filings', {}).get('8-K', [])),
                        'sec_proxy_count': len(filings_data.get('categorized_filings', {}).get('DEF 14A', [])),
                        
                        # Derived features
                        'sec_filing_frequency': filings_data.get('sec_filings_volume', 0) / 30,  # filings per day
                        'sec_major_filings': len(filings_data.get('categorized_filings', {}).get('10-K', [])) + len(filings_data.get('categorized_filings', {}).get('10-Q', [])),
                        'sec_current_events': len(filings_data.get('categorized_filings', {}).get('8-K', [])),
                        
                        # Risk indicators
                        'sec_high_activity': 1 if filings_data.get('sec_filings_volume', 0) > 10 else 0,
                        'sec_recent_10k': 1 if len(filings_data.get('categorized_filings', {}).get('10-K', [])) > 0 else 0,
                        'sec_recent_8k_spike': 1 if len(filings_data.get('categorized_filings', {}).get('8-K', [])) > 3 else 0,
                    }
                    
                    sec_features_list.append(features)
                    
                except Exception as e:
                    logger.warning(f"Error getting SEC features for {ticker} on {date}: {e}")
                    # Add default values on error
                    default_features = {
                        'sec_sentiment': 0.0, 'sec_volume': 0, 'sec_confidence': 0.0,
                        'sec_10k_count': 0, 'sec_10q_count': 0, 'sec_8k_count': 0, 'sec_proxy_count': 0,
                        'sec_filing_frequency': 0.0, 'sec_major_filings': 0, 'sec_current_events': 0,
                        'sec_high_activity': 0, 'sec_recent_10k': 0, 'sec_recent_8k_spike': 0
                    }
                    sec_features_list.append(default_features)
            
            if sec_features_list:
                sec_df = pd.DataFrame(sec_features_list)
                sec_df.index = df.index
                df = pd.concat([df, sec_df], axis=1)
                logger.info(f"Added comprehensive SEC filings features for {ticker}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding SEC filings features for {ticker}: {e}")
            return df

    def add_seeking_alpha_advanced_features(self, df: pd.DataFrame, ticker: str, mongo_client=None) -> pd.DataFrame:
        """Add advanced Seeking Alpha features beyond basic sentiment."""
        try:
            from .seeking_alpha import SeekingAlphaAnalyzer
            
            df = df.copy()
            sa_analyzer = SeekingAlphaAnalyzer(mongo_client)
            
            # Get Seeking Alpha data for each date
            sa_features_list = []
            for date in df['date']:
                try:
                    # Get both article sentiment and comments
                    comments_data = sa_analyzer.analyze_comments_sentiment(ticker, lookback_days=7)
                    
                    features = {
                        # Basic sentiment from comments
                        'sa_comments_sentiment': comments_data.get('sentiment_score', 0.0),
                        'sa_comments_volume': comments_data.get('comment_count', 0),
                        'sa_comments_confidence': comments_data.get('confidence', 0.0),
                        
                        # Advanced comment features
                        'sa_positive_ratio': comments_data.get('positive_ratio', 0.5),
                        'sa_negative_ratio': comments_data.get('negative_ratio', 0.5),
                        'sa_engagement_score': comments_data.get('comment_count', 0) / 7,  # comments per day
                        
                        # Comment quality indicators
                        'sa_high_engagement': 1 if comments_data.get('comment_count', 0) > 50 else 0,
                        'sa_sentiment_consensus': 1 if abs(comments_data.get('sentiment_score', 0)) > 0.3 else 0,
                        'sa_polarized_discussion': 1 if comments_data.get('sentiment_std', 0) > 0.5 else 0,
                        
                        # Derived features
                        'sa_bullish_signal': 1 if (comments_data.get('sentiment_score', 0) > 0.2 and comments_data.get('comment_count', 0) > 20) else 0,
                        'sa_bearish_signal': 1 if (comments_data.get('sentiment_score', 0) < -0.2 and comments_data.get('comment_count', 0) > 20) else 0,
                    }
                    
                    sa_features_list.append(features)
                    
                except Exception as e:
                    logger.warning(f"Error getting Seeking Alpha features for {ticker} on {date}: {e}")
                    # Add default values on error
                    default_features = {
                        'sa_comments_sentiment': 0.0, 'sa_comments_volume': 0, 'sa_comments_confidence': 0.0,
                        'sa_positive_ratio': 0.5, 'sa_negative_ratio': 0.5, 'sa_engagement_score': 0.0,
                        'sa_high_engagement': 0, 'sa_sentiment_consensus': 0, 'sa_polarized_discussion': 0,
                        'sa_bullish_signal': 0, 'sa_bearish_signal': 0
                    }
                    sa_features_list.append(default_features)
            
            if sa_features_list:
                sa_df = pd.DataFrame(sa_features_list)
                sa_df.index = df.index
                df = pd.concat([df, sa_df], axis=1)
                logger.info(f"Added comprehensive Seeking Alpha features for {ticker}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding Seeking Alpha features for {ticker}: {e}")
            return df

    def create_prediction_features(
        self,
        df: pd.DataFrame,
        ticker: str,
        window: str,
        mongo_client=None,
        use_stored_pipeline: bool = True
    ) -> np.ndarray:
        """
        Create features for prediction using the exact same pipeline as training.
        This ensures feature consistency between training and prediction phases.
        """
        try:
            window_size_map = {'next_day': 1, '7_day': 7, '30_day': 30, '90_day': 90}
            window_size = window_size_map.get(window, 1)
            
            # Load stored feature pipeline if available
            feature_pipeline_path = f"models/{ticker}/feature_pipeline_{ticker}_{window}.json"
            
            if use_stored_pipeline and os.path.exists(feature_pipeline_path):
                with open(feature_pipeline_path, 'r') as f:
                    pipeline_info = json.load(f)
                
                # Apply the exact same feature engineering steps as training
                logger.info(f"Using stored feature pipeline for {ticker}-{window}")
                
                # Step 1: Technical indicators (always consistent)
                df = self.add_technical_indicators(df)
                
                # Step 2: Get current values for external features that were used in training
                feature_columns = pipeline_info.get('feature_columns', [])
                external_features = pipeline_info.get('external_features', {})
                
                # Add placeholder external features with current/latest values
                for feature_name, default_value in external_features.items():
                    if feature_name not in df.columns:
                        # Try to get current value from MongoDB or use default
                        if mongo_client and 'sentiment' in feature_name:
                            current_sentiment = mongo_client.get_latest_sentiment(ticker)
                            if current_sentiment and feature_name in current_sentiment:
                                df[feature_name] = current_sentiment[feature_name]
                            else:
                                df[feature_name] = default_value
                        else:
                            df[feature_name] = default_value
                
                # Step 3: Rolling features
                df = self.add_rolling_features(df)
                df = self.add_lagged_and_volatility_features(df)
                
                # Step 4: Handle outliers if it was done during training
                if pipeline_info.get('outlier_handling_applied', False):
                    df = self.handle_outliers(df)
                
                # Step 5: Select only the features that were used during training
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                available_features = [col for col in feature_columns if col in numeric_cols]
                
                if len(available_features) < len(feature_columns) * 0.8:
                    logger.warning(f"Only {len(available_features)}/{len(feature_columns)} features available for prediction")
                
                # Fill missing features with zeros or stored defaults
                for col in feature_columns:
                    if col not in df.columns:
                        default_val = external_features.get(col, 0.0)
                        df[col] = default_val
                        logger.debug(f"Added missing feature {col} with default value {default_val}")
                
                # Reorder columns to match training
                df_features = df[feature_columns].copy()
                
            else:
                # Fallback: Create features using current pipeline (may have dimension mismatch)
                logger.warning(f"No stored pipeline found for {ticker}-{window}, using current pipeline")
                df_features, _ = self.prepare_features(
                    df,
                    window_size=window_size,
                    ticker=ticker,
                    mongo_client=mongo_client,
                    handle_outliers=True
                )
                return df_features
            
            # Handle NaN values
            df_features = df_features.ffill().bfill().fillna(0)
            
            # Convert to numpy array
            features = df_features.values
            
            # Save feature pipeline if requested (during training)
            if save_pipeline and ticker and window:
                # Collect external feature defaults
                external_features = {}
                for col in feature_df.columns:
                    if col not in ['Close', 'Open', 'High', 'Low', 'Volume'] and 'SMA' not in col and 'EMA' not in col and 'RSI' not in col:
                        # This is likely an external feature
                        external_features[col] = float(feature_df[col].iloc[-1]) if not feature_df[col].empty else 0.0
                
                self.save_feature_pipeline(
                    ticker=ticker,
                    window=window,
                    feature_columns=list(feature_df.columns),
                    external_features=external_features,
                    outlier_handling_applied=handle_outliers
                )
            
            # Create targets (price changes for next day, next 30 days, next 90 days)
            targets = None
            if 'Close' in df.columns:
                close_prices = df['Close'].values
                if window_size == 1:
                    # Next day prediction: tomorrow's price - today's price
                    targets = np.diff(close_prices)  # Shape: (n-1,)
                    features = features[:-1]  # Align features with targets
                elif window_size > 1:
                    # Multi-day prediction: price N days in future - current price
                    targets = []
                    for i in range(len(close_prices) - window_size):
                        current_price = close_prices[i]
                        future_price = close_prices[i + window_size] if i + window_size < len(close_prices) else current_price
                        price_change = future_price - current_price
                        targets.append(price_change)
                    targets = np.array(targets)
                    
                    # Align features - we need to remove the last window_size-1 samples
                    # because we can't predict future prices for them
                    features = features[:-window_size] if len(features) > window_size else features
                
            # Handle reshaping based on model type expectations
            if window_size > 1:
                # For LSTM models: Create 3D arrays (samples, time_steps, features)
                n_samples = len(targets) if targets is not None else features.shape[0] - window_size + 1
                n_features = features.shape[1]
                
                if n_samples <= 0:
                    raise ValueError(f"Not enough data for window_size {window_size}. Need at least {window_size} samples, got {features.shape[0]}")
                
                # Create sliding windows for LSTM - each sample contains window_size time steps
                max_samples = features.shape[0] - window_size + 1
                n_samples = min(n_samples, max_samples)
                
                if n_samples <= 0:
                    raise ValueError(f"Not enough data for windowing. Features: {features.shape[0]}, window_size: {window_size}")
                
                windowed_features = np.zeros((n_samples, window_size, n_features))
                for i in range(n_samples):
                    end_idx = i + window_size
                    if end_idx <= features.shape[0]:
                        windowed_features[i] = features[i:end_idx]
                    else:
                        # Handle edge case by padding with last available features
                        available_features = features[i:]
                        windowed_features[i, :len(available_features)] = available_features
                        # Pad remaining with last feature
                        if len(available_features) > 0:
                            windowed_features[i, len(available_features):] = available_features[-1]
                
                # For LightGBM/XGBoost: We'll flatten in the predictor, keep 3D here for LSTM
                features = windowed_features
                
                # Ensure targets align with features
                if targets is not None and len(targets) > n_samples:
                    targets = targets[:n_samples]
                    
            return features, targets if targets is not None else feature_stats
            
        except Exception as e:
            logger.error(f"Error creating prediction features for {ticker}-{window}: {e}")
            raise
    
    def save_feature_pipeline(
        self,
        ticker: str,
        window: str,
        feature_columns: List[str],
        external_features: Dict[str, float],
        outlier_handling_applied: bool = False
    ):
        """Save the feature engineering pipeline for consistent prediction."""
        try:
            os.makedirs(f"models/{ticker}", exist_ok=True)
            
            pipeline_info = {
                'ticker': ticker,
                'window': window,
                'feature_columns': feature_columns,
                'external_features': external_features,
                'outlier_handling_applied': outlier_handling_applied,
                'created_at': datetime.utcnow().isoformat(),
                'feature_count': len(feature_columns)
            }
            
            pipeline_path = f"models/{ticker}/feature_pipeline_{ticker}_{window}.json"
            with open(pipeline_path, 'w') as f:
                json.dump(pipeline_info, f, indent=2)
            
            logger.info(f"Saved feature pipeline for {ticker}-{window}: {len(feature_columns)} features")
            
        except Exception as e:
            logger.error(f"Error saving feature pipeline for {ticker}-{window}: {e}")

    def prepare_features(
        self,
        df: pd.DataFrame,
        sentiment_dict: Dict = None,
        alpha_vantage_dict: Dict = None,
        window_size: int = 1,
        mongo_client = None,
        ticker: str = None,
        handle_outliers: bool = True,
        enable_shap_selection: bool = False,
        feature_select_k: int = 20,
        save_pipeline: bool = False,
        window: str = None
    ) -> Tuple[np.ndarray, Dict]:
        """Prepare features for model training/prediction."""
        try:
            # Validate input
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
                
            if df.empty:
                raise ValueError("Input DataFrame is empty")
                
            # Check for required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            # Create a copy to avoid modifying the original
            df = df.copy()
            
            # Add date column if not present
            if 'date' not in df.columns:
                df['date'] = pd.to_datetime(df.index)
                
            # Add technical indicators
            df = self.add_technical_indicators(df)

            # Add external features if ticker is provided
            if ticker:
                # Add macro and sector data
                df = self.merge_external_features(df, ticker=ticker, alpha_vantage_dict=alpha_vantage_dict)
                # Add event features (earnings, dividends, FOMC)
                df = self.add_event_features(df, ticker=ticker, apikey=os.getenv('ALPHAVANTAGE_API_KEY'))
                # Add economic event features
                df = self.add_economic_event_features(df, ticker)
                # Add short interest features
                df = self.add_short_interest_features(df, ticker, mongo_client)
                # Add comprehensive SEC filings features
                df = self.add_sec_filings_features(df, ticker, mongo_client)
                # Add advanced Seeking Alpha features
                df = self.add_seeking_alpha_advanced_features(df, ticker, mongo_client)

            # Inject flat sentiment fields as columns (no double-counting, no re-normalization)
            if sentiment_dict:
                flat_sent = {
                    k: v for k, v in sentiment_dict.items()
                    if any(k.endswith(suffix) for suffix in ["_sentiment", "_volume", "_confidence"])
                }
                for field, value in flat_sent.items():
                    df[field] = value if not isinstance(value, list) else len(value)

            # Add rolling features
            df = self.add_rolling_features(df)
            
            # Add lagged and volatility features
            df = self.add_lagged_and_volatility_features(df)
            
            # Handle outliers if requested
            if handle_outliers:
                df = self.handle_outliers(df)
                
            # Auto-optimize features based on historical performance
            if ticker and mongo_client:
                df = self.auto_feature_selection_and_optimization(df, ticker, mongo_client)
                
            # Remove duplicate features
            df = df.loc[:, ~df.columns.duplicated()]
            
            # Exclude non-numeric columns (like MongoDB _id ObjectId) before processing
            non_numeric_cols = ['_id', 'ticker', 'date']
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Keep only numeric columns for feature processing
            feature_df = df[numeric_cols].copy()
            
            # Remove features with all NaN values
            feature_df = feature_df.dropna(axis=1, how='all')
            
            # Fill remaining NaN values using modern pandas syntax
            feature_df = feature_df.ffill().bfill().fillna(0)
            
            # Normalize features
            feature_stats = self._normalize_features(feature_df)
            
            # Select features if requested
            if enable_shap_selection and ticker:
                # Get feature importance from MongoDB if available
                if mongo_client:
                    feature_importance = mongo_client.db['feature_importance'].find_one({'ticker': ticker})
                    if feature_importance:
                        # Use stored feature importance
                        selected_features = list(feature_importance.get('features', {}).keys())[:feature_select_k]
                        # Filter to only include features that exist in our dataframe
                        selected_features = [f for f in selected_features if f in feature_df.columns]
                        if selected_features:
                            feature_df = feature_df[selected_features]
                else:
                    # Use SHAP for feature selection
                    self.select_features(feature_df.values, None, k=feature_select_k)
                    if self.selected_features is not None:
                        feature_df = feature_df.iloc[:, self.selected_features]
                    
            # Convert to numpy array
            features = feature_df.values
            
            # Save feature pipeline if requested (during training)
            if save_pipeline and ticker and window:
                # Collect external feature defaults
                external_features = {}
                for col in feature_df.columns:
                    if col not in ['Close', 'Open', 'High', 'Low', 'Volume'] and 'SMA' not in col and 'EMA' not in col and 'RSI' not in col:
                        # This is likely an external feature
                        external_features[col] = float(feature_df[col].iloc[-1]) if not feature_df[col].empty else 0.0
                
                self.save_feature_pipeline(
                    ticker=ticker,
                    window=window,
                    feature_columns=list(feature_df.columns),
                    external_features=external_features,
                    outlier_handling_applied=handle_outliers
                )
            
            # Create targets (price changes for next day, next 30 days, next 90 days)
            targets = None
            if 'Close' in df.columns:
                close_prices = df['Close'].values
                if window_size == 1:
                    # Next day prediction: tomorrow's price - today's price
                    targets = np.diff(close_prices)  # Shape: (n-1,)
                    features = features[:-1]  # Align features with targets
                elif window_size > 1:
                    # Multi-day prediction: price N days in future - current price
                    targets = []
                    for i in range(len(close_prices) - window_size):
                        current_price = close_prices[i]
                        future_price = close_prices[i + window_size] if i + window_size < len(close_prices) else current_price
                        price_change = future_price - current_price
                        targets.append(price_change)
                    targets = np.array(targets)
                    
                    # Align features - we need to remove the last window_size-1 samples
                    # because we can't predict future prices for them
                    features = features[:-window_size] if len(features) > window_size else features
                
            # Handle reshaping based on model type expectations
            if window_size > 1:
                # For LSTM models: Create 3D arrays (samples, time_steps, features)
                n_samples = len(targets) if targets is not None else features.shape[0] - window_size + 1
                n_features = features.shape[1]
                
                if n_samples <= 0:
                    raise ValueError(f"Not enough data for window_size {window_size}. Need at least {window_size} samples, got {features.shape[0]}")
                
                # Create sliding windows for LSTM - each sample contains window_size time steps
                max_samples = features.shape[0] - window_size + 1
                n_samples = min(n_samples, max_samples)
                
                if n_samples <= 0:
                    raise ValueError(f"Not enough data for windowing. Features: {features.shape[0]}, window_size: {window_size}")
                
                windowed_features = np.zeros((n_samples, window_size, n_features))
                for i in range(n_samples):
                    end_idx = i + window_size
                    if end_idx <= features.shape[0]:
                        windowed_features[i] = features[i:end_idx]
                    else:
                        # Handle edge case by padding with last available features
                        available_features = features[i:]
                        windowed_features[i, :len(available_features)] = available_features
                        # Pad remaining with last feature
                        if len(available_features) > 0:
                            windowed_features[i, len(available_features):] = available_features[-1]
                
                # For LightGBM/XGBoost: We'll flatten in the predictor, keep 3D here for LSTM
                features = windowed_features
                
                # Ensure targets align with features
                if targets is not None and len(targets) > n_samples:
                    targets = targets[:n_samples]
                    
            return features, targets if targets is not None else feature_stats
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise

    def _normalize_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Normalize features for model input, returning only scalars (last value in sequence)."""
        try:
            normalized = {}
            
            # Only process numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Price features (if they exist)
            price_cols = ['Close', 'Open', 'High', 'Low']
            for col in price_cols:
                if col in numeric_cols and col in df.columns:
                    if df[col].std() != 0:
                        series = (df[col] - df[col].mean()) / df[col].std()
                        normalized[f'{col}_norm'] = float(series.iloc[-1]) if not series.empty else 0.0
                    else:
                        normalized[f'{col}_norm'] = 0.0
                        
            # Volume features (if it exists)
            if 'Volume' in numeric_cols and 'Volume' in df.columns:
                if df['Volume'].std() != 0:
                    series = (df['Volume'] - df['Volume'].mean()) / df['Volume'].std()
                    normalized['Volume_norm'] = float(series.iloc[-1]) if not series.empty else 0.0
                else:
                    normalized['Volume_norm'] = 0.0
                    
            # Technical indicators (all other numeric columns)
            tech_cols = [col for col in numeric_cols if col not in price_cols + ['Volume']]
            for col in tech_cols:
                if col in df.columns:
                    try:
                        if df[col].std() != 0:
                            series = (df[col] - df[col].mean()) / df[col].std()
                            normalized[f'{col}_norm'] = float(series.iloc[-1]) if not series.empty else 0.0
                        else:
                            normalized[f'{col}_norm'] = 0.0
                    except (TypeError, ValueError) as e:
                        # Skip columns that can't be normalized (e.g., ObjectId, strings)
                        logger.warning(f"Skipping normalization for column {col}: {e}")
                        continue
                        
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing features: {str(e)}")
            return {}  # Return empty dict instead of trying to process all columns

    def select_features(self, features: np.ndarray, targets: np.ndarray, k: int = 20) -> None:
        """Select the most important features using SelectKBest."""
        try:
            # Initialize feature selector
            self.feature_selector = SelectKBest(score_func=f_regression, k=k)
            
            # Fit selector
            self.feature_selector.fit(features, targets)
            
            # Get selected feature indices
            self.selected_features = self.feature_selector.get_support(indices=True)
            
            logger.info(f"Selected {len(self.selected_features)} most important features")
            
        except Exception as e:
            logger.error(f"Error selecting features: {str(e)}")
            self.feature_selector = None
            self.selected_features = None

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        try:
            if self.feature_selector is None:
                return {}
            
            # Get feature scores
            scores = self.feature_selector.scores_
            
            # Normalize scores
            scores = scores / scores.sum()
            
            # Create feature importance dictionary
            importance = {
                f"feature_{i}": score
                for i, score in enumerate(scores)
            }
            
            return importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}

    def _calculate_growth_rate(self, series: pd.Series) -> float:
        """Calculate growth rate from a series of values."""
        if len(series) < 2:
            return np.nan
        return (series.iloc[0] - series.iloc[-1]) / abs(series.iloc[-1]) if series.iloc[-1] != 0 else np.nan 

    def store_llm_explanation(self, ticker: str, prediction_date: datetime, explanation_data: Dict) -> bool:
        """
        Store LLM explanation data in MongoDB.
        
        Args:
            ticker: Stock ticker symbol
            prediction_date: Date for which the prediction was made
            explanation_data: Dictionary containing explanation data including:
                - prediction_range: The predicted price range
                - confidence_score: Model's confidence in the prediction
                - key_factors: List of factors that influenced the prediction
                - technical_analysis: Technical indicators that contributed
                - sentiment_analysis: Sentiment factors that influenced
                - market_context: Broader market conditions
                - llm_explanation: The actual LLM-generated explanation
                - feature_importance: Dictionary of important features
                - timestamp: When the explanation was generated
                
        Returns:
            bool: True if storage was successful, False otherwise
        """
        if not self.mongo_client:
            logger.warning("MongoDB client not initialized, skipping LLM explanation storage")
            return False
            
        try:
            collection = self.mongo_client.db['llm_explanations']
            
            # Prepare the document
            document = {
                'ticker': ticker,
                'prediction_date': prediction_date,
                'timestamp': datetime.utcnow(),
                **explanation_data
            }
            
            # Store in MongoDB with upsert to avoid duplicates
            collection.replace_one(
                {
                    'ticker': ticker,
                    'prediction_date': prediction_date
                },
                document,
                upsert=True
            )
            
            logger.info(f"Stored LLM explanation for {ticker} on {prediction_date}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing LLM explanation for {ticker}: {e}")
            return False
            
    def get_llm_explanation(self, ticker: str, prediction_date: datetime) -> Dict:
        """
        Retrieve LLM explanation data from MongoDB.
        
        Args:
            ticker: Stock ticker symbol
            prediction_date: Date for which the prediction was made
            
        Returns:
            Dict: The stored explanation data or None if not found
        """
        if not self.mongo_client:
            logger.warning("MongoDB client not initialized, skipping LLM explanation retrieval")
            return None
            
        try:
            collection = self.mongo_client.db['llm_explanations']
            
            # Query MongoDB for the explanation
            explanation = collection.find_one({
                'ticker': ticker,
                'prediction_date': prediction_date
            })
            
            if explanation:
                # Remove MongoDB's _id field
                explanation.pop('_id', None)
                logger.info(f"Retrieved LLM explanation for {ticker} on {prediction_date}")
                return explanation
            else:
                logger.warning(f"No LLM explanation found for {ticker} on {prediction_date}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving LLM explanation for {ticker}: {e}")
            return None
            
    def get_recent_llm_explanations(self, ticker: str, limit: int = 5) -> List[Dict]:
        """
        Get recent LLM explanations for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of recent explanations to return
            
        Returns:
            List[Dict]: List of recent explanations
        """
        if not self.mongo_client:
            logger.warning("MongoDB client not initialized, skipping LLM explanation retrieval")
            return []
            
        try:
            collection = self.mongo_client.db['llm_explanations']
            
            # Query MongoDB for recent explanations
            explanations = list(collection.find(
                {'ticker': ticker},
                sort=[('prediction_date', -1)],
                limit=limit
            ))
            
            # Remove MongoDB's _id field from each document
            for exp in explanations:
                exp.pop('_id', None)
                
            logger.info(f"Retrieved {len(explanations)} recent LLM explanations for {ticker}")
            return explanations
            
        except Exception as e:
            logger.error(f"Error retrieving recent LLM explanations for {ticker}: {e}")
            return []

    def auto_feature_selection_and_optimization(self, df: pd.DataFrame, ticker: str, mongo_client=None) -> pd.DataFrame:
        """Automatically select and optimize features based on prediction performance."""
        try:
            if not mongo_client:
                return df
                
            # Get stored feature importance for this ticker
            feature_importance_doc = mongo_client.db['feature_importance'].find_one(
                {'ticker': ticker},
                sort=[('timestamp', -1)]
            )
            
            if feature_importance_doc and 'feature_scores' in feature_importance_doc:
                feature_scores = feature_importance_doc['feature_scores']
                
                # More aggressive feature selection with higher threshold
                threshold = 0.02  # Increased from 0.01 - Features must contribute at least 2%
                important_features = [
                    feature for feature, score in feature_scores.items() 
                    if score > threshold
                ]
                
                # Keep essential features always
                essential_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'date']
                all_important = list(set(important_features + essential_features))
                
                # Filter dataframe to only important features
                available_features = [col for col in all_important if col in df.columns]
                if len(available_features) > len(essential_features):
                    df = df[available_features]
                    logger.info(f"Auto-selected {len(available_features)} important features for {ticker}")
                
            # Remove highly correlated features with lower threshold for more aggressive removal
            df = self._remove_correlated_features(df, correlation_threshold=0.90)  # Reduced from 0.95
            
            # Reduce feature interactions for efficiency
            df = self._add_feature_interactions(df, max_interactions=5)  # Reduced from 10
            
            return df
            
        except Exception as e:
            logger.error(f"Error in auto feature selection for {ticker}: {e}")
            return df
    
    def _remove_correlated_features(self, df: pd.DataFrame, correlation_threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features to reduce multicollinearity."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return df
                
            correlation_matrix = df[numeric_cols].corr().abs()
            
            # Find highly correlated pairs
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            
            # Identify features to drop
            to_drop = [
                column for column in upper_triangle.columns 
                if any(upper_triangle[column] > correlation_threshold)
            ]
            
            # Keep essential features
            essential = ['Open', 'High', 'Low', 'Close', 'Volume']
            to_drop = [col for col in to_drop if col not in essential]
            
            if to_drop:
                df = df.drop(columns=to_drop)
                logger.info(f"Removed {len(to_drop)} highly correlated features: {to_drop[:5]}...")
                
            return df
            
        except Exception as e:
            logger.error(f"Error removing correlated features: {e}")
            return df
    
    def _add_feature_interactions(self, df: pd.DataFrame, max_interactions: int = 10) -> pd.DataFrame:
        """Add feature interaction terms for important features."""
        try:
            # Select top features for interactions (price and volume related)
            interaction_candidates = [
                col for col in df.columns 
                if any(keyword in col.lower() for keyword in ['close', 'volume', 'rsi', 'macd', 'sentiment'])
            ][:5]  # Limit to top 5 to avoid explosion
            
            interaction_count = 0
            for i in range(len(interaction_candidates)):
                for j in range(i+1, len(interaction_candidates)):
                    if interaction_count >= max_interactions:
                        break
                        
                    col1, col2 = interaction_candidates[i], interaction_candidates[j]
                    if col1 in df.columns and col2 in df.columns:
                        # Add interaction term
                        interaction_name = f"{col1}_x_{col2}"
                        df[interaction_name] = df[col1] * df[col2]
                        interaction_count += 1
                        
                if interaction_count >= max_interactions:
                    break
            
            if interaction_count > 0:
                logger.info(f"Added {interaction_count} feature interaction terms")
                
            return df
            
        except Exception as e:
            logger.error(f"Error adding feature interactions: {e}")
            return df

    def store_feature_importance(self, ticker: str, feature_scores: Dict[str, float], window: str, mongo_client=None):
        """Store feature importance scores for future optimization."""
        try:
            if not mongo_client:
                return
                
            # Normalize scores
            total_score = sum(abs(score) for score in feature_scores.values())
            if total_score > 0:
                normalized_scores = {
                    feature: abs(score) / total_score 
                    for feature, score in feature_scores.items()
                }
            else:
                normalized_scores = feature_scores
            
            # Store in MongoDB
            mongo_client.db['feature_importance'].replace_one(
                {'ticker': ticker, 'window': window},
                {
                    'ticker': ticker,
                    'window': window,
                    'feature_scores': normalized_scores,
                    'timestamp': datetime.utcnow(),
                    'total_features': len(feature_scores)
                },
                upsert=True
            )
            
            logger.info(f"Stored feature importance for {ticker} ({window}): top feature = {max(normalized_scores, key=normalized_scores.get)}")
            
        except Exception as e:
            logger.error(f"Error storing feature importance: {e}")

    def get_optimized_feature_set(self, ticker: str, mongo_client=None) -> Optional[List[str]]:
        """Get the optimized feature set for a ticker based on historical performance."""
        try:
            if not mongo_client:
                return None
                
            # Get latest feature importance
            importance_doc = mongo_client.db['feature_importance'].find_one(
                {'ticker': ticker},
                sort=[('timestamp', -1)]
            )
            
            if importance_doc and 'feature_scores' in importance_doc:
                # Return features above the significance threshold
                threshold = 0.005  # 0.5% minimum contribution
                important_features = [
                    feature for feature, score in importance_doc['feature_scores'].items()
                    if score > threshold
                ]
                return important_features
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting optimized feature set: {e}")
            return None 