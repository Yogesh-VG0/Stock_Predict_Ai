"""
Feature engineering module for generating technical indicators and other features.
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
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
import asyncio
import concurrent.futures
import joblib

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
                            # Convert cached dict to DataFrame-like structure
                            cached_series = pd.Series(cached_data)
                            cached_series.index = pd.to_datetime(cached_series.index)
                            cached_series = cached_series.reindex(dates, method='ffill')
                            
                            df[f"{etf}_close"] = cached_series.values
                            df[f"{etf}_volume"] = cached_series.values  # Use same for volume (placeholder)
                            
                            # Calculate derived features
                            if len(cached_series) > 1:
                                df[f"{etf}_return"] = cached_series.pct_change().values
                                df[f"{etf}_volatility"] = cached_series.rolling(20).std().values
                                df[f"{etf}_ma5"] = cached_series.rolling(5).mean().values
                                df[f"{etf}_ma20"] = cached_series.rolling(20).mean().values
                            continue
                    
                    # Fetch from yfinance if not cached
                    data = yf.download(etf, start=start_date, end=end_date, progress=False, auto_adjust=True)
                    
                    if data.empty:
                        logger.warning(f"No data returned from yfinance for {etf}")
                        df[f"{etf}_close"] = np.nan
                        df[f"{etf}_volume"] = np.nan
                        continue
                    
                    # Store in MongoDB if successful
                    if mongo_client and not data.empty:
                        try:
                            data_dict = {d.strftime('%Y-%m-%d'): float(v) for d, v in data['Close'].items() if pd.notna(v)}
                            mongo_client.store_sector_data(etf, data_dict)
                        except Exception as e:
                            logger.warning(f"Failed to store {etf} data in MongoDB: {e}")

                    data = data.reindex(dates, method='ffill')
                    df[f"{etf}_close"] = data['Close'].values if 'Close' in data.columns else np.nan
                    df[f"{etf}_volume"] = data['Volume'].values if 'Volume' in data.columns else np.nan

                    # Calculate derived features
                    if not data.empty and 'Close' in data.columns and len(data) > 1:
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
        """
        Add short interest features as predictive indicators for stock sentiment and potential price movements.
        Short interest indicates bearish sentiment and potential short squeeze opportunities.
        """
        try:
            logger.info(f"Adding short interest features for {ticker}")
            
            # Initialize short interest analyzer if needed
            self._init_short_interest_analyzer(mongo_client or self.mongo_client)
            
            # Default values for short interest features
            df['short_interest_ratio'] = 0.0  # Short interest as % of float
            df['days_to_cover'] = 0.0  # Days to cover short positions
            df['short_volume_ratio'] = 0.0  # Short volume as % of total volume
            df['short_interest_change'] = 0.0  # Change in short interest
            df['short_squeeze_potential'] = 0.0  # Calculated squeeze potential score
            
            # Fetch latest short interest data
            for idx, row in df.iterrows():
                try:
                    date = row.get('date', idx)
                    if isinstance(date, str):
                        date = pd.to_datetime(date)
                    
                    # Get short interest data from analyzer
                    short_data = self.short_interest_analyzer.get_short_interest_data(ticker, date)
                    
                    if short_data:
                        # Extract short interest metrics
                        df.at[idx, 'short_interest_ratio'] = short_data.get('short_interest_ratio', 0.0)
                        df.at[idx, 'days_to_cover'] = short_data.get('days_to_cover', 0.0)
                        df.at[idx, 'short_volume_ratio'] = short_data.get('short_volume_ratio', 0.0)
                        df.at[idx, 'short_interest_change'] = short_data.get('short_interest_change', 0.0)
                        
                        # Calculate short squeeze potential
                        # High short interest + low days to cover + increasing price = potential squeeze
                        short_ratio = short_data.get('short_interest_ratio', 0.0)
                        days_cover = short_data.get('days_to_cover', 0.0)
                        
                        if short_ratio > 10 and days_cover > 3:  # High short interest and difficult to cover
                            squeeze_score = min((short_ratio / 20.0) + (days_cover / 10.0), 1.0)
                        else:
                            squeeze_score = 0.0
                            
                        df.at[idx, 'short_squeeze_potential'] = squeeze_score
                        
                except Exception as e:
                    logger.warning(f"Error processing short interest for {ticker} on {date}: {e}")
                    # Keep default values
                    
            # Calculate rolling features for short interest trends
            df['short_interest_ma5'] = df['short_interest_ratio'].rolling(5).mean()
            df['short_interest_ma20'] = df['short_interest_ratio'].rolling(20).mean()
            df['short_interest_trend'] = df['short_interest_ratio'] - df['short_interest_ma20']
            
            # Forward fill missing values
            short_cols = ['short_interest_ratio', 'days_to_cover', 'short_volume_ratio', 
                         'short_interest_change', 'short_squeeze_potential']
            df[short_cols] = df[short_cols].fillna(method='ffill').fillna(0)
            
            logger.info(f"Successfully added short interest features for {ticker}")
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
            
            # NEW: Advanced Technical Indicators for Better Accuracy
            
            # 1. Momentum Oscillators
            df['Stochastic_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14)
            df['Stochastic_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=14)
            df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)
            
            # 2. Trend Strength Indicators
            df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
            df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)
            
            # 3. Volume-Price Indicators (Critical for accuracy)
            df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            df['Volume_Price_Trend'] = ta.volume.volume_price_trend(df['Close'], df['Volume'])
            df['Money_Flow_Index'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
            
            # 4. Market Regime Detection Features
            # Volatility Regime
            df['VIX_Proxy'] = df['Daily_Return'].rolling(20).std() * np.sqrt(252)  # Annualized volatility
            df['Volatility_Regime'] = (df['VIX_Proxy'] > df['VIX_Proxy'].rolling(60).quantile(0.7)).astype(int)
            
            # Trend Regime
            df['Trend_Strength'] = (df['Close'] > df['SMA_50']).astype(int)
            df['Long_Term_Trend'] = (df['SMA_50'] > df['SMA_200']).astype(int)
            
            # 5. Price Action Features
            # Gap detection
            df['Gap_Up'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) > 0.02).astype(int)
            df['Gap_Down'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) < -0.02).astype(int)
            
            # Breakout detection
            df['Breakout_High'] = (df['Close'] > df['High'].rolling(20).max().shift(1)).astype(int)
            df['Breakout_Low'] = (df['Close'] < df['Low'].rolling(20).min().shift(1)).astype(int)
            
            # 6. Multi-timeframe Features
            # Weekly indicators (using 5-day aggregation)
            df['Weekly_Return'] = df['Close'].pct_change(5)
            df['Weekly_High'] = df['High'].rolling(5).max()
            df['Weekly_Low'] = df['Low'].rolling(5).min()
            df['Weekly_Volume'] = df['Volume'].rolling(5).mean()
            
            # 7. Divergence Detection (Advanced)
            # Price vs RSI divergence
            df['Price_RSI_Divergence'] = self._detect_divergence(df['Close'], df['RSI'])
            
            # Price vs Volume divergence
            df['Price_Volume_Divergence'] = self._detect_divergence(df['Close'], df['OBV'])
            
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

    def _detect_divergence(self, price_series: pd.Series, indicator_series: pd.Series, window: int = 20) -> pd.Series:
        """Detect bullish/bearish divergence between price and indicator."""
        try:
            # Calculate rolling correlations
            correlation = price_series.rolling(window).corr(indicator_series)
            
            # Detect divergence: negative correlation indicates divergence
            divergence = (correlation < -0.3).astype(int)  # -1 = bearish divergence, 1 = bullish divergence
            
            return divergence.fillna(0)
        except:
            return pd.Series(0, index=price_series.index)

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

    def add_macro_economic_features(self, df: pd.DataFrame, mongo_client=None) -> pd.DataFrame:
        """
        Add macroeconomic features from FRED data that affect stock predictions.
        These features provide economic context for individual stock movements.
        """
        try:
            logger.info("Adding macro economic features from FRED data")
            
            # Get date range from dataframe
            if 'date' in df.columns:
                start_date = df['date'].min()
                end_date = df['date'].max()
            else:
                start_date = df.index.min()
                end_date = df.index.max()
            
            # Fetch macro data using the fetcher
            macro_df = self.macro_fetcher.fetch_all(start_date, end_date, mongo_client)
            
            if macro_df is not None and not macro_df.empty and len(macro_df.columns) > 1:
                logger.info(f"Successfully fetched macro data with {len(macro_df.columns)} indicators")
                
                # Ensure date alignment
                if 'date' in df.columns:
                    df_for_merge = df.set_index('date')
                    macro_df_for_merge = macro_df.set_index('date')
                else:
                    df_for_merge = df.copy()
                    macro_df_for_merge = macro_df.set_index('date')
                
                # Merge macro data with stock data
                df_merged = pd.merge(
                    df_for_merge, 
                    macro_df_for_merge, 
                    left_index=True, 
                    right_index=True, 
                    how='left'
                )
                
                # Restore date column if it existed
                if 'date' in df.columns:
                    df_merged = df_merged.reset_index()
                
                # Forward fill macro data (it's lower frequency than daily stock data)
                macro_columns = [col for col in macro_df.columns if col != 'date']
                df_merged[macro_columns] = df_merged[macro_columns].fillna(method='ffill')
                
                # Add macro trend features
                for col in macro_columns:
                    if col.endswith('_change') or col.endswith('_ma5') or col.endswith('_ma20'):
                        continue  # Skip derived features to avoid recalculation
                    
                    # Add trend indicators
                    if col in df_merged.columns:
                        try:
                            # Current vs 1-month ago trend
                            df_merged[f'{col}_trend_1m'] = df_merged[col] - df_merged[col].shift(21)
                            # Current vs 3-month ago trend  
                            df_merged[f'{col}_trend_3m'] = df_merged[col] - df_merged[col].shift(63)
                            # Acceleration (change in trend)
                            df_merged[f'{col}_acceleration'] = df_merged[f'{col}_trend_1m'] - df_merged[f'{col}_trend_1m'].shift(21)
                        except Exception as e:
                            logger.warning(f"Error calculating trends for {col}: {e}")
                
                logger.info(f"Successfully merged macro features. Final shape: {df_merged.shape}")
                return df_merged
                
            else:
                logger.warning("No macro data available, skipping macro features")
                # Add placeholder columns for consistency
                macro_placeholders = [
                    'GDP_trend_1m', 'GDP_trend_3m', 'GDP_acceleration',
                    'UNEMPLOYMENT_trend_1m', 'UNEMPLOYMENT_trend_3m', 'UNEMPLOYMENT_acceleration', 
                    'INFLATION_trend_1m', 'INFLATION_trend_3m', 'INFLATION_acceleration',
                    'INTEREST_RATES_trend_1m', 'INTEREST_RATES_trend_3m', 'INTEREST_RATES_acceleration'
                ]
                for col in macro_placeholders:
                    df[col] = 0.0
                return df
                
        except Exception as e:
            logger.error(f"Error adding macro economic features: {e}")
            return df
    
    def add_sector_performance_features(self, df: pd.DataFrame, ticker: str, mongo_client=None) -> pd.DataFrame:
        """
        Add sector performance features to understand relative stock performance.
        Sector rotation and relative performance are key predictive factors.
        """
        try:
            logger.info(f"Adding sector performance features for {ticker}")
            
            # Get date range from dataframe
            if 'date' in df.columns:
                start_date = df['date'].min()
                end_date = df['date'].max()
            else:
                start_date = df.index.min()
                end_date = df.index.max()
            
            # Fetch sector data using the fetcher
            sector_df = self.sector_fetcher.fetch_all(start_date, end_date, mongo_client)
            
            if sector_df is not None and not sector_df.empty and len(sector_df.columns) > 1:
                logger.info(f"Successfully fetched sector data with {len(sector_df.columns)} ETFs")
                
                # Determine which sector this stock belongs to (simplified mapping)
                sector_mapping = {
                    # Technology
                    'AAPL': 'XLK', 'MSFT': 'XLK', 'NVDA': 'XLK', 'GOOGL': 'XLK', 'META': 'XLK',
                    'ORCL': 'XLK', 'CRM': 'XLK', 'INTC': 'XLK', 'AMD': 'XLK', 'QCOM': 'XLK',
                    # Healthcare  
                    'JNJ': 'XLV', 'UNH': 'XLV', 'ABBV': 'XLV', 'PFE': 'XLV', 'MRK': 'XLV',
                    'ABT': 'XLV', 'TMO': 'XLV', 'LLY': 'XLV', 'MDT': 'XLV', 'BMY': 'XLV',
                    # Financials
                    'JPM': 'XLF', 'BAC': 'XLF', 'WFC': 'XLF', 'GS': 'XLF', 'MS': 'XLF',
                    'BLK': 'XLF', 'SCHW': 'XLF', 'AXP': 'XLF', 'USB': 'XLF', 'COF': 'XLF',
                    # Consumer Discretionary
                    'AMZN': 'XLY', 'HD': 'XLY', 'MCD': 'XLY', 'SBUX': 'XLY', 'NKE': 'XLY',
                    'LOW': 'XLY', 'TGT': 'XLY', 'DIS': 'XLY',
                    # Consumer Staples
                    'WMT': 'XLP', 'PG': 'XLP', 'KO': 'XLP', 'PEP': 'XLP', 'COST': 'XLP',
                    # Energy
                    'XOM': 'XLE', 'CVX': 'XLE', 'COP': 'XLE',
                    # Industrials  
                    'BA': 'XLI', 'CAT': 'XLI', 'GE': 'XLI', 'HON': 'XLI', 'MMM': 'XLI',
                    'UPS': 'XLI', 'FDX': 'XLI', 'DE': 'XLI', 'RTX': 'XLI', 'LMT': 'XLI'
                }
                
                primary_sector = sector_mapping.get(ticker, 'XLK')  # Default to tech
                
                # Ensure date alignment
                if 'date' in df.columns:
                    df_for_merge = df.set_index('date')
                    sector_df_for_merge = sector_df.set_index('date')
                else:
                    df_for_merge = df.copy()
                    sector_df_for_merge = sector_df.set_index('date')
                
                # Merge sector data
                df_merged = pd.merge(
                    df_for_merge,
                    sector_df_for_merge,
                    left_index=True,
                    right_index=True, 
                    how='left'
                )
                
                # Restore date column if it existed
                if 'date' in df.columns:
                    df_merged = df_merged.reset_index()
                
                # Forward fill sector data
                sector_columns = [col for col in sector_df.columns if col != 'date']
                df_merged[sector_columns] = df_merged[sector_columns].fillna(method='ffill')
                
                # Calculate relative performance features
                if 'Close' in df_merged.columns:
                    # Stock vs primary sector performance
                    primary_sector_close = f'{primary_sector}_close'
                    if primary_sector_close in df_merged.columns:
                        # Relative strength vs sector
                        df_merged['sector_relative_strength'] = (
                            df_merged['Close'].pct_change(20) - 
                            df_merged[primary_sector_close].pct_change(20)
                        )
                        
                        # Current relative position
                        df_merged['sector_relative_position'] = (
                            df_merged['Close'] / df_merged['Close'].rolling(252).mean() -
                            df_merged[primary_sector_close] / df_merged[primary_sector_close].rolling(252).mean()
                        )
                    
                    # Calculate sector rotation signals
                    sector_etfs = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU']
                    sector_returns = []
                    
                    for etf in sector_etfs:
                        etf_close = f'{etf}_close'
                        if etf_close in df_merged.columns:
                            df_merged[f'{etf}_return_5d'] = df_merged[etf_close].pct_change(5)
                            df_merged[f'{etf}_return_20d'] = df_merged[etf_close].pct_change(20)
                            sector_returns.append(f'{etf}_return_5d')
                    
                    # Sector momentum rank (where does primary sector rank?)
                    if sector_returns:
                        df_merged['sector_momentum_rank'] = 0.0
                        primary_return_col = f'{primary_sector}_return_5d'
                        if primary_return_col in df_merged.columns:
                            for idx, row in df_merged.iterrows():
                                try:
                                    sector_perf = [row[col] for col in sector_returns if pd.notna(row[col])]
                                    if len(sector_perf) > 0 and pd.notna(row[primary_return_col]):
                                        rank = sum(1 for x in sector_perf if x < row[primary_return_col])
                                        df_merged.at[idx, 'sector_momentum_rank'] = rank / len(sector_perf)
                                except:
                                    df_merged.at[idx, 'sector_momentum_rank'] = 0.5  # Neutral rank
                
                logger.info(f"Successfully merged sector features. Final shape: {df_merged.shape}")
                return df_merged
                
            else:
                logger.warning("No sector data available, skipping sector features")
                # Add placeholder columns for consistency
                sector_placeholders = [
                    'sector_relative_strength', 'sector_relative_position', 'sector_momentum_rank'
                ]
                for col in sector_placeholders:
                    df[col] = 0.0
                return df
                
        except Exception as e:
            logger.error(f"Error adding sector performance features: {e}")
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
            
            # SEC sentiment features (already processed in sentiment pipeline)
            'sec_sentiment': sentiment_dict.get('sec_sentiment', 0.0),
            'sec_volume': sentiment_dict.get('sec_volume', 0),
            'sec_confidence': sentiment_dict.get('sec_confidence', 0.0),
            
            # Short interest sentiment (processed sentiment, not raw data)
            'short_interest_sentiment': sentiment_dict.get('short_interest_sentiment', 0.0),
            'short_interest_confidence': sentiment_dict.get('short_interest_confidence', 0.0),
            
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

    def add_finnhub_financial_features(self, df: pd.DataFrame, ticker: str, mongo_client=None) -> pd.DataFrame:
        """
        Add comprehensive Finnhub financial features with proper MongoDB storage.
        All fetched data is stored to avoid API waste.
        """
        if not mongo_client:
            logger.warning("No MongoDB client provided for Finnhub data")
            return df
        
        try:
            from ml_backend.data.sentiment import get_stored_data_from_mongodb
            
            logger.info(f"Adding Finnhub financial features for {ticker}")
            
            # Get all Finnhub data with caching
            basic_financials = get_stored_data_from_mongodb(mongo_client, ticker, 'basic_financials', 'finnhub')
            company_peers = get_stored_data_from_mongodb(mongo_client, ticker, 'company_peers', 'finnhub')
            insider_sentiment = get_stored_data_from_mongodb(mongo_client, ticker, 'insider_sentiment', 'finnhub')
            recommendation_trends = get_stored_data_from_mongodb(mongo_client, ticker, 'recommendation_trends', 'finnhub')
            
            # If no cached data, fetch fresh (but store it!)
            if not basic_financials or not company_peers or not insider_sentiment or not recommendation_trends:
                logger.info(f"Fetching fresh Finnhub data for {ticker}")
                
                # Import the sentiment analyzer to fetch data
                from ml_backend.data.sentiment import SentimentAnalyzer
                
                # Create temporary analyzer for data fetching
                temp_analyzer = SentimentAnalyzer(mongo_client)
                
                # Fetch data (this will automatically store in MongoDB)
                async def fetch_all_finnhub_data():
                    tasks = []
                    if not basic_financials:
                        tasks.append(temp_analyzer.get_finnhub_basic_financials(ticker))
                    if not company_peers:
                        tasks.append(temp_analyzer.get_finnhub_company_peers(ticker))
                    if not insider_sentiment:
                        tasks.append(temp_analyzer.get_finnhub_insider_sentiment_direct(ticker))
                    if not recommendation_trends:
                        tasks.append(temp_analyzer.analyze_finnhub_recommendation_trends(ticker))
                    
                    if tasks:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        return results
                    return []
                
                # Run the async fetch
                import asyncio
                try:
                    # Check if we're in an event loop
                    asyncio.get_running_loop()
                    # If we are, create a new thread to run the async code
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, fetch_all_finnhub_data())
                        fetch_results = future.result()
                except RuntimeError:
                    # No event loop running, safe to use asyncio.run
                    fetch_results = asyncio.run(fetch_all_finnhub_data())
                
                # Retrieve the stored data
                basic_financials = get_stored_data_from_mongodb(mongo_client, ticker, 'basic_financials', 'finnhub') or {}
                company_peers = get_stored_data_from_mongodb(mongo_client, ticker, 'company_peers', 'finnhub') or {}
                insider_sentiment = get_stored_data_from_mongodb(mongo_client, ticker, 'insider_sentiment', 'finnhub') or {}
                recommendation_trends = get_stored_data_from_mongodb(mongo_client, ticker, 'recommendation_trends', 'finnhub') or {}
            
            # Process basic financials
            if basic_financials and 'metric' in basic_financials:
                metrics = basic_financials.get('metric', {})
                series = basic_financials.get('series', {})
                
                # Extract key financial metrics for feature engineering
                financial_features = {
                    # Valuation metrics
                    'pe_ratio': metrics.get('peBasicExclExtraTTM'),
                    'pe_forward': metrics.get('peInclExtraTTM'),
                    'price_to_book': metrics.get('pbAnnual'),
                    'price_to_sales': metrics.get('psAnnual'),
                    'ev_to_ebitda': metrics.get('evToEbitdaTTM'),
                    'market_cap': metrics.get('marketCapitalization'),
                    
                    # Profitability metrics
                    'roe': metrics.get('roeTTM'),
                    'roa': metrics.get('roaTTM'),
                    'gross_margin': metrics.get('grossMarginTTM'),
                    'operating_margin': metrics.get('operatingMarginTTM'),
                    'net_margin': metrics.get('netProfitMarginTTM'),
                    
                    # Financial health
                    'current_ratio': metrics.get('currentRatioAnnual'),
                    'debt_to_equity': metrics.get('totalDebtToEquityAnnual'),
                    'quick_ratio': metrics.get('quickRatioAnnual'),
                    'cash_ratio': metrics.get('cashRatioAnnual'),
                    
                    # Growth metrics
                    'revenue_growth_3m': metrics.get('revenueGrowthTTMYoy'),
                    'eps_growth_3m': metrics.get('epsGrowthTTMYoy'),
                    'revenue_per_share': metrics.get('revenuePerShareTTM'),
                    'book_value_per_share': metrics.get('bookValuePerShareAnnual'),
                    
                    # Trading metrics
                    'beta': metrics.get('beta'),
                    '52w_high': metrics.get('52WeekHigh'),
                    '52w_low': metrics.get('52WeekLow'),
                    '52w_return': metrics.get('52WeekPriceReturnDaily'),
                    'avg_volume_10d': metrics.get('10DayAverageTradingVolume'),
                    'avg_volume_3m': metrics.get('3MonthAverageTradingVolume'),
                    
                    # Dividend metrics
                    'dividend_yield': metrics.get('dividendYieldIndicatedAnnual'),
                    'dividend_per_share': metrics.get('dividendPerShareAnnual'),
                    'payout_ratio': metrics.get('payoutRatioAnnual'),
                    
                    # Additional metrics
                    'shares_outstanding': metrics.get('sharesOutstanding'),
                    'float_shares': metrics.get('freeCashFlowTTM'),  # Often correlated with float
                    'insider_ownership': metrics.get('insiderOwnership'),
                    'institutional_ownership': metrics.get('institutionOwnership')
                }
                
                # Extract time series data for trend analysis
                annual_series = series.get('annual', {})
                quarterly_series = series.get('quarterly', {})
                
                # Calculate recent trends if data available
                trends = {}
                for metric_name, metric_data in annual_series.items():
                    if metric_data and len(metric_data) >= 2:
                        recent = metric_data[0]['v'] if metric_data[0]['v'] is not None else 0
                        previous = metric_data[1]['v'] if metric_data[1]['v'] is not None else 0
                        if previous != 0:
                            trends[f"{metric_name}_yoy_change"] = (recent - previous) / previous
                
                financial_features['trends'] = trends
                financial_features['data_timestamp'] = datetime.utcnow().isoformat()
                financial_features['metrics_count'] = len([v for v in financial_features.values() if v is not None])
                
                # Add all financial features to dataframe
                for feature_name, feature_value in financial_features.items():
                    if feature_name not in ['trends', 'data_timestamp', 'metrics_count'] and feature_value is not None:
                        df[feature_name] = float(feature_value)
                    elif feature_name == 'trends':
                        # Add trend features
                        for trend_name, trend_value in feature_value.items():
                            if trend_value is not None:
                                df[trend_name] = float(trend_value)
                
                logger.info(f"Finnhub Basic Financials for {ticker}: {financial_features['metrics_count']} metrics retrieved")
                
                # Legacy column mapping for compatibility
                df['pe_ratio'] = df.get('pe_ratio', 0)
                df['pb_ratio'] = df.get('price_to_book', 0)
                df['ps_ratio'] = df.get('price_to_sales', 0)
                df['ev_ebitda'] = df.get('ev_to_ebitda', 0)
                df['52_week_high'] = df.get('52w_high', 0)
                df['52_week_low'] = df.get('52w_low', 0)
                df['revenue_growth'] = df.get('revenue_growth_3m', 0)
                df['eps_growth'] = df.get('eps_growth_3m', 0)
                df['debt_equity'] = df.get('debt_to_equity', 0)
            
            # Process company peers
            if company_peers and isinstance(company_peers, list):
                # Remove self from peers list
                peer_list = [p for p in company_peers if p.upper() != ticker.upper()]
                
                peer_features = {
                    'peer_count': len(peer_list),
                    'peer_list': peer_list[:10],  # Limit to top 10 peers
                    'has_peers': len(peer_list) > 0,
                    'peer_sector_size': len(peer_list)  # Indicator of sector competition
                }
                
                # Add peer features to dataframe using concat for better performance
                peer_feature_df = pd.DataFrame({
                    'peer_count': [peer_features['peer_count']] * len(df),
                    'sector_competition': [min(len(peer_list) / 10, 1.0)] * len(df),
                    'has_peers': [1 if peer_features['has_peers'] else 0] * len(df),
                    'peer_sector_size': [peer_features['peer_sector_size']] * len(df)
                }, index=df.index)
                df = pd.concat([df, peer_feature_df], axis=1)
                
                logger.info(f"Added company peers data: {len(peer_list)} peers")
            
            # Process insider sentiment
            if insider_sentiment and 'data' in insider_sentiment:
                # Fetch insider sentiment using Finnhub's pre-calculated MSPR (Monthly Share Purchase Ratio).
                # This is more reliable than parsing individual transactions.
                sentiment_data = insider_sentiment['data']
                if sentiment_data:
                    # Calculate weighted average MSPR (more recent months weighted higher)
                    total_weighted_mspr = 0
                    total_weights = 0
                    total_volume = 0
                    for i, item in enumerate(sentiment_data):
                        mspr = item.get('mspr', 0)
                        change = abs(item.get('change', 0))
                        
                        # Weight recent months higher (last 3 months = 3x weight, last 6 months = 2x weight)
                        weight = 3 if i < 3 else (2 if i < 6 else 1)
                        
                        total_weighted_mspr += mspr * weight
                        total_weights += weight
                        total_volume += change
                    
                    if total_weights == 0:
                        insider_values = {
                            'insider_mspr': 0,
                            'insider_sentiment_normalized': 0,
                            'insider_mspr_trend': 0,
                            'insider_confidence': 0,
                            'insider_sentiment_months': 0,
                            'insider_volume': 0
                        }
                    else:
                        # Calculate final metrics
                        weighted_avg_mspr = total_weighted_mspr / total_weights
                        
                        # Normalize MSPR to [-1, 1] range (MSPR typically ranges from -100 to 100)
                        normalized_sentiment = max(-1.0, min(1.0, weighted_avg_mspr / 100))
                        
                        # Calculate trend (comparing recent vs older periods)
                        recent_mspr = sum(item.get('mspr', 0) for item in sentiment_data[:3]) / min(3, len(sentiment_data))
                        older_mspr = sum(item.get('mspr', 0) for item in sentiment_data[3:6]) / max(1, len(sentiment_data[3:6]))
                        trend = recent_mspr - older_mspr if len(sentiment_data) > 3 else 0
                        
                        # Confidence based on data points and consistency
                        data_confidence = min(len(sentiment_data) / 6, 1.0)  # 6+ months = full confidence
                        mspr_consistency = 1 - (abs(trend) / 50)  # Penalize high volatility in MSPR
                        final_confidence = (data_confidence + max(0, mspr_consistency)) / 2
                        
                        # Prepare insider values for batch addition
                        insider_values = {
                            'insider_mspr': weighted_avg_mspr,
                            'insider_sentiment_normalized': normalized_sentiment,
                            'insider_mspr_trend': trend,
                            'insider_confidence': final_confidence,
                            'insider_sentiment_months': len(sentiment_data),
                            'insider_volume': int(total_volume)
                        }
                        
                        logger.info(f"Enhanced Finnhub Insider Sentiment for {ticker}:")
                        logger.info(f"  Weighted MSPR: {weighted_avg_mspr:.2f}, Normalized: {normalized_sentiment:.3f}")
                        logger.info(f"  Trend: {trend:.2f}, Confidence: {final_confidence:.2f}")
                    
                    # Add insider features using concat for better performance
                    insider_feature_df = pd.DataFrame({
                        col: [val] * len(df) for col, val in insider_values.items()
                    }, index=df.index)
                    df = pd.concat([df, insider_feature_df], axis=1)
            
            # Process recommendation trends
            if recommendation_trends and isinstance(recommendation_trends, dict):
                analyst_features = {
                    'analyst_sentiment': recommendation_trends.get('sentiment', 0),
                    'analyst_count': recommendation_trends.get('total_analysts', 0),
                    'analyst_confidence': recommendation_trends.get('confidence', 0),
                    'analyst_consensus': 1 if recommendation_trends.get('consensus') == 'BUY' else (-1 if recommendation_trends.get('consensus') == 'SELL' else 0)
                }
                analyst_feature_df = pd.DataFrame({
                    col: [val] * len(df) for col, val in analyst_features.items()
                }, index=df.index)
                df = pd.concat([df, analyst_feature_df], axis=1)
                logger.info(f"Added analyst recommendations: {recommendation_trends.get('total_analysts', 0)} analysts, sentiment {recommendation_trends.get('sentiment', 0):.3f}")
            
            logger.info(f"✅ Successfully added Finnhub financial features for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error adding Finnhub financial features for {ticker}: {e}")
        return df

    def add_alpha_vantage_features(self, df: pd.DataFrame, alpha_vantage_dict: Dict = None) -> pd.DataFrame:
        """Add Alpha Vantage features to the dataframe."""
        if alpha_vantage_dict is None:
            return df
            
        try:
            # Process earnings data
            if 'alpha_earnings' in alpha_vantage_dict and alpha_vantage_dict['alpha_earnings']:
                earnings_data = alpha_vantage_dict['alpha_earnings']
                if 'quarterlyEarnings' in earnings_data:
                    quarterly = earnings_data['quarterlyEarnings']
                    if quarterly:
                        # Get the most recent quarter
                        recent_earnings = quarterly[0]
                        earnings_features = {
                            'eps_reported': float(recent_earnings.get('reportedEPS', 0)),
                            'eps_estimate': float(recent_earnings.get('estimatedEPS', 0)),
                            'eps_surprise': float(recent_earnings.get('surprise', 0)),
                            'eps_surprise_pct': float(recent_earnings.get('surprisePercentage', 0))
                        }
                        earnings_df = pd.DataFrame({
                            col: [val] * len(df) for col, val in earnings_features.items()
                        }, index=df.index)
                        df = pd.concat([df, earnings_df], axis=1)
                        
            # Process dividends data
            if 'alpha_dividends' in alpha_vantage_dict and alpha_vantage_dict['alpha_dividends']:
                dividends = alpha_vantage_dict['alpha_dividends']
                if dividends:
                    recent_dividend = list(dividends.values())[0] if dividends else {}
                    dividend_features = {'dividend_amount': float(recent_dividend.get('amount', 0))}
                    dividend_df = pd.DataFrame({
                        col: [val] * len(df) for col, val in dividend_features.items()
                    }, index=df.index)
                    df = pd.concat([df, dividend_df], axis=1)
                    
            # Process insider transactions
            if 'alpha_insider_transactions' in alpha_vantage_dict and alpha_vantage_dict['alpha_insider_transactions']:
                pass  # Fixed indentation error
                insider_data = alpha_vantage_dict['alpha_insider_transactions']
                if 'data' in insider_data:
                    transactions = insider_data['data']
                    if transactions:
                        # Analyze recent transactions (last 10)
                        recent_transactions = transactions[:10]
                        buy_volume = sum(float(t.get('transactionShares', 0)) for t in recent_transactions 
                                       if t.get('transactionType', '').lower() in ['p-purchase', 'purchase'])
                        sell_volume = sum(float(t.get('transactionShares', 0)) for t in recent_transactions 
                                        if t.get('transactionType', '').lower() in ['s-sale', 'sale'])
                        
                        insider_transaction_features = {
                            'insider_buy_volume': buy_volume,
                            'insider_sell_volume': sell_volume,
                            'insider_buy_sell_ratio': buy_volume / (sell_volume + 1)  # +1 to avoid division by zero
                        }
                        insider_transaction_df = pd.DataFrame({
                            col: [val] * len(df) for col, val in insider_transaction_features.items()
                        }, index=df.index)
                        df = pd.concat([df, insider_transaction_df], axis=1)
                        
            # Process Alpha Vantage sentiment if available
            if 'alphavantage_sentiment' in alpha_vantage_dict and alpha_vantage_dict['alphavantage_sentiment']:
                sentiment_data = alpha_vantage_dict['alphavantage_sentiment']
                if 'feed' in sentiment_data:
                    feed = sentiment_data['feed']
                    if feed:
                        # Calculate aggregate sentiment
                        total_sentiment = 0
                        total_relevance = 0
                        
                        for article in feed:
                            relevance = float(article.get('relevance_score', 0))
                            sentiment = float(article.get('overall_sentiment_score', 0))
                            total_sentiment += sentiment * relevance
                            total_relevance += relevance
                            
                        sentiment_features = {
                            'alphavantage_sentiment_score': total_sentiment / total_relevance if total_relevance > 0 else 0,
                            'alphavantage_news_volume': len(feed) if total_relevance > 0 else 0
                        }
                        sentiment_df = pd.DataFrame({
                            col: [val] * len(df) for col, val in sentiment_features.items()
                        }, index=df.index)
                        df = pd.concat([df, sentiment_df], axis=1)
                            
        except Exception as e:
            logger.error(f"Error processing Alpha Vantage features: {e}")
            
        return df

    def add_historical_options_features(self, df: pd.DataFrame, ticker: str, mongo_client=None) -> pd.DataFrame:
        """
        Add comprehensive historical options features using Alpha Vantage HISTORICAL_OPTIONS API.
        Extracts market sentiment, volatility expectations, and options flow insights.
        """
        try:
            if not mongo_client:
                logger.warning("No MongoDB client provided for options data")
                return df
            
            logger.info(f"Adding historical options features for {ticker}")
            
            # Get options data from MongoDB cache or fetch from API
            options_data = self._fetch_historical_options_data(ticker, mongo_client)
            
            if not options_data:
                logger.warning(f"No options data available for {ticker}")
                return df
            
            # Process options data into features
            options_features = self._process_options_chain_features(options_data, ticker)
            
            # Add options features to dataframe using concat for better performance
            if options_features:
                options_feature_df = pd.DataFrame({
                    col: [val] * len(df) for col, val in options_features.items()
                }, index=df.index)
                df = pd.concat([df, options_feature_df], axis=1)
            
            logger.info(f"Added {len(options_features)} options features for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error adding historical options features for {ticker}: {e}")
            return df
    
    def _fetch_historical_options_data(self, ticker: str, mongo_client) -> Dict:
        """
        Fetch Alpha Vantage options data from the correct MongoDB collection.
        Based on your MongoDB structure: collection 'alpha_vantage_data' with endpoint 'Historical Options'
        """
        try:
            from datetime import datetime, timedelta
            
            # Query the correct collection using the actual MongoDB structure you provided
            # Collection: alpha_vantage_data
            # Document structure: {ticker: "AAPL", endpoint: "Historical Options", data: [...]}
            options_doc = mongo_client.db['alpha_vantage_data'].find_one({
                'ticker': ticker,
                'endpoint': 'Historical Options'
            }, sort=[('timestamp', -1)])  # Get most recent
            
            if options_doc and options_doc.get('data'):
                logger.info(f"Retrieved Alpha Vantage options data for {ticker}")
                return {'data': options_doc['data']}
            
            logger.warning(f"No Alpha Vantage options data found for {ticker}")
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage options data for {ticker}: {e}")
            return {}
    
    def _process_options_chain_features(self, options_data: Dict, ticker: str) -> Dict[str, float]:
        """
        Process options chain data into meaningful features for ML model.
        """
        features = {}
        
        try:
            if 'data' not in options_data:
                logger.warning(f"No 'data' field in options data for {ticker}")
                return features
            
            chain_data = options_data['data']
            
            # Initialize counters and accumulators
            total_call_volume = 0
            total_put_volume = 0
            total_call_oi = 0  # Open Interest
            total_put_oi = 0
            
            call_iv_sum = 0
            put_iv_sum = 0
            call_count = 0
            put_count = 0
            
            # Greeks accumulation
            total_delta = 0
            total_gamma = 0
            total_theta = 0
            total_vega = 0
            
            # Strike price analysis
            strikes = []
            volumes = []
            
            # Moneyness analysis (ITM/OTM distribution)
            itm_call_volume = 0
            otm_call_volume = 0
            itm_put_volume = 0
            otm_put_volume = 0
            
            # Get current stock price for moneyness calculation
            current_price = self._get_current_stock_price(ticker, options_data)
            
            # Process each contract in the options chain
            for contract in chain_data:
                try:
                    contract_type = contract.get('type', '').lower()
                    volume = float(contract.get('volume', 0))
                    open_interest = float(contract.get('open_interest', 0))
                    implied_vol = float(contract.get('implied_volatility', 0))
                    strike = float(contract.get('strike', 0))
                    
                    # Greeks
                    delta = float(contract.get('delta', 0))
                    gamma = float(contract.get('gamma', 0))
                    theta = float(contract.get('theta', 0))
                    vega = float(contract.get('vega', 0))
                    
                    strikes.append(strike)
                    volumes.append(volume)
                    
                    # Accumulate Greeks
                    total_delta += delta * volume  # Volume-weighted
                    total_gamma += gamma * volume
                    total_theta += theta * volume
                    total_vega += vega * volume
                    
                    if contract_type == 'call':
                        total_call_volume += volume
                        total_call_oi += open_interest
                        call_iv_sum += implied_vol
                        call_count += 1
                        
                        # Moneyness for calls
                        if current_price and strike < current_price:
                            itm_call_volume += volume
                        else:
                            otm_call_volume += volume
                            
                    elif contract_type == 'put':
                        total_put_volume += volume
                        total_put_oi += open_interest
                        put_iv_sum += implied_vol
                        put_count += 1
                        
                        # Moneyness for puts
                        if current_price and strike > current_price:
                            itm_put_volume += volume
                        else:
                            otm_put_volume += volume
                
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing contract data: {e}")
                    continue
            
            # Calculate derived features
            total_volume = total_call_volume + total_put_volume
            total_oi = total_call_oi + total_put_oi
            
            # 1. Put/Call Ratios (Key sentiment indicators)
            features['options_put_call_volume_ratio'] = (
                total_put_volume / (total_call_volume + 1) if total_call_volume > 0 else 0
            )
            features['options_put_call_oi_ratio'] = (
                total_put_oi / (total_call_oi + 1) if total_call_oi > 0 else 0
            )
            
            # 2. Implied Volatility Analysis
            features['options_avg_call_iv'] = call_iv_sum / call_count if call_count > 0 else 0
            features['options_avg_put_iv'] = put_iv_sum / put_count if put_count > 0 else 0
            features['options_iv_skew'] = features['options_avg_put_iv'] - features['options_avg_call_iv']
            
            # 3. Volume and Open Interest Metrics
            features['options_total_volume'] = total_volume
            features['options_total_open_interest'] = total_oi
            features['options_volume_oi_ratio'] = total_volume / (total_oi + 1) if total_oi > 0 else 0
            
            # 4. Greeks Analysis (Risk metrics)
            if total_volume > 0:
                features['options_weighted_delta'] = total_delta / total_volume
                features['options_weighted_gamma'] = total_gamma / total_volume
                features['options_weighted_theta'] = total_theta / total_volume
                features['options_weighted_vega'] = total_vega / total_volume
            else:
                features['options_weighted_delta'] = 0
                features['options_weighted_gamma'] = 0
                features['options_weighted_theta'] = 0
                features['options_weighted_vega'] = 0
            
            # 5. Moneyness Distribution (Positioning analysis)
            features['options_itm_call_volume_pct'] = (
                itm_call_volume / (total_call_volume + 1) if total_call_volume > 0 else 0
            )
            features['options_otm_put_volume_pct'] = (
                otm_put_volume / (total_put_volume + 1) if total_put_volume > 0 else 0
            )
            
            # 6. Strike Price Analysis
            if strikes and volumes and current_price:
                # Volume-weighted average strike
                total_volume_weighted_strike = sum(s * v for s, v in zip(strikes, volumes))
                features['options_volume_weighted_strike'] = (
                    total_volume_weighted_strike / total_volume if total_volume > 0 else current_price
                )
                
                # Distance from current price (market expectation)
                features['options_strike_price_bias'] = (
                    (features['options_volume_weighted_strike'] - current_price) / current_price 
                    if current_price > 0 else 0
                )
                
                # Strike dispersion (uncertainty measure)
                if len(strikes) > 1:
                    features['options_strike_dispersion'] = np.std(strikes) / current_price
                else:
                    features['options_strike_dispersion'] = 0
            
            # 7. Market Regime Indicators
            # High put/call ratio often indicates fear
            features['options_fear_indicator'] = min(features['options_put_call_volume_ratio'] / 0.7, 2.0)
            
            # High IV often indicates uncertainty/volatility expectations
            avg_iv = (features['options_avg_call_iv'] + features['options_avg_put_iv']) / 2
            features['options_volatility_expectation'] = min(avg_iv / 0.3, 3.0)  # Normalized to reasonable range
            
            # Volume surge indicator
            if total_volume > 0:
                # Compare to typical volume (rough heuristic)
                features['options_volume_surge'] = min(total_volume / 10000, 5.0)  # Normalized
            else:
                features['options_volume_surge'] = 0
            
            logger.info(f"Processed {len(chain_data)} options contracts into {len(features)} features")
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing options chain features: {e}")
            return features
    
    def _get_current_stock_price(self, ticker: str, options_data: Dict) -> float:
        """
        Extract current stock price from options data or use a fallback method.
        """
        try:
            # Try to get from options data metadata
            if 'meta' in options_data:
                meta = options_data['meta']
                if 'underlying_price' in meta:
                    return float(meta['underlying_price'])
                if 'last_price' in meta:
                    return float(meta['last_price'])
            
            # Try to get from individual contracts (use ATM strike as approximation)
            if 'data' in options_data:
                strikes = [float(contract.get('strike', 0)) for contract in options_data['data']]
                if strikes:
                    # Estimate current price as median strike (rough approximation)
                    return np.median(strikes)
            
            # Fallback: return 0 (will disable moneyness calculations)
            logger.warning(f"Could not determine current price for {ticker} from options data")
            return 0.0
            
        except Exception as e:
            logger.error(f"Error extracting current price for {ticker}: {e}")
            return 0.0



    def create_prediction_features(self, df: pd.DataFrame, ticker: str, window: str, mongo_client=None) -> np.ndarray:
        """Create features for prediction using stored pipeline from training to ensure consistency."""
        try:
            # Check if we have a stored feature pipeline for this ticker-window
            pipeline_key = f"{ticker}-{window}"
            
            # Try to load stored pipeline (feature columns, normalizer, etc.)
            pipeline_path = f"models/{ticker}/feature_pipeline_{window}.joblib"
            if os.path.exists(pipeline_path):
                try:
                    import joblib
                    stored_pipeline = joblib.load(pipeline_path)
                    logger.info(f"Loaded stored pipeline for {ticker}-{window}")
                    
                    # Use stored feature engineering parameters
                    if 'feature_columns' in stored_pipeline:
                        self.feature_columns = stored_pipeline['feature_columns']
                    if 'domain_normalizers' in stored_pipeline:
                        self.domain_normalizers = stored_pipeline['domain_normalizers']
                        
                except Exception as e:
                    logger.warning(f"Could not load stored pipeline for {ticker}-{window}: {e}")
            else:
                logger.warning(f"No stored pipeline found for {ticker}-{window}, using current pipeline")
            
            # Use consistent window size mapping
            window_size_map = {'next_day': 1, '7_day': 7, '30_day': 30}
            window_size = window_size_map.get(window, 1)
            
            # Prepare features using the same pipeline as training
            features, _ = self.prepare_features(
                df=df.copy(),
                sentiment_dict=None,  # Will be fetched inside prepare_features
                alpha_vantage_dict=None,  # Will be fetched inside prepare_features
                window_size=window_size,
                mongo_client=mongo_client,
                ticker=ticker
            )
            
            if features is None:
                logger.error(f"Failed to create features for {ticker}-{window}")
                return None
                
            # For prediction, we only need the last sample
            if len(features.shape) > 1 and features.shape[0] > 1:
                features = features[-1:, :]  # Take only last row
                
            logger.info(f"Created prediction features for {ticker}-{window}: shape {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"Error creating prediction features for {ticker}-{window}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def save_feature_pipeline(self, ticker: str, window: str):
        """Save feature pipeline for consistent prediction."""
        try:
            import joblib
            import os
            
            # Create directory if it doesn't exist
            model_dir = f"models/{ticker}"
            os.makedirs(model_dir, exist_ok=True)
            
            # Save feature pipeline components
            pipeline_data = {
                'feature_columns': getattr(self, 'feature_columns', []),
                'domain_normalizers': getattr(self, 'domain_normalizers', {}),
                'window_size': window,
                'ticker': ticker
            }
            
            pipeline_path = f"{model_dir}/feature_pipeline_{window}.joblib"
            joblib.dump(pipeline_data, pipeline_path)
            logger.info(f"Saved feature pipeline for {ticker}-{window}")
            
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
        """
        Prepare features for model training with domain-aware preprocessing.
        Features are organized by type for finance-aware model processing.
        """
        try:
            # Validate inputs
            if df is None or df.empty:
                raise ValueError("DataFrame is empty or None")
            
            logger.info(f"Starting feature preparation: df shape={df.shape}, window_size={window_size}")
            
            # Core feature engineering pipeline
            logger.info("Adding technical indicators...")
            df = self.add_technical_indicators(df)
            
            # Add sentiment features if available
            if sentiment_dict:
                logger.info("Adding sentiment features...")
                df = self.add_sentiment_features(df, sentiment_dict)
            else:
                logger.info("No sentiment data provided, skipping sentiment features")
            
            # Add Alpha Vantage features
            if alpha_vantage_dict:
                logger.info("Adding Alpha Vantage features...")
                df = self.add_alpha_vantage_features(df, alpha_vantage_dict)
            
            # Add Finnhub financial features (the new comprehensive financial data)
            if ticker and mongo_client:
                logger.info("Adding Finnhub financial features...")
                df = self.add_finnhub_financial_features(df, ticker, mongo_client)
            
            # Add historical options features (NEW: Critical for market sentiment and volatility expectations)
            if ticker and mongo_client:
                logger.info("Adding historical options features...")
                df = self.add_historical_options_features(df, ticker, mongo_client)
            
            # Add short interest features (FIXED: Now properly integrated)
            if ticker and mongo_client:
                logger.info("Adding short interest features...")
                df = self.add_short_interest_features(df, ticker, mongo_client)
            
            # Add macro economic features (FIXED: FRED data integration)
            logger.info("Adding macro economic features...")
            df = self.add_macro_economic_features(df, mongo_client or self.mongo_client)
            
            # Add sector performance features (FIXED: Sector ETF integration) 
            if ticker:
                logger.info("Adding sector performance features...")
                df = self.add_sector_performance_features(df, ticker, mongo_client or self.mongo_client)
            
            # NOTE: Economic calendar features are already included in sentiment_dict from sentiment pipeline
            # Skipping redundant economic calendar processing to avoid duplicate web scraping
            logger.info("Economic calendar features already included in sentiment data - skipping redundant processing")
            
            # NOTE: SEC sentiment features are already included in sentiment_dict from sentiment pipeline
            # The sentiment pipeline includes SEC data via analyze_sec_sentiment method
            # Skipping redundant SEC processing to avoid duplicate data fetching and async/await issues
            logger.info("SEC sentiment features already included in sentiment data - skipping redundant processing")
            
            # Add rolling and lagged features
            logger.info("Adding rolling and volatility features...")
            df = self.add_rolling_features(df)
            df = self.add_lagged_and_volatility_features(df)
            
            # Handle outliers if requested
            if handle_outliers:
                logger.info("Handling outliers...")
                df = self.handle_outliers(df)
            
            # Handle missing values
            logger.info("Handling missing values...")
            df = self._handle_nan_values(df)
                
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
            
            # Organize features by type for finance-aware model
            feature_columns = feature_df.columns.tolist()
            organized_features = self._organize_features_by_type(feature_columns)
            
            # Reorder DataFrame columns by feature type (prioritize based on predictive importance)
            ordered_columns = (
                organized_features['price_volume'] +
                organized_features['financial_ratios'] +
                organized_features['sentiment'] +
                organized_features['short_interest'] +
                organized_features['macro_economic'] +
                organized_features['sector_performance'] +
                organized_features['technical']
            )
            
            # Ensure all ordered columns exist in feature_df
            available_ordered_columns = [col for col in ordered_columns if col in feature_df.columns]
            remaining_columns = [col for col in feature_df.columns if col not in available_ordered_columns]
            final_columns = available_ordered_columns + remaining_columns
            
            feature_df = feature_df[final_columns]
            
            # Use finance-aware feature selection instead of generic SHAP
            if enable_shap_selection and ticker:
                logger.info("Applying finance-aware feature selection...")
                selected_features = self.finance_aware_feature_selection(
                    feature_df, feature_df.columns.tolist(), ticker, feature_select_k
                )
                feature_df = feature_df[selected_features]
                logger.info(f"Selected {len(selected_features)} features using finance-aware selection")
            
            # Normalize features using domain-aware normalization
            feature_stats = self._normalize_features(feature_df)
                    
            # Convert to numpy array
            features = feature_df.values
            
            # Ensure we have enough data points
            if len(features) < window_size + 10:
                raise ValueError(f"Not enough data points: {len(features)} < {window_size + 10}")
            
            # Create targets if we have price data
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
                
                # FIXED: Ensure proper alignment for different model types
                features = windowed_features
                
                # Ensure targets align with features
                if targets is not None and len(targets) > n_samples:
                    targets = targets[:n_samples]
                    
                logger.info(f"Windowed features shape: {features.shape}, targets shape: {targets.shape if targets is not None else 'None'}")
            else:
                # For next_day predictions, keep 2D but ensure alignment
                if targets is not None and len(targets) != len(features):
                    min_len = min(len(features), len(targets))
                    features = features[:min_len]
                    targets = targets[:min_len]
                    
                logger.info(f"2D features shape: {features.shape}, targets shape: {targets.shape if targets is not None else 'None'}")
            
            # FIXED: Store feature columns for model consistency
            if hasattr(self, 'feature_columns'):
                self.feature_columns = feature_df.columns.tolist()
            else:
                self.feature_columns = feature_df.columns.tolist()
            
            # Validate final shapes
            if features.shape[0] == 0:
                raise ValueError("No valid features generated")
            
            if targets is not None and targets.shape[0] == 0:
                raise ValueError("No valid targets generated")
            
            if targets is not None and features.shape[0] != targets.shape[0]:
                raise ValueError(f"Feature-target shape mismatch: features {features.shape[0]}, targets {targets.shape[0]}")

            # Save feature pipeline for consistency if requested
            if save_pipeline and ticker and window:
                self.save_feature_pipeline(
                    ticker=ticker,
                    window=window
                )
                
            logger.info(f"Feature preparation completed successfully: features {features.shape}, targets {targets.shape if targets is not None else 'None'}")
            return features, targets
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise

    def _organize_features_by_type(self, feature_columns: List[str]) -> Dict[str, List[str]]:
        """
        Organize features by type for the finance-aware model architecture.
        Returns features grouped by: price_volume, financial_ratios, sentiment, technical, macro_economic, sector_performance, short_interest
        """
        organized = {
            'price_volume': [],
            'financial_ratios': [],
            'sentiment': [],
            'technical': [],
            'macro_economic': [],
            'sector_performance': [],
            'short_interest': []
        }
        
        for col in feature_columns:
            col_lower = col.lower()
            
            # Price and volume features
            if any(keyword in col_lower for keyword in [
                'close', 'open', 'high', 'low', 'volume', 'price', 'avgdailyshare'
            ]):
                organized['price_volume'].append(col)
            
            # Financial ratios and fundamental metrics
            elif any(keyword in col_lower for keyword in [
                'ratio', 'pe', 'pb', 'ps', 'ev_ebitda', 'roe', 'roa', 'margin', 'yield',
                'current_ratio', 'debt_equity', 'quick_ratio', 'earnings', 'revenue',
                'book_value', 'beta', '52week', 'dividend', 'shares_outstanding',
                'market_cap', 'enterprise_value'
            ]):
                organized['financial_ratios'].append(col)
            
            # Sentiment and recommendation features (INCLUDING OPTIONS SENTIMENT)
            elif any(keyword in col_lower for keyword in [
                'sentiment', 'mspr', 'insider', 'recommendation', 'analyst',
                'grade', 'rating', 'target', 'estimate', 'earnings_call',
                # OPTIONS SENTIMENT INDICATORS
                'options_put_call', 'options_fear', 'options_volatility_expectation',
                'options_iv_skew', 'options_strike_price_bias'
            ]):
                organized['sentiment'].append(col)
            
            # Macro Economic Features (FRED indicators, interest rates, GDP, etc.)
            elif any(keyword in col_lower for keyword in [
                'gdp', 'unemployment', 'inflation', 'interest_rates', 'consumer_confidence',
                'industrial_production', 'retail_sales', 'fed', 'fomc', 'cpi', 'ppi',
                'unemployment_rate', 'federal_funds', 'treasury', 'yield_curve',
                'unrate', 'fedfunds', 'cpiaucsl', 'indpro', 'gs10', 'gs2',
                '_trend_1m', '_trend_3m', '_acceleration', '_ma5', '_ma20'
            ]):
                organized['macro_economic'].append(col)
                
            # Sector Performance Features (ETF comparisons, sector rotation)
            elif any(keyword in col_lower for keyword in [
                'xlk', 'xlf', 'xle', 'xlv', 'xly', 'xlp', 'xli', 'xlb', 'xlu', 'xlre',
                'sector_relative', 'sector_momentum', 'sector_rotation',
                '_return_5d', '_return_20d', '_close', '_volume'
            ]):
                organized['sector_performance'].append(col)
                
            # Short Interest Features (bearish sentiment, squeeze potential)
            elif any(keyword in col_lower for keyword in [
                'short_interest', 'days_to_cover', 'short_volume', 'short_squeeze',
                'squeeze_potential', 'short_ratio'
            ]):
                organized['short_interest'].append(col)
            
            # Technical indicators (INCLUDING OPTIONS TECHNICAL METRICS)
            elif any(keyword in col_lower for keyword in [
                'sma', 'ema', 'rsi', 'macd', 'bollinger', 'stoch', 'williams',
                'adx', 'cci', 'momentum', 'volatility', 'support', 'resistance',
                'atr', 'obv', 'vwap',
                # OPTIONS TECHNICAL METRICS  
                'options_total', 'options_volume', 'options_oi', 'options_weighted',
                'options_itm', 'options_otm', 'options_strike_dispersion', 'options_volume_surge'
            ]):
                organized['technical'].append(col)
            
            # Default to technical if unsure
            else:
                organized['technical'].append(col)
        
        # Log the organization
        for group, features in organized.items():
            logger.info(f"{group.title()} features ({len(features)}): {features[:5]}{'...' if len(features) > 5 else ''}")
        
        return organized

    def finance_aware_feature_selection(self, df: pd.DataFrame, feature_names: List[str], 
                                      ticker: str = None, k: int = 30) -> List[str]:
        """
        Finance-domain-aware feature selection that prioritizes features based on:
        1. Financial theory importance
        2. Market regime relevance
        3. Data quality and recency
        4. Cross-correlation with price movements
        """
        try:
            # Define feature importance tiers based on financial theory
            tier_1_critical = [
                # Core valuation metrics
                'pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_ebitda',
                # Profitability
                'roe', 'roa', 'gross_margin', 'operating_margin', 'net_margin',
                # Recent price action
                'close', 'volume', 'volatility',
                # Analyst sentiment
                'recommendation_mean', 'recommendation_trend',
                # Insider activity (recent)
                'mspr', 'insider_sentiment'
            ]
            
            tier_2_important = [
                # Growth metrics
                'earnings_growth', 'revenue_growth', 'book_value_growth',
                # Financial health
                'current_ratio', 'debt_equity', 'quick_ratio',
                # Market sentiment
                'sentiment_score', 'sentiment_momentum',
                # Technical indicators
                'rsi', 'macd', 'sma_20', 'ema_50',
                # Market context
                'beta', '52_week_high', '52_week_low',
                # OPTIONS CRITICAL METRICS (Tier 2 - Very Important)
                'options_put_call_volume_ratio', 'options_put_call_oi_ratio',
                'options_avg_call_iv', 'options_avg_put_iv', 'options_iv_skew',
                'options_fear_indicator', 'options_volatility_expectation',
                # SHORT INTEREST CRITICAL METRICS (Tier 2 - Very Important)
                'short_interest_ratio', 'days_to_cover', 'short_squeeze_potential',
                # MACRO ECONOMIC CRITICAL METRICS (Tier 2 - Very Important)
                'gdp_trend_3m', 'unemployment_trend_1m', 'interest_rates_trend_1m',
                'inflation_trend_1m', 'consumer_confidence', 'fed_policy_direction',
                # SECTOR PERFORMANCE CRITICAL METRICS (Tier 2 - Very Important)
                'sector_relative_strength', 'sector_momentum_rank', 'sector_rotation_signal'
            ]
            
            tier_3_supporting = [
                # Sector comparison
                'peer_comparison', 'sector_strength',
                # Options activity
                'put_call_ratio', 'options_volume',
                # News sentiment
                'news_sentiment', 'social_sentiment',
                # Macro indicators
                'interest_rates', 'gdp_growth', 'inflation',
                # OPTIONS SUPPORTING METRICS (Tier 3 - Supporting)
                'options_total_volume', 'options_total_open_interest',
                'options_weighted_delta', 'options_weighted_gamma', 'options_weighted_theta',
                'options_itm_call_volume_pct', 'options_otm_put_volume_pct',
                'options_strike_price_bias', 'options_volume_surge',
                # SHORT INTEREST SUPPORTING METRICS (Tier 3 - Supporting)
                'short_volume_ratio', 'short_interest_change', 'short_interest_ma5',
                'short_interest_ma20', 'short_interest_trend',
                # MACRO ECONOMIC SUPPORTING METRICS (Tier 3 - Supporting)
                'gdp_trend_1m', 'unemployment_trend_3m', 'inflation_acceleration',
                'industrial_production_trend', 'retail_sales_trend', 'cpi_trend',
                'federal_funds_rate_change', 'yield_curve_slope',
                # SECTOR PERFORMANCE SUPPORTING METRICS (Tier 3 - Supporting)
                'sector_relative_position', 'xlk_return_5d', 'xlf_return_5d',
                'xle_return_5d', 'xlv_return_5d', 'sector_etf_momentum'
            ]
            
            # Score each available feature
            feature_scores = {}
            
            for feature_name in feature_names:
                if feature_name not in df.columns:
                    continue
                    
                base_score = 0
                feature_lower = feature_name.lower()
                
                # Tier-based scoring
                if any(tier1 in feature_lower for tier1 in tier_1_critical):
                    base_score = 100
                elif any(tier2 in feature_lower for tier2 in tier_2_important):
                    base_score = 70
                elif any(tier3 in feature_lower for tier3 in tier_3_supporting):
                    base_score = 40
                else:
                    base_score = 20  # Default for unknown features
                
                # Data quality multipliers
                series = df[feature_name].dropna()
                if len(series) == 0:
                    feature_scores[feature_name] = 0
                    continue
                        
                quality_multiplier = 1.0
                
                # Penalize features with too many missing values
                missing_ratio = 1 - (len(series) / len(df))
                if missing_ratio > 0.5:
                    quality_multiplier *= 0.3  # Severely penalize
                elif missing_ratio > 0.2:
                    quality_multiplier *= 0.7
                
                # Penalize features with no variance
                if series.std() == 0:
                    quality_multiplier *= 0.1
                
                # Bonus for recent data (if timestamp info available)
                if any(time_word in feature_lower for time_word in ['recent', 'latest', 'current']):
                    quality_multiplier *= 1.3
                
                # Bonus for high-frequency updates
                if any(freq_word in feature_lower for freq_word in ['daily', 'intraday', 'realtime']):
                    quality_multiplier *= 1.2
                
                # Interaction bonus for key combinations
                interaction_bonus = 0
                if 'pe' in feature_lower and any('growth' in name.lower() for name in feature_names):
                    interaction_bonus += 10  # PEG ratio components
                if 'mspr' in feature_lower and any('volume' in name.lower() for name in feature_names):
                    interaction_bonus += 10  # Insider + volume
                if 'sentiment' in feature_lower and any('recommendation' in name.lower() for name in feature_names):
                    interaction_bonus += 8   # Sentiment alignment
                
                # Financial ratio validity check
                if any(ratio_word in feature_lower for ratio_word in ['ratio', 'margin', 'roe', 'roa']):
                    # Check for reasonable ranges
                    if feature_lower.startswith('pe') and series.median() > 100:
                        quality_multiplier *= 0.5  # Suspicious P/E ratios
                    if 'margin' in feature_lower and (series.min() < -1 or series.max() > 1):
                        quality_multiplier *= 1.2  # Good range for margins
                
                # Calculate final score
                final_score = base_score * quality_multiplier + interaction_bonus
                feature_scores[feature_name] = final_score
            
            # Sort features by score and select top k
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [name for name, score in sorted_features[:k]]
            
            # Ensure we have critical features even if scored low
            must_have = ['close', 'volume'] if any(f in feature_names for f in ['close', 'volume']) else []
            for must_feature in must_have:
                if must_feature in feature_names and must_feature not in selected_features:
                    selected_features.append(must_feature)
                    # Remove lowest scored feature to maintain k limit
                    if len(selected_features) > k:
                        selected_features = selected_features[:k]
            
            logger.info(f"Finance-aware feature selection: {len(selected_features)}/{len(feature_names)} features selected")
            logger.info(f"Top 5 selected features: {[f'{name} ({feature_scores.get(name, 0):.1f})' for name in selected_features[:5]]}")
            
            return selected_features
            
        except Exception as e:
            logger.error(f"Error in finance-aware feature selection: {e}")
            # Fallback to top k features by name priority
            priority_features = [name for name in feature_names if any(
                key in name.lower() for key in ['close', 'pe', 'roe', 'sentiment', 'volume', 'recommendation']
            )]
            return priority_features[:k] if len(priority_features) >= k else feature_names[:k]

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
            
            interaction_features = {}
            interaction_count = 0
            
            for i in range(len(interaction_candidates)):
                for j in range(i+1, len(interaction_candidates)):
                    if interaction_count >= max_interactions:
                        break
                        
                    col1, col2 = interaction_candidates[i], interaction_candidates[j]
                    if col1 in df.columns and col2 in df.columns:
                        # Calculate interaction term
                        interaction_name = f"{col1}_x_{col2}"
                        interaction_features[interaction_name] = df[col1] * df[col2]
                        interaction_count += 1
                        
                if interaction_count >= max_interactions:
                    break
            
            # Add all interaction features at once using concat
            if interaction_features:
                interaction_df = pd.DataFrame(interaction_features, index=df.index)
                df = pd.concat([df, interaction_df], axis=1)
                logger.info(f"Added {len(interaction_features)} feature interaction terms")
                
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

    def _normalize_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Domain-aware feature normalization that handles different data types appropriately.
        Groups features by type and applies appropriate scaling/transformation.
        """
        try:
            normalized = {}
            
            # Only process numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Define feature groups with appropriate normalization strategies
            feature_groups = {
                'price_features': {
                    'columns': ['Close', 'Open', 'High', 'Low', 'close_price', 'open_price', 'high_price', 'low_price'],
                    'method': 'log_return'  # Log transformation for prices
                },
                'volume_features': {
                    'columns': ['Volume', 'volume', 'avgDailyShareVolume', 'daily_volume'],
                    'method': 'log_normalize'  # Log + Z-score for volume (right-skewed)
                },
                'sentiment_features': {
                    'columns': [col for col in numeric_cols if 'sentiment' in col.lower() or 'mspr' in col.lower()],
                    'method': 'bounded_normalize',  # Min-max for bounded features
                    'bounds': (-1, 1)
                },
                'financial_ratios': {
                    'columns': [col for col in numeric_cols if any(x in col.lower() for x in [
                        'ratio', 'pe', 'pb', 'ps', 'ev_ebitda', 'roe', 'roa', 'margin', 'yield'
                    ])],
                    'method': 'robust_normalize'  # Robust scaling for ratios (outlier-resistant)
                },
                'percentage_features': {
                    'columns': [col for col in numeric_cols if any(x in col.lower() for x in [
                        'pct', 'change', 'growth', 'return', '_52week'
                    ])],
                    'method': 'bounded_normalize',
                    'bounds': (-100, 100)  # Most percentage changes within ±100%
                },
                'count_features': {
                    'columns': [col for col in numeric_cols if any(x in col.lower() for x in [
                        'count', 'days', 'shares', 'analysts', 'recommendations'
                    ])],
                    'method': 'sqrt_normalize'  # Square root for count data
                },
                'beta_features': {
                    'columns': [col for col in numeric_cols if 'beta' in col.lower()],
                    'method': 'bounded_normalize',
                    'bounds': (0, 3)  # Beta typically 0-3
                },
                'technical_indicators': {
                    'columns': [col for col in numeric_cols if any(x in col.lower() for x in [
                        'sma', 'ema', 'rsi', 'macd', 'bollinger', 'stoch', 'williams'
                    ])],
                    'method': 'z_normalize'  # Standard Z-score for technical indicators
                },
                'options_features': {
                    'columns': [col for col in numeric_cols if col.lower().startswith('options_')],
                    'method': 'options_normalize'  # Special handling for options data
                }
            }
            
            # Track which columns have been processed
            processed_cols = set()
            
            # Process each feature group
            for group_name, group_config in feature_groups.items():
                group_cols = [col for col in group_config['columns'] if col in numeric_cols]
                method = group_config['method']
                
                for col in group_cols:
                    if col in processed_cols or col not in df.columns:
                        continue
                        
                    try:
                        series = df[col].dropna()
                        if len(series) == 0:
                            normalized[f'{col}_norm'] = 0.0
                            continue
                            
                        # Apply appropriate normalization method
                        if method == 'log_return':
                            # Log return for prices
                            if series.min() > 0:  # Ensure positive values
                                log_series = np.log(series)
                                if log_series.std() != 0:
                                    norm_val = (log_series.iloc[-1] - log_series.mean()) / log_series.std()
                                else:
                                    norm_val = 0.0
                            else:
                                norm_val = 0.0
                                
                        elif method == 'log_normalize':
                            # Log + Z-score for right-skewed data
                            if series.min() > 0:
                                log_series = np.log(series + 1)  # +1 to handle zeros
                                if log_series.std() != 0:
                                    norm_val = (log_series.iloc[-1] - log_series.mean()) / log_series.std()
                                else:
                                    norm_val = 0.0
                            else:
                                norm_val = 0.0
                                
                        elif method == 'bounded_normalize':
                            # Min-max scaling for bounded features
                            bounds = group_config.get('bounds', (series.min(), series.max()))
                            min_val, max_val = bounds
                            if max_val != min_val:
                                norm_val = 2 * (series.iloc[-1] - min_val) / (max_val - min_val) - 1
                            else:
                                norm_val = 0.0
                            # Clip to [-1, 1]
                            norm_val = np.clip(norm_val, -1, 1)
                            
                        elif method == 'robust_normalize':
                            # Robust scaling using median and IQR
                            median_val = series.median()
                            q75, q25 = np.percentile(series, [75, 25])
                            iqr = q75 - q25
                            if iqr != 0:
                                norm_val = (series.iloc[-1] - median_val) / iqr
                            else:
                                norm_val = 0.0
                            # Clip extreme outliers
                            norm_val = np.clip(norm_val, -5, 5)
                            
                        elif method == 'sqrt_normalize':
                            # Square root + Z-score for count data
                            sqrt_series = np.sqrt(series + 1)  # +1 to handle zeros
                            if sqrt_series.std() != 0:
                                norm_val = (sqrt_series.iloc[-1] - sqrt_series.mean()) / sqrt_series.std()
                            else:
                                norm_val = 0.0
                                
                        elif method == 'options_normalize':
                            # Special normalization for options features
                            col_lower = col.lower()
                            
                            if 'put_call' in col_lower:
                                # Put/call ratios: 0.5-2.0 typical range, log-scale
                                if series.iloc[-1] > 0:
                                    norm_val = np.log(series.iloc[-1] + 0.1) - np.log(1.0)  # Center around 1.0
                                else:
                                    norm_val = 0.0
                            
                            elif 'iv' in col_lower or 'volatility' in col_lower:
                                # Implied volatility: 0.1-1.0 typical range
                                norm_val = (series.iloc[-1] - 0.3) / 0.2  # Center around 30% IV
                                norm_val = np.clip(norm_val, -3, 3)
                            
                            elif 'delta' in col_lower:
                                # Delta: -1 to +1 range for puts/calls
                                norm_val = series.iloc[-1]  # Already bounded
                            
                            elif 'gamma' in col_lower or 'theta' in col_lower or 'vega' in col_lower:
                                # Greeks: use robust scaling due to wide ranges
                                median_val = series.median()
                                q75, q25 = np.percentile(series, [75, 25])
                                iqr = q75 - q25
                                if iqr != 0:
                                    norm_val = (series.iloc[-1] - median_val) / iqr
                                else:
                                    norm_val = 0.0
                                norm_val = np.clip(norm_val, -5, 5)
                            
                            elif 'volume' in col_lower or 'oi' in col_lower:
                                # Options volume/OI: log transformation for right-skewed data
                                if series.iloc[-1] > 0:
                                    norm_val = np.log(series.iloc[-1] + 1) / 10  # Scale down
                                else:
                                    norm_val = 0.0
                            
                            elif 'strike' in col_lower or 'bias' in col_lower:
                                # Strike-related: percentage from current price
                                norm_val = np.clip(series.iloc[-1], -0.5, 0.5)  # ±50% max
                            
                            elif 'fear' in col_lower or 'expectation' in col_lower:
                                # Regime indicators: typically 0-3 range
                                norm_val = (series.iloc[-1] - 1.0) / 1.0  # Center around 1.0
                                norm_val = np.clip(norm_val, -2, 2)
                            
                            else:
                                # Default options normalization: robust scaling
                                if series.std() != 0:
                                    norm_val = (series.iloc[-1] - series.mean()) / series.std()
                                else:
                                    norm_val = 0.0
                                norm_val = np.clip(norm_val, -3, 3)
                        
                        else:  # method == 'z_normalize'
                            # Standard Z-score normalization
                            if series.std() != 0:
                                norm_val = (series.iloc[-1] - series.mean()) / series.std()
                            else:
                                norm_val = 0.0
                        
                        # Store normalized value
                        normalized[f'{col}_norm'] = float(norm_val) if not np.isnan(norm_val) else 0.0
                        processed_cols.add(col)
                        
                        # Add feature group indicator for model awareness
                        normalized[f'{col}_group'] = hash(group_name) % 100  # Group encoding
                        
                    except (TypeError, ValueError, AttributeError) as e:
                        logger.warning(f"Skipping normalization for {col} in group {group_name}: {e}")
                        normalized[f'{col}_norm'] = 0.0
                        continue
            
            # Handle any remaining unprocessed columns with standard normalization
            unprocessed_cols = [col for col in numeric_cols if col not in processed_cols]
            for col in unprocessed_cols:
                if col in df.columns:
                    try:
                        series = df[col].dropna()
                        if len(series) > 0 and series.std() != 0:
                            norm_val = (series.iloc[-1] - series.mean()) / series.std()
                            normalized[f'{col}_norm'] = float(norm_val) if not np.isnan(norm_val) else 0.0
                        else:
                            normalized[f'{col}_norm'] = 0.0
                        # Mark as miscellaneous group
                        normalized[f'{col}_group'] = 99  # Misc group
                    except Exception as e:
                        logger.warning(f"Skipping unprocessed column {col}: {e}")
                        continue
            
            # Add feature interaction terms for key relationships
            interaction_pairs = [
                ('pe_ratio_norm', 'earnings_growth_norm'),  # P/E vs growth
                ('mspr_norm', 'volume_norm'),  # Insider activity vs volume
                ('sentiment_score_norm', 'recommendation_mean_norm'),  # Sentiment vs analyst rec
                ('current_ratio_norm', 'debt_equity_norm'),  # Liquidity vs leverage
                # OPTIONS INTERACTION TERMS (Critical for options-based predictions)
                ('options_put_call_volume_ratio_norm', 'options_volatility_expectation_norm'),  # Fear vs volatility
                ('options_iv_skew_norm', 'sentiment_score_norm'),  # Options sentiment vs general sentiment
                ('options_weighted_delta_norm', 'volume_norm'),  # Options positioning vs stock volume
                ('options_fear_indicator_norm', 'beta_norm'),  # Market fear vs stock sensitivity
                ('options_strike_price_bias_norm', 'pe_ratio_norm')  # Price expectations vs valuation
            ]
            
            for feat1, feat2 in interaction_pairs:
                if feat1 in normalized and feat2 in normalized:
                    normalized[f'{feat1}_{feat2}_interaction'] = normalized[feat1] * normalized[feat2]
            
            logger.info(f"Normalized {len(normalized)} features across {len(feature_groups)} domain-specific groups")
            return normalized
            
        except Exception as e:
            logger.error(f"Error in domain-aware feature normalization: {str(e)}")
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

    def get_feature_documentation(self) -> Dict[str, Dict[str, str]]:
        """
        Comprehensive feature documentation for ML models to understand feature meanings and predictive significance.
        This helps with model interpretability and AI explainability.
        """
        return {
            'price_volume': {
                'description': 'Core stock price and trading volume metrics',
                'predictive_significance': 'Direct indicators of market demand and supply. High volume confirms price movements.',
                'examples': ['Close', 'Open', 'High', 'Low', 'Volume', 'VWAP'],
                'interpretation': 'Higher values generally indicate stronger market interest and price momentum'
            },
            
            'financial_ratios': {
                'description': 'Company valuation and financial health metrics',
                'predictive_significance': 'Fundamental indicators of company value and growth potential. Critical for long-term predictions.',
                'examples': ['PE_ratio', 'PB_ratio', 'ROE', 'debt_to_equity', 'current_ratio', 'profit_margin'],
                'interpretation': 'Lower P/E ratios may indicate undervaluation, higher ROE indicates profitability efficiency'
            },
            
            'sentiment': {
                'description': 'Market sentiment and analyst recommendations',
                'predictive_significance': 'Reflects market psychology and professional opinions. Leading indicator for price movements.',
                'examples': ['analyst_recommendation_mean', 'news_sentiment_score', 'insider_sentiment', 'social_media_sentiment'],
                'interpretation': 'Positive sentiment generally precedes price increases, negative sentiment precedes declines'
            },
            
            'short_interest': {
                'description': 'Bearish sentiment and short squeeze potential indicators',
                'predictive_significance': 'High short interest can indicate bearish sentiment OR potential for short squeeze rally',
                'examples': ['short_interest_ratio', 'days_to_cover', 'short_squeeze_potential', 'short_volume_ratio'],
                'interpretation': 'High short interest (>20%) + low days to cover (<3) + rising price = potential short squeeze'
            },
            
            'macro_economic': {
                'description': 'Economic indicators affecting overall market conditions',
                'predictive_significance': 'Macro trends drive sector rotation and overall market direction. Critical for timing.',
                'examples': ['GDP_trend_3m', 'unemployment_rate', 'inflation_rate', 'interest_rates', 'fed_policy'],
                'interpretation': 'Rising interest rates typically negative for growth stocks, positive for financial stocks'
            },
            
            'sector_performance': {
                'description': 'Relative performance vs sector ETFs and sector rotation signals',
                'predictive_significance': 'Sector rotation drives individual stock performance. Relative strength is key predictor.',
                'examples': ['sector_relative_strength', 'sector_momentum_rank', 'XLK_performance', 'sector_rotation_signal'],
                'interpretation': 'Stocks outperforming their sector tend to continue outperforming in the short term'
            },
            
            'technical': {
                'description': 'Technical analysis indicators and price patterns',
                'predictive_significance': 'Captures market momentum, trend strength, and reversal signals. Critical for timing entry/exit.',
                'examples': ['RSI', 'MACD', 'SMA_20', 'Bollinger_Bands', 'ATR', 'support_resistance'],
                'interpretation': 'RSI > 70 indicates overbought, RSI < 30 indicates oversold. MACD crossovers signal trend changes'
            },
            
            'options_activity': {
                'description': 'Options market activity indicating institutional sentiment and volatility expectations',
                'predictive_significance': 'Options activity often precedes stock movements. High IV indicates expected volatility.',
                'examples': ['put_call_ratio', 'implied_volatility', 'options_volume_surge', 'gamma_exposure'],
                'interpretation': 'High put/call ratio indicates bearish sentiment. Rising IV indicates expected price movement'
            }
        }
    
    def get_feature_predictive_relationships(self) -> Dict[str, str]:
        """
        Explains how different feature combinations affect predictions for AI explainability.
        """
        return {
            'bullish_signals': [
                'Low PE ratio + High ROE + Positive analyst sentiment = Undervalued growth stock',
                'High short interest + Rising price + High volume = Potential short squeeze',
                'Strong sector performance + Relative outperformance = Sector momentum play',
                'Low interest rates + High GDP growth = Growth stock favorable environment',
                'RSI < 30 + Positive news sentiment = Oversold bounce opportunity',
                'High call volume + Low put/call ratio = Institutional bullish positioning'
            ],
            
            'bearish_signals': [
                'High PE ratio + Declining earnings + Negative sentiment = Overvalued decline risk',
                'Rising interest rates + High debt ratio = Interest rate sensitive stock risk',
                'Sector underperformance + Weak relative strength = Sector rotation victim',
                'High unemployment + Consumer discretionary stock = Economic weakness impact',
                'RSI > 70 + High volume + Negative news = Overbought correction risk',
                'High put volume + Rising implied volatility = Institutional bearish hedging'
            ],
            
            'timing_signals': [
                'MACD bullish crossover + Volume confirmation = Entry signal',
                'Bollinger Band squeeze + Low volatility = Breakout setup',
                'Support level hold + Positive sentiment = Bounce opportunity',
                'Resistance break + High volume = Momentum continuation',
                'Economic calendar event + High IV = Volatility opportunity'
            ],
            
            'risk_signals': [
                'High correlation with market + High beta = Systematic risk exposure',
                'Low volume + Price divergence = Weak trend sustainability',
                'High insider selling + Negative guidance = Fundamental deterioration',
                'Yield curve inversion + Financial stock = Credit risk exposure',
                'High options gamma + Low liquidity = Volatility amplification risk'
            ]
        }