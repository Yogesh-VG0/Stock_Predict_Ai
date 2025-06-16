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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Placeholder macro and sector data fetchers
class MacroDataFetcher:
    def __init__(self):
        self.fred_api_key = os.getenv('FRED_API_KEY')
        self.fred = Fred(api_key=self.fred_api_key) if self.fred_api_key else None
        # Use comprehensive FRED indicators from fred_macro.py
        self.series = {
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
            'NONFARM_PAYROLL': 'PAYEMS'
        }
        self.cache = {}
        self.cache_expiry = 3600  # 1 hour

    def fetch_all(self, start_date, end_date, mongo_client=None):
        """Fetch all macro indicators with caching."""
        try:
            # Check cache first
            cache_key = f"{start_date}_{end_date}"
            if cache_key in self.cache:
                cache_time, cache_data = self.cache[cache_key]
                if (datetime.now() - cache_time).seconds < self.cache_expiry:
                    return cache_data

            dates = pd.date_range(start=start_date, end=end_date)
            df = pd.DataFrame({'date': dates})

            if self.fred:
                for name, series_id in self.series.items():
                    try:
                        # Try to get from MongoDB first
                        if mongo_client:
                            cached_data = mongo_client.get_macro_data(name, start_date, end_date)
                            if cached_data:
                                data = pd.Series(cached_data)
                            else:
                                data = self.fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
                                # Store in MongoDB
                                data_dict = {d.strftime('%Y-%m-%d'): float(v) for d, v in data.items() if pd.notna(v)}
                                mongo_client.store_macro_data(name, data_dict, source='FRED')
                        else:
                            data = self.fred.get_series(series_id, observation_start=start_date, observation_end=end_date)

                        data = data.reindex(dates, method='ffill')
                        df[name] = data.values

                        # Calculate derived features
                        if len(data) > 1:
                            df[f'{name}_change'] = data.pct_change().values
                            df[f'{name}_ma5'] = data.rolling(5).mean().values
                            df[f'{name}_ma20'] = data.rolling(20).mean().values

                    except Exception as e:
                        logger.error(f"FRED fetch failed for {name}: {e}")
                        df[name] = np.nan
            else:
                df[list(self.series.keys())] = np.nan

            # Fill missing values
            df = df.ffill().bfill()

            # Update cache
            self.cache[cache_key] = (datetime.now(), df)

            return df

        except Exception as e:
            logger.error(f"Error fetching macro data: {e}")
            return pd.DataFrame({'date': dates})

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
    def __init__(self, macro_sources: list = None, sentiment_window_days: int = None, sentiment_analyzer: SentimentAnalyzer = None, mongo_client: MongoDBClient = None):
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
        self.calendar_fetcher = EconomicCalendar(self.mongo_client)
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
            # Get macro and sector data from MongoDB
            macro_data = self.mongo_client.get_macro_data(ticker)
            sector_data = self.mongo_client.get_sector_data(ticker)
            
            # Merge macro data if available
            if macro_data is not None and not macro_data.empty:
                df = pd.merge(
                    df,
                    macro_data,
                    left_index=True,
                    right_index=True,
                    how='left'
                )
            
            # Merge sector data if available
            if sector_data is not None and not sector_data.empty:
                df = pd.merge(
                    df,
                    sector_data,
                    left_index=True,
                    right_index=True,
                    how='left'
                )
            
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
        """Add sentiment features to the DataFrame, avoiding double-counting of sentiment sources."""
        if sentiment_dict is None:
            return df

        df = df.copy()
        
        # Use only the blended sentiment score and its components
        sentiment_features = {
            'blended_sentiment_score': sentiment_dict.get('blended_sentiment_score', 0.0),
            'sentiment_confidence': sentiment_dict.get('sentiment_confidence', 0.5),
            'sentiment_volume': sentiment_dict.get('sentiment_volume', 0)
        }
        
        # Add economic calendar features separately (not part of sentiment)
        if 'economic_event_features' in sentiment_dict:
            for key, value in sentiment_dict['economic_event_features'].items():
                sentiment_features[f'event_{key}'] = value
        
        # Add features to DataFrame
        for feature, value in sentiment_features.items():
            df[feature] = value
            
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
        feature_select_k: int = 20
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
                
                # Add sentiment features (only if not already in event features)
                if sentiment_dict and 'economic_event_features' not in sentiment_dict:
                    df = self.add_sentiment_features(df, sentiment_dict)
                    
            # Add rolling features
            df = self.add_rolling_features(df)
            
            # Add lagged and volatility features
            df = self.add_lagged_and_volatility_features(df)
            
            # Handle outliers if requested
            if handle_outliers:
                df = self.handle_outliers(df)
                
            # Remove duplicate features
            df = df.loc[:, ~df.columns.duplicated()]
            
            # Remove features with all NaN values
            df = df.dropna(axis=1, how='all')
            
            # Fill remaining NaN values
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Normalize features
            feature_stats = self._normalize_features(df)
            
            # Select features if requested
            if enable_shap_selection and ticker:
                # Get feature importance from MongoDB if available
                if mongo_client:
                    feature_importance = mongo_client.db['feature_importance'].find_one({'ticker': ticker})
                    if feature_importance:
                        # Use stored feature importance
                        selected_features = list(feature_importance.get('features', {}).keys())[:feature_select_k]
                        df = df[selected_features]
                else:
                    # Use SHAP for feature selection
                    self.select_features(df.values, None, k=feature_select_k)
                    df = df[self.selected_features]
                    
            # Convert to numpy array
            features = df.values
            
            # Reshape for LSTM if needed
            if window_size > 1:
                features = features.reshape(-1, window_size, features.shape[1])
                
            return features, feature_stats
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise

    def _normalize_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Normalize features for model input, returning only scalars (last value in sequence)."""
        try:
            normalized = {}
            # Price features
            price_cols = ['Close', 'Open', 'High', 'Low']
            for col in price_cols:
                if df[col].std() != 0:
                    series = (df[col] - df[col].mean()) / df[col].std()
                    normalized[f'{col}_norm'] = series.iloc[-1] if not series.empty else 0
                else:
                    normalized[f'{col}_norm'] = 0
            # Volume features
            if df['Volume'].std() != 0:
                series = (df['Volume'] - df['Volume'].mean()) / df['Volume'].std()
                normalized['Volume_norm'] = series.iloc[-1] if not series.empty else 0
            else:
                normalized['Volume_norm'] = 0
            # Technical indicators
            tech_cols = [col for col in df.columns if col not in price_cols + ['Volume']]
            for col in tech_cols:
                if df[col].std() != 0:
                    series = (df[col] - df[col].mean()) / df[col].std()
                    normalized[f'{col}_norm'] = series.iloc[-1] if not series.empty else 0
                else:
                    normalized[f'{col}_norm'] = 0
            return normalized
        except Exception as e:
            logger.error(f"Error normalizing features: {str(e)}")
            return {col: 0 for col in df.columns}

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