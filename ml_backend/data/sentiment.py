"""
Sentiment analysis module for processing social media and news sentiment.

"""

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Dict, Any
import logging
import requests
import os
from dotenv import load_dotenv
from ..config.constants import (
    REDDIT_SUBREDDITS,
    RETRY_CONFIG,
    TOP_100_TICKERS,
    TICKER_SUBREDDITS
)
from ..utils.mongodb import MongoDBClient
from .sec_filings import SECFilingsAnalyzer
import aiohttp
import finnhub
from .seeking_alpha import SeekingAlphaAnalyzer
from starlette.concurrency import run_in_threadpool
import argparse
import asyncio
import pandas_market_calendars as mcal
import numpy as np
from .economic_calendar import EconomicCalendar
from .fred_macro import fetch_and_store_all_fred_indicators
import sys
import traceback

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global locks for sequential processing to prevent bot detection
SEEKING_ALPHA_LOCK = asyncio.Lock()
ECONOMIC_CALENDAR_LOCK = asyncio.Lock()
ECONOMIC_CALENDAR_CACHE = {}
ECONOMIC_CALENDAR_CACHE_DURATION = timedelta(hours=24)

# Global economic calendar instance to share across all tickers
SHARED_ECONOMIC_CALENDAR = None

# Cache global economic data to prevent multiple fetches
ECONOMIC_DATA_CACHE = {
    'data': None,
    'timestamp': None,
    'cache_duration': timedelta(hours=6)  # Cache for 6 hours
}

# Sentiment analysis configuration
SENTIMENT_CONFIG = {
    "min_engagement": {
        "reddit": 10,  # Minimum score for Reddit posts
        "news": 3      # Minimum articles for news sentiment
    },
    "confidence_threshold": 0.5,  # Minimum confidence for sentiment scores
    "model_path": os.getenv("BERT_MODEL_PATH", "models/bert-sentiment"),
    "cache_dir": os.getenv("MODEL_CACHE_DIR", "models/cache")
}

LABEL_MAP = {
    "POSITIVE": 0.8, "NEUTRAL": 0.0, "NEGATIVE": -0.8,
    "LABEL_2": 0.8,  # Positive (Roberta)
    "LABEL_1": 0.0,  # Neutral (Roberta)
    "LABEL_0": -0.8  # Negative (Roberta)
}

USE_ALPHA_OBJECTS_FOR_SENTIMENT = True  # Set to True to use Alpha Vantage objects for sentiment fallback
RECENT_DAYS = 7  # Use news/posts from the last 7 days

def get_cutoff_datetime():
    return datetime.utcnow() - timedelta(days=RECENT_DAYS)

def _map_sentiment_label(label):
    if label in LABEL_MAP:
        return LABEL_MAP[label]
    # Try to handle lowercase or unknown labels
    l = label.upper()
    if l in LABEL_MAP:
        return LABEL_MAP[l]
    logger.warning(f"Unknown sentiment label: {label}, defaulting to 0.0")
    return 0.0

def _is_recent(date_obj):
    if not date_obj:
        return False
    now = datetime.utcnow()
    # Always make both timezone-naive (UTC)
    if hasattr(date_obj, 'tz_localize') or hasattr(date_obj, 'tzinfo'):
        if getattr(date_obj, 'tzinfo', None) is not None:
            date_obj = date_obj.tz_convert(None) if hasattr(date_obj, 'tz_convert') else date_obj.tz_localize(None)
    if hasattr(now, 'tzinfo') and now.tzinfo is not None:
        now = now.tz_convert(None) if hasattr(now, 'tz_convert') else now.tz_localize(None)
    if isinstance(date_obj, str):
        try:
            date_obj = pd.to_datetime(date_obj).tz_localize(None)
        except Exception:
            return False
    return (now - date_obj).days < RECENT_DAYS

class SentimentAnalyzer:
    _macro_data_fetched = False  # Class variable to ensure macro data is fetched only once per process

    def __init__(self, mongo_client: MongoDBClient, calendar_fetcher=None):
        
        self.mongo_client = mongo_client
        # Initialize FinBERT model if available
        try:
            from transformers import pipeline
            self.finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            logger.info("FinBERT model loaded for enhanced sentiment analysis")
        except ImportError:
            logger.warning("Transformers/FinBERT not available, using VADER only")
            self.finbert = None
        except Exception as e:
            logger.warning(f"Failed to load FinBERT: {e}, using VADER only")
            self.finbert = None
        
        # Initialize VADER sentiment analyzer
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer initialized")
        except ImportError:
            logger.error("VADER sentiment analyzer not available")
            self.vader = None
        
        # Initialize Twitter-RoBERTa for Reddit sentiment analysis
        try:
            from transformers import pipeline
            self.twitter_roberta = pipeline("sentiment-analysis", 
                                           model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                                           device=-1)  # Use CPU
            logger.info("Twitter-RoBERTa model loaded for Reddit sentiment analysis")
        except Exception as e:
            logger.warning(f"Failed to load Twitter-RoBERTa: {e}, using VADER only for Reddit")
            self.twitter_roberta = None
        
        # Initialize SEC analyzer
        try:
            from .sec_filings import SECFilingsAnalyzer
            self.sec_analyzer = SECFilingsAnalyzer(mongo_client)
            logger.info("SEC analyzer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize SEC analyzer: {e}")
            self.sec_analyzer = None
        
        # Initialize Seeking Alpha analyzer
        try:
            from .seeking_alpha import SeekingAlphaAnalyzer
            self.seeking_alpha_analyzer = SeekingAlphaAnalyzer(mongo_client)
            logger.info("Seeking Alpha analyzer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Seeking Alpha analyzer: {e}")
            self.seeking_alpha_analyzer = None
        
        # Initialize FMP API Manager
        self.fmp_manager = FMPAPIManager(mongo_client)
        logger.info("FMP API Manager initialized")
        
        # Initialize macro data once per class (not per instance)
        if not SentimentAnalyzer._macro_data_fetched:
            try:
                # Get macro data once and store in MongoDB
                from .fred_macro import fetch_and_store_all_fred_indicators
                from datetime import datetime, timedelta
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365 * 2)  # 2 years of data
                
                logger.info("Fetching macro data (one-time initialization)...")
                fetch_and_store_all_fred_indicators(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    mongo_client
                )
                SentimentAnalyzer._macro_data_fetched = True
                logger.info("Macro data fetched and stored in MongoDB")
            except Exception as e:
                logger.warning(f"Failed to fetch macro data: {e}")
        
        # Initialize calendar fetcher
        self.calendar_fetcher = calendar_fetcher
        
        # Initialize short interest analyzer (will be created on-demand)
        self.short_interest_analyzer = None
        
        # Track API health status for rate limiting and error handling
        self.api_status = {
            'finviz': {'working': True, 'last_error': None, 'cooldown_until': None},
            'yahoo': {'working': True, 'last_error': None, 'cooldown_until': None},
            'marketaux': {'working': True, 'last_error': None, 'cooldown_until': None},
            'alpha_vantage': {'working': True, 'last_error': None, 'cooldown_until': None},
            'finnhub': {'working': True, 'last_error': None, 'cooldown_until': None},
            'fmp': {'working': True, 'last_error': None, 'cooldown_until': None},
            'sec': {'working': True, 'last_error': None, 'cooldown_until': None},
            'yahoo_news': {'working': True, 'last_error': None, 'cooldown_until': None},
            'rss_news': {'working': True, 'last_error': None, 'cooldown_until': None},
            'reddit': {'working': True, 'last_error': None, 'cooldown_until': None},
            'seekingalpha': {'working': True, 'last_error': None, 'cooldown_until': None},
            'seekingalpha_comments': {'working': True, 'last_error': None, 'cooldown_until': None},
            'alpha_earnings_call': {'working': True, 'last_error': None, 'cooldown_until': None},
            'alphavantage': {'working': True, 'last_error': None, 'cooldown_until': None},
        }
        
        logger.info(f"SentimentAnalyzer initialized with FinBERT: {self.finbert is not None}, VADER: {self.vader is not None}")
        logger.info(f"Additional models: Twitter-RoBERTa: {self.twitter_roberta is not None}, SEC: {self.sec_analyzer is not None}, SeekingAlpha: {self.seeking_alpha_analyzer is not None}")
        
    def _check_api_health(self, api_name: str) -> bool:
        """Check if an API is healthy and not in cooldown."""
        status = self.api_status.get(api_name, {})
        if not status.get('working', True):
            cooldown_until = status.get('cooldown_until')
            if cooldown_until and datetime.utcnow() < cooldown_until:
                return False
            else:
                # Reset status if cooldown period is over
                self.api_status[api_name]['working'] = True
                self.api_status[api_name]['cooldown_until'] = None
        return True
    
    def _mark_api_error(self, api_name: str, error: str, cooldown_minutes: int = 15):
        """Mark an API as having an error and set cooldown period."""
        self.api_status[api_name] = {
            'working': False,
            'last_error': error,
            'cooldown_until': datetime.utcnow() + timedelta(minutes=cooldown_minutes)
        }
        logger.warning(f"API {api_name} marked as down for {cooldown_minutes} minutes due to: {error}")
    
    def _mark_api_success(self, api_name: str):
        """Mark an API as working successfully."""
        self.api_status[api_name] = {
            'working': True,
            'last_error': None,
            'cooldown_until': None
        }

    def _extract_section(self, transcript: str, section_name: str) -> str:
        """Extract a specific section from the transcript."""
        try:
            # Look for section headers
            section_patterns = [
                f"{section_name}:",
                f"{section_name} -",
                f"{section_name}—",
                f"{section_name} –"
            ]
            
            for pattern in section_patterns:
                if pattern in transcript:
                    # Find the start of this section
                    start_idx = transcript.find(pattern)
                    if start_idx != -1:
                        # Find the start of the next section
                        next_section_idx = len(transcript)
                        for next_pattern in section_patterns:
                            next_idx = transcript.find(next_pattern, start_idx + len(pattern))
                            if next_idx != -1:
                                next_section_idx = min(next_section_idx, next_idx)
                        
                        # Extract the section text
                        section_text = transcript[start_idx + len(pattern):next_section_idx].strip()
                        return section_text
            
            return ""
            
        except Exception as e:
            logger.warning(f"Error extracting section {section_name}: {e}")
            return ""

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

    def analyze_news_sentiment(self, ticker: str) -> Dict[str, float]:
        """
        This method is deprecated. Use analyze_rss_news_sentiment instead for stock-specific news.
        Returns empty sentiment data.
        """
        logger.warning(f"analyze_news_sentiment is deprecated. Use analyze_rss_news_sentiment for {ticker}")
        return {
            "news_sentiment": 0.0,
            "news_volume": 0,
            "news_confidence": 0.0,
            "news_api_status": "deprecated",
            "news_error": "Use analyze_rss_news_sentiment for stock-specific news"
        }

    def is_sentiment_empty(self, sentiment: Dict[str, float], min_volume=3, min_confidence=0.2) -> bool:
        # Try to find a volume/confidence key in the dict
        volume = 0
        confidence = 0.0
        for k in sentiment:
            if 'volume' in k:
                volume = sentiment[k]
            if 'confidence' in k:
                confidence = sentiment[k]
        return (volume < min_volume) or (confidence < min_confidence)

    async def analyze_rss_news_sentiment(self, ticker: str) -> Dict[str, float]:
        """
        Analyze sentiment from Yahoo Finance and SeekingAlpha RSS feeds using NLP models.
        """
        sentiment_scores = []
        total_volume = 0
        raw_data = []
        nlp_results = []
        
        # Yahoo Finance RSS
        yahoo_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        # SeekingAlpha RSS
        sa_url = f"https://seekingalpha.com/api/sa/combined/{ticker}.xml"
        
        for source_name, url in [("Yahoo Finance", yahoo_url), ("SeekingAlpha", sa_url)]:
            try:
                feed = feedparser.parse(url)
                entries = feed.entries[:15]  # Limit to latest 15
                
                for entry in entries:
                    title = getattr(entry, 'title', '')
                    summary = getattr(entry, 'summary', '')
                    date_obj = None
                    
                    # Parse date
                    for date_field in ['published', 'updated', 'published_parsed', 'updated_parsed']:
                        if hasattr(entry, date_field):
                            try:
                                date_obj = pd.to_datetime(getattr(entry, date_field), errors='coerce')
                                if date_obj is not None:
                                    date_obj = date_obj.tz_localize(None)
                            except Exception:
                                date_obj = None
                            break
                    
                    if not _is_recent(date_obj):
                        continue
                    
                    text = f"{title} {summary}".strip()
                    text_trunc = text[:512]
                    
                    # Run sentiment analysis
                    if self.finbert is not None:
                        sentiment = self.finbert(text_trunc)[0]
                        if 'label' in sentiment:
                            sentiment_score = _map_sentiment_label(sentiment['label'])
                        else:
                            sentiment_score = sentiment['score']
                    else:
                        sentiment = self.vader.polarity_scores(text_trunc)
                        sentiment_score = sentiment["compound"]
                    
                    sentiment_scores.append(sentiment_score)
                    raw_data.append({"source": source_name, "title": title, "summary": summary})
                    nlp_results.append({
                        "text": text_trunc,
                        "sentiment": sentiment_score,
                        "model": "finbert" if self.finbert is not None else "vader",
                        "source": source_name
                    })
                    total_volume += 1
                    
            except Exception as e:
                logger.error(f"Error analyzing RSS news sentiment for {ticker} from {source_name}: {e}")
        
        if total_volume < 1:
            return {
                "rss_news_sentiment": 0.0,
                "rss_news_volume": 0,
                "rss_news_confidence": 0.0,
                "rss_news_raw_data": [],
                "rss_news_nlp_results": []
            }
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        logger.info(f"RSS news sentiment: {total_volume} articles, avg score: {avg_sentiment:.3f}")
        
        return {
            "rss_news_sentiment": avg_sentiment,
            "rss_news_volume": total_volume,
            "rss_news_confidence": min(total_volume / 20, 1.0),
            "rss_news_raw_data": raw_data,
            "rss_news_nlp_results": nlp_results
        }

    def blend_sentiment_scores(self, sentiment: Dict[str, float]) -> float:
        """Blend sentiment scores from multiple sources with proper type checking."""
        if not sentiment:
            return 0.0

        total_score = 0.0
        total_weight = 0.0

        # Standardize source keys
        standard_keys = {
            'fmp_price_target_summary': 'fmp_price_target',
            'fmp_grades_summary': 'fmp_grades',
            'fmp_ratings_snapshot': 'fmp_ratings',
            'fmp_financial_estimates': 'fmp_estimates',
            'seekingalpha_comments_sentiment': 'seeking_alpha_comments'
        }

        # Source weights for blending
        weights = {
            'rss_news': 0.15,
            'seeking_alpha': 0.15,
            'yahoo_news': 0.15,
            'marketaux': 0.15,
            'reddit': 0.1,
            'finnhub_insider': 0.1,
            'earnings_call': 0.1,
            'seeking_alpha_comments': 0.05,
            'fmp_estimates': 0.05,
            'fmp_ratings': 0.05,
            'fmp_price_targets': 0.05,
            'fmp_grades': 0.05,
            'fmp_dividends': 0.05,
            'sec_filings': 0.1
        }

        for source, data in sentiment.items():
            # Get standardized source key
            source_key = standard_keys.get(source, source)
            weight = weights.get(source_key, 0.05)  # Default weight for unknown sources

            # Extract score with proper type checking
            score = 0.0
            if isinstance(data, dict):
                # Try different possible score keys
                score = (data.get('sentiment_score') or 
                        data.get('sentiment') or 
                        data.get('score') or 
                        0.0)
            elif isinstance(data, (int, float)):
                score = float(data)
            else:
                logger.warning(f"Unexpected data type for {source}: {type(data)}")
                continue

            # Validate score is within [-1, 1]
            try:
                score = float(score)
            except Exception:
                score = 0.0
            score = max(-1.0, min(1.0, score))
            
            # Add to weighted average
            total_score += score * weight
            total_weight += weight

        # Return normalized score
        return total_score / total_weight if total_weight > 0 else 0.0

    async def fetch_alpha_vantage_earnings_call(self, ticker: str, quarter: str = None) -> dict:
        """Keep earnings call transcript from Alpha Vantage as requested."""
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not quarter or not isinstance(quarter, str) or not quarter.startswith("20") or "Q" not in quarter:
            logger.error(f"Invalid or missing quarter for earnings call transcript for {ticker}. Must be in format YYYYQn.")
            return {"error": "Invalid or missing quarter. Must be in format YYYYQn."}
        url = f"https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPT&symbol={ticker}&quarter={quarter}&apikey={api_key}"
        try:
            cached = self.mongo_client.get_alpha_vantage_data(ticker, f'earnings_call_transcript_{quarter}')
            if cached:
                return cached
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as resp:
                    if resp.status != 200:
                        logger.warning(f"Alpha Vantage earnings call transcript returned status {resp.status} for {ticker} {quarter}")
                        return {}
                    data = await resp.json()
                    self.mongo_client.store_alpha_vantage_data(ticker, f'earnings_call_transcript_{quarter}', data)
                    return data
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage earnings call transcript: {e}")
            return {}

    async def fetch_finnhub_insider_transactions(self, ticker: str) -> dict:
        """Fetch insider transactions from Finnhub."""
        api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key:
            logger.warning("FINNHUB_API_KEY not set. Skipping Finnhub insider transactions.")
            return {}
        
        try:
            finnhub_client = finnhub.Client(api_key=api_key)
            from_date = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
            to_date = datetime.utcnow().strftime("%Y-%m-%d")
            
            # Fetch insider transactions
            data = finnhub_client.stock_insider_transactions(ticker, from_date, to_date)
            
            # Store in MongoDB
            if data and data.get('data'):
                for transaction in data['data']:
                    self.mongo_client.db['insider_transactions'].replace_one(
                        {
                            "symbol": ticker,
                            "filingDate": transaction.get("filingDate"),
                            "transactionDate": transaction.get("transactionDate"),
                            "name": transaction.get("name")
                        },
                        {**transaction, "symbol": ticker},
                        upsert=True
                    )
                logger.info(f"Stored {len(data['data'])} insider transactions for {ticker}")
            
            return data
        except Exception as e:
            logger.error(f"Error fetching Finnhub insider transactions for {ticker}: {e}")
            return {}

    async def fetch_fmp_dividends(self, ticker: str) -> dict:
        """Fetch dividend data from FMP using centralized manager."""
        return await self.fmp_manager.get_dividends_historical(ticker)

    async def fetch_fmp_earnings(self, ticker: str) -> dict:
        """Fetch earnings data from FMP using centralized manager."""
        return await self.fmp_manager.get_earnings_historical(ticker)

    async def fetch_fmp_earnings_calendar(self, ticker: str = None) -> dict:
        """Fetch earnings calendar from FMP using centralized manager."""
        return await self.fmp_manager.get_earnings_calendar(ticker=ticker)

    async def fetch_fmp_dividends_calendar(self, ticker: str = None) -> dict:
        """Fetch dividends calendar from FMP using centralized manager."""
        return await self.fmp_manager.get_dividends_calendar(ticker=ticker)

    async def fetch_finnhub_quote(self, ticker: str) -> dict:
        """Fetch real-time quote from Finnhub."""
        api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key:
            logger.warning("FINNHUB_API_KEY not set. Skipping Finnhub quote.")
            return {}
        
        try:
            cached = self.mongo_client.get_alpha_vantage_data(ticker, 'quote')
            if cached:
                return cached
                
            finnhub_client = finnhub.Client(api_key=api_key)
            data = finnhub_client.quote(ticker)
            
            # Transform to match expected format
            quote_data = {
                "Global Quote": {
                    "01. symbol": ticker,
                    "02. open": str(data.get('o', 0)),
                    "03. high": str(data.get('h', 0)),
                    "04. low": str(data.get('l', 0)),
                    "05. price": str(data.get('c', 0)),
                    "06. volume": "0",  # Finnhub doesn't provide volume in quote
                    "07. latest trading day": datetime.utcnow().strftime("%Y-%m-%d"),
                    "08. previous close": str(data.get('pc', 0)),
                    "09. change": str(data.get('d', 0)),
                    "10. change percent": f"{data.get('dp', 0)}%"
                }
            }
            
            self.mongo_client.store_alpha_vantage_data(ticker, 'quote', quote_data)
            return quote_data
        except Exception as e:
            logger.error(f"Error fetching Finnhub quote for {ticker}: {e}")
            return {}

    async def analyze_finviz_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Analyze sentiment from FinViz."""
        try:
            # Run in threadpool since this is a blocking operation
            return await run_in_threadpool(self._analyze_finviz_sentiment_sync, ticker)
        except Exception as e:
            logger.warning(f"FinViz sentiment analysis failed for {ticker}: {e}")
            return {"finviz_sentiment": 0.0, "finviz_volume": 0, "finviz_confidence": 0.0}

    def _analyze_finviz_sentiment_sync(self, ticker: str) -> Dict[str, Any]:
        """Synchronous implementation of FinViz sentiment analysis."""
        import requests
        from bs4 import BeautifulSoup
        sentiment_scores = []
        headlines = []
        sentiment_results = []
        total_volume = 0
        try:
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                logger.warning(f"FinViz returned status {resp.status_code} for {ticker}")
                return {
                    "finviz_sentiment": 0.0,
                    "finviz_volume": 0,
                    "finviz_confidence": 0.0,
                    "finviz_raw_data": [],
                    "finviz_nlp_results": [],
                    "finviz_api_status": "api_error",
                    "finviz_error": f"HTTP {resp.status_code}"
                }
            soup = BeautifulSoup(resp.text, "html.parser")
            news_table = soup.find("table", class_="fullview-news-outer")
            if not news_table:
                logger.warning(f"No news table found on FinViz for {ticker}")
                return {
                    "finviz_sentiment": 0.0,
                    "finviz_volume": 0,
                    "finviz_confidence": 0.0,
                    "finviz_raw_data": [],
                    "finviz_nlp_results": [],
                    "finviz_api_status": "no_data",
                    "finviz_error": "No news table found"
                }
            rows = news_table.find_all("tr")[:15]  # Limit to latest 15
            for row in rows:
                headline_td = row.find_all("td")
                if len(headline_td) < 2:
                    continue
                headline = headline_td[1].get_text(strip=True)
                date_str = headline_td[0].get_text(strip=True)
                try:
                    date_obj = pd.to_datetime(date_str, errors='coerce')
                except Exception:
                    date_obj = None
                if not _is_recent(date_obj):
                    continue
                headlines.append(headline)
                text_trunc = headline[:512]
                if self.finbert is not None:
                    sentiment = self.finbert(text_trunc)[0]
                    if 'label' in sentiment:
                        sentiment_score = _map_sentiment_label(sentiment['label'])
                        logger.debug(f"FinViz label '{sentiment['label']}' mapped to score {sentiment_score}")
                    else:
                        sentiment_score = sentiment['score']
                else:
                    sentiment = self.vader.polarity_scores(text_trunc)
                    sentiment_score = sentiment["compound"]
                sentiment_scores.append(sentiment_score)
                sentiment_results.append({"headline": headline, "sentiment": sentiment_score, "model": "finbert" if self.finbert is not None else "vader"})
                total_volume += 1
        except Exception as e:
            logger.error(f"Error analyzing FinViz sentiment for {ticker}: {str(e)}")
            return {
                "finviz_sentiment": 0.0,
                "finviz_volume": 0,
                "finviz_confidence": 0.0,
                "finviz_raw_data": headlines,
                "finviz_nlp_results": sentiment_results,
                "finviz_api_status": "exception",
                "finviz_error": str(e)
            }
        if total_volume < 1:
            logger.info(f"FinViz volume {total_volume} below threshold 1, setting sentiment to 0.")
            return {
                "finviz_sentiment": 0.0,
                "finviz_volume": total_volume,
                "finviz_confidence": 0.0,
                "finviz_raw_data": headlines,
                "finviz_nlp_results": sentiment_results,
                "finviz_api_status": "no_data",
                "finviz_error": "Insufficient volume (<1)"
            }
        logger.info(f"FinViz sentiment: {total_volume} headlines, avg score: {sum(sentiment_scores)/len(sentiment_scores) if sentiment_scores else 0.0:.3f}")
        return {"finviz_sentiment": sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0, "finviz_volume": total_volume, "finviz_confidence": min(total_volume / 20, 1.0), "finviz_raw_data": headlines, "finviz_nlp_results": sentiment_results, "finviz_api_status": "ok", "finviz_error": None}

    async def analyze_seekingalpha_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Analyze sentiment from SeekingAlpha."""
        try:
            # Run in threadpool since this is a blocking operation
            return await run_in_threadpool(self._analyze_seekingalpha_sentiment_sync, ticker)
        except Exception as e:
            logger.warning(f"SeekingAlpha sentiment analysis failed for {ticker}: {e}")
            return {"seekingalpha_sentiment": 0.0, "seekingalpha_volume": 0, "seekingalpha_confidence": 0.0}

    def _analyze_seekingalpha_sentiment_sync(self, ticker: str) -> Dict[str, Any]:
        """Synchronous implementation of SeekingAlpha sentiment analysis."""
        import feedparser
        sentiment_scores = []
        headlines = []
        sentiment_results = []
        total_volume = 0
        try:
            rss_url = f"https://seekingalpha.com/feed.xml?symbol={ticker}"
            feed = feedparser.parse(rss_url)
            entries = feed.entries[:10]  # Limit to latest 10
            for entry in entries:
                title = entry.title
                summary = getattr(entry, 'summary', '')
                date_obj = None
                for date_field in ['published', 'updated', 'published_parsed', 'updated_parsed']:
                    if hasattr(entry, date_field):
                        try:
                            date_obj = pd.to_datetime(getattr(entry, date_field), errors='coerce')
                            if date_obj is not None:
                                date_obj = date_obj.tz_localize(None)
                        except Exception:
                            date_obj = None
                        break
                if not _is_recent(date_obj):
                    continue
                text = f"{title} {summary}".strip()
                headlines.append({"title": title, "summary": summary})
                text_trunc = text[:512]
                if self.finbert is not None:
                    sentiment = self.finbert(text_trunc)[0]
                    if 'label' in sentiment:
                        sentiment_score = _map_sentiment_label(sentiment['label'])
                        logger.debug(f"SeekingAlpha label '{sentiment['label']}' mapped to score {sentiment_score}")
                    else:
                        sentiment_score = sentiment['score']
                else:
                    sentiment = self.vader.polarity_scores(text_trunc)
                    sentiment_score = sentiment["compound"]
                sentiment_scores.append(sentiment_score)
                sentiment_results.append({"headline": title, "summary": summary, "sentiment": sentiment_score, "model": "finbert" if self.finbert is not None else "vader"})
                total_volume += 1
        except Exception as e:
            logger.error(f"Error analyzing SeekingAlpha sentiment for {ticker}: {str(e)}")
            return {"seekingalpha_sentiment": 0.0, "seekingalpha_volume": 0, "seekingalpha_confidence": 0.0, "seekingalpha_raw_data": headlines, "seekingalpha_nlp_results": sentiment_results, "seekingalpha_api_status": "exception", "seekingalpha_error": str(e)}
        if total_volume < 1:
            logger.info(f"SeekingAlpha volume {total_volume} below threshold 1, setting sentiment to 0.")
            return {"seekingalpha_sentiment": 0.0, "seekingalpha_volume": total_volume, "seekingalpha_confidence": 0.0, "seekingalpha_raw_data": headlines, "seekingalpha_nlp_results": sentiment_results, "seekingalpha_api_status": "no_data", "seekingalpha_error": "Insufficient volume (<1)"}
        logger.info(f"SeekingAlpha sentiment: {total_volume} headlines, avg score: {sum(sentiment_scores)/len(sentiment_scores) if sentiment_scores else 0.0:.3f}")
        return {"seekingalpha_sentiment": sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0, "seekingalpha_volume": total_volume, "seekingalpha_confidence": min(total_volume / 20, 1.0), "seekingalpha_raw_data": headlines, "seekingalpha_nlp_results": sentiment_results, "seekingalpha_api_status": "ok", "seekingalpha_error": None}

    def _analyze_yahoo_news_sentiment_sync(self, ticker: str) -> Dict[str, Any]:
        """Synchronous implementation of Yahoo Finance news sentiment analysis."""
        import feedparser
        sentiment_scores = []
        headlines = []
        sentiment_results = []
        total_volume = 0
        try:
            rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            feed = feedparser.parse(rss_url)
            entries = feed.entries[:15]  # Limit to latest 15
            for entry in entries:
                title = entry.title
                summary = getattr(entry, 'summary', '')
                date_obj = None
                for date_field in ['published', 'updated', 'published_parsed', 'updated_parsed']:
                    if hasattr(entry, date_field):
                        try:
                            date_obj = pd.to_datetime(getattr(entry, date_field), errors='coerce')
                            if date_obj is not None:
                                date_obj = date_obj.tz_localize(None)
                        except Exception:
                            date_obj = None
                        break
                if not _is_recent(date_obj):
                    continue
                text = f"{title} {summary}".strip()
                headlines.append({"title": title, "summary": summary})
                text_trunc = text[:512]
                if self.finbert is not None:
                    sentiment = self.finbert(text_trunc)[0]
                    if 'label' in sentiment:
                        sentiment_score = _map_sentiment_label(sentiment['label'])
                        logger.debug(f"Yahoo News label '{sentiment['label']}' mapped to score {sentiment_score}")
                    else:
                        sentiment_score = sentiment['score']
                else:
                    sentiment = self.vader.polarity_scores(text_trunc)
                    sentiment_score = sentiment["compound"]
                sentiment_scores.append(sentiment_score)
                sentiment_results.append({"headline": title, "summary": summary, "sentiment": sentiment_score, "model": "finbert" if self.finbert is not None else "vader"})
                total_volume += 1
        except Exception as e:
            logger.error(f"Error analyzing Yahoo Finance news sentiment for {ticker}: {str(e)}")
            return {"yahoo_news_sentiment": 0.0, "yahoo_news_volume": 0, "yahoo_news_confidence": 0.0, "yahoo_news_raw_data": headlines, "yahoo_news_nlp_results": sentiment_results, "yahoo_news_api_status": "exception", "yahoo_news_error": str(e)}
        if total_volume < 1:
            logger.info(f"Yahoo News volume {total_volume} below threshold 1, setting sentiment to 0.")
            return {"yahoo_news_sentiment": 0.0, "yahoo_news_volume": total_volume, "yahoo_news_confidence": 0.0, "yahoo_news_raw_data": headlines, "yahoo_news_nlp_results": sentiment_results, "yahoo_news_api_status": "no_data", "yahoo_news_error": "Insufficient volume (<1)"}
        logger.info(f"Yahoo News sentiment: {total_volume} headlines, avg score: {sum(sentiment_scores)/len(sentiment_scores) if sentiment_scores else 0.0:.3f}")
        return {"yahoo_news_sentiment": sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0, "yahoo_news_volume": total_volume, "yahoo_news_confidence": min(total_volume / 20, 1.0), "yahoo_news_raw_data": headlines, "yahoo_news_nlp_results": sentiment_results, "yahoo_news_api_status": "ok", "yahoo_news_error": None}

    async def analyze_marketaux_sentiment(self, ticker: str) -> dict:
        """Analyze sentiment from Marketaux."""
        try:
            # Run in threadpool since this is a blocking operation
            return await run_in_threadpool(self._analyze_marketaux_sentiment_sync, ticker)
        except Exception as e:
            logger.warning(f"Marketaux sentiment analysis failed for {ticker}: {e}")
            return {"marketaux_sentiment": 0.0, "marketaux_volume": 0, "marketaux_confidence": 0.0}

    def _analyze_marketaux_sentiment_sync(self, ticker: str) -> dict:
        """Synchronous implementation of Marketaux sentiment analysis."""
        api_key = os.getenv("MARKETAUX_API_KEY")
        if not api_key:
            logger.warning("MARKETAUX_API_KEY not set. Skipping Marketaux sentiment.")
            return {"marketaux_sentiment": 0.0, "marketaux_volume": 0, "marketaux_confidence": 0.0, "marketaux_raw_data": [], "marketaux_nlp_results": []}
        url = f"https://api.marketaux.com/v1/news/all"
        params = {
            "symbols": ticker,
            "filter_entities": "true",
            "language": "en",
            "sentiment_gte": -1,  # get all articles with sentiment >= -1
            "sentiment_lte": 1,   # get all articles with sentiment <= 1
            "api_token": api_key
            }
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                logger.warning(f"Marketaux returned status {resp.status_code} for {ticker}")
                return {"marketaux_sentiment": 0.0, "marketaux_volume": 0, "marketaux_confidence": 0.0, "marketaux_raw_data": [], "marketaux_nlp_results": []}
            data = resp.json()
            articles = data.get("data", [])
            sentiment_scores = []
            sentiment_results = []
            for article in articles:
                # Only consider entities relevant to the ticker
                for entity in article.get("entities", []):
                    if entity.get("symbol", "").upper() == ticker.upper() and "sentiment_score" in entity:
                        score = entity["sentiment_score"]
                        sentiment_scores.append(score)
                sentiment_results.append({
                    "headline": article.get("title", ""),
                    "url": article.get("url", ""),
                    "sentiment": score
                })
            total_volume = len(sentiment_scores)
            if total_volume < 1:
                logger.info(f"Marketaux volume {total_volume} below threshold 1, setting sentiment to 0.")
                return {"marketaux_sentiment": 0.0, "marketaux_volume": 0, "marketaux_confidence": 0.0, "marketaux_raw_data": articles, "marketaux_nlp_results": sentiment_results}
            avg_sentiment = sum(sentiment_scores) / total_volume
            logger.info(f"Marketaux sentiment: {total_volume} articles/entities, avg score: {avg_sentiment:.3f}")
            return {"marketaux_sentiment": avg_sentiment, "marketaux_volume": total_volume, "marketaux_confidence": min(total_volume/20, 1.0), "marketaux_raw_data": articles, "marketaux_nlp_results": sentiment_results}
        except Exception as e:
            logger.error(f"Error analyzing Marketaux sentiment for {ticker}: {e}")
            return {"marketaux_sentiment": 0.0, "marketaux_volume": 0, "marketaux_confidence": 0.0, "marketaux_raw_data": [], "marketaux_nlp_results": []}

    async def analyze_fmp_financial_estimates(self, ticker: str) -> dict:
        """Analyze sentiment from FMP financial estimates using centralized manager."""
        try:
            result = await self.fmp_manager.get_analyst_estimates(ticker)
            estimates = result.get('estimates', [])
            
            if not estimates:
                return {"fmp_estimates_sentiment": 0.0, "fmp_estimates_volume": 0, "fmp_estimates_confidence": 0.0}
            
            # Use the most recent estimate
            est = estimates[0]
            eps_avg = est.get('estimatedEpsAvg', 0.0)
            # Normalize EPS to [-1, 1] using a soft cap (e.g., 0-10 for large caps)
            norm_eps = max(-1, min(1, (eps_avg - 5) / 5))
            logger.info(f"FMP Financial Estimates: epsAvg={eps_avg}, norm={norm_eps}")
            return {"fmp_estimates_sentiment": norm_eps, "fmp_estimates_volume": est.get('numberAnalystEstimatedEps', 1), "fmp_estimates_confidence": min(est.get('numberAnalystEstimatedEps', 1)/10, 1.0)}
        except Exception as e:
            logger.error(f"Error analyzing FMP financial estimates for {ticker}: {e}")
            return {"fmp_estimates_sentiment": 0.0, "fmp_estimates_volume": 0, "fmp_estimates_confidence": 0.0}

    async def analyze_fmp_ratings_snapshot(self, ticker: str) -> dict:
        """Analyze sentiment from FMP ratings snapshot using centralized manager."""
        try:
            result = await self.fmp_manager.get_ratings_snapshot(ticker)
            ratings = result.get('ratings', [])
            
            if not ratings:
                return {"fmp_ratings_sentiment": 0.0, "fmp_ratings_volume": 0, "fmp_ratings_confidence": 0.0}
            
            rating = ratings[0]
            overall = rating.get('ratingScore', 0)
            # Normalize overallScore (1-5) to [-1, 1]
            norm_score = (overall - 3) / 2
            logger.info(f"FMP Ratings Snapshot: overallScore={overall}, norm={norm_score}")
            return {"fmp_ratings_sentiment": norm_score, "fmp_ratings_volume": 1, "fmp_ratings_confidence": 1.0}
        except Exception as e:
            logger.error(f"Error analyzing FMP ratings snapshot for {ticker}: {e}")
            return {"fmp_ratings_sentiment": 0.0, "fmp_ratings_volume": 0, "fmp_ratings_confidence": 0.0}

    async def analyze_fmp_price_target_summary(self, ticker: str) -> dict:
        """Analyze sentiment from FMP price target summary using centralized manager."""
        try:
            result = await self.fmp_manager.get_price_target_summary(ticker)
            price_targets = result.get('price_targets', [])
            
            if not price_targets:
                return {"fmp_price_target_sentiment": 0.0, "fmp_price_target_volume": 0, "fmp_price_target_confidence": 0.0}
            
            pt = price_targets[0]
            avg_target = pt.get('lastYearAvgPriceTarget', 0.0)
            # For normalization, you may want to compare to current price (fetch if available)
            # For now, use a soft cap: (avg_target - 200) / 100 for large caps
            norm_pt = max(-1, min(1, (avg_target - 200) / 100))
            logger.info(f"FMP Price Target Summary: lastYearAvgPriceTarget={avg_target}, norm={norm_pt}")
            return {"fmp_price_target_sentiment": norm_pt, "fmp_price_target_volume": pt.get('lastYearCount', 1), "fmp_price_target_confidence": min(pt.get('lastYearCount', 1)/10, 1.0)}
        except Exception as e:
            logger.error(f"Error analyzing FMP price target summary for {ticker}: {e}")
            return {"fmp_price_target_sentiment": 0.0, "fmp_price_target_volume": 0, "fmp_price_target_confidence": 0.0}

    async def analyze_fmp_grades_summary(self, ticker: str) -> dict:
        """Analyze sentiment from FMP grades summary using centralized manager."""
        try:
            result = await self.fmp_manager.get_grades_consensus(ticker)
            grades = result.get('grades', [])
            
            if not grades:
                return {"fmp_grades_sentiment": 0.0, "fmp_grades_volume": 0, "fmp_grades_confidence": 0.0}
            
            gs = grades[0]
            consensus = gs.get('gradingCompany', '').lower()
            # Map consensus to sentiment
            mapping = {'strong buy': 1.0, 'buy': 0.7, 'hold': 0.0, 'sell': -0.7, 'strong sell': -1.0}
            norm_score = mapping.get(consensus, 0.0)
            logger.info(f"FMP Grades Summary: consensus={consensus}, norm={norm_score}")
            return {"fmp_grades_sentiment": norm_score, "fmp_grades_volume": 1, "fmp_grades_confidence": 1.0}
        except Exception as e:
            logger.error(f"Error analyzing FMP grades summary for {ticker}: {e}")
            return {"fmp_grades_sentiment": 0.0, "fmp_grades_volume": 0, "fmp_grades_confidence": 0.0}

    def _safe_dict_convert(self, value: Any) -> Any:
        """
        Safely convert a value to a JSON-serializable format.
        Handles nested dictionaries and ensures all keys are strings.
        
        Args:
            value: The value to convert
            
        Returns:
            A JSON-serializable version of the value
        """
        try:
            if isinstance(value, (int, float, str, bool, type(None))):
                return value
            elif isinstance(value, (list, tuple)):
                return [self._safe_dict_convert(v) for v in value]
            elif isinstance(value, dict):
                safe_dict = {}
                for k, v in value.items():
                    try:
                        sk = str(k)  # ensure key is a string
                        safe_dict[sk] = self._safe_dict_convert(v)
                    except Exception as e:
                        logger.warning(f"Failed to process nested key={k}: {e}")
                        safe_dict[sk] = str(v)
                return safe_dict
            else:
                return str(value)
        except Exception as e:
            logger.warning(f"Failed to convert value: {e}")
            return str(value)

    async def get_combined_sentiment(self, ticker: str, force_refresh: bool = False):
        """
        Get combined sentiment from all sources with proper MongoDB storage.
        
        Args:
            ticker: Stock ticker symbol
            force_refresh: Force refresh of cached data
            
        Returns:
            Dictionary containing comprehensive sentiment analysis
        """
        sentiment_dict = {
            "ticker": ticker,
            "timestamp": datetime.utcnow(),
        }

        # Define sentiment sources and their corresponding keys
        sources = [
            self.get_finviz_sentiment,
            self.get_sec_sentiment,
            self.get_yahoo_news_sentiment,
            self.get_marketaux_sentiment,
            self.get_rss_news_sentiment,
            self.get_reddit_sentiment,
            self.get_fmp_sentiment,
            self.get_finnhub_sentiment,
            self.get_seekingalpha_sentiment,
            self.get_seekingalpha_comments_sentiment,
            self.get_alpha_earnings_call_sentiment,
            self.get_alphavantage_news_sentiment
        ]
        keys = [
            "finviz", "sec", "yahoo_news", "marketaux", "rss_news", "reddit",
            "fmp", "finnhub", "seekingalpha", "seekingalpha_comments",
            "alpha_earnings_call", "alphavantage"
        ]
        
        # Process each sentiment source
        for source_func, key in zip(sources, keys):
            try:
                # Check API health before making request
                if not self._check_api_health(key):
                    logger.info(f"Skipping {key} sentiment - API in cooldown")
                    continue
                
                result = await source_func(ticker)
                
                # Mark API as successful if we get a result
                if result:
                    self._mark_api_success(key)
                
                # Try different possible keys for sentiment score, volume, and confidence
                score = result.get("sentiment_score", result.get(f"{key}_sentiment", 0))
                volume = result.get("volume", result.get(f"{key}_volume", 0))
                confidence = result.get("confidence", result.get(f"{key}_confidence", 0.5))
                
                sentiment_dict[f"{key}_sentiment"] = score
                sentiment_dict[f"{key}_volume"] = volume
                sentiment_dict[f"{key}_confidence"] = confidence
                
                # Store raw data and NLP results for debugging and transparency
                for suffix in ["raw_data", "nlp_results", "api_status", "error"]:
                    result_key = f"{key}_{suffix}"
                    if result_key in result:
                        sentiment_dict[result_key] = result[result_key]
                    elif suffix in result:
                        sentiment_dict[result_key] = result[suffix]
                        
            except Exception as e:
                logger.warning(f"Sentiment fetch failed for {key}: {e}")
                # Mark API as having an error
                self._mark_api_error(key, str(e))
                
                # Set default values for failed sources
                sentiment_dict[f"{key}_sentiment"] = 0.0
                sentiment_dict[f"{key}_volume"] = 0
                sentiment_dict[f"{key}_confidence"] = 0.0
                sentiment_dict[f"{key}_error"] = str(e)
                
        # Add Economic Calendar Sentiment (global events, not ticker-specific)
        try:
            sentiment_dict = await self.integrate_economic_events_sentiment(
                sentiment_dict, ticker, mongo_client=self.mongo_client
            )
            logger.info(f"Economic events sentiment added for {ticker}")
        except Exception as e:
            logger.error(f"Failed to add economic calendar sentiment for {ticker}: {e}")
            sentiment_dict['economic_event_sentiment'] = 0.0
            sentiment_dict['economic_event_volume'] = 0
            sentiment_dict['economic_event_confidence'] = 0.0
            
        # Add Short Interest Sentiment
        try:
            short_sentiment = await self.analyze_short_interest_sentiment(ticker)
            sentiment_dict.update(short_sentiment)
            logger.info(f"Short interest sentiment added for {ticker}")
        except Exception as e:
            logger.error(f"Failed to fetch short interest sentiment for {ticker}: {e}")
            sentiment_dict['short_interest_sentiment'] = 0.0
            sentiment_dict['short_interest_volume'] = 0
            sentiment_dict['short_interest_confidence'] = 0.0
            
        # Calculate final blended sentiment score
        try:
            blended_sentiment = self.blend_sentiment_scores(sentiment_dict)
            sentiment_dict["blended_sentiment"] = blended_sentiment
            
            # Calculate overall confidence and volume
            sentiment_dict["sentiment_confidence"] = self._calculate_sentiment_confidence(sentiment_dict)
            sentiment_dict["sentiment_volume"] = self._calculate_sentiment_volume(sentiment_dict)
            
            logger.info(f"Final blended sentiment for {ticker}: {blended_sentiment:.3f} "
                       f"(confidence: {sentiment_dict['sentiment_confidence']:.2f}, "
                       f"volume: {sentiment_dict['sentiment_volume']})")
        except Exception as e:
            logger.error(f"Error calculating blended sentiment for {ticker}: {e}")
            sentiment_dict["blended_sentiment"] = 0.0
            sentiment_dict["sentiment_confidence"] = 0.0
            sentiment_dict["sentiment_volume"] = 0
        
        # Add metadata for storage
        sentiment_dict["date"] = sentiment_dict["timestamp"].replace(hour=0, minute=0, second=0, microsecond=0)
        sentiment_dict["last_updated"] = datetime.utcnow()
        sentiment_dict["api_status"] = self.api_status  # Store API health status
        
        # Store in MongoDB with error handling
        try:
            self.mongo_client.store_sentiment(ticker, sentiment_dict)
            logger.info(f"Sentiment data for {ticker} stored in MongoDB successfully")
        except Exception as e:
            logger.error(f"Failed to store sentiment data for {ticker} in MongoDB: {e}")
            
        return sentiment_dict

    def _calculate_sentiment_confidence(self, sources: Dict) -> float:
        """Calculate overall sentiment confidence based on source reliability and data quality."""
        if not sources:
            return 0.0

        confidence_scores = []
        weights = {
            'rss_news': 0.15,
            'seeking_alpha': 0.15,
            'yahoo_news': 0.15,
            'marketaux': 0.15,
            'reddit': 0.1,
            'finnhub_insider': 0.1,
            'earnings_call': 0.1,
            'seeking_alpha_comments': 0.05,
            'fmp_estimates': 0.05,
            'fmp_ratings': 0.05,
            'fmp_price_targets': 0.05,
            'fmp_grades': 0.05,
            'fmp_dividends': 0.05,  # Added weight for FMP dividends
            'sec_filings': 0.1
        }

        for source, data in sources.items():
            if isinstance(data, dict):
                confidence = data.get('confidence', 0.0)
                weight = weights.get(source, 0.05)
                confidence_scores.append(confidence * weight)

        return sum(confidence_scores) if confidence_scores else 0.0

    def _calculate_sentiment_volume(self, sources: Dict) -> int:
        """Calculate total sentiment volume across all sources."""
        if not sources:
            return 0

        total_volume = 0
        for source, data in sources.items():
            if isinstance(data, dict):
                volume = data.get('volume', 0)
                if isinstance(volume, (int, float)):
                    total_volume += volume

        return total_volume

    async def fetch_fmp_dividends_and_store(self, ticker: str) -> dict:
        """Fetch dividend history for a ticker from FMP using centralized manager."""
        logger.warning("fetch_fmp_dividends_and_store is deprecated. Use fmp_manager.get_dividends_historical instead.")
        return await self.fmp_manager.get_dividends_historical(ticker)

    async def fetch_fmp_analyst_ratings_and_store(self, ticker: str) -> dict:
        """Fetch analyst ratings for a ticker from FMP using centralized manager."""
        logger.warning("fetch_fmp_analyst_ratings_and_store is deprecated. Use fmp_manager.get_all_fmp_data instead.")
        return await self.fmp_manager.get_all_fmp_data(ticker)

    def get_latest_quarter(self) -> str:
        """Return the most recent completed quarter in format YYYYQn (e.g., 2024Q1)."""
        now = datetime.utcnow()
        year = now.year
        month = now.month
        if month <= 3:
            year -= 1
            quarter = 4
        elif month <= 6:
            quarter = 1
        elif month <= 9:
            quarter = 2
        else:
            quarter = 3
        return f"{year}Q{quarter}"

    def _extract_sentiment_from_alpha_objects(self, alpha_data: dict) -> float:
        """
        Extracts a simple sentiment score from Alpha Vantage objects if no news is found.
        Looks for positive/negative signals in earnings, dividends, insider transactions, etc.
        Returns a float sentiment score between -1 and 1.
        """
        score = 0.0
        count = 0
        # Earnings call: look for 'surprise' or positive/negative words
        earnings_call = alpha_data.get('alpha_earnings_call', {})
        if isinstance(earnings_call, dict):
            text = str(earnings_call)
            if any(w in text.lower() for w in ['beat', 'strong', 'positive', 'growth', 'record']):
                score += 0.5
                count += 1
            if any(w in text.lower() for w in ['miss', 'weak', 'negative', 'decline', 'loss']):
                score -= 0.5
                count += 1
        # Insider transactions: more buys than sells is positive
        insider = alpha_data.get('finnhub_insider_transactions', {})
        if isinstance(insider, dict):
            text = str(insider)
            buys = text.lower().count('buy')
            sells = text.lower().count('sell')
            if buys > sells:
                score += 0.3
                count += 1
            elif sells > buys:
                score -= 0.3
                count += 1
        # Dividends: increase is positive
        dividends = alpha_data.get('fmp_dividends', {})
        if isinstance(dividends, dict):
            text = str(dividends)
            if 'increase' in text.lower():
                score += 0.2
                count += 1
            if 'decrease' in text.lower():
                score -= 0.2
                count += 1
        # Earnings: surprise positive/negative
        earnings = alpha_data.get('fmp_earnings', {})
        if isinstance(earnings, dict):
            text = str(earnings)
            if 'surprise' in text.lower() and 'positive' in text.lower():
                score += 0.4
                count += 1
            if 'surprise' in text.lower() and 'negative' in text.lower():
                score -= 0.4
                count += 1
        # Quote: price change up/down
        quote = alpha_data.get('finnhub_quote', {})
        if isinstance(quote, dict) and quote.get('Global Quote'):
            change = None
            try:
                change = float(quote['Global Quote'].get('10. change percent', '0').replace('%',''))
            except Exception:
                pass
            if change is not None:
                if change > 0:
                    score += 0.1
                    count += 1
                elif change < 0:
                    score -= 0.1
                    count += 1
        return score / count if count > 0 else 0.0

    async def analyze_finnhub_insider_sentiment(self, ticker: str) -> dict:
        """Analyze insider sentiment using Finnhub data."""
        try:
            # First try to fetch fresh data
            insider_transactions = await self.fetch_finnhub_insider_transactions(ticker)
            if not insider_transactions or not insider_transactions.get('data'):
                logger.warning(f"No insider transactions data available for {ticker}")
                
            # Get insider transactions from MongoDB (including freshly stored data)
            insider_data = self.mongo_client.get_insider_trading(ticker)
            if not insider_data:
                logger.warning(f"No insider transactions found in MongoDB for {ticker}")
                return {
                    'source': 'finnhub_insider',
                    'sentiment': 0.0,
                    'volume': 0,
                    'confidence': 0.0,
                    'transactions': 0,
                    'error': 'No insider transaction data available'
                }
            
            logger.info(f"Found {len(insider_data)} insider transactions for {ticker}")
            
            # Calculate sentiment based on transaction types and volumes
            total_volume = 0
            weighted_sentiment = 0
            buy_transactions = 0
            sell_transactions = 0
            
            for transaction in insider_data:
                try:
                    # Finnhub insider API structure: https://finnhub.io/docs/api/insider-sentiment
                    volume = abs(float(transaction.get('share', 0))) or abs(float(transaction.get('change', 0)))
                    price = float(transaction.get('transactionPrice', 0)) or 1.0  # Use 1.0 if no price
                    transaction_value = volume * price if price > 0 else volume
                    
                    # Check transaction code and change
                    transaction_code = transaction.get('transactionCode', '').upper()
                    change = float(transaction.get('change', 0))
                    
                    # Enhanced transaction type detection based on Finnhub data structure
                    if transaction_code in ['P', 'M', 'A'] or change > 0:  # Purchase, Multiple, Acquisition or positive change
                        weighted_sentiment += transaction_value
                        buy_transactions += 1
                    elif transaction_code in ['S', 'D'] or change < 0:  # Sale, Disposition or negative change
                        weighted_sentiment -= transaction_value
                        sell_transactions += 1
                    
                    total_volume += volume
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing transaction for {ticker}: {e}")
                    continue
            
            # Normalize sentiment
            if total_volume > 0:
                sentiment = weighted_sentiment / (total_volume * 100)  # Scale down for normalization
                sentiment = max(-1.0, min(1.0, sentiment))  # Clamp to [-1, 1]
                confidence = min(1.0, len(insider_data) / 5)  # Higher volume = higher confidence
            else:
                sentiment = 0.0
                confidence = 0.0
            
            logger.info(f"Finnhub insider sentiment for {ticker}: {sentiment:.3f} (buys: {buy_transactions}, sells: {sell_transactions})")
            
            return {
                'source': 'finnhub_insider',
                'sentiment': sentiment,
                'volume': int(total_volume),
                'confidence': confidence,
                'transactions': len(insider_data),
                'buy_transactions': buy_transactions,
                'sell_transactions': sell_transactions
            }
        except Exception as e:
            logger.error(f"Error analyzing Finnhub insider sentiment for {ticker}: {e}")
            return {
                'source': 'finnhub_insider',
                'sentiment': 0.0,
                'volume': 0,
                'confidence': 0.0,
                'error': str(e)
            }

    async def analyze_finnhub_recommendation_trends(self, ticker: str) -> dict:
        """
        Fetch Finnhub Recommendation Trends and compute a normalized sentiment score.
        """
        api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key:
            logger.warning("FINNHUB_API_KEY not set. Skipping Finnhub Recommendation Trends.")
            return {"finnhub_recommendation_sentiment": 0.0, "finnhub_recommendation_volume": 0, "finnhub_recommendation_confidence": 0.0}
        try:
            finnhub_client = finnhub.Client(api_key=api_key)
            data = finnhub_client.recommendation_trends(ticker)
            if not data:
                logger.info(f"Finnhub Recommendation Trends: No data for {ticker}")
                return {"finnhub_recommendation_sentiment": 0.0, "finnhub_recommendation_volume": 0, "finnhub_recommendation_confidence": 0.0}
            # Use the most recent period
            rec = data[0]
            strong_buy = rec.get('strongBuy', 0)
            buy = rec.get('buy', 0)
            hold = rec.get('hold', 0)
            sell = rec.get('sell', 0)
            strong_sell = rec.get('strongSell', 0)
            total = strong_buy + buy + hold + sell + strong_sell
            if total == 0:
                return {"finnhub_recommendation_sentiment": 0.0, "finnhub_recommendation_volume": 0, "finnhub_recommendation_confidence": 0.0}
            # Weighted score: (strongBuy*2 + buy - sell - strongSell*2) / total
            score = (2*strong_buy + buy - sell - 2*strong_sell) / total
            # Normalize to [-1, 1]
            norm_score = max(-1, min(1, score / 2))
            logger.info(f"Finnhub Recommendation Trends: score={score:.2f}, norm={norm_score:.3f}, total={total}")
            return {"finnhub_recommendation_sentiment": norm_score, "finnhub_recommendation_volume": total, "finnhub_recommendation_confidence": min(total/20, 1.0)}
        except Exception as e:
            logger.error(f"Error fetching Finnhub Recommendation Trends for {ticker}: {e}")
            return {"finnhub_recommendation_sentiment": 0.0, "finnhub_recommendation_volume": 0, "finnhub_recommendation_confidence": 0.0}

    async def analyze_reddit_sentiment(self, ticker: str) -> Dict[str, float]:
        """Analyze sentiment from Reddit posts using Twitter-Roberta."""
        try:
            # Run in threadpool for async compatibility
            logger.info("Running Reddit sentiment analysis in threadpool for async compatibility.")
            return await run_in_threadpool(self._analyze_reddit_sentiment_sync, ticker)
        except Exception as e:
            logger.error(f"Error analyzing Reddit sentiment: {e}")
            return None

    def _analyze_reddit_sentiment_sync(self, ticker: str) -> Dict[str, float]:
        """Synchronous Reddit sentiment analysis using Twitter-Roberta and ticker-specific subreddits."""
        try:
            import praw
            from ..config.constants import TICKER_SUBREDDITS, REDDIT_SUBREDDITS
            
            sentiment_scores = []
            total_volume = 0
            raw_data = []
            nlp_results = []
            max_retries = 3
            
            # Get ticker-specific subreddits or use general ones
            target_subreddits = TICKER_SUBREDDITS.get(ticker, REDDIT_SUBREDDITS)
            logger.info(f"📊 Reddit sentiment for {ticker} using subreddits: {target_subreddits}")
            
            for attempt in range(max_retries):
                try:
                    reddit = praw.Reddit(
                        client_id=os.getenv("REDDIT_CLIENT_ID"),
                        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                        user_agent=os.getenv("REDDIT_USER_AGENT")
                    )
                    
                    for subreddit_name in target_subreddits:
                        try:
                            logger.info(f"🔍 Searching r/{subreddit_name} for {ticker}")
                            posts = reddit.subreddit(subreddit_name).search(ticker, sort="new", time_filter="week", limit=20)
                            posts_found = 0
                            
                            for post in posts:
                                if post.score < 10:
                                    continue
                                posts_found += 1
                                text = post.title + " " + post.selftext
                                text = text[:512]
                                
                                sentiment_score = 0.0  # Initialize with default value
                                
                                if self.twitter_roberta is not None:
                                    sentiment = self.twitter_roberta(text)[0]
                                    if 'label' in sentiment:
                                        sentiment_score = LABEL_MAP.get(sentiment['label'], 0.0)
                                    else:
                                        sentiment_score = sentiment.get('score', 0.0)
                                    sentiment_scores.append(sentiment_score * min(post.score / 100, 1.0))
                                else:
                                    s = self.vader.polarity_scores(text)
                                    sentiment_score = s["compound"]
                                    sentiment_scores.append(sentiment_score * min(post.score / 100, 1.0))
                                
                                total_volume += 1
                                raw_data.append({
                                    "type": "post",
                                    "subreddit": subreddit_name,
                                    "title": post.title,
                                    "selftext": post.selftext[:200],  # Truncate for storage
                                    "score": post.score
                                })
                                nlp_results.append({
                                    "text": text[:200],  # Truncate for storage
                                    "sentiment": sentiment_score,
                                    "model": "twitter_roberta" if self.twitter_roberta is not None else "vader",
                                    "subreddit": subreddit_name
                                })
                                
                                # Process top comments
                                post.comments.replace_more(limit=0)
                                for comment in post.comments.list()[:5]:
                                    if comment.score < 5:
                                        continue
                                    comment_text = comment.body[:512]
                                    
                                    comment_sentiment_score = 0.0  # Initialize with default value
                                    
                                    if self.twitter_roberta is not None:
                                        sentiment = self.twitter_roberta(comment_text)[0]
                                        if 'label' in sentiment:
                                            comment_sentiment_score = LABEL_MAP.get(sentiment['label'], 0.0)
                                        else:
                                            comment_sentiment_score = sentiment.get('score', 0.0)
                                        sentiment_scores.append(comment_sentiment_score * min(comment.score / 50, 1.0))
                                    else:
                                        s = self.vader.polarity_scores(comment_text)
                                        comment_sentiment_score = s["compound"]
                                        sentiment_scores.append(comment_sentiment_score * min(comment.score / 50, 1.0))
                                    
                                    total_volume += 1
                                    raw_data.append({
                                        "type": "comment",
                                        "subreddit": subreddit_name,
                                        "body": comment.body[:200],  # Truncate for storage
                                        "score": comment.score
                                    })
                                    nlp_results.append({
                                        "text": comment_text[:200],  # Truncate for storage
                                        "sentiment": comment_sentiment_score,
                                        "model": "twitter_roberta" if self.twitter_roberta is not None else "vader",
                                        "subreddit": subreddit_name
                                    })
                            
                            logger.info(f"✅ Found {posts_found} posts in r/{subreddit_name} for {ticker}")
                            
                        except Exception as e:
                            logger.error(f"❌ Error analyzing Reddit sentiment for {ticker} in r/{subreddit_name}: {str(e)}")
                            continue
                    break  # Success
                except praw.exceptions.APIException as e:
                    logger.warning(f"⚠️ Reddit API rate limit for {ticker}, attempt {attempt+1}")
                    time.sleep(self._exponential_backoff(attempt))
                except Exception as e:
                    logger.error(f"💥 Error initializing Reddit client: {str(e)}")
                    break
            
            if total_volume < 1:
                logger.info(f"📊 Reddit volume {total_volume} below threshold 1 for {ticker}, setting sentiment to 0.")
                return {
                    "reddit_sentiment": 0.0, 
                    "reddit_volume": total_volume, 
                    "reddit_confidence": 0.0, 
                    "reddit_raw_data": raw_data, 
                    "reddit_nlp_results": nlp_results,
                    "reddit_subreddits_used": target_subreddits
                }
            
            average_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
            logger.info(f"🎯 Reddit sentiment for {ticker}: {average_sentiment:.3f} (volume: {total_volume})")
            
            return {
                "reddit_sentiment": average_sentiment,
                "reddit_volume": total_volume,
                "reddit_confidence": min(total_volume / 50, 1.0),
                "reddit_raw_data": raw_data,
                "reddit_nlp_results": nlp_results,
                "reddit_subreddits_used": target_subreddits
            }
        except Exception as e:
            logger.error(f"💥 Error in Reddit sentiment analysis for {ticker}: {e}")
            return {
                "reddit_sentiment": 0.0, 
                "reddit_volume": 0, 
                "reddit_confidence": 0.0, 
                "reddit_raw_data": [], 
                "reddit_nlp_results": [],
                "reddit_error": str(e)
            }

    def fetch_fomc_meetings_and_store(self):
        """Fetch FOMC meeting dates from the Fed website, cache in MongoDB, and return the dates."""
        cache_key = 'fomc_meetings'
        cached = self.mongo_client.get_alpha_vantage_data('market', cache_key)
        if cached and 'dates' in cached:
            return cached['dates']
        url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"FOMC calendar returned status {resp.status_code}")
                return []
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            dates = []
            for tag in soup.find_all("td", class_="fomc-meeting__date"):
                text = tag.get_text(strip=True)
                try:
                    dt = pd.to_datetime(text, errors='coerce')
                    if pd.notnull(dt):
                        dates.append(dt.strftime('%Y-%m-%d'))
                except Exception:
                    continue
            self.mongo_client.store_alpha_vantage_data('market', cache_key, {'dates': dates})
            return dates
        except Exception as e:
            logger.error(f"Error fetching FOMC meetings: {e}")
            return []

    def fetch_alpha_vantage_earnings_call_and_store(self, ticker: str, quarter: str = None) -> dict:
        """Fetch earnings call transcript for a ticker from Alpha Vantage and store in MongoDB."""
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not api_key:
            logger.warning("ALPHAVANTAGE_API_KEY not set. Skipping earnings call transcript fetch.")
            return {"status": "no_api_key", "transcript": None}
        url = f"https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPT&symbol={ticker}"
        if quarter:
            url += f"&quarter={quarter}"
        url += f"&apikey={api_key}"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"Alpha Vantage earnings call transcript endpoint returned status {resp.status_code} for {ticker}")
                return {"status": "api_error", "transcript": None}
            data = resp.json()
            self.mongo_client.store_alpha_vantage_data(ticker, 'earnings_call_transcript', data)
            logger.info(f"Stored Alpha Vantage earnings call transcript for {ticker} in MongoDB.")
            return {"status": "ok", "transcript": data}
        except Exception as e:
            logger.error(f"Error fetching earnings call transcript for {ticker} from Alpha Vantage: {e}")
            return {"status": "exception", "transcript": None}

    async def analyze_earnings_call_sentiment(self, ticker: str) -> Dict[str, float]:
        """Analyze sentiment from earnings call transcripts."""
        try:
            collection = self.mongo_client.db['earnings_calls']
            latest_call = collection.find_one(
                {'ticker': ticker},
                sort=[('date', -1)]
            )
            if not latest_call:
                return {'sentiment': 0.0, 'confidence': 0.0}
            
            # Analyze transcript using FinBERT
            transcript = latest_call.get('transcript', '')
            if not transcript:
                return {'sentiment': 0.0, 'confidence': 0.0}
                
            # Use the sync version in a thread pool
            sentiment = await asyncio.to_thread(
                self._analyze_earnings_call_sentiment_sync,
                transcript
            )
            return sentiment
        except Exception as e:
            logger.error(f"Error analyzing earnings call sentiment for {ticker}: {str(e)}")
            return {'sentiment': 0.0, 'confidence': 0.0}

    def _analyze_earnings_call_sentiment_sync(self, transcript: str) -> Dict[str, float]:
        """Synchronous version of earnings call sentiment analysis."""
        try:
            # Use FinBERT for analysis if available
            if hasattr(self, 'finbert') and self.finbert is not None:
                sentiment = self.finbert(transcript[:512])[0]  # Truncate to avoid memory issues
                if 'label' in sentiment:
                    sentiment_score = _map_sentiment_label(sentiment['label'])
                    confidence = sentiment.get('score', 0.8)
                else:
                    sentiment_score = sentiment.get('score', 0.0)
                    confidence = sentiment.get('confidence', 0.8)
                return {
                    'sentiment': float(sentiment_score),
                    'confidence': float(confidence)
                }
            else:
                # Fallback to VADER
                sentiment = self.vader.polarity_scores(transcript[:512])
                return {
                    'sentiment': float(sentiment['compound']),
                    'confidence': 0.6  # Lower confidence for VADER
                }
        except Exception as e:
            logger.error(f"Error in sync earnings call analysis: {str(e)}")
            return {'sentiment': 0.0, 'confidence': 0.0}

    async def analyze_seekingalpha_comments_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Analyze sentiment from Seeking Alpha comments with sequential processing to prevent bot detection."""
        try:
            if not self.seeking_alpha_analyzer:
                return {
                    'sentiment': 0.0,
                    'volume': 0,
                    'confidence': 0.0,
                    'error': 'SeekingAlpha analyzer not available'
                }
            
            # Use global lock to ensure sequential processing across all tickers
            async with SEEKING_ALPHA_LOCK:
                logger.info(f"Sequential processing: Analyzing Seeking Alpha comments for {ticker}")
                
                # Add delay between requests to avoid bot detection
                await asyncio.sleep(2)
                
                result = await self.seeking_alpha_analyzer.analyze_comments_sentiment(ticker)
                
                # Add additional delay after processing
                await asyncio.sleep(3)
            
            if not result:
                return {
                    'sentiment': 0.0,
                    'volume': 0,
                    'confidence': 0.0,
                    'error': 'No comments analyzed'
                }
                
            return {
                'sentiment': result.get('seeking_alpha_comments_sentiment', 0.0),
                'volume': result.get('seeking_alpha_comments_volume', 0),
                'confidence': result.get('seeking_alpha_comments_confidence', 0.0),
                'engagement': result.get('seeking_alpha_comments_engagement', 0.0),
                'analyzed': result.get('seeking_alpha_comments_analyzed', 0),
                'sentiment_std': result.get('seeking_alpha_comments_sentiment_std', 0.0),
                'raw_comments': result.get('seeking_alpha_comments', [])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing Seeking Alpha comments sentiment for {ticker}: {str(e)}")
            return {
                'sentiment': 0.0,
                'volume': 0,
                'confidence': 0.0,
                'error': str(e)
            }
        

    async def get_finviz_sentiment(self, ticker: str):
        return await self.analyze_finviz_sentiment(ticker)

    async def get_sec_sentiment(self, ticker: str):
        return await self.analyze_sec_sentiment(ticker)

    async def get_yahoo_news_sentiment(self, ticker: str):
        return await self.analyze_yahoo_news_sentiment(ticker)

    async def get_marketaux_sentiment(self, ticker: str):
        return await self.analyze_marketaux_sentiment(ticker)

    async def get_rss_news_sentiment(self, ticker: str):
        return await self.analyze_rss_news_sentiment(ticker)

    async def get_reddit_sentiment(self, ticker: str):
        return await self.analyze_reddit_sentiment(ticker)

    async def get_fmp_sentiment(self, ticker: str):
        return await self.analyze_fmp_sentiment(ticker)

    async def get_finnhub_sentiment(self, ticker: str):
        return await self.analyze_finnhub_sentiment(ticker)

    async def get_seekingalpha_sentiment(self, ticker: str):
        return await self.analyze_seekingalpha_sentiment(ticker)

    async def get_seekingalpha_comments_sentiment(self, ticker: str):
        return await self.analyze_seekingalpha_comments_sentiment(ticker)

    async def get_alpha_earnings_call_sentiment(self, ticker: str):
        return await self.analyze_alpha_earnings_call_sentiment(ticker)

    async def get_alphavantage_news_sentiment(self, ticker: str):
        return await self.analyze_alphavantage_news_sentiment(ticker)

    async def analyze_sec_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Analyze sentiment from SEC filings using the SEC analyzer."""
        try:
            if not self.sec_analyzer:
                return {
                    'sentiment_score': 0.0,
                    'volume': 0,
                    'confidence': 0.0,
                    'error': 'SEC analyzer not available'
                }
            
            result = await self.sec_analyzer.analyze_filings_sentiment(ticker)
            if not result:
                return {
                    'sentiment_score': 0.0,
                    'volume': 0,
                    'confidence': 0.0,
                    'error': 'No SEC filings analyzed'
                }
            
            return {
                'sentiment_score': result.get('sec_filings_sentiment', 0.0),
                'volume': result.get('sec_filings_volume', 0),
                'confidence': result.get('sec_filings_confidence', 0.0),
                'analyzed': result.get('sec_filings_analyzed', 0),
                'sentiment_std': result.get('sec_filings_sentiment_std', 0.0),
                'raw_data': result.get('categorized_filings', {}),
                'processed_filings': result.get('processed_filings', []),
                'source': result.get('source', 'unknown'),
                'error': result.get('sec_filings_error', None)
            }
        except Exception as e:
            logger.error(f"Error analyzing SEC sentiment for {ticker}: {str(e)}")
            return {
                'sentiment_score': 0.0,
                'volume': 0,
                'confidence': 0.0,
                'error': str(e)
            }

    async def analyze_yahoo_news_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Analyze sentiment from Yahoo News."""
        try:
            # Run in threadpool since this is a blocking operation
            return await run_in_threadpool(self._analyze_yahoo_news_sentiment_sync, ticker)
        except Exception as e:
            logger.warning(f"Yahoo News sentiment analysis failed for {ticker}: {e}")
            return {
                'sentiment_score': 0.0,
                'volume': 0,
                'confidence': 0.0,
                'error': str(e)
            }

    async def analyze_fmp_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Analyze sentiment from FMP data sources using consolidated manager."""
        try:
            # Get all FMP data in one consolidated call to avoid duplicate API requests
            fmp_data = await self.fmp_manager.get_all_fmp_data(ticker)
            
            if not fmp_data:
                return {
                    'sentiment_score': 0.0,
                    'volume': 0,
                    'confidence': 0.0,
                    'error': 'No FMP data available'
                }
            
            # Calculate sentiment from different components
            sentiments = []
            weights = []
            total_volume = 0
            
            # Process estimates
            estimates = fmp_data.get('estimates', [])
            if estimates:
                est = estimates[0]
                eps_avg = est.get('estimatedEpsAvg', 0.0)
                norm_eps = max(-1, min(1, (eps_avg - 5) / 5))
                sentiments.append(norm_eps)
                weights.append(0.3)
                total_volume += est.get('numberAnalystEstimatedEps', 1)
            
            # Process ratings
            ratings = fmp_data.get('ratings', [])
            if ratings:
                rating = ratings[0]
                overall = rating.get('ratingScore', 0)
                norm_score = (overall - 3) / 2
                sentiments.append(norm_score)
                weights.append(0.3)
                total_volume += 1
            
            # Process price targets
            price_targets = fmp_data.get('price_targets', [])
            if price_targets:
                pt = price_targets[0]
                avg_target = pt.get('lastYearAvgPriceTarget', 0.0)
                norm_pt = max(-1, min(1, (avg_target - 200) / 100))
                sentiments.append(norm_pt)
                weights.append(0.25)
                total_volume += pt.get('lastYearCount', 1)
            
            # Process grades
            grades = fmp_data.get('grades', [])
            if grades:
                gs = grades[0]
                consensus = gs.get('gradingCompany', '').lower()
                mapping = {'strong buy': 1.0, 'buy': 0.7, 'hold': 0.0, 'sell': -0.7, 'strong sell': -1.0}
                norm_score = mapping.get(consensus, 0.0)
                sentiments.append(norm_score)
                weights.append(0.15)
                total_volume += 1
            
            # Calculate weighted sentiment
            if sentiments:
                weighted_sentiment = sum(s * w for s, w in zip(sentiments, weights)) / sum(weights)
                confidence = sum(weights) / 4  # Normalize confidence
            else:
                weighted_sentiment = 0.0
                confidence = 0.0
            
            logger.info(f"FMP consolidated sentiment for {ticker}: {weighted_sentiment:.3f} from {len(sentiments)} sources")
            
            return {
                'sentiment_score': weighted_sentiment,
                'volume': total_volume,
                'confidence': confidence,
                'raw_data': fmp_data,  # This will include calendar data for economic calendar
                'components_analyzed': len(sentiments)
            }
        except Exception as e:
            logger.error(f"Error analyzing FMP sentiment for {ticker}: {str(e)}")
            return {
                'sentiment_score': 0.0,
                'volume': 0,
                'confidence': 0.0,
                'error': str(e)
            }

    async def analyze_finnhub_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Analyze sentiment from Finnhub data sources."""
        try:
            # Combine insider transactions and recommendation trends
            insider_sentiment = await self.analyze_finnhub_insider_sentiment(ticker)
            recommendation_sentiment = await self.analyze_finnhub_recommendation_trends(ticker)
            
            # Calculate weighted average sentiment
            sentiments = []
            weights = []
            volume = 0
            
            if insider_sentiment.get('sentiment', 0) != 0:
                sentiments.append(insider_sentiment['sentiment'])
                weights.append(0.4)
                volume += insider_sentiment.get('volume', 0)
            
            if recommendation_sentiment.get('finnhub_recommendation_sentiment', 0) != 0:
                sentiments.append(recommendation_sentiment['finnhub_recommendation_sentiment'])
                weights.append(0.6)
                volume += recommendation_sentiment.get('finnhub_recommendation_volume', 0)
            
            if sentiments:
                weighted_sentiment = sum(s * w for s, w in zip(sentiments, weights)) / sum(weights)
                confidence = sum(weights) / 2  # Normalize confidence
            else:
                weighted_sentiment = 0.0
                confidence = 0.0
            
            return {
                'sentiment_score': weighted_sentiment,
                'volume': volume,
                'confidence': confidence,
                'raw_data': {
                    'insider': insider_sentiment,
                    'recommendations': recommendation_sentiment
                },
                'components': {
                    'insider': insider_sentiment,
                    'recommendations': recommendation_sentiment
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing Finnhub sentiment for {ticker}: {str(e)}")
            return {
                'sentiment_score': 0.0,
                'volume': 0,
                'confidence': 0.0,
                'error': str(e)
            }

    async def analyze_alpha_earnings_call_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Analyze sentiment from Alpha Vantage earnings call transcripts."""
        try:
            quarter = self.get_latest_quarter()
            logger.info(f"Fetching earnings call for {ticker} quarter: {quarter}")
            
            # Try current quarter first
            earnings_call = await self.fetch_alpha_vantage_earnings_call(ticker, quarter)
            
            # If no data for current quarter, try previous quarter
            if not earnings_call or 'error' in earnings_call:
                logger.info(f"No data for {quarter}, trying previous quarter")
                # Try previous quarter
                year = int(quarter[:4])
                q = int(quarter[5:])
                if q == 1:
                    prev_quarter = f"{year-1}Q4"
                else:
                    prev_quarter = f"{year}Q{q-1}"
                
                earnings_call = await self.fetch_alpha_vantage_earnings_call(ticker, prev_quarter)
                logger.info(f"Tried previous quarter {prev_quarter}")
            
            if not earnings_call or 'error' in earnings_call:
                logger.warning(f"No earnings call data available for {ticker} in recent quarters")
                return {
                    'sentiment_score': 0.0,
                    'volume': 0,
                    'confidence': 0.0,
                    'error': f'No earnings call data available for {ticker} in quarters {quarter} or previous'
                }
            
            # Check if we have transcript data
            transcript = None
            if isinstance(earnings_call, dict):
                transcript = earnings_call.get('transcript', earnings_call.get('data', {}))
            
            if not transcript:
                return {
                    'sentiment_score': 0.0,
                    'volume': 0,
                    'confidence': 0.0,
                    'error': 'No transcript available in earnings call data'
                }
            
            # Check if transcript is structured (array of speaker segments with pre-calculated sentiment)
            if isinstance(transcript, list) and len(transcript) > 0:
                # Use pre-calculated sentiment scores from structured transcript
                sentiment_scores = []
                for segment in transcript:
                    if isinstance(segment, dict) and 'sentiment' in segment:
                        try:
                            sentiment = float(segment['sentiment'])
                            sentiment_scores.append(sentiment)
                        except (ValueError, TypeError):
                            continue
                
                if sentiment_scores:
                    sentiment_score = sum(sentiment_scores) / len(sentiment_scores)
                    volume = len(sentiment_scores)
                    logger.info(f"Alpha Vantage earnings call: {volume} segments, avg sentiment: {sentiment_score:.3f}")
                else:
                    # Fall back to text analysis
                    text_content = ' '.join([seg.get('content', '') for seg in transcript if isinstance(seg, dict)])
                    sentiment_score = self._analyze_sentiment(text_content) if text_content else 0.0
                    volume = 1
            else:
                # Extract text content for sentiment analysis
                text_content = str(transcript)
                if len(text_content) < 100:  # Too short to be meaningful
                    return {
                        'sentiment_score': 0.0,
                        'volume': 0,
                        'confidence': 0.0,
                        'error': 'Transcript too short'
                    }
                
                # Analyze sentiment
                sentiment_score = self._analyze_sentiment(text_content)
                volume = 1
            
            return {
                'sentiment_score': sentiment_score,
                'volume': volume,
                'confidence': 0.8,  # High confidence in earnings call data
                'transcript_segments': len(transcript) if isinstance(transcript, list) else 1,
                'quarter_analyzed': quarter
            }
        except Exception as e:
            logger.error(f"Error analyzing Alpha Vantage earnings call sentiment for {ticker}: {str(e)}")
            return {
                'sentiment_score': 0.0,
                'volume': 0,
                'confidence': 0.0,
                'error': str(e)
            }

    async def analyze_alphavantage_news_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Analyze sentiment from Alpha Vantage NEWS_SENTIMENT API."""
        try:
            api_key = os.getenv("ALPHAVANTAGE_API_KEY")
            if not api_key:
                logger.warning("ALPHAVANTAGE_API_KEY not set. Skipping Alpha Vantage news sentiment.")
                return {
                    'alphavantage_sentiment': 0.0,
                    'alphavantage_volume': 0,
                    'alphavantage_confidence': 0.0,
                    'alphavantage_error': 'API key not set'
                }
            
            # Alpha Vantage NEWS_SENTIMENT API
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                'sort': 'LATEST',
                'limit': '50',  # Get up to 50 recent articles
                'apikey': api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status != 200:
                        logger.warning(f"Alpha Vantage NEWS_SENTIMENT returned status {response.status}")
                        return {
                            'alphavantage_sentiment': 0.0,
                            'alphavantage_volume': 0,
                            'alphavantage_confidence': 0.0,
                            'alphavantage_error': f'API returned status {response.status}'
                        }
                    
                    data = await response.json()
                    
                    # Check for API errors
                    if 'Error Message' in data:
                        logger.error(f"Alpha Vantage NEWS_SENTIMENT error: {data['Error Message']}")
                        return {
                            'alphavantage_sentiment': 0.0,
                            'alphavantage_volume': 0,
                            'alphavantage_confidence': 0.0,
                            'alphavantage_error': data['Error Message']
                        }
                    
                    # Check for rate limit
                    if 'Note' in data:
                        logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                        return {
                            'alphavantage_sentiment': 0.0,
                            'alphavantage_volume': 0,
                            'alphavantage_confidence': 0.0,
                            'alphavantage_error': 'Rate limit exceeded'
                        }
                    
                    # Process news sentiment data
                    feed = data.get('feed', [])
                    if not feed:
                        logger.info(f"No news sentiment data from Alpha Vantage for {ticker}")
                        return {
                            'alphavantage_sentiment': 0.0,
                            'alphavantage_volume': 0,
                            'alphavantage_confidence': 0.0,
                            'alphavantage_note': 'No news articles found'
                        }
                    
                    sentiment_scores = []
                    ticker_relevance_scores = []
                    raw_data = []
                    
                    for article in feed:
                        try:
                            # Get overall sentiment score
                            overall_sentiment = article.get('overall_sentiment_score', 0.0)
                            
                            # Get ticker-specific sentiment
                            ticker_sentiments = article.get('ticker_sentiment', [])
                            ticker_sentiment = 0.0
                            ticker_relevance = 0.0
                            
                            for ts in ticker_sentiments:
                                if ts.get('ticker', '').upper() == ticker.upper():
                                    ticker_sentiment = float(ts.get('sentiment_score', 0.0))
                                    ticker_relevance = float(ts.get('relevance_score', 0.0))
                                    break
                            
                            # Use ticker-specific sentiment if available, otherwise overall
                            final_sentiment = ticker_sentiment if ticker_sentiment != 0.0 else overall_sentiment
                            sentiment_scores.append(final_sentiment)
                            ticker_relevance_scores.append(ticker_relevance)
                            
                            raw_data.append({
                                'title': article.get('title', ''),
                                'url': article.get('url', ''),
                                'time_published': article.get('time_published', ''),
                                'source': article.get('source', ''),
                                'overall_sentiment_score': overall_sentiment,
                                'ticker_sentiment_score': ticker_sentiment,
                                'relevance_score': ticker_relevance
                            })
                            
                        except Exception as e:
                            logger.warning(f"Error processing Alpha Vantage article: {e}")
                            continue
                    
                    if not sentiment_scores:
                        return {
                            'alphavantage_sentiment': 0.0,
                            'alphavantage_volume': 0,
                            'alphavantage_confidence': 0.0,
                            'alphavantage_raw_data': raw_data,
                            'alphavantage_note': 'No valid sentiment scores found'
                        }
                    
                    # Calculate weighted average sentiment (weighted by relevance)
                    if ticker_relevance_scores and any(r > 0 for r in ticker_relevance_scores):
                        # Weight by relevance scores
                        weighted_sentiment = sum(s * r for s, r in zip(sentiment_scores, ticker_relevance_scores))
                        total_weight = sum(ticker_relevance_scores)
                        avg_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0.0
                    else:
                        # Simple average if no relevance scores
                        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                    
                    volume = len(sentiment_scores)
                    confidence = min(volume / 20, 1.0)  # Higher confidence with more articles
                    
                    logger.info(f"Alpha Vantage news sentiment for {ticker}: {avg_sentiment:.3f} ({volume} articles)")
                    
                    return {
                        'alphavantage_sentiment': avg_sentiment,
                        'alphavantage_volume': volume,
                        'alphavantage_confidence': confidence,
                        'alphavantage_raw_data': raw_data[:10],  # Store top 10 articles
                        'alphavantage_error': None
                    }
            
        except Exception as e:
            logger.error(f"Error analyzing Alpha Vantage news sentiment for {ticker}: {str(e)}")
            return {
                'alphavantage_sentiment': 0.0,
                'alphavantage_volume': 0,
                'alphavantage_confidence': 0.0,
                'alphavantage_error': str(e)
            }

    async def analyze_short_interest_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Analyze sentiment from short interest data with sequential processing."""
        try:
            from .short_interest import ShortInterestAnalyzer, SHORT_INTEREST_LOCK
            if not hasattr(self, 'short_interest_analyzer') or self.short_interest_analyzer is None:
                try:
                    self.short_interest_analyzer = ShortInterestAnalyzer(self.mongo_client)
                except Exception as e:
                    logger.warning(f"Failed to initialize short interest analyzer: {e}")
                    self.short_interest_analyzer = None
            
            if not self.short_interest_analyzer:
                return {
                    'short_interest_sentiment': 0.0,
                    'short_interest_volume': 0,
                    'short_interest_confidence': 0.0,
                    'short_interest_error': 'Short interest analyzer not available'
                }
            
            # Use global lock to ensure sequential processing across all tickers
            async with SHORT_INTEREST_LOCK:
                logger.info(f"Sequential processing: Fetching short interest data for {ticker}")
                
                # Add delay between requests to avoid bot detection
                await asyncio.sleep(3)
                
                # Get recent short interest data
                recent_data = await self.short_interest_analyzer.fetch_short_interest(ticker)
                
                # Add additional delay after processing
                await asyncio.sleep(2)
            
            if not recent_data:
                return {
                    'short_interest_sentiment': 0.0,
                    'short_interest_volume': 0,
                    'short_interest_confidence': 0.0,
                    'short_interest_error': 'No short interest data available'
                }
            
            # Calculate sentiment based on short interest trends
            sentiment_score = 0.0
            volume = len(recent_data)
            
            if len(recent_data) >= 2:
                # Compare latest vs previous
                latest = recent_data[0]
                previous = recent_data[1]
                
                latest_si = latest.get('short_interest', 0)
                previous_si = previous.get('short_interest', 0)
                
                if previous_si > 0:
                    change_pct = (latest_si - previous_si) / previous_si
                    # Increasing short interest is bearish (negative sentiment)
                    sentiment_score = -min(abs(change_pct), 0.5) if change_pct > 0 else min(abs(change_pct), 0.5)
            
            confidence = min(volume / 5, 1.0)  # Higher confidence with more data points
            
            return {
                'short_interest_sentiment': sentiment_score,
                'short_interest_volume': volume,
                'short_interest_confidence': confidence,
                'short_interest_data': recent_data[:3]  # Include latest 3 data points
            }
        except Exception as e:
            logger.error(f"Error analyzing short interest sentiment for {ticker}: {str(e)}")
            return {
                'short_interest_sentiment': 0.0,
                'short_interest_volume': 0,
                'short_interest_confidence': 0.0,
                'short_interest_error': str(e)
            }

    async def integrate_economic_events_sentiment(self, sentiment_dict: Dict, ticker: str, mongo_client=None) -> Dict:
        """
        Integrate economic event data into sentiment analysis with improved caching to prevent multiple web scraping.
        Economic calendar events are global (not ticker-specific) so they only need to be fetched once per day.
        """
        try:
            # Use global lock to ensure only one economic calendar fetch at a time
            async with ECONOMIC_CALENDAR_LOCK:
                # Check if we have cached global economic data for today
                global ECONOMIC_DATA_CACHE
                today_key = datetime.now().strftime('%Y-%m-%d')
                
                # Check if cache is still valid
                if (ECONOMIC_DATA_CACHE['data'] is not None and 
                    ECONOMIC_DATA_CACHE['timestamp'] is not None and
                    (datetime.now() - ECONOMIC_DATA_CACHE['timestamp']) < ECONOMIC_DATA_CACHE['cache_duration']):
                    
                    logger.info(f"Using cached global economic calendar data for {ticker}")
                    
                    # Apply cached economic events to this ticker
                    cached_data = ECONOMIC_DATA_CACHE['data']
                    economic_keys = ['economic_event_sentiment', 'economic_event_volume', 
                                   'economic_event_confidence', 'economic_event_volatility',
                                   'economic_event_features']
                    
                    for key in economic_keys:
                        if key in cached_data:
                            sentiment_dict[key] = cached_data[key]
                    
                    return sentiment_dict
                
                # Cache miss or expired, fetch new data (THIS WILL ONLY HAPPEN ONCE PER 6 HOURS)
                logger.info(f"Fetching fresh economic calendar data (will be shared across all tickers)")
                
                # Use thread pool for the sync function from economic_calendar.py
                from .economic_calendar import integrate_economic_events_sentiment
                updated_dict = await run_in_threadpool(integrate_economic_events_sentiment, sentiment_dict, ticker, mongo_client)
                
                # Cache the global economic events data (not ticker-specific)
                economic_keys = ['economic_event_sentiment', 'economic_event_volume', 
                               'economic_event_confidence', 'economic_event_volatility',
                               'economic_event_features']
                economic_data = {k: updated_dict.get(k, 0.0) for k in economic_keys if k in updated_dict}
                
                # Update global cache
                ECONOMIC_DATA_CACHE['data'] = economic_data
                ECONOMIC_DATA_CACHE['timestamp'] = datetime.now()
                
                logger.info(f"Cached global economic calendar data for all tickers (valid for 6 hours)")
                
                return updated_dict
            
        except Exception as e:
            logger.error(f"Error integrating economic events sentiment for {ticker}: {e}")
            return sentiment_dict

    def _analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using VADER or FinBERT if available.
        Returns a sentiment score between -1 and 1.
        """
        try:
            if not text or not text.strip():
                return 0.0
            
            # Truncate text to avoid memory issues
            text = text[:2048]
            
            # Use FinBERT if available, otherwise fall back to VADER
            if hasattr(self, 'finbert') and self.finbert is not None:
                try:
                    result = self.finbert(text)[0]
                    if 'label' in result:
                        return _map_sentiment_label(result['label'])
                    else:
                        return result.get('score', 0.0)
                except Exception as e:
                    logger.warning(f"FinBERT analysis failed, falling back to VADER: {e}")
            
            # Use VADER sentiment analyzer
            if hasattr(self, 'vader') and self.vader is not None:
                sentiment = self.vader.polarity_scores(text)
                return sentiment.get('compound', 0.0)
            else:
                # Initialize VADER if not available
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                vader = SentimentIntensityAnalyzer()
                sentiment = vader.polarity_scores(text)
                return sentiment.get('compound', 0.0)
                
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0

    async def fetch_finnhub_insider_sentiment_direct(self, ticker: str) -> dict:
        """
        Fetch insider sentiment directly from Finnhub's insider sentiment API.
        This provides pre-calculated MSPR (Monthly Share Purchase Ratio) values.
        """
        api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key:
            logger.warning("FINNHUB_API_KEY not set. Skipping Finnhub insider sentiment.")
            return {}
        
        try:
            # Get data for the last 12 months
            from_date = (datetime.utcnow() - timedelta(days=365)).strftime('%Y-%m-%d')
            to_date = datetime.utcnow().strftime('%Y-%m-%d')
            
            import finnhub
            finnhub_client = finnhub.Client(api_key=api_key)
            
            # Use Finnhub's insider sentiment API
            data = finnhub_client.stock_insider_sentiment(ticker, from_date, to_date)
            
            if data and 'data' in data and data['data']:
                logger.info(f"Finnhub insider sentiment API returned {len(data['data'])} data points for {ticker}")
                return data
            else:
                logger.warning(f"No insider sentiment data from Finnhub API for {ticker}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching Finnhub insider sentiment API for {ticker}: {e}")
            return {}

    async def analyze_finnhub_insider_sentiment_enhanced(self, ticker: str) -> dict:
        """
        Enhanced insider sentiment analysis using both the direct API and transaction parsing.
        """
        try:
            # Try the direct insider sentiment API first
            direct_sentiment = await self.fetch_finnhub_insider_sentiment_direct(ticker)
            
            if direct_sentiment and 'data' in direct_sentiment and direct_sentiment['data']:
                # Use the pre-calculated MSPR values from Finnhub
                sentiment_data = direct_sentiment['data']
                
                # Calculate average sentiment from recent months
                total_mspr = sum(item.get('mspr', 0) for item in sentiment_data)
                avg_mspr = total_mspr / len(sentiment_data) if sentiment_data else 0
                
                # Normalize MSPR to [-1, 1] range (MSPR typically ranges from -100 to 100)
                normalized_sentiment = max(-1.0, min(1.0, avg_mspr / 100))
                
                # Calculate volume as sum of absolute changes
                total_volume = sum(abs(item.get('change', 0)) for item in sentiment_data)
                
                logger.info(f"Finnhub insider sentiment API for {ticker}: MSPR={avg_mspr:.2f}, normalized={normalized_sentiment:.3f}")
                
                return {
                    'source': 'finnhub_insider_api',
                    'sentiment': normalized_sentiment,
                    'volume': int(total_volume),
                    'confidence': 0.9,  # High confidence for official API
                    'mspr_average': avg_mspr,
                    'data_points': len(sentiment_data),
                    'raw_data': sentiment_data
                }
            else:
                # Fallback to transaction parsing method
                logger.info(f"Finnhub insider sentiment API had no data for {ticker}, falling back to transaction parsing")
                return await self.analyze_finnhub_insider_sentiment(ticker)
                
        except Exception as e:
            logger.error(f"Error in enhanced Finnhub insider sentiment for {ticker}: {e}")
            # Fallback to original method
            return await self.analyze_finnhub_insider_sentiment(ticker)


def get_previous_trading_day(date_str):
    nyse = mcal.get_calendar('NYSE')
    date = pd.to_datetime(date_str)
    # Get the last valid trading day before or on the given date
    schedule = nyse.schedule(start_date=date - pd.Timedelta(days=7), end_date=date)
    if schedule.empty:
        return None
    last_trading_day = schedule.index[-1]
    return last_trading_day.strftime('%Y-%m-%d')

def fetch_and_store_options_sentiment(mongo_client, ticker, date):
    api_key = os.getenv('ALPHAVANTAGE_API_KEY')
    # Only one request for options data: no date param (most recent session)
    options_url = f'https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol={ticker}&apikey={api_key}'
    try:
        r = requests.get(options_url, timeout=10)
        options_data = r.json()
    except Exception as e:
        logger.warning(f"Options API error for {ticker}: {e}")
        options_data = {}

    # Store only if options_data has non-empty 'data'
    if options_data.get("data"):
        doc = {
            'ticker': ticker,
            'date': date,
            'options_data': options_data,
            'news_data': news_data,
            'fetched_at': datetime.utcnow()
        }
        mongo_client.db['options_sentiment'].update_one(
            {'ticker': ticker, 'date': date},
            {'$set': doc},
            upsert=True
        )
        logger.info(f"Stored options/news sentiment for {ticker} {date}")
    else:
        logger.info(f"No options data for {ticker}, skipping storage.")

class FMPAPIManager:
    """Centralized FMP API manager using correct stable endpoints."""
    
    def __init__(self, mongo_client: MongoDBClient):
        self.api_key = os.getenv("FMP_API_KEY")
        self.mongo_client = mongo_client
        self.base_url = "https://financialmodelingprep.com/stable"  # ✅ CORRECT STABLE BASE URL
        self.cache_duration = 3600  # 1 hour cache
        
        # Track API status
        self.api_working = True
        self.last_error_time = None
        self.error_cooldown = 300  # 5 minutes cooldown after errors
        
        # Free tier supported tickers - only these will work
        self.FREE_TIER_TICKERS = {
            'AAPL', 'TSLA', 'AMZN', 'MSFT', 'NVDA', 'GOOGL', 'META', 'NFLX', 'JPM', 'V', 
            'BAC', 'AMD', 'PYPL', 'DIS', 'T', 'PFE', 'COST', 'INTC', 'KO', 'TGT', 'NKE', 
            'SPY', 'BA', 'BABA', 'XOM', 'WMT', 'GE', 'CSCO', 'VZ', 'JNJ', 'CVX', 'PLTR', 
            'SQ', 'SHOP', 'SBUX', 'SOFI', 'HOOD', 'RBLX', 'SNAP', 'UBER', 'FDX', 'ABBV', 
            'ETSY', 'MRNA', 'LMT', 'GM', 'F', 'RIVN', 'LCID', 'CCL', 'DAL', 'UAL', 'AAL', 
            'TSM', 'SONY', 'ET', 'NOK', 'MRO', 'COIN', 'SIRI', 'RIOT', 'CPRX', 'VWO', 
            'SPYG', 'ROKU', 'VIAC', 'ATVI', 'BIDU', 'DOCU', 'ZM', 'PINS', 'TLRY', 'WBA', 
            'MGM', 'NIO', 'C', 'GS', 'WFC', 'ADBE', 'PEP', 'UNH', 'CARR', 'FUBO', 'HCA', 
            'TWTR', 'BILI', 'RKT'
        }
        
    def _is_ticker_supported(self, ticker: str) -> bool:
        """Check if ticker is supported on free tier."""
        return ticker.upper() in self.FREE_TIER_TICKERS
        
    async def _make_fmp_request(self, endpoint: str, ticker: str = None, **params) -> dict:
        """Make centralized FMP API request with caching and improved error handling."""
        if not self.api_key:
            logger.warning("FMP_API_KEY not set. Skipping FMP request.")
            return {}
        
        # Check if ticker is supported on free tier
        if ticker and not self._is_ticker_supported(ticker):
            logger.warning(f"❌ Ticker {ticker} not supported on FMP free tier")
            return {}
        
        # Check if we're in cooldown period
        if (not self.api_working and self.last_error_time and 
            (datetime.utcnow() - self.last_error_time).total_seconds() < self.error_cooldown):
            logger.warning(f"FMP API in cooldown period, skipping {endpoint}")
            return {}
            
        # Create cache key
        cache_key = f"fmp_{endpoint.replace('/', '_')}_{ticker or 'global'}_{hash(str(sorted(params.items())))}"
        
        # Check cache first
        try:
            cached = self.mongo_client.get_alpha_vantage_data(ticker or 'global', cache_key)
            if cached and 'timestamp' in cached:
                age = (datetime.utcnow() - cached['timestamp']).total_seconds()
                if age < self.cache_duration:
                    logger.info(f"Using cached FMP data for {endpoint} - {ticker or 'global'}")
                    return cached.get('data', {})
        except Exception as e:
            logger.warning(f"Error checking FMP cache: {e}")
        
        # Make API request
        try:
            url = f"{self.base_url}/{endpoint}"
            
            # Add API key to params
            all_params = dict(params)
            all_params['apikey'] = self.api_key
            
            # Log the exact URL being called for debugging
            logger.info(f"🔗 FMP STABLE API Call: {url} with params: {all_params}")
                
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=all_params, timeout=15) as resp:
                    
                    if resp.status == 403:
                        logger.error(f"❌ FMP 403 Forbidden: {endpoint} - Check API subscription level or ticker support")
                        self.api_working = False
                        self.last_error_time = datetime.utcnow()
                        return {}
                    elif resp.status == 429:
                        logger.warning(f"⚠️ FMP 429 Rate Limited: {endpoint}")
                        self.api_working = False
                        self.last_error_time = datetime.utcnow()
                        return {}
                    elif resp.status != 200:
                        logger.warning(f"⚠️ FMP {endpoint} returned status {resp.status}")
                        return {}
                    
                    # Try to parse JSON response
                    try:
                        data = await resp.json()
                    except Exception as e:
                        logger.error(f"Failed to parse FMP response as JSON: {e}")
                        return {}
                    
                    # Check for FMP error messages
                    if isinstance(data, dict):
                        if 'Error Message' in data:
                            logger.error(f"❌ FMP API error: {data['Error Message']}")
                            if 'limit' in data['Error Message'].lower() or 'subscription' in data['Error Message'].lower():
                                self.api_working = False
                                self.last_error_time = datetime.utcnow()
                            return {}
                        elif 'error' in data:
                            logger.error(f"❌ FMP API error: {data['error']}")
                            return {}
                    
                    # Reset API status on successful request
                    if not self.api_working:
                        logger.info("✅ FMP API is working again")
                        self.api_working = True
                        self.last_error_time = None
                    
                    logger.info(f"✅ FMP STABLE API Success: {endpoint} returned {len(data) if isinstance(data, list) else 'object'}")
                    
                    # Store in cache
                    try:
                        self.mongo_client.store_alpha_vantage_data(
                            ticker or 'global', 
                            cache_key, 
                            {'data': data, 'timestamp': datetime.utcnow()}
                        )
                    except Exception as e:
                        logger.warning(f"Failed to cache FMP data: {e}")
                    
                    return data
                    
        except asyncio.TimeoutError:
            logger.error(f"⏰ FMP {endpoint} request timed out")
            return {}
        except Exception as e:
            logger.error(f"💥 Error fetching FMP {endpoint}: {e}")
            return {}
    
    async def get_dividends_company(self, ticker: str) -> dict:
        """Get company dividends using correct stable endpoint."""
        # ✅ CORRECT STABLE ENDPOINT: https://financialmodelingprep.com/stable/dividends?symbol=AAPL
        data = await self._make_fmp_request("dividends", ticker, symbol=ticker)
        return {"company_dividends": data if isinstance(data, list) else []}
    
    async def get_dividends_calendar(self, from_date: str = None, to_date: str = None) -> dict:
        """Get dividends calendar using correct stable endpoint."""
        # ✅ CORRECT STABLE ENDPOINT: https://financialmodelingprep.com/stable/dividends-calendar
        if not from_date:
            from_date = datetime.utcnow().strftime("%Y-%m-%d")
        if not to_date:
            to_date = (datetime.utcnow() + timedelta(days=30)).strftime("%Y-%m-%d")  # Free tier: 1 month max
        
        params = {"from": from_date, "to": to_date}
        data = await self._make_fmp_request("dividends-calendar", None, **params)
        return {"dividends_calendar": data if isinstance(data, list) else []}
    
    async def get_earnings_company(self, ticker: str) -> dict:
        """Get company earnings using correct stable endpoint."""
        # ✅ CORRECT STABLE ENDPOINT: https://financialmodelingprep.com/stable/earnings?symbol=AAPL&limit=5
        data = await self._make_fmp_request("earnings", ticker, symbol=ticker, limit=5)  # Free tier: max 5
        return {"company_earnings": data if isinstance(data, list) else []}
    
    async def get_earnings_calendar(self, from_date: str = None, to_date: str = None) -> dict:
        """Get earnings calendar using correct stable endpoint."""
        # ✅ CORRECT STABLE ENDPOINT: https://financialmodelingprep.com/stable/earnings-calendar
        if not from_date:
            from_date = datetime.utcnow().strftime("%Y-%m-%d")
        if not to_date:
            to_date = (datetime.utcnow() + timedelta(days=30)).strftime("%Y-%m-%d")  # Free tier: 1 month max
        
        params = {"from": from_date, "to": to_date}
        data = await self._make_fmp_request("earnings-calendar", None, **params)
        return {"earnings_calendar": data if isinstance(data, list) else []}
    
    async def get_analyst_estimates(self, ticker: str, period: str = "annual") -> dict:
        """Get analyst estimates using correct stable endpoint."""
        # ✅ CORRECT STABLE ENDPOINT: https://financialmodelingprep.com/stable/analyst-estimates?symbol=AAPL&period=annual&page=0&limit=10
        data = await self._make_fmp_request("analyst-estimates", ticker, symbol=ticker, period=period, page=0, limit=10)
        return {"analyst_estimates": data if isinstance(data, list) else []}
    
    async def get_ratings_snapshot(self, ticker: str) -> dict:
        """Get ratings snapshot using correct stable endpoint."""
        # ✅ CORRECT STABLE ENDPOINT: https://financialmodelingprep.com/stable/ratings-snapshot?symbol=AAPL
        data = await self._make_fmp_request("ratings-snapshot", ticker, symbol=ticker)
        return {"ratings_snapshot": data if isinstance(data, list) else []}
    
    async def get_price_target_summary(self, ticker: str) -> dict:
        """Get price target summary using correct stable endpoint."""
        # ✅ CORRECT STABLE ENDPOINT: https://financialmodelingprep.com/stable/price-target-summary?symbol=AAPL
        data = await self._make_fmp_request("price-target-summary", ticker, symbol=ticker)
        return {"price_target_summary": data if isinstance(data, list) else []}
    
    async def get_price_target_consensus(self, ticker: str) -> dict:
        """Get price target consensus using correct stable endpoint."""
        # ✅ CORRECT STABLE ENDPOINT: https://financialmodelingprep.com/stable/price-target-consensus?symbol=AAPL
        data = await self._make_fmp_request("price-target-consensus", ticker, symbol=ticker)
        return {"price_target_consensus": data if isinstance(data, list) else []}
    
    async def get_all_fmp_data(self, ticker: str) -> dict:
        """Get all FMP data for a ticker using correct stable endpoints."""
        logger.info(f"🔄 Fetching consolidated FMP STABLE data for {ticker}")
        
        # Check if ticker is supported
        if not self._is_ticker_supported(ticker):
            logger.warning(f"❌ Ticker {ticker} not supported on FMP free tier")
            return {"error": f"Ticker {ticker} not supported on free tier"}
        
        # Check if API is working before making multiple calls
        if not self.api_working:
            logger.warning(f"⚠️ FMP API not working, returning empty data for {ticker}")
            return {}
        
        # Make all calls concurrently to avoid sequential delays
        import asyncio
        results = await asyncio.gather(
            self.get_dividends_company(ticker),
            self.get_dividends_calendar(),
            self.get_earnings_company(ticker),
            self.get_earnings_calendar(),
            self.get_analyst_estimates(ticker),
            self.get_ratings_snapshot(ticker),
            self.get_price_target_summary(ticker),
            self.get_price_target_consensus(ticker),
            return_exceptions=True
        )
        
        # Combine all results
        consolidated_data = {}
        api_call_names = [
            'company_dividends', 'dividends_calendar', 'company_earnings', 'earnings_calendar',
            'analyst_estimates', 'ratings_snapshot', 'price_target_summary', 'price_target_consensus'
        ]
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ FMP STABLE {api_call_names[i]} failed: {result}")
                consolidated_data[api_call_names[i]] = []
            else:
                consolidated_data.update(result)
                logger.info(f"✅ FMP STABLE {api_call_names[i]} success")
        
        logger.info(f"🎯 FMP STABLE consolidated data for {ticker}: {list(consolidated_data.keys())}")
        return consolidated_data

# Add FMP manager to SentimentAnalyzer
# ... existing code ...

if __name__ == "__main__":
    import argparse
    import os
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description="Fetch/store options/news sentiment for tickers.")
    parser.add_argument('--ticker', type=str, default=None, help='Single ticker (default: all)')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    args = parser.parse_args()
    from ml_backend.utils.mongodb import MongoDBClient
    from ml_backend.config.constants import TOP_100_TICKERS
    mongo_uri = os.getenv("MONGODB_URI")
    mongo_client = MongoDBClient(mongo_uri)
    tickers = [args.ticker] if args.ticker else TOP_100_TICKERS
    dates = pd.date_range(args.start, args.end)
    for ticker in tickers:
        for d in dates:
            fetch_and_store_options_sentiment(mongo_client, ticker, d.strftime('%Y-%m-%d'))
    mongo_client.close() 