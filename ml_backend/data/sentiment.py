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
    TOP_100_TICKERS
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
        
        """
        Initialize the SentimentAnalyzer class.
        
        Args:
            mongo_client: MongoDB client instance
            calendar_fetcher: Optional shared EconomicCalendar instance
        """
        self.mongo_client = mongo_client
        self.api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        self.vader = SentimentIntensityAnalyzer()
        self.recent_cutoff = get_cutoff_datetime()
        self.news_fetcher = None
        
        # Initialize FMP API manager
        self.fmp_manager = FMPAPIManager(mongo_client)
        
        self.max_retries = RETRY_CONFIG["max_retries"]
        self.base_delay = RETRY_CONFIG["base_delay"]
        self.max_delay = RETRY_CONFIG["max_delay"]
        
        # Initialize sentiment analysis tools
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize SEC filings analyzer
        try:
            self.sec_analyzer = SECFilingsAnalyzer(mongo_client)
            logger.info("SEC analyzer initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize SEC analyzer: {e}")
            self.sec_analyzer = None

        # Initialize Seeking Alpha analyzer
        try:
            self.seeking_alpha_analyzer = SeekingAlphaAnalyzer(mongo_client)
            logger.info("SeekingAlpha analyzer initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize SeekingAlpha analyzer: {e}")
            self.seeking_alpha_analyzer = None
        
        # Initialize BERT models (load from cache if available)
        try:
            # Load FinBERT for news/SEC/earnings (PyTorch only)
            self.finbert = pipeline(
                "sentiment-analysis",
                model="yiyanghkust/finbert-tone",
                framework="pt",
                device=-1
            )
            # Load DistilBERT as fallback (PyTorch only)
            self.distilbert = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                framework="pt",
                device=-1
            )
            logger.info("Loaded FinBERT and DistilBERT models for sentiment analysis (PyTorch only).")
        except Exception as e:
            logger.error(f"Error initializing sentiment models: {str(e)}")
            logger.warning("Falling back to VADER sentiment analysis only")
            self.finbert = None
            self.distilbert = None

        # Load Twitter-Roberta model
        self.twitter_roberta = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            framework="pt",
            device=-1
        )
        logger.info("Loaded Twitter-Roberta model for sentiment analysis (PyTorch only).")

        # Use shared economic calendar instance to prevent multiple browser sessions
        global SHARED_ECONOMIC_CALENDAR
        if calendar_fetcher:
            self.calendar_fetcher = calendar_fetcher
            logger.info("Using provided EconomicCalendar instance")
        elif SHARED_ECONOMIC_CALENDAR is not None:
            self.calendar_fetcher = SHARED_ECONOMIC_CALENDAR
            logger.info("Using shared EconomicCalendar instance")
        else:
            try:
                SHARED_ECONOMIC_CALENDAR = EconomicCalendar(mongo_client)
                self.calendar_fetcher = SHARED_ECONOMIC_CALENDAR
                logger.info("EconomicCalendar initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize EconomicCalendar: {e}")
                self.calendar_fetcher = None
                    
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
        """Fetch dividend data from FMP."""
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            logger.warning("FMP_API_KEY not set. Skipping FMP dividends.")
            return {}
        
        url = f"https://financialmodelingprep.com/stable/dividends?symbol={ticker}&limit=5&apikey={api_key}"
        
        try:
            cached = self.mongo_client.get_alpha_vantage_data(ticker, 'dividends')
            if cached:
                return cached
                
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as resp:
                    if resp.status != 200:
                        logger.warning(f"FMP dividends returned status {resp.status} for {ticker}")
                        return {}
                    data = await resp.json()
                    self.mongo_client.store_alpha_vantage_data(ticker, 'dividends', data)
                    self.mongo_client.store_alpha_vantage_data(ticker, 'ratings', data)
                    return {"dividends": data}
        except Exception as e:
            logger.error(f"Error fetching FMP dividends for {ticker}: {e}")
            return {}

    async def fetch_fmp_earnings(self, ticker: str) -> dict:
        """Fetch earnings data from FMP (limited to 5 responses per call)."""
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            logger.warning("FMP_API_KEY not set. Skipping FMP earnings.")
            return {}
        
        url = f"https://financialmodelingprep.com/stable/earnings?symbol={ticker}&limit=5&apikey={api_key}"
        
        try:
            cached = self.mongo_client.get_alpha_vantage_data(ticker, 'earnings')
            if cached:
                return cached
                
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as resp:
                    if resp.status != 200:
                        logger.warning(f"FMP earnings returned status {resp.status} for {ticker}")
                        return {}
                    data = await resp.json()
                    self.mongo_client.store_alpha_vantage_data(ticker, 'earnings', data)
                    return {"earnings": data}
        except Exception as e:
            logger.error(f"Error fetching FMP earnings for {ticker}: {e}")
            return {}

    async def fetch_fmp_earnings_calendar(self, ticker: str = None) -> dict:
        """Fetch earnings calendar from FMP for next 3 months."""
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            logger.warning("FMP_API_KEY not set. Skipping FMP earnings calendar.")
            return {}
        
        # Calculate date range (today to 3 months from now)
        from_date = datetime.utcnow().strftime("%Y-%m-%d")
        to_date = (datetime.utcnow() + timedelta(days=90)).strftime("%Y-%m-%d")
        
        # Use only 'to' parameter as 'from' doesn't work
        url = f"https://financialmodelingprep.com/stable/earnings-calendar?to={to_date}&apikey={api_key}"
        
        try:
            cache_key = f'earnings_calendar_{to_date}'
            cached = self.mongo_client.get_alpha_vantage_data('market', cache_key)
            if cached:
                if ticker:
                    return {"earnings_calendar": [e for e in cached if e.get('symbol') == ticker]}
                return {"earnings_calendar": cached}
                
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as resp:
                    if resp.status != 200:
                        logger.warning(f"FMP earnings calendar returned status {resp.status}")
                        return {}
                    data = await resp.json()
                    
                    # Store in cache with date-specific key
                    self.mongo_client.store_alpha_vantage_data('market', cache_key, data)
                    
                    if ticker:
                        return {"earnings_calendar": [e for e in data if e.get('symbol') == ticker]}
                    return {"earnings_calendar": data}
        except Exception as e:
            logger.error(f"Error fetching FMP earnings calendar: {e}")
            return {}

    async def fetch_fmp_dividends_calendar(self, ticker: str = None) -> dict:
        """Fetch dividends calendar from FMP for next 3 months."""
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            logger.warning("FMP_API_KEY not set. Skipping FMP dividends calendar.")
            return {}
        
        # Calculate date range (today to 3 months from now)
        to_date = (datetime.utcnow() + timedelta(days=90)).strftime("%Y-%m-%d")
        
        # Use only 'to' parameter as 'from' doesn't work
        url = f"https://financialmodelingprep.com/stable/dividends-calendar?to={to_date}&apikey={api_key}"
        
        try:
            cache_key = f'dividends_calendar_{to_date}'
            cached = self.mongo_client.get_alpha_vantage_data('market', cache_key)
            if cached:
                if ticker:
                    return {"dividends_calendar": [d for d in cached if d.get('symbol') == ticker]}
                return {"dividends_calendar": cached}
                
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as resp:
                    if resp.status != 200:
                        logger.warning(f"FMP dividends calendar returned status {resp.status}")
                        return {}
                    data = await resp.json()
                    
                    # Store in cache with date-specific key
                    self.mongo_client.store_alpha_vantage_data('market', cache_key, data)
                    
                    if ticker:
                        return {"dividends_calendar": [d for d in data if d.get('symbol') == ticker]}
                    return {"dividends_calendar": data}
        except Exception as e:
            logger.error(f"Error fetching FMP dividends calendar: {e}")
            return {}

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
        """Analyze sentiment from FMP financial estimates."""
        try:
            # Run in threadpool since this is a blocking operation
            return await run_in_threadpool(self._analyze_fmp_financial_estimates_sync, ticker)
        except Exception as e:
            logger.warning(f"FMP financial estimates analysis failed for {ticker}: {e}")
            return {"fmp_financial_estimates_sentiment": 0.0, "fmp_financial_estimates_volume": 0, "fmp_financial_estimates_confidence": 0.0}

    def _analyze_fmp_financial_estimates_sync(self, ticker: str) -> dict:
        """Synchronous implementation of FMP financial estimates analysis."""
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            logger.warning("FMP_API_KEY not set. Skipping FMP Financial Estimates.")
            return {"fmp_estimates_sentiment": 0.0, "fmp_estimates_volume": 0, "fmp_estimates_confidence": 0.0}
        url = f"https://financialmodelingprep.com/stable/analyst-estimates?symbol={ticker}&period=annual&apikey={api_key}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                logger.warning(f"FMP Financial Estimates returned status {resp.status_code} for {ticker}")
                return {"fmp_estimates_sentiment": 0.0, "fmp_estimates_volume": 0, "fmp_estimates_confidence": 0.0}
            data = resp.json()
            if not data:
                return {"fmp_estimates_sentiment": 0.0, "fmp_estimates_volume": 0, "fmp_estimates_confidence": 0.0}
            # Use the most recent estimate
            est = data[0]
            eps_avg = est.get('epsAvg', 0.0)
            # Normalize EPS to [-1, 1] using a soft cap (e.g., 0-10 for large caps)
            norm_eps = max(-1, min(1, (eps_avg - 5) / 5))
            logger.info(f"FMP Financial Estimates: epsAvg={eps_avg}, norm={norm_eps}")
            return {"fmp_estimates_sentiment": norm_eps, "fmp_estimates_volume": est.get('numAnalystsEps', 1), "fmp_estimates_confidence": min(est.get('numAnalystsEps', 1)/10, 1.0)}
        except Exception as e:
            logger.error(f"Error fetching FMP Financial Estimates for {ticker}: {e}")
            return {"fmp_estimates_sentiment": 0.0, "fmp_estimates_volume": 0, "fmp_estimates_confidence": 0.0}

    async def analyze_fmp_ratings_snapshot(self, ticker: str) -> dict:
        """Analyze sentiment from FMP ratings snapshot."""
        try:
            # Run in threadpool since this is a blocking operation
            return await run_in_threadpool(self._analyze_fmp_ratings_snapshot_sync, ticker)
        except Exception as e:
            logger.warning(f"FMP ratings snapshot analysis failed for {ticker}: {e}")
            return {"fmp_ratings_snapshot_sentiment": 0.0, "fmp_ratings_snapshot_volume": 0, "fmp_ratings_snapshot_confidence": 0.0}

    def _analyze_fmp_ratings_snapshot_sync(self, ticker: str) -> dict:
        """Synchronous implementation of FMP ratings snapshot analysis."""
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            logger.warning("FMP_API_KEY not set. Skipping FMP Ratings Snapshot.")
            return {"fmp_ratings_sentiment": 0.0, "fmp_ratings_volume": 0, "fmp_ratings_confidence": 0.0}
        url = f"https://financialmodelingprep.com/stable/ratings-snapshot?symbol={ticker}&apikey={api_key}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                logger.warning(f"FMP Ratings Snapshot returned status {resp.status_code} for {ticker}")
                return {"fmp_ratings_sentiment": 0.0, "fmp_ratings_volume": 0, "fmp_ratings_confidence": 0.0}
            data = resp.json()
            if not data:
                return {"fmp_ratings_sentiment": 0.0, "fmp_ratings_volume": 0, "fmp_ratings_confidence": 0.0}
            snap = data[0]
            overall = snap.get('overallScore', 0)
            # Normalize overallScore (1-5) to [-1, 1]
            norm_score = (overall - 3) / 2
            logger.info(f"FMP Ratings Snapshot: overallScore={overall}, norm={norm_score}")
            return {"fmp_ratings_sentiment": norm_score, "fmp_ratings_volume": 1, "fmp_ratings_confidence": 1.0}
        except Exception as e:
            logger.error(f"Error fetching FMP Ratings Snapshot for {ticker}: {e}")
            return {"fmp_ratings_sentiment": 0.0, "fmp_ratings_volume": 0, "fmp_ratings_confidence": 0.0}

    async def analyze_fmp_price_target_summary(self, ticker: str) -> dict:
        """Analyze sentiment from FMP price target summary."""
        try:
            # Run in threadpool since this is a blocking operation
            return await run_in_threadpool(self._analyze_fmp_price_target_summary_sync, ticker)
        except Exception as e:
            logger.warning(f"FMP price target summary analysis failed for {ticker}: {e}")
            return {"fmp_price_target_summary_sentiment": 0.0, "fmp_price_target_summary_volume": 0, "fmp_price_target_summary_confidence": 0.0}

    def _analyze_fmp_price_target_summary_sync(self, ticker: str) -> dict:
        """Synchronous implementation of FMP price target summary analysis."""
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            logger.warning("FMP_API_KEY not set. Skipping FMP Price Target Summary.")
            return {"fmp_price_target_sentiment": 0.0, "fmp_price_target_volume": 0, "fmp_price_target_confidence": 0.0}
        url = f"https://financialmodelingprep.com/stable/price-target-summary?symbol={ticker}&apikey={api_key}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                logger.warning(f"FMP Price Target Summary returned status {resp.status_code} for {ticker}")
                return {"fmp_price_target_sentiment": 0.0, "fmp_price_target_volume": 0, "fmp_price_target_confidence": 0.0}
            data = resp.json()
            if not data:
                return {"fmp_price_target_sentiment": 0.0, "fmp_price_target_volume": 0, "fmp_price_target_confidence": 0.0}
            pt = data[0]
            avg_target = pt.get('lastYearAvgPriceTarget', 0.0)
            # For normalization, you may want to compare to current price (fetch if available)
            # For now, use a soft cap: (avg_target - 200) / 100 for large caps
            norm_pt = max(-1, min(1, (avg_target - 200) / 100))
            logger.info(f"FMP Price Target Summary: lastYearAvgPriceTarget={avg_target}, norm={norm_pt}")
            return {"fmp_price_target_sentiment": norm_pt, "fmp_price_target_volume": pt.get('lastYearCount', 1), "fmp_price_target_confidence": min(pt.get('lastYearCount', 1)/10, 1.0)}
        except Exception as e:
            logger.error(f"Error fetching FMP Price Target Summary for {ticker}: {e}")
            return {"fmp_price_target_sentiment": 0.0, "fmp_price_target_volume": 0, "fmp_price_target_confidence": 0.0}

    async def analyze_fmp_grades_summary(self, ticker: str) -> dict:
        """Analyze sentiment from FMP grades summary."""
        try:
            # Run in threadpool since this is a blocking operation
            return await run_in_threadpool(self._analyze_fmp_grades_summary_sync, ticker)
        except Exception as e:
            logger.warning(f"FMP grades summary analysis failed for {ticker}: {e}")
            return {"fmp_grades_summary_sentiment": 0.0, "fmp_grades_summary_volume": 0, "fmp_grades_summary_confidence": 0.0}

    def _analyze_fmp_grades_summary_sync(self, ticker: str) -> dict:
        """Synchronous implementation of FMP grades summary analysis."""
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            logger.warning("FMP_API_KEY not set. Skipping FMP Grades Summary.")
            return {"fmp_grades_sentiment": 0.0, "fmp_grades_volume": 0, "fmp_grades_confidence": 0.0}
        url = f"https://financialmodelingprep.com/stable/grades-consensus?symbol={ticker}&apikey={api_key}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                logger.warning(f"FMP Grades Summary returned status {resp.status_code} for {ticker}")
                return {"fmp_grades_sentiment": 0.0, "fmp_grades_volume": 0, "fmp_grades_confidence": 0.0}
            data = resp.json()
            if not data:
                return {"fmp_grades_sentiment": 0.0, "fmp_grades_volume": 0, "fmp_grades_confidence": 0.0}
            gs = data[0]
            consensus = gs.get('consensus', '').lower()
            # Map consensus to sentiment
            mapping = {'strong buy': 1.0, 'buy': 0.7, 'hold': 0.0, 'sell': -0.7, 'strong sell': -1.0}
            norm_score = mapping.get(consensus, 0.0)
            logger.info(f"FMP Grades Summary: consensus={consensus}, norm={norm_score}")
            return {"fmp_grades_sentiment": norm_score, "fmp_grades_volume": 1, "fmp_grades_confidence": 1.0}
        except Exception as e:
            logger.error(f"Error fetching FMP Grades Summary for {ticker}: {e}")
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
        sentiment_dict = {
            "ticker": ticker,
            "timestamp": datetime.utcnow(),
        }

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
        for source_func, key in zip(sources, keys):
            try:
                result = await source_func(ticker)
                # Try different possible keys for sentiment score, volume, and confidence
                score = result.get("sentiment_score", result.get(f"{key}_sentiment", 0))
                volume = result.get("volume", result.get(f"{key}_volume", 0))
                confidence = result.get("confidence", result.get(f"{key}_confidence", 0.5))
                
                sentiment_dict[f"{key}_sentiment"] = score
                sentiment_dict[f"{key}_volume"] = volume
                sentiment_dict[f"{key}_confidence"] = confidence
                
                # Store raw data and NLP results
                for suffix in ["raw_data", "nlp_results", "api_status", "error"]:
                    result_key = f"{key}_{suffix}"
                    if result_key in result:
                        sentiment_dict[result_key] = result[result_key]
                    elif suffix in result:
                        sentiment_dict[result_key] = result[suffix]
                        
            except Exception as e:
                logger.warning(f"Sentiment fetch failed for {key}: {e}")
        # Economic Calendar Sentiment
        try:
            sentiment_dict = await self.integrate_economic_events_sentiment(sentiment_dict, ticker, mongo_client=self.mongo_client)
        except Exception as e:
            logger.error(f"Failed to add economic calendar sentiment for {ticker}: {e}")
        # Short Interest Sentiment
        try:
            short_sentiment = await self.analyze_short_interest_sentiment(ticker)
            sentiment_dict.update(short_sentiment)
        except Exception as e:
            logger.error(f"Failed to fetch short interest sentiment for {ticker}: {e}")
        # Calculate final blended sentiment score
        try:
            blended_sentiment = self.blend_sentiment_scores(sentiment_dict)
            sentiment_dict["blended_sentiment"] = blended_sentiment
            logger.info(f"Final blended sentiment for {ticker}: {blended_sentiment:.3f}")
        except Exception as e:
            logger.error(f"Error calculating blended sentiment for {ticker}: {e}")
            sentiment_dict["blended_sentiment"] = 0.0
        
        # Finalize fields
        sentiment_dict["date"] = sentiment_dict["timestamp"].replace(hour=0, minute=0, second=0, microsecond=0)
        sentiment_dict["last_updated"] = datetime.utcnow()
        # Store in DB
        self.mongo_client.store_sentiment(ticker, sentiment_dict)
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
        """Fetch dividend history for a ticker from FMP and store in MongoDB."""
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            logger.warning("FMP_API_KEY not set. Skipping dividends fetch.")
            return {"status": "no_api_key", "dividends": None}
        url = f"https://financialmodelingprep.com/stable/dividends?symbol={ticker}&limit=5&apikey={api_key}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as resp:
                    if resp.status != 200:
                        logger.warning(f"FMP dividends endpoint returned status {resp.status} for {ticker}")
                        return {"status": "api_error", "dividends": None}
                    data = await resp.json()
                    self.mongo_client.db['alpha_vantage_data'].replace_one(
                        {"symbol": ticker, "endpoint": 'dividends'},
                        {"symbol": ticker, "endpoint": 'dividends', "data": data, "timestamp": datetime.utcnow()},
                        upsert=True
                    )
            logger.info(f"Stored FMP dividends for {ticker} in MongoDB.")
            return {"status": "ok", "dividends": data}
        except Exception as e:
            logger.error(f"Error fetching dividends for {ticker} from FMP: {e}")
            return {"status": "exception", "dividends": None}

    async def fetch_fmp_analyst_ratings_and_store(self, ticker: str) -> dict:
        """Fetch analyst ratings for a ticker from FMP and store in MongoDB."""
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            logger.warning("FMP_API_KEY not set. Skipping analyst ratings fetch.")
            return {"status": "no_api_key", "ratings": {}}
        endpoints = {
            "grades_consensus": f"https://financialmodelingprep.com/stable/grades-consensus?symbol={ticker}&apikey={api_key}",
            "price_target_summary": f"https://financialmodelingprep.com/stable/price-target-summary?symbol={ticker}&apikey={api_key}",
            "price_target_consensus": f"https://financialmodelingprep.com/stable/price-target-consensus?symbol={ticker}&apikey={api_key}",
            "grades": f"https://financialmodelingprep.com/stable/grades?symbol={ticker}&apikey={api_key}",
            "grades_historical": f"https://financialmodelingprep.com/stable/grades-historical?symbol={ticker}&apikey={api_key}"
        }
        results = {}
        for key, url in endpoints.items():
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    self.mongo_client.db['analyst_ratings'].replace_one(
                        {"symbol": ticker, "endpoint": key},
                        {"symbol": ticker, "endpoint": key, "data": data, "timestamp": datetime.utcnow()},
                        upsert=True
                    )
                    results[key] = data
                    logger.info(f"Stored analyst ratings {key} for {ticker} in MongoDB.")
                else:
                    logger.warning(f"FMP analyst ratings endpoint {key} returned status {resp.status_code} for {ticker}")
            except Exception as e:
                logger.error(f"Error fetching analyst ratings {key} for {ticker} from FMP: {e}")
        return {"status": "ok", "ratings": results}

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
            # Get insider transactions from MongoDB
            insider_data = self.mongo_client.get_insider_trading(ticker)
            if not insider_data:
                return {
                    'source': 'finnhub_insider',
                    'sentiment': 0.0,
                    'volume': 0,
                    'confidence': 0.0
                }
            # Calculate sentiment based on transaction types and volumes
            total_volume = 0
            weighted_sentiment = 0
            for transaction in insider_data:
                volume = abs(float(transaction.get('transactionShares', 0)))
                price = float(transaction.get('transactionPrice', 0))
                transaction_value = volume * price
                # Weight by transaction value
                if transaction.get('transactionType') == 'Buy':
                    weighted_sentiment += transaction_value
                elif transaction.get('transactionType') == 'Sell':
                    weighted_sentiment -= transaction_value
                total_volume += volume
            # Normalize sentiment
            if total_volume > 0:
                sentiment = weighted_sentiment / (total_volume * 100)  # Scale down for normalization
                confidence = min(1.0, total_volume / 10000)  # Higher volume = higher confidence
            else:
                sentiment = 0.0
                confidence = 0.0
            return {
                'source': 'finnhub_insider',
                'sentiment': sentiment,
                'volume': total_volume,
                'confidence': confidence,
                'transactions': len(insider_data)
            }
        except Exception as e:
            logger.error(f"Error analyzing Finnhub insider sentiment for {ticker}: {e}")
            return {
                'source': 'finnhub_insider',
                'sentiment': 0.0,
                'volume': 0,
                'confidence': 0.0
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
        """Synchronous Reddit sentiment analysis using Twitter-Roberta."""
        try:
            import praw
            sentiment_scores = []
            total_volume = 0
            raw_data = []
            nlp_results = []
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    reddit = praw.Reddit(
                        client_id=os.getenv("REDDIT_CLIENT_ID"),
                        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                        user_agent=os.getenv("REDDIT_USER_AGENT")
                    )
                    for subreddit in REDDIT_SUBREDDITS:
                        try:
                            posts = reddit.subreddit(subreddit).search(ticker, sort="new", time_filter="week", limit=20)
                            for post in posts:
                                if post.score < 10:
                                    continue
                                text = post.title + " " + post.selftext
                                text = text[:512]
                                if self.twitter_roberta is not None:
                                    sentiment = self.twitter_roberta(text)[0]
                                    if 'label' in sentiment:
                                        sentiment_score = LABEL_MAP.get(sentiment['label'], 0.0)
                                    else:
                                        sentiment_score = sentiment['score']
                                    sentiment_scores.append(sentiment_score * min(post.score / 100, 1.0))
                                else:
                                    s = self.vader.polarity_scores(text)
                                    sentiment_scores.append(s["compound"] * min(post.score / 100, 1.0))
                                total_volume += 1
                                raw_data.append({
                                    "type": "post",
                                    "title": post.title,
                                    "selftext": post.selftext,
                                    "score": post.score
                                })
                                nlp_results.append({
                                    "text": text,
                                    "sentiment": sentiment_score,
                                    "model": "twitter_roberta" if self.twitter_roberta is not None else "vader"
                                })
                                post.comments.replace_more(limit=0)
                                for comment in post.comments.list()[:5]:
                                    if comment.score < 5:
                                        continue
                                    comment_text = comment.body[:512]
                                    if self.twitter_roberta is not None:
                                        sentiment = self.twitter_roberta(comment_text)[0]
                                        if 'label' in sentiment:
                                            sentiment_score = LABEL_MAP.get(sentiment['label'], 0.0)
                                        else:
                                            sentiment_score = sentiment['score']
                                        sentiment_scores.append(sentiment_score * min(comment.score / 50, 1.0))
                                    else:
                                        s = self.vader.polarity_scores(comment_text)
                                        sentiment_scores.append(s["compound"] * min(comment.score / 50, 1.0))
                                    total_volume += 1
                                    raw_data.append({
                                        "type": "comment",
                                        "body": comment.body,
                                        "score": comment.score
                                    })
                                    nlp_results.append({
                                        "text": comment_text,
                                        "sentiment": sentiment_score,
                                        "model": "twitter_roberta" if self.twitter_roberta is not None else "vader"
                                    })
                        except Exception as e:
                            logger.error(f"Error analyzing Reddit sentiment for {ticker} in {subreddit}: {str(e)}")
                            continue
                    break  # Success
                except praw.exceptions.APIException as e:
                    logger.warning(f"Reddit API rate limit for {ticker}, attempt {attempt+1}")
                    time.sleep(self._exponential_backoff(attempt))
                except Exception as e:
                    logger.error(f"Error initializing Reddit client: {str(e)}")
                    break
            if total_volume < 1:
                logger.info(f"Reddit volume {total_volume} below threshold 1, setting sentiment to 0.")
                return {"reddit_sentiment": 0.0, "reddit_volume": total_volume, "reddit_confidence": 0.0, "reddit_raw_data": raw_data, "reddit_nlp_results": nlp_results}
            return {
                "reddit_sentiment": sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0,
                "reddit_volume": total_volume,
                "reddit_confidence": min(total_volume / 50, 1.0),
                "reddit_raw_data": raw_data,
                "reddit_nlp_results": nlp_results
            }
        except Exception as e:
            logger.error(f"Error in Reddit sentiment analysis: {e}")
            return None

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
        """Analyze sentiment from FMP data sources including calendar data."""
        try:
            # Combine multiple FMP sentiment sources
            estimates = await self.analyze_fmp_financial_estimates(ticker)
            ratings = await self.analyze_fmp_ratings_snapshot(ticker)
            price_targets = await self.analyze_fmp_price_target_summary(ticker)
            grades = await self.analyze_fmp_grades_summary(ticker)
            
            # Fetch calendar data for economic calendar integration
            earnings_calendar = await self.fetch_fmp_earnings_calendar(ticker)
            dividends_calendar = await self.fetch_fmp_dividends_calendar(ticker)
            
            # Calculate weighted average sentiment
            sentiments = []
            weights = []
            
            if estimates.get('fmp_estimates_sentiment', 0) != 0:
                sentiments.append(estimates['fmp_estimates_sentiment'])
                weights.append(0.3)
            
            if ratings.get('fmp_ratings_sentiment', 0) != 0:
                sentiments.append(ratings['fmp_ratings_sentiment'])
                weights.append(0.3)
            
            if price_targets.get('fmp_price_target_sentiment', 0) != 0:
                sentiments.append(price_targets['fmp_price_target_sentiment'])
                weights.append(0.25)
            
            if grades.get('fmp_grades_sentiment', 0) != 0:
                sentiments.append(grades['fmp_grades_sentiment'])
                weights.append(0.15)
            
            if sentiments:
                weighted_sentiment = sum(s * w for s, w in zip(sentiments, weights)) / sum(weights)
                volume = sum([
                    estimates.get('fmp_estimates_volume', 0),
                    ratings.get('fmp_ratings_volume', 0),
                    price_targets.get('fmp_price_target_volume', 0),
                    grades.get('fmp_grades_volume', 0)
                ])
                confidence = sum([
                    estimates.get('fmp_estimates_confidence', 0),
                    ratings.get('fmp_ratings_confidence', 0),
                    price_targets.get('fmp_price_target_confidence', 0),
                    grades.get('fmp_grades_confidence', 0)
                ]) / 4
            else:
                weighted_sentiment = 0.0
                volume = 0
                confidence = 0.0
            
            # Structure raw data with proper calendar data for economic calendar
            raw_data = {
                'estimates': estimates,
                'ratings': ratings,
                'price_targets': price_targets,
                'grades': grades
            }
            
            # Add calendar data in the expected structure for economic calendar
            if earnings_calendar and 'earnings_calendar' in earnings_calendar:
                raw_data['earnings'] = earnings_calendar['earnings_calendar']
                logger.info(f"Added {len(earnings_calendar['earnings_calendar'])} earnings entries to FMP raw data")
            
            if dividends_calendar and 'dividends_calendar' in dividends_calendar:
                raw_data['dividends'] = dividends_calendar['dividends_calendar']
                logger.info(f"Added {len(dividends_calendar['dividends_calendar'])} dividends entries to FMP raw data")
            
            return {
                'sentiment_score': weighted_sentiment,
                'volume': volume,
                'confidence': confidence,
                'raw_data': raw_data,
                'components': {
                    'estimates': estimates,
                    'ratings': ratings,
                    'price_targets': price_targets,
                    'grades': grades,
                    'earnings_calendar': earnings_calendar,
                    'dividends_calendar': dividends_calendar
                }
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
            earnings_call = await self.fetch_alpha_vantage_earnings_call(ticker, quarter)
            
            if not earnings_call or earnings_call.get('status') != 'ok':
                return {
                    'sentiment_score': 0.0,
                    'volume': 0,
                    'confidence': 0.0,
                    'error': 'No earnings call data available'
                }
            
            transcript = earnings_call.get('transcript', {})
            if not transcript:
                return {
                    'sentiment_score': 0.0,
                    'volume': 0,
                    'confidence': 0.0,
                    'error': 'No transcript available'
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
                'transcript_segments': len(transcript) if isinstance(transcript, list) else 1
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
        """Analyze sentiment from Alpha Vantage news (placeholder implementation)."""
        try:
            # This is a placeholder implementation since Alpha Vantage doesn't have a news sentiment API
            # We can use the earnings call sentiment as a proxy or return neutral
            logger.info(f"Alpha Vantage news sentiment not available for {ticker}, returning neutral")
            return {
                'sentiment_score': 0.0,
                'volume': 0,
                'confidence': 0.0,
                'note': 'Alpha Vantage news sentiment not implemented'
            }
        except Exception as e:
            logger.error(f"Error analyzing Alpha Vantage news sentiment for {ticker}: {str(e)}")
            return {
                'sentiment_score': 0.0,
                'volume': 0,
                'confidence': 0.0,
                'error': str(e)
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
    """Centralized FMP API manager to eliminate duplicate calls and improve caching."""
    
    def __init__(self, mongo_client: MongoDBClient):
        self.api_key = os.getenv("FMP_API_KEY")
        self.mongo_client = mongo_client
        self.base_url = "https://financialmodelingprep.com"
        self.cache_duration = 3600  # 1 hour cache
        
    async def _make_fmp_request(self, endpoint: str, ticker: str = None, **params) -> dict:
        """Make centralized FMP API request with caching."""
        if not self.api_key:
            logger.warning("FMP_API_KEY not set. Skipping FMP request.")
            return {}
            
        # Create cache key
        cache_key = f"fmp_{endpoint}_{ticker or 'global'}_{hash(str(sorted(params.items())))}"
        
        # Check cache first
        cached = self.mongo_client.get_alpha_vantage_data(ticker or 'global', cache_key)
        if cached and 'timestamp' in cached:
            age = (datetime.utcnow() - cached['timestamp']).total_seconds()
            if age < self.cache_duration:
                return cached.get('data', {})
        
        # Make API request
        try:
            url = f"{self.base_url}/{endpoint}"
            params['apikey'] = self.api_key
            if ticker:
                params['symbol'] = ticker
                
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as resp:
                    if resp.status != 200:
                        logger.warning(f"FMP {endpoint} returned status {resp.status}")
                        return {}
                    data = await resp.json()
                    
                    # Store in cache
                    self.mongo_client.store_alpha_vantage_data(
                        ticker or 'global', 
                        cache_key, 
                        {'data': data, 'timestamp': datetime.utcnow()}
                    )
                    return data
                    
        except Exception as e:
            logger.error(f"Error fetching FMP {endpoint}: {e}")
            return {}
    
    async def get_dividends(self, ticker: str, limit: int = 5) -> dict:
        """Get dividend data from FMP."""
        data = await self._make_fmp_request("stable/dividends", ticker, limit=limit)
        return {"dividends": data}
    
    async def get_earnings(self, ticker: str, limit: int = 5) -> dict:
        """Get earnings data from FMP."""
        data = await self._make_fmp_request("stable/earnings", ticker, limit=limit)
        return {"earnings": data}
    
    async def get_earnings_calendar(self, ticker: str = None) -> dict:
        """Get earnings calendar from FMP."""
        params = {"limit": 100}
        if ticker:
            params["symbol"] = ticker
        data = await self._make_fmp_request("v3/earning_calendar", ticker, **params)
        return {"earnings_calendar": data}
    
    async def get_dividends_calendar(self, ticker: str = None) -> dict:
        """Get dividends calendar from FMP."""
        params = {"limit": 100}
        if ticker:
            params["symbol"] = ticker
        data = await self._make_fmp_request("v3/stock_dividend_calendar", ticker, **params)
        return {"dividends_calendar": data}

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