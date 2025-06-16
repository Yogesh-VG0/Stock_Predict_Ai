"""
Module for scraping and analyzing Seeking Alpha comments using Playwright.
"""

import asyncio
from playwright.async_api import async_playwright
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
import random
import os
from dotenv import load_dotenv
from ..utils.mongodb import MongoDBClient
import aiohttp
from bs4 import BeautifulSoup
import re
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import json
from urllib.parse import urljoin
import aiofiles
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import signal
import atexit
from functools import wraps
import backoff
from tenacity import retry, stop_after_attempt, wait_exponential
from fake_useragent import UserAgent
from aiohttp_proxy import ProxyConnector
from aiohttp_socks import ProxyConnector as SocksProxyConnector

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SeekingAlphaAnalyzer:
    def __init__(self, mongo_client: Optional[MongoClient] = None):
        """
        Initialize the Seeking Alpha analyzer with MongoDB client and proxy configuration.
        
        Args:
            mongo_client: MongoDB client instance
        """
        self.mongo_client = mongo_client
        self.base_url = "https://seekingalpha.com"
        self.session = None
        self.browser = None
        self.context = None
        self.page = None
        self.is_running = False
        self.stop_event = threading.Event()
        self.comment_queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.last_request_time = 0
        self.min_request_interval = 2.0  # Minimum seconds between requests
        self.max_retries = 3
        self.retry_delay = 5
        self.max_comments_per_ticker = 100
        self.comment_cache = {}
        self.cache_expiry = 3600  # 1 hour
        self.rate_limit_window = 60  # 1 minute
        self.rate_limit_max_requests = 30
        self.request_timestamps = []
        self.proxy_list = self._load_proxies()
        self.current_proxy_index = 0
        self.user_agent = UserAgent()
        self.proxy_rotation_interval = 300  # 5 minutes
        self.last_proxy_rotation = time.time()
        
        # Initialize MongoDB collection
        if self.mongo_client:
            self.collection = self.mongo_client.db['seeking_alpha_sentiment']
            logger.info("Initialized MongoDB collection for Seeking Alpha data")

    def _load_proxies(self) -> List[Dict[str, str]]:
        """
        Load proxy list from environment variables or configuration file.
        
        Returns:
            List of proxy configurations
        """
        proxies = []
        
        # Try to load from environment variables
        proxy_list = os.getenv('PROXY_LIST')
        if proxy_list:
            try:
                proxies = json.loads(proxy_list)
            except json.JSONDecodeError:
                logger.error("Failed to parse PROXY_LIST environment variable")
        
        # If no proxies in environment, try to load from file
        if not proxies:
            try:
                with open('config/proxies.json', 'r') as f:
                    proxies = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                logger.warning("No proxy configuration found")
        
        return proxies

    def _get_next_proxy(self) -> Optional[Dict[str, str]]:
        """
        Get the next proxy from the rotation.
        
        Returns:
            Proxy configuration or None if no proxies available
        """
        if not self.proxy_list:
            return None
            
        # Check if it's time to rotate proxies
        current_time = time.time()
        if current_time - self.last_proxy_rotation >= self.proxy_rotation_interval:
            self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxy_list)
            self.last_proxy_rotation = current_time
            
        return self.proxy_list[self.current_proxy_index]

    def _get_random_user_agent(self) -> str:
        """
        Get a random user agent string.
        
        Returns:
            Random user agent string
        """
        return self.user_agent.random

    async def _create_browser_context(self) -> None:
        """
        Create a new browser context with proxy and user agent rotation.
        """
        if self.browser:
            await self.browser.close()
            
        proxy = self._get_next_proxy()
        user_agent = self._get_random_user_agent()
        
        if proxy:
            logger.info(f"Using proxy: {proxy['host']}:{proxy['port']}")
            self.browser = await self.playwright.chromium.launch(
                proxy={
                    "server": f"{proxy['protocol']}://{proxy['host']}:{proxy['port']}",
                    "username": proxy.get('username'),
                    "password": proxy.get('password')
                }
            )
        else:
            self.browser = await self.playwright.chromium.launch()
            
        self.context = await self.browser.new_context(
            user_agent=user_agent,
            viewport={'width': 1920, 'height': 1080},
            java_script_enabled=True
        )
        
        # Add additional headers to appear more like a real browser
        await self.context.set_extra_http_headers({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        })
        
        self.page = await self.context.new_page()
        
        # Add request interception to handle rate limiting
        await self.page.route("**/*", self._handle_request)

    async def _handle_request(self, route) -> None:
        """
        Handle request interception to implement rate limiting and retry logic.
        
        Args:
            route: Playwright route object
        """
        # Check rate limiting
        current_time = time.time()
        self.request_timestamps = [t for t in self.request_timestamps if current_time - t < self.rate_limit_window]
        
        if len(self.request_timestamps) >= self.rate_limit_max_requests:
            wait_time = self.rate_limit_window - (current_time - self.request_timestamps[0])
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        # Add random delay between requests
        await self._random_delay()
        
        # Update request timestamps
        self.request_timestamps.append(current_time)
        
        # Continue with the request
        await route.continue_()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def _navigate_to_comments(self, page, ticker: str) -> bool:
        """Navigate to the comments page for a given ticker."""
        try:
            # Navigate to the stock page
            url = f"https://seekingalpha.com/symbol/{ticker}/comments"
            logger.info(f"Navigating to {url}")
            
            # Set a longer timeout for navigation
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            
            # Wait for the page to load
            try:
                # First check if we're on a login page or blocked
                if await page.query_selector('form[action*="login"]') or await page.query_selector('div.captcha'):
                    logger.error("Access blocked by Seeking Alpha - login or captcha required")
                    return False
                
                # Wait for comments to load
                await page.wait_for_selector('div[data-test-id="comment-content"]', timeout=10000)
                logger.info("Found comments section")
                
                # Check if we have comments
                no_comments = await page.query_selector('div.no-comments')
                if no_comments:
                    logger.info(f"No comments found for {ticker}")
                    return True
                
                return True
                
            except Exception as e:
                logger.error(f"Error waiting for comments to load: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Navigation to comments failed for {ticker}: {str(e)}")
            return False
        
    async def _random_delay(self):
        """Add random delay to mimic human behavior."""
        delay = random.uniform(1.5, 3.0)
        await asyncio.sleep(delay)
        
    async def _store_comments_in_mongodb(self, ticker: str, comments: List[Dict]):
        """Store comments in MongoDB with timestamp."""
        if not self.mongo_client:
            logger.warning("MongoDB client not initialized, skipping comment storage")
            return
            
        try:
            # Add metadata to each comment
            for comment in comments:
                comment.update({
                    'ticker': ticker,
                    'fetched_at': datetime.utcnow(),
                    'source': 'seeking_alpha'
                })
                
            # Store in MongoDB
            collection = self.mongo_client.db['seeking_alpha_comments']
            if comments:
                collection.insert_many(comments)
                logger.info(f"Stored {len(comments)} comments for {ticker} in MongoDB")
        except Exception as e:
            logger.error(f"Error storing comments in MongoDB for {ticker}: {str(e)}")
            
    def _store_sentiment_in_mongodb(self, ticker: str, sentiment_data: Dict):
        """Store sentiment analysis results in MongoDB."""
        if not self.mongo_client:
            logger.warning("MongoDB client not initialized, skipping sentiment storage")
            return
            
        try:
            # Add metadata
            sentiment_data.update({
                'ticker': ticker,
                'fetched_at': datetime.utcnow(),
                'source': 'seeking_alpha'
            })
            
            # Store in MongoDB
            collection = self.mongo_client.db['seeking_alpha_sentiment']
            collection.replace_one(
                {'ticker': ticker, 'fetched_at': sentiment_data['fetched_at']},
                sentiment_data,
                upsert=True
            )
            logger.info(f"Stored Seeking Alpha sentiment for {ticker} in MongoDB")
            
        except Exception as e:
            logger.error(f"Error storing sentiment in MongoDB for {ticker}: {str(e)}")
            
    async def _init_browser(self):
        """Initialize Playwright browser."""
        playwright = await async_playwright().start()
        browser = await playwright.firefox.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0"
        )
        return playwright, browser, context
        
    async def _parse_comment(self, comment_element) -> Optional[Dict]:
        """Parse a single comment element."""
        try:
            # Get comment text
            text = await comment_element.inner_text()
            if not text:
                logger.warning("Empty comment text")
                return None
            
            # Get username
            username = "Anonymous"
            try:
                username_element = await comment_element.query_selector('a[data-test-id="user-name"]')
                if username_element:
                    username = await username_element.inner_text()
            except Exception as e:
                logger.warning(f"Error parsing username: {str(e)}")
            
            # Get timestamp
            timestamp = None
            try:
                time_element = await comment_element.query_selector('time')
                if time_element:
                    timestamp_str = await time_element.get_attribute('datetime')
                    if timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except Exception as e:
                logger.warning(f"Error parsing timestamp: {str(e)}")
            
            # Get likes count
            likes = 0
            try:
                likes_element = await comment_element.query_selector('button[data-test-id="like-button"]')
                if likes_element:
                    likes_text = await likes_element.inner_text()
                    if likes_text:
                        likes = int(likes_text.split()[0])
            except Exception as e:
                logger.warning(f"Error parsing likes: {str(e)}")
            
            # Clean up the text by removing the username and any extra whitespace
            text = text.replace(username, '').strip()
            
            return {
                'text': text,
                'username': username,
                'timestamp': timestamp or datetime.utcnow(),
                'likes': likes,
                'fetched_at': datetime.utcnow()
            }
                
        except Exception as e:
            logger.warning(f"Error parsing comment: {str(e)}")
            return None
            
    async def get_comments(self, ticker: str, max_comments: int = 30) -> List[Dict]:
        """Scrape comments from Seeking Alpha."""
        comments = []
        playwright = None
        browser = None
        context = None
        
        try:
            playwright, browser, context = await self._init_browser()
            page = await context.new_page()
            
            # Navigate to comments page
            navigation_success = await self._navigate_to_comments(page, ticker)
            if not navigation_success:
                return comments
            
            # Get all comment elements
            comment_elements = await page.query_selector_all('div[data-test-id="comment-content"]')
            logger.info(f"Found {len(comment_elements)} comment elements")
            
            # Parse each comment
            for element in comment_elements[:max_comments]:
                comment = await self._parse_comment(element)
                if comment:
                    comments.append(comment)
                    
            return comments
            
        except Exception as e:
            logger.error(f"Error fetching Seeking Alpha comments for {ticker}: {str(e)}")
            return comments
            
        finally:
            if playwright:
                await playwright.stop()
                
    async def analyze_comments_sentiment(self, ticker: str, lookback_days: int = 7) -> Dict:
        """
        Analyze sentiment from Seeking Alpha comments using VADER sentiment analysis.
        
        Args:
            ticker: Stock ticker symbol
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary containing sentiment metrics
        """
        try:
            # Initialize VADER sentiment analyzer
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            vader = SentimentIntensityAnalyzer()
            
            # First, fetch new comments
            logger.info(f"Fetching Seeking Alpha comments for {ticker}")
            comments = await self.get_comments(ticker, max_comments=100)
            
            if not comments:
                logger.warning(f"No comments found for {ticker}")
                return {
                    "seeking_alpha_comments_sentiment": 0.0,
                    "seeking_alpha_comments_volume": 0,
                    "seeking_alpha_comments_confidence": 0.0,
                    "seeking_alpha_comments_engagement": 0.0,
                    "seeking_alpha_comments_analyzed": 0,
                    "seeking_alpha_comments_sentiment_std": 0.0,
                    "seeking_alpha_comments_error": "No comments found"
                }
            
            logger.info(f"Analyzing sentiment for {len(comments)} Seeking Alpha comments")
            
            # Calculate sentiment metrics using VADER
            total_sentiment = 0.0
            total_weight = 0
            sentiment_scores = []  # List to store compound scores
            analyzed_comments = []  # List to store comments with their sentiment scores
            
            for comment in comments:
                text = comment.get('text', '')
                if not text:
                    continue
                    
                # Get VADER sentiment scores
                vs = vader.polarity_scores(text)
                compound_score = vs['compound']
                
                # Calculate time decay factor (more recent comments have higher weight)
                comment_date = comment.get('fetched_at', datetime.utcnow())
                days_old = (datetime.utcnow() - comment_date).days
                time_decay = max(0.1, 1.0 - (days_old / lookback_days))
                
                # Use time decay as weight
                weight = time_decay
                
                total_sentiment += compound_score * weight
                total_weight += weight
                sentiment_scores.append(compound_score)
                
                # Store comment with its sentiment analysis
                analyzed_comments.append({
                    'text': text,
                    'username': comment.get('username', 'Anonymous'),
                    'timestamp': comment.get('timestamp', datetime.utcnow()),
                    'likes': comment.get('likes', 0),
                    'sentiment_score': compound_score,
                    'sentiment_components': {
                        'pos': vs['pos'],
                        'neu': vs['neu'],
                        'neg': vs['neg']
                    }
                })
            
            # Calculate final sentiment score
            sentiment_score = total_sentiment / total_weight if total_weight > 0 else 0.0
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            
            # Calculate confidence based on volume and sentiment consistency
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
            sentiment_std = (sum((s - avg_sentiment) ** 2 for s in sentiment_scores) / len(sentiment_scores)) ** 0.5 if sentiment_scores else 1.0
            confidence = min(1.0, (len(comments) / 10) * (1 - min(1.0, sentiment_std)))
            
            result = {
                "seeking_alpha_comments_sentiment": sentiment_score,
                "seeking_alpha_comments_volume": len(comments),
                "seeking_alpha_comments_confidence": confidence,
                "seeking_alpha_comments_engagement": len(comments),
                "seeking_alpha_comments_analyzed": len(comments),
                "seeking_alpha_comments_sentiment_std": sentiment_std,
                "seeking_alpha_comments": analyzed_comments,  # Store individual comments with their sentiment
                "ticker": ticker,
                "fetched_at": datetime.utcnow(),
                "source": "seeking_alpha"
            }
            
            # Store sentiment in MongoDB if client is available
            if self.mongo_client:
                self._store_sentiment_in_mongodb(ticker, result)
            
            logger.info(f"Seeking Alpha comments sentiment analysis complete for {ticker}: score={sentiment_score:.3f}, volume={len(comments)}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing Seeking Alpha sentiment for {ticker}: {str(e)}")
            return {
                "seeking_alpha_comments_sentiment": 0.0,
                "seeking_alpha_comments_volume": 0,
                "seeking_alpha_comments_confidence": 0.0,
                "seeking_alpha_comments_engagement": 0.0,
                "seeking_alpha_comments_analyzed": 0,
                "seeking_alpha_comments_sentiment_std": 0.0,
                "seeking_alpha_comments_error": str(e)
            } 