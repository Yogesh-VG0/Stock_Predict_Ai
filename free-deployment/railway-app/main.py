"""
Railway app for heavy sentiment processing and model training tasks.
Handles computationally intensive operations that don't fit in serverless functions.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import aiohttp
import feedparser
import logging
import os
from datetime import datetime, timedelta
import json
import redis.asyncio as redis
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis connection for caching
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global redis_client
    if os.getenv('REDIS_URL'):
        redis_client = redis.from_url(os.getenv('REDIS_URL'))
        logger.info("Connected to Redis")
    yield
    # Shutdown
    if redis_client:
        await redis_client.close()

app = FastAPI(
    title="Stock Sentiment Heavy Processing API",
    description="Heavy sentiment processing for multiple sources",
    version="1.0.0",
    lifespan=lifespan
)

class SentimentRequest(BaseModel):
    tickers: List[str]
    sources: Optional[List[str]] = ["reddit", "sec", "marketaux", "seeking_alpha"]
    force_refresh: bool = False

class BatchSentimentResponse(BaseModel):
    results: Dict[str, Dict[str, Any]]
    processing_time: float
    cached_count: int
    processed_count: int

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "sentiment-processor"}

@app.post("/sentiment/batch", response_model=BatchSentimentResponse)
async def process_batch_sentiment(request: SentimentRequest):
    """Process sentiment for multiple tickers from multiple sources."""
    start_time = datetime.now()
    results = {}
    cached_count = 0
    processed_count = 0
    
    try:
        # Process each ticker
        tasks = []
        for ticker in request.tickers:
            task = process_ticker_sentiment(ticker, request.sources, request.force_refresh)
            tasks.append(task)
        
        # Execute all tasks concurrently
        ticker_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        for i, ticker in enumerate(request.tickers):
            result = ticker_results[i]
            if isinstance(result, Exception):
                logger.error(f"Error processing {ticker}: {result}")
                results[ticker] = {"error": str(result)}
            else:
                results[ticker] = result
                if result.get("cached"):
                    cached_count += 1
                else:
                    processed_count += 1
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchSentimentResponse(
            results=results,
            processing_time=processing_time,
            cached_count=cached_count,
            processed_count=processed_count
        )
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_ticker_sentiment(ticker: str, sources: List[str], force_refresh: bool = False):
    """Process sentiment for a single ticker from multiple sources."""
    cache_key = f"sentiment_heavy:{ticker}"
    
    # Check cache first
    if not force_refresh and redis_client:
        cached = await redis_client.get(cache_key)
        if cached:
            return {**json.loads(cached), "cached": True}
    
    sentiment_data = {
        "ticker": ticker,
        "timestamp": datetime.utcnow().isoformat(),
        "sources": {},
        "overall_sentiment": 0.0,
        "confidence": 0.0,
        "cached": False
    }
    
    # Process each source
    source_tasks = []
    if "reddit" in sources:
        source_tasks.append(analyze_reddit_sentiment(ticker))
    if "sec" in sources:
        source_tasks.append(analyze_sec_sentiment(ticker))
    if "marketaux" in sources:
        source_tasks.append(analyze_marketaux_sentiment(ticker))
    if "seeking_alpha" in sources:
        source_tasks.append(analyze_seeking_alpha_sentiment(ticker))
    
    # Execute source analysis concurrently
    source_results = await asyncio.gather(*source_tasks, return_exceptions=True)
    
    # Aggregate results
    total_sentiment = 0.0
    total_volume = 0
    valid_sources = 0
    
    for i, result in enumerate(source_results):
        if isinstance(result, Exception):
            logger.warning(f"Source {sources[i]} failed for {ticker}: {result}")
            continue
            
        source_name = sources[i]
        sentiment_data["sources"][source_name] = result
        
        if result.get("sentiment") is not None:
            weight = min(result.get("volume", 1), 10) / 10  # Weight by volume
            total_sentiment += result["sentiment"] * weight
            total_volume += result.get("volume", 1)
            valid_sources += 1
    
    # Calculate overall sentiment
    if valid_sources > 0:
        sentiment_data["overall_sentiment"] = total_sentiment / valid_sources
        sentiment_data["confidence"] = min(total_volume / 20, 1.0)
    
    # Cache result for 2 hours
    if redis_client:
        await redis_client.setex(cache_key, 7200, json.dumps(sentiment_data))
    
    return sentiment_data

async def analyze_reddit_sentiment(ticker: str) -> Dict[str, Any]:
    """Analyze Reddit sentiment using PRAW API."""
    try:
        # Simulate Reddit API call (replace with actual PRAW implementation)
        await asyncio.sleep(0.1)  # Simulate API delay
        
        # Mock data - replace with actual Reddit scraping
        return {
            "sentiment": 0.15,
            "volume": 25,
            "confidence": 0.7,
            "source": "reddit",
            "posts_analyzed": 25,
            "avg_score": 45
        }
        
    except Exception as e:
        logger.error(f"Reddit sentiment error for {ticker}: {e}")
        return {"sentiment": 0.0, "volume": 0, "confidence": 0.0, "error": str(e)}

async def analyze_sec_sentiment(ticker: str) -> Dict[str, Any]:
    """Analyze SEC filings sentiment."""
    try:
        # Simulate SEC API processing
        await asyncio.sleep(0.2)
        
        # Mock data - replace with actual SEC filing analysis
        return {
            "sentiment": 0.05,
            "volume": 3,
            "confidence": 0.8,
            "source": "sec",
            "filings_analyzed": 3,
            "recent_filing_date": "2024-01-15"
        }
        
    except Exception as e:
        logger.error(f"SEC sentiment error for {ticker}: {e}")
        return {"sentiment": 0.0, "volume": 0, "confidence": 0.0, "error": str(e)}

async def analyze_marketaux_sentiment(ticker: str) -> Dict[str, Any]:
    """Analyze MarketAux news sentiment."""
    try:
        async with aiohttp.ClientSession() as session:
            # Use MarketAux API (free tier available)
            api_key = os.getenv('MARKETAUX_API_KEY')
            if not api_key:
                return {"sentiment": 0.0, "volume": 0, "confidence": 0.0, "error": "No API key"}
            
            url = f"https://api.marketaux.com/v1/news/all"
            params = {
                "symbols": ticker,
                "filter_entities": "true",
                "language": "en",
                "api_token": api_key,
                "limit": 10
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get("data", [])
                    
                    # Simple sentiment analysis on headlines
                    sentiment_scores = []
                    for article in articles:
                        title = article.get("title", "")
                        score = calculate_simple_sentiment(title)
                        sentiment_scores.append(score)
                    
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
                    
                    return {
                        "sentiment": avg_sentiment,
                        "volume": len(articles),
                        "confidence": min(len(articles) / 10, 1.0),
                        "source": "marketaux",
                        "articles_analyzed": len(articles)
                    }
                else:
                    return {"sentiment": 0.0, "volume": 0, "confidence": 0.0, "error": f"API error {response.status}"}
                    
    except Exception as e:
        logger.error(f"MarketAux sentiment error for {ticker}: {e}")
        return {"sentiment": 0.0, "volume": 0, "confidence": 0.0, "error": str(e)}

async def analyze_seeking_alpha_sentiment(ticker: str) -> Dict[str, Any]:
    """Analyze Seeking Alpha RSS sentiment."""
    try:
        # Use feedparser for RSS feeds
        rss_url = f"https://seekingalpha.com/api/sa/combined/{ticker}.xml"
        
        # Parse RSS feed
        feed = feedparser.parse(rss_url)
        entries = feed.entries[:10]  # Limit to latest 10
        
        sentiment_scores = []
        for entry in entries:
            title = entry.get('title', '')
            summary = entry.get('summary', '')
            text = f"{title} {summary}"
            
            # Check if recent (last 7 days)
            published = entry.get('published_parsed')
            if published:
                entry_date = datetime(*published[:6])
                if datetime.now() - entry_date > timedelta(days=7):
                    continue
            
            score = calculate_simple_sentiment(text)
            sentiment_scores.append(score)
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        
        return {
            "sentiment": avg_sentiment,
            "volume": len(sentiment_scores),
            "confidence": min(len(sentiment_scores) / 10, 1.0),
            "source": "seeking_alpha",
            "articles_analyzed": len(sentiment_scores)
        }
        
    except Exception as e:
        logger.error(f"Seeking Alpha sentiment error for {ticker}: {e}")
        return {"sentiment": 0.0, "volume": 0, "confidence": 0.0, "error": str(e)}

def calculate_simple_sentiment(text: str) -> float:
    """Simple rule-based sentiment analysis."""
    positive_words = ['buy', 'bullish', 'growth', 'profit', 'gain', 'rise', 'strong', 'positive', 'upgrade', 'beat']
    negative_words = ['sell', 'bearish', 'loss', 'fall', 'weak', 'negative', 'downgrade', 'miss', 'decline', 'poor']
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    total_sentiment_words = pos_count + neg_count
    if total_sentiment_words == 0:
        return 0.0
    
    # Normalize between -1 and 1
    sentiment = (pos_count - neg_count) / total_sentiment_words
    return max(-1.0, min(1.0, sentiment))

@app.get("/sentiment/{ticker}")
async def get_single_sentiment(ticker: str, sources: str = "reddit,sec,marketaux,seeking_alpha"):
    """Get sentiment for a single ticker."""
    source_list = [s.strip() for s in sources.split(",")]
    request = SentimentRequest(tickers=[ticker], sources=source_list)
    response = await process_batch_sentiment(request)
    return response.results.get(ticker, {})

@app.post("/models/train/{ticker}")
async def trigger_model_training(ticker: str, background_tasks: BackgroundTasks):
    """Trigger background model training for a ticker."""
    background_tasks.add_task(train_model_background, ticker)
    return {"message": f"Model training started for {ticker}", "status": "processing"}

async def train_model_background(ticker: str):
    """Background task for model training."""
    try:
        logger.info(f"Starting model training for {ticker}")
        # Simulate model training (replace with actual training logic)
        await asyncio.sleep(10)  # Simulate training time
        logger.info(f"Model training completed for {ticker}")
    except Exception as e:
        logger.error(f"Model training failed for {ticker}: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 