#!/usr/bin/env python3
"""
Sentiment Pipeline Cron Job
Runs every 4 hours to fetch and store sentiment data for top stocks
"""

import os
import sys
import logging
import time
from datetime import datetime, timezone

# Add the ml_backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.sentiment import SentimentAnalyzer
from utils.mongodb import MongoDBClient
from config.constants import TOP_100_TICKERS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/sentiment_pipeline.log', 'a')
    ]
)
logger = logging.getLogger(__name__)

def run_sentiment_pipeline():
    """Main function to run the sentiment analysis pipeline"""
    
    start_time = time.time()
    logger.info("=" * 60)
    logger.info(f"Starting sentiment pipeline at {datetime.now(timezone.utc).isoformat()}")
    logger.info("=" * 60)
    
    try:
        # Initialize MongoDB client
        mongodb_uri = os.getenv('MONGODB_URI')
        if not mongodb_uri:
            raise ValueError("MONGODB_URI environment variable not set")
            
        mongo_client = MongoDBClient(mongodb_uri)
        sentiment_analyzer = SentimentAnalyzer(mongo_client)
        
        # Top tickers to process (can expand this list)
        tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX', 
            'DIS', 'AMD', 'BABA', 'CRM', 'ORCL', 'ADBE', 'INTC', 'PYPL',
            'UBER', 'SNOW', 'COIN', 'PLTR'
        ]
        
        successful_count = 0
        failed_count = 0
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"[{i}/{len(tickers)}] Processing sentiment for {ticker}")
            
            try:
                # Fetch sentiment from all sources
                sentiment_data = {}
                
                # News sentiment (primary source)
                try:
                    news_sentiment = sentiment_analyzer.fetch_and_store_news_sentiment(ticker)
                    sentiment_data['news'] = news_sentiment
                    logger.info(f"✓ {ticker} news sentiment: {news_sentiment}")
                except Exception as e:
                    logger.warning(f"✗ {ticker} news sentiment failed: {e}")
                
                # Reddit sentiment
                try:
                    reddit_sentiment = sentiment_analyzer.fetch_and_store_reddit_sentiment(ticker)
                    sentiment_data['reddit'] = reddit_sentiment
                    logger.info(f"✓ {ticker} reddit sentiment: {reddit_sentiment}")
                except Exception as e:
                    logger.warning(f"✗ {ticker} reddit sentiment failed: {e}")
                
                # SEC filings sentiment
                try:
                    sec_sentiment = sentiment_analyzer.fetch_and_store_sec_sentiment(ticker)
                    sentiment_data['sec'] = sec_sentiment
                    logger.info(f"✓ {ticker} SEC sentiment: {sec_sentiment}")
                except Exception as e:
                    logger.warning(f"✗ {ticker} SEC sentiment failed: {e}")
                
                # Seeking Alpha sentiment
                try:
                    sa_sentiment = sentiment_analyzer.fetch_and_store_seeking_alpha_sentiment(ticker)
                    sentiment_data['seeking_alpha'] = sa_sentiment
                    logger.info(f"✓ {ticker} Seeking Alpha sentiment: {sa_sentiment}")
                except Exception as e:
                    logger.warning(f"✗ {ticker} Seeking Alpha sentiment failed: {e}")
                
                if sentiment_data:
                    successful_count += 1
                    logger.info(f"✅ {ticker} completed successfully")
                else:
                    failed_count += 1
                    logger.error(f"❌ {ticker} failed completely")
                
                # Small delay to avoid rate limiting
                time.sleep(2)
                
            except Exception as e:
                failed_count += 1
                logger.error(f"❌ Error processing {ticker}: {e}")
                continue
        
        # Pipeline completion summary
        elapsed_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("SENTIMENT PIPELINE COMPLETED")
        logger.info(f"✅ Successful: {successful_count}/{len(tickers)} tickers")
        logger.info(f"❌ Failed: {failed_count}/{len(tickers)} tickers")
        logger.info(f"⏱️  Total time: {elapsed_time:.2f} seconds")
        logger.info(f"🕐 Completed at: {datetime.now(timezone.utc).isoformat()}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed with critical error: {e}")
        return False

if __name__ == "__main__":
    success = run_sentiment_pipeline()
    sys.exit(0 if success else 1) 