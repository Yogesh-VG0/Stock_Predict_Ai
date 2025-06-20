"""
Test script for short interest scraper with visible browser.
"""

import asyncio
import logging
import sys
import platform
from ml_backend.data.short_interest import ShortInterestAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_short_interest():
    """Test the short interest scraping functionality."""
    logger.info("\n=== System Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Operating System: {platform.system()} - {platform.release()}")
    logger.info("=== Starting Test ===\n")

    logger.info("Starting test with Chrome version detection...")
    logger.info("Browser will be visible for debugging purposes")
    
    analyzer = ShortInterestAnalyzer()
    ticker = "AAPL"  # Example ticker
    
    try:
        logger.info(f"\nFetching short interest data for {ticker}...")
        data = await analyzer.fetch_short_interest(ticker)
        
        if data:
            logger.info("Successfully fetched short interest data:")
            logger.info(f"Current data: {data}")
        else:
            logger.error("No data returned from scraper")
            
    except Exception as e:
        logger.error(f"Test failed with error:\n{str(e)}")
        logger.error("\nFull traceback:", exc_info=True)
    
    logger.info("\nTest completed")

if __name__ == "__main__":
    asyncio.run(test_short_interest()) 