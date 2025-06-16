"""
Simple test script for short interest scraper with visible browser.
"""

import logging
from datetime import datetime
from ..data.short_interest import ShortInterestAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Test short interest scraper with visible browser."""
    # Initialize analyzer
    analyzer = ShortInterestAnalyzer()
    
    # Test with a single stock first
    test_ticker = 'AAPL'  # You can change this to any ticker you want to test
    
    try:
        logger.info(f"\nFetching short interest data for {test_ticker}...")
        
        # Fetch data
        result = analyzer.fetch_short_interest(test_ticker)
        
        if result['status'] == 'success':
            data = result['short_interest_data']
            metrics = data['metrics']
            
            # Print results
            logger.info("\n=== Results ===")
            logger.info(f"Status: {result['status']}")
            
            if data['current']:
                logger.info("\nCurrent Data:")
                logger.info(f"Date: {data['current']['date']}")
                logger.info(f"Short Interest: {data['current']['short_interest']:,}")
                logger.info(f"Average Volume: {data['current']['avg_volume']:,}")
                logger.info(f"Days to Cover: {data['current']['days_to_cover']:.2f}")
            
            logger.info("\nMetrics:")
            logger.info(f"Average Short Interest: {metrics['avg_short_interest']:,}")
            logger.info(f"Average Volume: {metrics['avg_volume']:,}")
            logger.info(f"Average Days to Cover: {metrics['avg_days_to_cover']:.2f}")
            logger.info(f"Short Interest Change: {metrics['short_interest_change']:.2f}%")
            logger.info(f"Data Points: {metrics['data_points']}")
            
            if data['historical']:
                logger.info(f"\nHistorical Data Points: {len(data['historical'])}")
                logger.info("\nRecent History:")
                for entry in data['historical'][:3]:  # Show last 3 entries
                    logger.info(f"Date: {entry['date']}")
                    logger.info(f"Short Interest: {entry['short_interest']:,}")
                    logger.info(f"Average Volume: {entry['avg_volume']:,}")
                    logger.info(f"Days to Cover: {entry['days_to_cover']:.2f}")
                    logger.info("---")
        else:
            logger.error(f"Failed to fetch data. Status: {result['status']}")
            if 'error' in result:
                logger.error(f"Error: {result['error']}")
                
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        raise

if __name__ == "__main__":
    main() 