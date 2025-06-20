#!/usr/bin/env python3
"""
Daily data ingestion script for keeping the database updated.
Runs daily as a cron job to fetch latest market data.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import traceback

# Add ml_backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml_backend.utils.mongodb import MongoDBClient
from ml_backend.data.ingestion import DataIngestion
from ml_backend.config.constants import TOP_100_TICKERS, MONGODB_URI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function for daily data ingestion."""
    logger.info("Starting daily data ingestion process...")
    
    try:
        # Initialize components
        mongo_client = MongoDBClient(MONGODB_URI)
        if mongo_client.db is None:
            logger.error("Failed to connect to MongoDB")
            return 1
        
        data_ingestion = DataIngestion(mongo_client)
        
        # Calculate date range (last 7 days to ensure we catch any missing data)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        logger.info(f"Ingesting data from {start_date.date()} to {end_date.date()}")
        
        # Ingest data for all tickers
        successful_ingestion = 0
        failed_ingestion = 0
        
        for ticker in TOP_100_TICKERS:
            try:
                logger.info(f"Ingesting data for {ticker} ({TOP_100_TICKERS.index(ticker)+1}/{len(TOP_100_TICKERS)})")
                
                # Ingest historical data
                success = data_ingestion.ingest_historical_data(
                    ticker=ticker,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
                if success:
                    successful_ingestion += 1
                    logger.info(f"Successfully ingested data for {ticker}")
                else:
                    failed_ingestion += 1
                    logger.error(f"Failed to ingest data for {ticker}")
                    
            except Exception as e:
                logger.error(f"Error ingesting data for {ticker}: {e}")
                failed_ingestion += 1
                continue
        
        logger.info(f"Data ingestion completed: {successful_ingestion} successful, {failed_ingestion} failed")
        
        # Log ingestion metadata to MongoDB
        try:
            ingestion_metadata = {
                'ingestion_date': datetime.utcnow(),
                'start_date': start_date,
                'end_date': end_date,
                'successful_tickers': successful_ingestion,
                'failed_tickers': failed_ingestion,
                'total_tickers': len(TOP_100_TICKERS)
            }
            
            collection = mongo_client.db['data_ingestion_log']
            collection.insert_one(ingestion_metadata)
            logger.info("Ingestion metadata logged to MongoDB")
            
        except Exception as e:
            logger.error(f"Failed to log metadata: {e}")
        
        return 0 if failed_ingestion == 0 else 1
        
    except Exception as e:
        logger.error(f"Critical error in daily data ingestion: {e}")
        logger.error(traceback.format_exc())
        return 1
    
    finally:
        if 'mongo_client' in locals():
            mongo_client.close()

if __name__ == '__main__':
    sys.exit(main()) 