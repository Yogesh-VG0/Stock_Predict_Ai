"""
MongoDB integration for data storage and retrieval.
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ServerSelectionTimeoutError, BulkWriteError
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import os
import pandas as pd
from dotenv import load_dotenv
try:
    from ml_backend.config.constants import (
        MONGO_COLLECTIONS,
        RETRY_CONFIG,
        PREDICTION_WINDOWS
    )
except ImportError:
    # Fallback for direct script usage
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.abspath(os.path.join(current_dir, '..', 'config'))
    if config_dir not in sys.path:
        sys.path.insert(0, config_dir)
    from constants import (
        MONGO_COLLECTIONS,
        RETRY_CONFIG,
        PREDICTION_WINDOWS
    )

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDBClient:
    def __init__(self, connection_string: str = None):
        """Initialize MongoDB client with connection string."""
        self.connection_string = connection_string or os.getenv("MONGODB_URI")
        if not self.connection_string:
            raise ValueError("MongoDB connection string not provided and MONGODB_URI environment variable not set")
            
        self.database_name = "stock_predictor"
        self.client = None
        self.db = None
        self.collections = {}
        self.max_retries = RETRY_CONFIG["max_retries"]
        self.base_delay = RETRY_CONFIG["base_delay"]
        self.max_delay = RETRY_CONFIG["max_delay"]
        
        try:
            self.connect()
            if self.client:
                self.initialize_collections()
                self.create_indexes()
                self.setup_historical_data_schema()
        except Exception as e:
            logger.error(f"Error initializing MongoDB client: {str(e)}")

    def connect(self) -> None:
        """Connect to MongoDB server."""
        try:
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=5000
            )
            # Test the connection
            self.client.server_info()
            self.db = self.client[self.database_name]
            logger.info(f"Connected to MongoDB database: {self.database_name}")
        except ServerSelectionTimeoutError as e:
            logger.warning(f"Could not connect to MongoDB: {str(e)}")
            self.client = None
            self.db = None
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {str(e)}")
            self.client = None
            self.db = None

    def initialize_collections(self) -> None:
        """Initialize MongoDB collections."""
        try:
            for collection_name in MONGO_COLLECTIONS.values():
                self.collections[collection_name] = self.db[collection_name]
                logger.info(f"Initialized collection: {collection_name}")
            # Add reference for historical_data collection
            self.historical_data_collection = self.collections[MONGO_COLLECTIONS["historical_data"]]
        except Exception as e:
            logger.error(f"Error initializing collections: {str(e)}")

    def create_indexes(self) -> None:
        """Create indexes on frequently queried fields."""
        try:
            # Predictions collection indexes
            self.collections[MONGO_COLLECTIONS["predictions"]].create_index([
                ("ticker", ASCENDING),
                ("timestamp", DESCENDING)
            ])
            
            # Historical data collection indexes
            self.collections[MONGO_COLLECTIONS["historical_data"]].create_index([
                ("ticker", ASCENDING),
                ("date", DESCENDING)
            ])
            
            # Sentiment data collection indexes
            self.collections[MONGO_COLLECTIONS["sentiment_data"]].create_index([
                ("ticker", ASCENDING),
                ("last_updated", DESCENDING)
            ])
            
            # Model versions collection indexes
            self.collections[MONGO_COLLECTIONS["model_versions"]].create_index([
                ("ticker", ASCENDING),
                ("window", ASCENDING),
                ("version", DESCENDING)
            ])
            
            logger.info("Created indexes on collections")
        except Exception as e:
            logger.error(f"Error creating indexes: {str(e)}")

    def setup_historical_data_schema(self):
        """Set up JSON schema validation for the historical_data collection."""
        schema = {
            "bsonType": "object",
            "required": ["ticker", "date", "Open", "High", "Low", "Close", "Volume"],
            "properties": {
                "ticker": {"bsonType": "string"},
                "date": {"bsonType": ["date", "string"]},
                "Open": {"bsonType": ["double", "int", "decimal", "long"]},
                "High": {"bsonType": ["double", "int", "decimal", "long"]},
                "Low": {"bsonType": ["double", "int", "decimal", "long"]},
                "Close": {"bsonType": ["double", "int", "decimal", "long"]},
                "Volume": {"bsonType": ["double", "int", "decimal", "long"]}
            }
        }
        try:
            self.db.command({
                "collMod": MONGO_COLLECTIONS["historical_data"],
                "validator": {"$jsonSchema": schema},
                "validationLevel": "moderate"
            })
            logger.info("Set up schema validation for historical_data collection")
        except Exception as e:
            logger.warning(f"Could not set up schema validation: {e}")

    def store_predictions(self, ticker: str, predictions: Dict[str, Dict[str, float]]) -> bool:
        """Store predictions in bulk."""
        try:
            collection = self.collections[MONGO_COLLECTIONS["predictions"]]
            
            # Prepare documents for bulk insert
            documents = []
            timestamp = datetime.utcnow()
            
            for window, data in predictions.items():
                document = {
                    "ticker": ticker,
                    "window": window,
                    "prediction": float(data["prediction"]),
                    "confidence": float(data["confidence"]),
                    "timestamp": timestamp
                }
                # Add range if present
                if "range" in data:
                    document["range"] = data["range"]
                documents.append(document)
            
            # Bulk insert
            result = collection.insert_many(documents)
            logger.info(f"Stored {len(result.inserted_ids)} predictions for {ticker}")
            return True
            
        except BulkWriteError as e:
            logger.error(f"Error storing predictions for {ticker}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error storing predictions for {ticker}: {str(e)}")
            return False

    def store_historical_data(self, ticker: str, data: pd.DataFrame) -> bool:
        """Store historical data in bulk, in a single collection."""
        try:
            collection = self.historical_data_collection
            if isinstance(data, list):
                data = pd.DataFrame(data)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ['_'.join([str(i) for i in col if i]) for col in data.columns.values]
            # Always ensure 'date' column exists and is datetime
            if 'date' not in data.columns:
                possible_date_cols = [col for col in data.columns if str(col).lower() in ['date', 'datetime', 'index']]
                if possible_date_cols:
                    data = data.rename(columns={possible_date_cols[0]: 'date'})
                else:
                    logger.error(f"No 'date' column found in DataFrame for {ticker} before saving. Columns: {data.columns}")
                    return False
            data['date'] = pd.to_datetime(data['date'])
            documents = data.to_dict('records')
            for doc in documents:
                doc['ticker'] = ticker
                doc['date'] = pd.to_datetime(doc['date'])
            if documents:
                result = collection.insert_many(documents)
                logger.info(f"Stored {len(result.inserted_ids)} historical data points for {ticker} in historical_data collection")
            return True
        except BulkWriteError as e:
            logger.error(f"Error storing historical data for {ticker}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error storing historical data for {ticker}: {str(e)}")
            return False

    def store_sentiment_data(self, ticker: str, sentiment: Dict[str, Any]) -> bool:
        """Store sentiment data with proper validation and error handling."""
        try:
            if not ticker or not sentiment:
                logger.error("Invalid ticker or sentiment data")
                return False

            # Validate required fields
            required_fields = ['ticker', 'timestamp', 'sources']
            if not all(field in sentiment for field in required_fields):
                logger.error(f"Missing required fields in sentiment data: {required_fields}")
                return False

            # Store main sentiment data
            sentiment_collection = self.db['sentiment']
            sentiment_collection.update_one(
                {'ticker': ticker, 'timestamp': sentiment['timestamp']},
                {'$set': sentiment},
                upsert=True
            )

            # Store individual source data
            sources_collection = self.db['sentiment_sources']
            for source, data in sentiment.get('sources', {}).items():
                if isinstance(data, dict):
                    source_doc = {
                        'ticker': ticker,
                        'timestamp': sentiment['timestamp'],
                        'source': source,
                        'data': data
                    }
                    sources_collection.update_one(
                        {'ticker': ticker, 'timestamp': sentiment['timestamp'], 'source': source},
                        {'$set': source_doc},
                        upsert=True
                    )

            # Store economic event features if present
            if 'economic_event_features' in sentiment:
                events_collection = self.db['economic_events']
                events_doc = {
                    'ticker': ticker,
                    'timestamp': sentiment['timestamp'],
                    'features': sentiment['economic_event_features']
                }
                events_collection.update_one(
                    {'ticker': ticker, 'timestamp': sentiment['timestamp']},
                    {'$set': events_doc},
                    upsert=True
                )

            return True

        except Exception as e:
            logger.error(f"Error storing sentiment data for {ticker}: {e}")
            return False

    def get_latest_predictions(self, ticker: str) -> Dict[str, Dict[str, float]]:
        """Get latest predictions for a ticker."""
        try:
            collection = self.collections[MONGO_COLLECTIONS["predictions"]]
            
            # Find latest predictions
            cursor = collection.find(
                {"ticker": ticker},
                sort=[("timestamp", DESCENDING)],
                limit=len(PREDICTION_WINDOWS)
            )
            
            predictions = {}
            for doc in cursor:
                predictions[doc["window"]] = {
                    "prediction": doc["prediction"],
                    "confidence": doc["confidence"]
                }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting latest predictions for {ticker}: {str(e)}")
            return {}

    def get_historical_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get historical data for a ticker within date range from the single collection."""
        try:
            collection = self.historical_data_collection
            cursor = collection.find({
                "ticker": ticker,
                "date": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }, sort=[("date", ASCENDING)])
            data = pd.DataFrame(list(cursor))
            if not data.empty:
                # Defensive: ensure 'date' column exists and is datetime
                if 'date' not in data.columns:
                    logger.error(f"No 'date' column found in loaded DataFrame for {ticker}. Columns: {data.columns}")
                    return pd.DataFrame()
                data['date'] = pd.to_datetime(data['date'])
                data = data.sort_values('date')
                data = data.reset_index(drop=True)
            return data
        except Exception as e:
            logger.error(f"Error getting historical data for {ticker}: {str(e)}")
            return pd.DataFrame()

    def get_latest_sentiment(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get latest sentiment data with proper validation."""
        try:
            if not ticker:
                return None

            sentiment_collection = self.db['sentiment']
            latest_sentiment = sentiment_collection.find_one(
                {'ticker': ticker},
                sort=[('timestamp', -1)]
            )

            if not latest_sentiment:
                return None

            # Validate sentiment data
            if not self._validate_sentiment_data(latest_sentiment):
                logger.warning(f"Invalid sentiment data found for {ticker}")
                return None

            return latest_sentiment

        except Exception as e:
            logger.error(f"Error getting latest sentiment for {ticker}: {e}")
            return None

    def _validate_sentiment_data(self, sentiment: Dict[str, Any]) -> bool:
        """Validate sentiment data structure."""
        try:
            required_fields = ['ticker', 'timestamp', 'sources']
            if not all(field in sentiment for field in required_fields):
                return False

            if not isinstance(sentiment['sources'], dict):
                return False

            # Validate each source
            for source, data in sentiment['sources'].items():
                if not isinstance(data, dict):
                    return False
                if not all(key in data for key in ['sentiment_score', 'volume', 'confidence']):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating sentiment data: {e}")
            return False

    def store_model_version(self, ticker: str, window: str, version: str, metrics: Dict[str, float]) -> bool:
        """Store model version information."""
        try:
            collection = self.collections[MONGO_COLLECTIONS["model_versions"]]
            
            document = {
                "ticker": ticker,
                "window": window,
                "version": version,
                "metrics": metrics,
                "timestamp": datetime.utcnow()
            }
            
            result = collection.insert_one(document)
            logger.info(f"Stored model version {version} for {ticker} {window}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing model version for {ticker} {window}: {str(e)}")
            return False

    def store_prediction_metrics(self, ticker: str, window: str, metrics: Dict[str, float]) -> bool:
        """Store prediction metrics."""
        try:
            collection = self.collections[MONGO_COLLECTIONS["prediction_metrics"]]
            
            document = {
                "ticker": ticker,
                "window": window,
                "metrics": metrics,
                "timestamp": datetime.utcnow()
            }
            
            result = collection.insert_one(document)
            logger.info(f"Stored prediction metrics for {ticker} {window}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing prediction metrics for {ticker} {window}: {str(e)}")
            return False

    def get_latest_date(self, ticker: str) -> Optional[datetime]:
        """Get the latest date for which data is stored for a ticker from the single collection."""
        try:
            collection = self.historical_data_collection
            doc = collection.find_one(
                {"ticker": ticker},
                sort=[("date", -1)]
            )
            if doc and "date" in doc:
                date_val = doc["date"]
                if pd.isna(date_val):
                    logger.warning(f"NaT/null date found in DB for {ticker}, returning None.")
                    return None
                return date_val
            return None
        except Exception as e:
            logger.error(f"Error getting latest date for {ticker}: {str(e)}")
            return None

    def store_alpha_vantage_data(self, ticker: str, endpoint: str, data: dict) -> bool:
        """Store Alpha Vantage data with proper validation"""
        try:
            if not ticker or not endpoint or not data:
                return False
                
            # Store raw data
            self.db['alpha_vantage_raw'].update_one(
                {
                    'ticker': ticker,
                    'endpoint': endpoint,
                    'timestamp': data.get('timestamp')
                },
                {'$set': data},
                upsert=True
            )
            
            # Store processed data if available
            if 'processed' in data:
                self.db['alpha_vantage_processed'].update_one(
                    {
                        'ticker': ticker,
                        'endpoint': endpoint,
                        'timestamp': data.get('timestamp')
                    },
                    {'$set': data['processed']},
                    upsert=True
                )
                
            return True
            
        except Exception as e:
            logger.error(f"Error storing Alpha Vantage data: {e}")
            return False

    def get_alpha_vantage_data(self, ticker: str, endpoint: str) -> dict:
        """Retrieve Alpha Vantage data for a ticker and endpoint, if it exists."""
        try:
            collection = self.db['alpha_vantage_data']
            doc = collection.find_one({'ticker': ticker, 'endpoint': endpoint}, sort=[('timestamp', -1)])
            if doc and 'data' in doc:
                return doc['data']
            return None
        except Exception as e:
            logger.error(f"Error retrieving Alpha Vantage data for {ticker} - {endpoint}: {e}")
            return None

    def get_sentiment_data(self, ticker: str, date: str) -> dict:
        """Fetch the sentiment document for a given ticker and date (ISO format yyyy-mm-dd)."""
        try:
            collection = self.collections[MONGO_COLLECTIONS["sentiment_data"]]
            # Find the most recent sentiment document for the ticker on or before the date
            dt = pd.to_datetime(date)
            doc = collection.find_one(
                {"ticker": ticker, "last_updated": {"$lte": dt}},
                sort=[("last_updated", -1)]
            )
            return doc if doc else {}
        except Exception as e:
            logger.error(f"Error getting sentiment data for {ticker} on {date}: {str(e)}")
            return {}

    def get_prediction(self, ticker: str, date: str) -> dict:
        """Fetch the prediction document for a given ticker and date (ISO format yyyy-mm-dd)."""
        try:
            collection = self.collections[MONGO_COLLECTIONS["predictions"]]
            dt = pd.to_datetime(date)
            doc = collection.find_one(
                {"ticker": ticker, "timestamp": {"$lte": dt}},
                sort=[("timestamp", -1)]
            )
            return doc if doc else {}
        except Exception as e:
            logger.error(f"Error getting prediction for {ticker} on {date}: {str(e)}")
            return {}

    def store_sec_filing(self, filing: dict) -> bool:
        """Store SEC filing with proper validation"""
        try:
            if not filing or not filing.get('ticker'):
                return False
                
            # Store raw filing
            self.db['sec_filings_raw'].update_one(
                {
                    'ticker': filing['ticker'],
                    'form_type': filing.get('form_type'),
                    'filing_date': filing.get('filing_date')
                },
                {'$set': filing},
                upsert=True
            )
            
            # Store filing sentiment
            if 'sentiment' in filing:
                self.db['sec_filings_sentiment'].update_one(
                    {
                        'ticker': filing['ticker'],
                        'filing_date': filing.get('filing_date')
                    },
                    {'$set': filing['sentiment']},
                    upsert=True
                )
                
            return True
            
        except Exception as e:
            logger.error(f"Error storing SEC filing: {e}")
            return False

    def store_macro_data(self, indicator: str, data: dict, source: str = 'AlphaVantage') -> bool:
        """Store macroeconomic data with proper validation"""
        try:
            if not indicator or not data:
                return False
                
            # Store raw data
            self.db['macro_data_raw'].update_one(
                {
                    'indicator': indicator,
                    'source': source,
                    'date': data.get('date')
                },
                {'$set': data},
                upsert=True
            )
            
            # Store processed data
            if 'processed' in data:
                self.db['macro_data_processed'].update_one(
                    {
                        'indicator': indicator,
                        'source': source,
                        'date': data.get('date')
                    },
                    {'$set': data['processed']},
                    upsert=True
                )
                
            return True
            
        except Exception as e:
            logger.error(f"Error storing macro data: {e}")
            return False

    def get_macro_data(self, indicator: str, start_date: str, end_date: str, source: str = None) -> dict:
        """Retrieve macro data for an indicator between start_date and end_date (inclusive). If source is given, only return that source. Returns {date: value}."""
        try:
            collection = self.db['macro_data']
            query = {
                'indicator': indicator,
                'date': {'$gte': pd.to_datetime(start_date), '$lte': pd.to_datetime(end_date)}
            }
            if source:
                query['source'] = source
            cursor = collection.find(query)
            # Only keep the latest value per date (if multiple sources, prefer source if given)
            result = {}
            for doc in cursor:
                date_str = doc['date'].strftime('%Y-%m-%d')
                if date_str not in result or (source and doc.get('source') == source):
                    result[date_str] = doc['value']
            return result
        except Exception as e:
            logger.error(f"Error retrieving macro data for {indicator}: {e}")
            return {}

    def get_insider_trading(self, ticker: str) -> list:
        """Return a list of insider trading transactions for a ticker from the 'insider_transactions' collection."""
        try:
            collection = self.db['insider_transactions']
            cursor = collection.find({'symbol': ticker})
            return list(cursor)
        except Exception as e:
            logger.error(f"Error fetching insider trading data for {ticker}: {e}")
            return []

    def close(self) -> None:
        """Close MongoDB connection."""
        try:
            if self.client:
                self.client.close()
                logger.info("Closed MongoDB connection")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {str(e)}")

if __name__ == "__main__":
    # Test MongoDB connection
    mongo_client = MongoDBClient()
    
    # Test storing predictions
    test_predictions = {
        "next_day": {"prediction": 100.0, "confidence": 0.8},
        "30_days": {"prediction": 105.0, "confidence": 0.7},
        "90_days": {"prediction": 110.0, "confidence": 0.6}
    }
    mongo_client.store_predictions("AAPL", test_predictions)
    
    # Test retrieving predictions
    predictions = mongo_client.get_latest_predictions("AAPL")
    print("Retrieved Predictions:")
    print(predictions)
    
    mongo_client.close() 