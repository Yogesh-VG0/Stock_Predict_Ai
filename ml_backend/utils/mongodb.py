"""
MongoDB integration for data storage and retrieval.
"""

from pymongo import MongoClient, ASCENDING, DESCENDING, UpdateOne
from pymongo.errors import ServerSelectionTimeoutError, BulkWriteError
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
try:
    from ml_backend.config.constants import (
        MONGO_COLLECTIONS,
        RETRY_CONFIG,
        PREDICTION_WINDOWS,
        ARTICLE_COUNT_VOLUME_KEYS,
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
        PREDICTION_WINDOWS,
        ARTICLE_COUNT_VOLUME_KEYS,
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
        """Connect to MongoDB server with connection pooling."""
        try:
            self.client = MongoClient(
                self.connection_string,
                # Connection pool
                maxPoolSize=50,
                minPoolSize=10,
                maxIdleTimeMS=45000,
                waitQueueTimeoutMS=5000,
                # Timeouts
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=20000,
                # Reliability
                retryWrites=True,
                retryReads=True,
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
            preds = self.collections[MONGO_COLLECTIONS["predictions"]]
            preds.create_index(
                [("ticker", ASCENDING), ("timestamp", DESCENDING)],
                name="idx_ticker_timestamp",
            )
            preds.create_index(
                [("ticker", ASCENDING), ("window", ASCENDING), ("timestamp", DESCENDING)],
                name="idx_ticker_window_timestamp",
            )
            # History index: one prediction per ticker-window-day (asof_date upsert key)
            preds.create_index(
                [("ticker", ASCENDING), ("window", ASCENDING), ("asof_date", DESCENDING)],
                name="idx_ticker_window_asof",
            )
            hist = self.collections[MONGO_COLLECTIONS["historical_data"]]
            try:
                hist.create_index(
                    [("ticker", ASCENDING), ("date", ASCENDING)],
                    name="idx_ticker_date",
                    unique=True,
                )
            except Exception as idx_err:
                # Fallback if duplicates exist: create non-unique index
                logger.warning(
                    "Could not create unique index on historical_data (duplicates may exist): %s. "
                    "Creating non-unique index. Run deduplication to enable upsert.",
                    idx_err,
                )
                hist.create_index(
                    [("ticker", ASCENDING), ("date", ASCENDING)],
                    name="idx_ticker_date",
                )
            sent = self.db["sentiment"]
            sent.create_index(
                [("ticker", ASCENDING), ("last_updated", DESCENDING)],
                name="idx_ticker_last_updated",
            )
            # Sentiment timeseries index (for ML feature queries by date range)
            sent.create_index(
                [("ticker", ASCENDING), ("date", ASCENDING)],
                name="idx_ticker_date_sent",
            )
            model_versions = self.collections[MONGO_COLLECTIONS["model_versions"]]
            model_versions.create_index(
                [("ticker", ASCENDING), ("window", ASCENDING), ("version", DESCENDING)],
                name="idx_ticker_window_version",
            )
            logger.info("Created indexes on collections")
        except Exception as e:
            logger.warning(f"Could not create indexes: {e}")

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
        """Store complete predictions data in MongoDB with all details.

        Each document is keyed on {ticker, window, asof_date} so we retain
        one prediction per ticker-window-day (history) while still allowing
        fast "get latest" queries via the existing timestamp index.
        """
        try:
            collection = self.collections[MONGO_COLLECTIONS["predictions"]]
            
            # Prepare documents for bulk insert
            documents = []
            timestamp = datetime.utcnow()
            # asof_date = calendar day the prediction was generated (midnight UTC).
            # This becomes part of the upsert key so re-runs on the same day
            # update in-place but different days accumulate history.
            asof_date = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            
            for window, data in predictions.items():
                # Skip _meta — it's metadata, not a prediction window
                if window == "_meta":
                    continue
                # Store COMPLETE prediction data instead of simplified version
                document = {
                    "ticker": ticker,
                    "window": window,
                    "asof_date": asof_date,
                    "timestamp": timestamp,
                    
                    # Core prediction data
                    "predicted_price": data.get("predicted_price"),
                    "price_change": data.get("price_change"),
                    "current_price": data.get("current_price"),
                    "confidence": float(data.get("confidence", 0.0)),
                    
                    # Price range data
                    "price_range": data.get("price_range", {}),
                    
                    # Individual model predictions
                    "model_predictions": data.get("model_predictions", {}),
                    
                    # Ensemble weights
                    "ensemble_weights": data.get("ensemble_weight", data.get("ensemble_weights", {})),
                    
                    # prediction must remain the model signal (log-return), not dollars
                    "prediction": float(data.get("prediction", data.get("alpha", 0.0)))
                }
                
                # Add any additional fields that might be present, excluding duplicates
                excluded_fields = ['prediction', 'predicted_price', 'price_change', 'current_price', 
                                 'confidence', 'price_range', 'model_predictions', 'ensemble_weight', 'ensemble_weights']
                
                for key, value in data.items():
                    if key not in document and key not in excluded_fields:
                        try:
                            # Ensure all numeric values are properly converted
                            if isinstance(value, (int, float)):
                                document[key] = float(value)
                            elif isinstance(value, dict):
                                document[key] = value
                            elif isinstance(value, list):
                                document[key] = value
                            else:
                                document[key] = str(value)
                        except Exception as e:
                            logger.warning(f"Could not store field {key} for {ticker}-{window}: {e}")
                
                documents.append(document)
            
            # Upsert: replace the existing doc for each ticker+window+asof_date.
            # This keeps one prediction per day (history) while preventing
            # unbounded growth from multiple intra-day reruns.
            from pymongo import UpdateOne as _UpdateOne
            ops = [
                _UpdateOne(
                    {"ticker": doc["ticker"], "window": doc["window"], "asof_date": doc["asof_date"]},
                    {"$set": doc},
                    upsert=True,
                )
                for doc in documents
            ]
            result = collection.bulk_write(ops)
            logger.info(
                f"Upserted {result.upserted_count + result.modified_count} predictions for {ticker} "
                f"(upserted={result.upserted_count}, modified={result.modified_count})"
            )
            
            # Log what was stored for verification
            for doc in documents:
                logger.info(f"Stored prediction for {ticker}-{doc['window']}: "
                          f"Price ${doc.get('predicted_price', 'N/A')}, "
                          f"Change ${doc.get('price_change', 'N/A')}, "
                          f"Confidence {doc.get('confidence', 'N/A'):.3f}")
            
            return True
            
        except BulkWriteError as e:
            logger.error(f"Error storing predictions for {ticker}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error storing predictions for {ticker}: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
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
                ops = [
                    UpdateOne(
                        {"ticker": doc["ticker"], "date": doc["date"]},
                        {"$set": doc},
                        upsert=True,
                    )
                    for doc in documents
                ]
                result = collection.bulk_write(ops, ordered=False)
                logger.info(
                    f"Upserted {result.upserted_count + result.modified_count + result.matched_count} "
                    f"historical data points for {ticker} in historical_data collection"
                )
            return True
        except BulkWriteError as e:
            logger.error(f"Error storing historical data for {ticker}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error storing historical data for {ticker}: {str(e)}")
            return False

    def store_sentiment(self, ticker: str, sentiment: dict):
        collection = self.db['sentiment']
        sentiment['ticker'] = ticker
        sentiment['last_updated'] = datetime.utcnow()
        # Respect pre-set trading-day-aligned date from get_combined_sentiment().
        # Only fall back to UTC midnight if caller didn't set it.
        if 'date' not in sentiment or sentiment['date'] is None:
            sentiment['date'] = sentiment['timestamp'].replace(
                hour=0, minute=0, second=0, microsecond=0
            )

        # Stale fields from disabled sources (alphavantage, alpha_earnings_call)
        # linger in existing docs after $set upserts.  Explicitly remove them so
        # the stored document only contains live-source data.
        _stale_fields = [
            "alphavantage_sentiment", "alphavantage_volume", "alphavantage_confidence",
            "alpha_earnings_call_sentiment", "alpha_earnings_call_volume",
            "alpha_earnings_call_confidence",
        ]
        # Strip stale keys from the $set payload (shouldn't be there but guard)
        for k in _stale_fields:
            sentiment.pop(k, None)

        # Upsert on {ticker, date} so only the latest sentiment per ticker per day is kept
        collection.update_one(
            {"ticker": ticker, "date": sentiment["date"]},
            {
                "$set": sentiment,
                "$unset": {k: "" for k in _stale_fields},
            },
            upsert=True,
        )

    def get_latest_predictions(self, ticker: str) -> Dict[str, Dict[str, float]]:
        """Get latest complete predictions for a ticker."""
        try:
            collection = self.collections[MONGO_COLLECTIONS["predictions"]]
            
            # Find latest predictions for each window
            predictions = {}
            
            # Get the most recent timestamp
            latest_doc = collection.find_one(
                {"ticker": ticker},
                sort=[("timestamp", DESCENDING)]
            )
            
            if not latest_doc:
                return {}
            
            latest_timestamp = latest_doc["timestamp"]
            
            # Get all predictions from the latest timestamp
            cursor = collection.find(
                {
                    "ticker": ticker,
                    "timestamp": latest_timestamp
                }
            )
            
            for doc in cursor:
                window = doc["window"]
                # Skip _meta rows that may exist from older pipeline runs
                if window == "_meta":
                    continue
                
                # Return COMPLETE prediction data
                prediction_data = {
                    "predicted_price": doc.get("predicted_price"),
                    "price_change": doc.get("price_change"),
                    "current_price": doc.get("current_price"),
                    "confidence": doc.get("confidence", 0.0),
                    "price_range": doc.get("price_range", {}),
                    "model_predictions": doc.get("model_predictions", {}),
                    "ensemble_weights": doc.get("ensemble_weights", {}),
                    "timestamp": doc.get("timestamp"),
                    
                    # Backward compatibility
                    "prediction": doc.get("prediction", doc.get("price_change", 0.0))
                }
                
                # Add any additional fields that were stored
                for key, value in doc.items():
                    if key not in prediction_data and key not in ['_id', 'ticker', 'window']:
                        prediction_data[key] = value
                
                predictions[window] = prediction_data
            
            logger.info(f"Retrieved {len(predictions)} complete predictions for {ticker}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting latest predictions for {ticker}: {str(e)}")
            return {}

    def get_prediction_history(
        self,
        ticker: str,
        window: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 90,
    ) -> list:
        """Return prediction history for a ticker-window pair, newest first.

        Each document includes asof_date, prediction, confidence,
        predicted_price, etc.  Used for drift detection and accuracy tracking.
        """
        try:
            collection = self.collections[MONGO_COLLECTIONS["predictions"]]
            query: Dict[str, Any] = {"ticker": ticker, "window": window}
            if start_date or end_date:
                date_filter: Dict[str, Any] = {}
                if start_date:
                    date_filter["$gte"] = start_date
                if end_date:
                    date_filter["$lte"] = end_date
                query["asof_date"] = date_filter

            cursor = (
                collection.find(query, {"_id": 0})
                .sort("asof_date", DESCENDING)
                .limit(limit)
            )
            return list(cursor)
        except Exception as e:
            logger.error("Error getting prediction history for %s-%s: %s", ticker, window, e)
            return []

    def get_sentiment_timeseries(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Return daily sentiment rows for a ticker in [start_date, end_date].

        Returns DataFrame with columns: date, composite_sentiment, news_count.
        Used by the ML feature engine to build sentiment features.
        """
        try:
            collection = self.db["sentiment"]
            cursor = collection.find(
                {
                    "ticker": ticker,
                    "date": {"$gte": start_date, "$lte": end_date},
                },
                sort=[("date", ASCENDING)],
            )
            rows = []
            for doc in cursor:
                # Prefer the explicit composite_sentiment field (written by
                # get_combined_sentiment) which is the weighted blend.
                if "composite_sentiment" in doc and isinstance(doc["composite_sentiment"], (int, float)):
                    composite = float(doc["composite_sentiment"])
                elif "blended_sentiment" in doc and isinstance(doc["blended_sentiment"], (int, float)):
                    composite = float(doc["blended_sentiment"])
                else:
                    # Fallback: average per-source *_sentiment keys.
                    # Exclude the aggregate keys so we don't double-count.
                    _exclude = {"blended_sentiment", "composite_sentiment", "sentiment_confidence"}
                    sent_fields = [
                        v for k, v in doc.items()
                        if k.endswith("_sentiment")
                        and k not in _exclude
                        and isinstance(v, (int, float))
                    ]
                    if sent_fields:
                        composite = float(np.mean(sent_fields))
                    elif "sources" in doc and isinstance(doc["sources"], dict):
                        scores = [
                            s["sentiment_score"]
                            for s in doc["sources"].values()
                            if isinstance(s, dict) and "sentiment_score" in s
                        ]
                        composite = float(np.mean(scores)) if scores else 0.0
                    else:
                        composite = 0.0

                # Prefer the explicit news_count field.
                if "news_count" in doc and isinstance(doc["news_count"], (int, float)):
                    news_count = int(doc["news_count"])
                else:
                    # Fallback: allowlist-only (single source of truth).
                    # ARTICLE_COUNT_VOLUME_KEYS lives in config/constants.py.
                    count_fields = [
                        v for k, v in doc.items()
                        if k in ARTICLE_COUNT_VOLUME_KEYS
                        and isinstance(v, (int, float))
                    ]
                    news_count = int(sum(count_fields)) if count_fields else 1

                rows.append({
                    "date": pd.Timestamp(doc["date"]).normalize(),
                    "composite_sentiment": composite,
                    "news_count": news_count,
                })
            if not rows:
                return pd.DataFrame(columns=["date", "composite_sentiment", "news_count"])
            df = pd.DataFrame(rows)
            df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
            return df
        except Exception as e:
            logger.error("Error getting sentiment timeseries for %s: %s", ticker, e)
            return pd.DataFrame(columns=["date", "composite_sentiment", "news_count"])

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
        """Validate sentiment data structure (supports both flat and nested formats)."""
        try:
            required_fields = ['ticker', 'timestamp']
            if not all(field in sentiment for field in required_fields):
                return False

            # Check if it's the new flat format (preferred)
            if any(key.endswith('_sentiment') for key in sentiment.keys()):
                # Flat format - look for sentiment fields like finviz_sentiment, sec_sentiment, etc.
                sentiment_fields = [k for k in sentiment.keys() if k.endswith('_sentiment')]
                if len(sentiment_fields) > 0:
                    return True
            
            # Check if it's the old nested format (for backward compatibility)
            elif 'sources' in sentiment:
                if not isinstance(sentiment['sources'], dict):
                    return False

                # Validate each source
                for source, data in sentiment['sources'].items():
                    if not isinstance(data, dict):
                        return False
                    if not all(key in data for key in ['sentiment_score', 'volume', 'confidence']):
                        return False
                return True

            return False

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
            
            # Upsert on {ticker, window, version} so retrains update in place
            collection.update_one(
                {"ticker": ticker, "window": window, "version": version},
                {"$set": document},
                upsert=True,
            )
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
            
            # Upsert on {ticker, window} so only the latest metrics are kept
            collection.update_one(
                {"ticker": ticker, "window": window},
                {"$set": document},
                upsert=True,
            )
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
        """Store Alpha Vantage data with proper validation and error handling"""
        try:
            if not ticker or not endpoint:
                return False
            
            # Validate data structure - handle both dict and list cases
            if isinstance(data, list):
                logger.warning(f"Alpha Vantage data is a list, converting to dict structure for {ticker}/{endpoint}")
                data = {
                    'data': data,
                    'timestamp': datetime.utcnow(),
                    'ticker': ticker,
                    'endpoint': endpoint
                }
            elif not isinstance(data, dict):
                logger.warning(f"Alpha Vantage data is not dict or list, skipping storage for {ticker}/{endpoint}")
                return False
            
            # Ensure required fields exist
            if 'timestamp' not in data:
                data['timestamp'] = datetime.utcnow()
            if 'ticker' not in data:
                data['ticker'] = ticker
            if 'endpoint' not in data:
                data['endpoint'] = endpoint
                
            # Store raw data
            self.db['alpha_vantage_data'].update_one(
                {
                    'ticker': ticker,
                    'endpoint': endpoint
                },
                {'$set': data},
                upsert=True
            )
            
            # Store processed data if available
            if isinstance(data, dict) and 'processed' in data:
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
            logger.error(f"Error storing Alpha Vantage data for {ticker}/{endpoint}: {e}")
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

    def optimize_sentiment_storage(self, ticker: str, sentiment: dict):
        """Optimized sentiment storage with deduplication and compression."""
        try:
            collection = self.db['sentiment']
            
            # Check for recent duplicate data
            recent_sentiment = collection.find_one(
                {
                    'ticker': ticker,
                    'timestamp': {'$gte': datetime.utcnow() - timedelta(hours=1)}
                },
                sort=[('timestamp', -1)]
            )
            
            # Only store if data has changed significantly
            if recent_sentiment:
                # Compare sentiment scores for significant changes
                threshold = 0.05  # 5% change threshold
                has_significant_change = False
                
                for key in sentiment.keys():
                    if key.endswith('_sentiment'):
                        old_val = recent_sentiment.get(key, 0)
                        new_val = sentiment.get(key, 0)
                        if abs(new_val - old_val) > threshold:
                            has_significant_change = True
                            break
                
                if not has_significant_change:
                    logger.info(f"No significant sentiment change for {ticker}, skipping storage")
                    return True
            
            # Store with compression for large sentiment objects
            if len(str(sentiment)) > 10000:  # If sentiment data is large
                import gzip
                import pickle
                compressed_data = gzip.compress(pickle.dumps(sentiment))
                sentiment['_compressed'] = True
                sentiment['_data'] = compressed_data
            
            collection.insert_one({
                **sentiment,
                'ticker': ticker,
                'timestamp': datetime.utcnow()
            })
            
            # Cleanup old data (keep only last 30 days)
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            collection.delete_many({
                'ticker': ticker,
                'timestamp': {'$lt': cutoff_date}
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error in optimized sentiment storage for {ticker}: {e}")
            return False

    def get_cached_data(self, cache_key: str, max_age_hours: int = 1) -> Optional[Dict]:
        """Generic cached data retrieval with age validation."""
        try:
            collection = self.db['api_cache']
            
            cached = collection.find_one({'cache_key': cache_key})
            if cached:
                age = (datetime.utcnow() - cached['timestamp']).total_seconds() / 3600
                if age < max_age_hours:
                    # Decompress if needed
                    if cached.get('_compressed'):
                        import gzip
                        import pickle
                        return pickle.loads(gzip.decompress(cached['_data']))
                    return cached.get('data')
                else:
                    # Remove expired data
                    collection.delete_one({'cache_key': cache_key})
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving cached data for {cache_key}: {e}")
            return None

    def store_cached_data(self, cache_key: str, data: Dict, expire_hours: int = 1):
        """Generic cached data storage with compression for large objects."""
        try:
            collection = self.db['api_cache']
            
            cache_doc = {
                'cache_key': cache_key,
                'timestamp': datetime.utcnow(),
                'expire_hours': expire_hours
            }
            
            # Compress large data
            if len(str(data)) > 5000:
                import gzip
                import pickle
                cache_doc['_compressed'] = True
                cache_doc['_data'] = gzip.compress(pickle.dumps(data))
            else:
                cache_doc['data'] = data
            
            collection.replace_one(
                {'cache_key': cache_key},
                cache_doc,
                upsert=True
            )
            
            # Cleanup expired cache entries
            expiry_time = datetime.utcnow() - timedelta(hours=24)  # Clean entries older than 24h
            collection.delete_many({'timestamp': {'$lt': expiry_time}})
            
        except Exception as e:
            logger.error(f"Error storing cached data for {cache_key}: {e}")

    def create_advanced_indexes(self):
        """Create advanced indexes for optimal performance."""
        try:
            # Sentiment data indexes
            sentiment_collection = self.db['sentiment']
            sentiment_collection.create_index([
                ("ticker", 1), 
                ("timestamp", -1)
            ], background=True)
            
            # Cache collection indexes
            self.db['api_cache'].create_index([
                ("cache_key", 1)
            ], unique=True, background=True)
            
            self.db['api_cache'].create_index([
                ("timestamp", 1)
            ], expireAfterSeconds=86400*2, background=True)  # Auto-expire after 2 days
            
            # Feature importance index
            self.db['feature_importance'].create_index([
                ("ticker", 1),
                ("window", 1),
                ("timestamp", -1)
            ], background=True)
            
            logger.info("Created advanced MongoDB indexes for optimization")
            
        except Exception as e:
            logger.error(f"Error creating advanced indexes: {e}")

    def store_prediction_explanation(self, ticker: str, window: str, explanation_data: Dict) -> bool:
        """Store prediction explanation data including feature importance and LLM explanations."""
        try:
            collection = self.db['prediction_explanations']
            
            # Convert numpy types to native Python types for MongoDB compatibility
            def convert_numpy_types(obj):
                """Recursively convert numpy types to native Python types."""
                import numpy as np
                
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_numpy_types(item) for item in obj)
                else:
                    return obj
            
            # Convert the explanation data
            converted_explanation_data = convert_numpy_types(explanation_data)
            
            document = {
                "ticker": ticker,
                "window": window,
                "timestamp": datetime.utcnow(),
                "explanation_data": converted_explanation_data
            }
            
            result = collection.insert_one(document)
            logger.info(f"Stored prediction explanation for {ticker}-{window}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing prediction explanation for {ticker}-{window}: {e}")
            return False

    def get_prediction_explanation(self, ticker: str, window: str) -> Dict:
        """Get latest prediction explanation for a ticker and window."""
        try:
            collection = self.db['prediction_explanations']
            
            explanation = collection.find_one(
                {"ticker": ticker, "window": window},
                sort=[("timestamp", DESCENDING)]
            )
            
            if explanation:
                return explanation.get("explanation_data", {})
            return {}
            
        except Exception as e:
            logger.error(f"Error getting prediction explanation for {ticker}-{window}: {e}")
            return {}

    def store_complete_prediction_session(self, ticker: str, session_data: Dict) -> bool:
        """Store complete prediction session with all models, features, and explanations."""
        try:
            collection = self.db['prediction_sessions']
            
            document = {
                "ticker": ticker,
                "timestamp": datetime.utcnow(),
                "session_id": f"{ticker}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                **session_data
            }
            
            result = collection.insert_one(document)
            logger.info(f"Stored complete prediction session for {ticker}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing prediction session for {ticker}: {e}")
            return False

    def get_prediction_history_simple(self, ticker: str, days: int = 30) -> List[Dict]:
        """Get prediction history for a ticker over specified days (API use)."""
        try:
            collection = self.collections[MONGO_COLLECTIONS["predictions"]]
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            cursor = collection.find(
                {
                    "ticker": ticker,
                    "timestamp": {"$gte": cutoff_date}
                },
                sort=[("timestamp", DESCENDING)]
            )
            
            return list(cursor)
            
        except Exception as e:
            logger.error(f"Error getting prediction history for {ticker}: {e}")
            return []
    
    def store_sector_data(self, etf: str, data_dict: Dict) -> bool:
        """Store sector ETF data in MongoDB."""
        try:
            collection = self.db['sector_data']
            
            doc = {
                'etf': etf,
                'data': data_dict,
                'timestamp': datetime.utcnow(),
                'last_updated': datetime.utcnow()
            }
            
            # Use upsert to update if exists
            result = collection.replace_one(
                {'etf': etf},
                doc,
                upsert=True
            )
            
            logger.info(f"Stored sector data for {etf} with {len(data_dict)} data points")
            return result.acknowledged
            
        except Exception as e:
            logger.error(f"Error storing sector data for {etf}: {e}")
            return False
    
    def get_sector_data(self, etf: str, start_date=None, end_date=None) -> Dict:
        """Get sector ETF data from MongoDB."""
        try:
            collection = self.db['sector_data']
            query = {'etf': etf}
            
            doc = collection.find_one(query, sort=[('timestamp', -1)])
            
            if doc and 'data' in doc:
                data = doc['data']
                
                # Filter by date range if specified
                if start_date and end_date:
                    start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
                    end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
                    
                    filtered_data = {
                        date: value for date, value in data.items()
                        if start_str <= date <= end_str
                    }
                    logger.info(f"Retrieved filtered sector data for {etf}: {len(filtered_data)} points")
                    return filtered_data
                    
                logger.info(f"Retrieved sector data for {etf}: {len(data)} points")
                return data
                
            logger.warning(f"No sector data found for {etf}")
            return {}
            
        except Exception as e:
            logger.error(f"Error getting sector data for {etf}: {e}")
            return {}

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