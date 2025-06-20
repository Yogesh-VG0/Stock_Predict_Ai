#!/usr/bin/env python3
"""
Train models and upload them to cloud storage.
Runs weekly as a cron job to keep models updated.
"""

import os
import sys
import boto3
from google.cloud import storage as gcs
import logging
from pathlib import Path
import json
from datetime import datetime
import traceback

# Add ml_backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml_backend.utils.mongodb import MongoDBClient
from ml_backend.data.features import FeatureEngineer
from ml_backend.models.predictor import StockPredictor
from ml_backend.config.constants import TOP_100_TICKERS, MONGODB_URI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_to_s3(local_path, s3_key):
    """Upload a file to AWS S3."""
    try:
        bucket_name = os.getenv('AWS_S3_BUCKET_MODELS')
        if not bucket_name:
            return False
            
        s3_client = boto3.client('s3')
        s3_client.upload_file(str(local_path), bucket_name, s3_key)
        logger.info(f"Uploaded {local_path} to s3://{bucket_name}/{s3_key}")
        return True
        
    except Exception as e:
        logger.error(f"Error uploading {local_path} to S3: {e}")
        return False

def upload_to_gcs(local_path, blob_name):
    """Upload a file to Google Cloud Storage."""
    try:
        bucket_name = os.getenv('GOOGLE_CLOUD_STORAGE_BUCKET')
        if not bucket_name:
            return False
            
        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        blob.upload_from_filename(str(local_path))
        logger.info(f"Uploaded {local_path} to gs://{bucket_name}/{blob_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error uploading {local_path} to GCS: {e}")
        return False

def upload_models():
    """Upload all model files to cloud storage."""
    models_dir = Path('models')
    if not models_dir.exists():
        logger.error("Models directory doesn't exist")
        return False
    
    uploaded_count = 0
    failed_count = 0
    
    # Upload all model files
    for model_file in models_dir.rglob('*'):
        if model_file.is_file() and model_file.suffix in ['.h5', '.joblib', '.json']:
            relative_path = model_file.relative_to(Path('.'))
            s3_key = str(relative_path).replace('\\', '/')  # Normalize path separators
            
            # Upload to both S3 and GCS (redundancy)
            s3_success = upload_to_s3(model_file, s3_key)
            gcs_success = upload_to_gcs(model_file, s3_key)
            
            if s3_success or gcs_success:
                uploaded_count += 1
            else:
                failed_count += 1
                logger.error(f"Failed to upload {model_file}")
    
    logger.info(f"Model upload completed: {uploaded_count} successful, {failed_count} failed")
    return failed_count == 0

def load_historical_data(mongo_client, ticker, start_date=None, end_date=None):
    """Load historical data for a ticker from MongoDB."""
    try:
        collection = mongo_client.db['historical_data']
        
        query = {'ticker': ticker}
        if start_date:
            query['date'] = {'$gte': start_date}
        if end_date:
            if 'date' in query:
                query['date']['$lte'] = end_date
            else:
                query['date'] = {'$lte': end_date}
        
        cursor = collection.find(query).sort('date', 1)
        data = list(cursor)
        
        if not data:
            logger.warning(f"No historical data found for {ticker}")
            return None
        
        # Convert to DataFrame format expected by the models
        import pandas as pd
        df = pd.DataFrame(data)
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column {col} missing for {ticker}")
                return None
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading historical data for {ticker}: {e}")
        return None

def train_models_for_ticker(ticker, mongo_client, feature_engineer, predictor):
    """Train all models for a specific ticker."""
    try:
        logger.info(f"Training models for {ticker}")
        
        # Load historical data
        df = load_historical_data(mongo_client, ticker)
        if df is None:
            logger.error(f"Cannot train models for {ticker} - no data available")
            return False
        
        # Ensure minimum data requirements
        if len(df) < 500:  # Need sufficient data for training
            logger.warning(f"Insufficient data for {ticker}: {len(df)} rows")
            return False
        
        # Engineer features
        features_data = feature_engineer.engineer_features(df, ticker)
        if features_data is None or features_data.empty:
            logger.error(f"Feature engineering failed for {ticker}")
            return False
        
        # Train models for all prediction windows
        success_count = 0
        for window in ['next_day', '30_day', '90_day']:
            try:
                logger.info(f"Training {ticker} - {window} models")
                
                # Prepare target variable based on window
                if window == 'next_day':
                    target = features_data['Close'].shift(-1)
                elif window == '30_day':
                    target = features_data['Close'].rolling(30).mean().shift(-30)
                elif window == '90_day':
                    target = features_data['Close'].rolling(90).mean().shift(-90)
                
                # Remove rows with NaN targets
                valid_indices = ~target.isna()
                X = features_data[valid_indices].select_dtypes(include=[float, int])
                y = target[valid_indices]
                
                if len(X) < 100:  # Minimum training samples
                    logger.warning(f"Insufficient samples for {ticker}-{window}: {len(X)}")
                    continue
                
                # Train models
                models_trained = predictor.train_models(ticker, X.values, y.values, window)
                
                if models_trained:
                    success_count += 1
                    logger.info(f"Successfully trained {ticker}-{window} models")
                else:
                    logger.error(f"Failed to train {ticker}-{window} models")
                    
            except Exception as e:
                logger.error(f"Error training {ticker}-{window}: {e}")
                continue
        
        if success_count > 0:
            # Save models
            predictor.save_models()
            logger.info(f"Training completed for {ticker}: {success_count}/3 windows successful")
            return True
        else:
            logger.error(f"All model training failed for {ticker}")
            return False
        
    except Exception as e:
        logger.error(f"Error in train_models_for_ticker for {ticker}: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function for training and uploading models."""
    logger.info("Starting model training and upload process...")
    
    try:
        # Initialize components
        mongo_client = MongoDBClient(MONGODB_URI)
        if mongo_client.db is None:
            logger.error("Failed to connect to MongoDB")
            return 1
        
        feature_engineer = FeatureEngineer()
        predictor = StockPredictor(feature_engineer=feature_engineer)
        
        # Create models directory
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        # Train models for top tickers (subset for efficiency)
        priority_tickers = TOP_100_TICKERS[:20]  # Train top 20 each week
        successful_training = 0
        failed_training = 0
        
        for ticker in priority_tickers:
            try:
                logger.info(f"Processing ticker {ticker} ({priority_tickers.index(ticker)+1}/{len(priority_tickers)})")
                
                success = train_models_for_ticker(ticker, mongo_client, feature_engineer, predictor)
                if success:
                    successful_training += 1
                else:
                    failed_training += 1
                    
            except Exception as e:
                logger.error(f"Failed to process {ticker}: {e}")
                failed_training += 1
                continue
        
        logger.info(f"Training completed: {successful_training} successful, {failed_training} failed")
        
        # Upload models to cloud storage
        if successful_training > 0:
            upload_success = upload_models()
            if upload_success:
                logger.info("All models uploaded successfully")
            else:
                logger.warning("Some model uploads failed")
        
        # Update training metadata
        training_metadata = {
            'last_training_date': datetime.utcnow().isoformat(),
            'successful_tickers': successful_training,
            'failed_tickers': failed_training,
            'total_tickers': len(priority_tickers)
        }
        
        metadata_path = models_dir / 'training_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(training_metadata, f, indent=2)
        
        # Upload metadata
        upload_to_s3(metadata_path, 'models/training_metadata.json')
        upload_to_gcs(metadata_path, 'models/training_metadata.json')
        
        logger.info("Model training and upload process completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Critical error in main process: {e}")
        logger.error(traceback.format_exc())
        return 1
    
    finally:
        if 'mongo_client' in locals():
            mongo_client.close()

if __name__ == '__main__':
    sys.exit(main()) 