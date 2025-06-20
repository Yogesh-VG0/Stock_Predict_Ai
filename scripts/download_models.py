#!/usr/bin/env python3
"""
Download pre-trained models from cloud storage during deployment.
Supports AWS S3 and Google Cloud Storage.
"""

import os
import sys
import boto3
from google.cloud import storage as gcs
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_from_s3():
    """Download models from AWS S3."""
    try:
        bucket_name = os.getenv('AWS_S3_BUCKET_MODELS')
        if not bucket_name:
            logger.warning("AWS_S3_BUCKET_MODELS not set, skipping S3 download")
            return False
            
        s3_client = boto3.client('s3')
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        # List all objects in the models/ prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix='models/')
        
        downloaded_count = 0
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    local_path = Path(key)
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    logger.info(f"Downloading {key} to {local_path}")
                    s3_client.download_file(bucket_name, key, str(local_path))
                    downloaded_count += 1
        
        logger.info(f"Downloaded {downloaded_count} model files from S3")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading from S3: {e}")
        return False

def download_from_gcs():
    """Download models from Google Cloud Storage."""
    try:
        bucket_name = os.getenv('GOOGLE_CLOUD_STORAGE_BUCKET')
        if not bucket_name:
            logger.warning("GOOGLE_CLOUD_STORAGE_BUCKET not set, skipping GCS download")
            return False
            
        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        # List all blobs with models/ prefix
        blobs = bucket.list_blobs(prefix='models/')
        
        downloaded_count = 0
        for blob in blobs:
            if blob.name.endswith('/'):  # Skip directories
                continue
                
            local_path = Path(blob.name)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading {blob.name} to {local_path}")
            blob.download_to_filename(str(local_path))
            downloaded_count += 1
        
        logger.info(f"Downloaded {downloaded_count} model files from GCS")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading from GCS: {e}")
        return False

def create_model_registry():
    """Create a model registry file for tracking available models."""
    models_dir = Path('models')
    if not models_dir.exists():
        logger.warning("Models directory doesn't exist")
        return
        
    registry = {}
    
    # Scan for model files
    for ticker_dir in models_dir.iterdir():
        if ticker_dir.is_dir():
            ticker = ticker_dir.name
            registry[ticker] = {}
            
            for model_file in ticker_dir.glob('*.h5'):
                window = model_file.stem.split('_')[-2]  # Extract window from filename
                registry[ticker][window] = {
                    'lstm_model': str(model_file.relative_to(models_dir)),
                    'last_updated': os.path.getmtime(model_file)
                }
            
            for model_file in ticker_dir.glob('*.joblib'):
                parts = model_file.stem.split('_')
                if len(parts) >= 4:
                    window = parts[-3] + '_' + parts[-2]  # e.g., "30_day"
                    model_type = parts[-1]  # e.g., "lgbm", "xgb", "rf"
                    
                    if window not in registry[ticker]:
                        registry[ticker][window] = {}
                    
                    registry[ticker][window][f'{model_type}_model'] = str(model_file.relative_to(models_dir))
    
    # Save registry
    registry_path = models_dir / 'model_registry.json'
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    logger.info(f"Created model registry with {len(registry)} tickers")

def main():
    """Main function to download models from cloud storage."""
    logger.info("Starting model download process...")
    
    # Try both cloud storage providers
    s3_success = download_from_s3()
    gcs_success = download_from_gcs()
    
    if not s3_success and not gcs_success:
        logger.warning("No models downloaded from cloud storage. Using local models if available.")
        
        # Check if we have any local models
        models_dir = Path('models')
        if models_dir.exists() and any(models_dir.rglob('*.h5')):
            logger.info("Found local model files")
        else:
            logger.warning("No model files found locally either!")
            return 1
    
    # Create model registry
    create_model_registry()
    
    logger.info("Model download process completed")
    return 0

if __name__ == '__main__':
    sys.exit(main()) 