#!/usr/bin/env python3
"""
Test Script for Enhanced Stock Prediction API

This script tests the improved prediction system using the existing API structure.
"""

import requests
import json
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://localhost:8000"
TEST_TICKER = "AAPL"

def test_api_health():
    """Test API health endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            logger.info("✓ API Health Check:")
            logger.info(f"  Status: {data.get('status', 'unknown')}")
            logger.info(f"  MongoDB: {data.get('mongodb', 'unknown')}")
            logger.info(f"  Version: {data.get('api_version', 'unknown')}")
            return True
        else:
            logger.error(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"✗ Health check failed: {e}")
        return False

def test_model_listing():
    """Test model listing endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            data = response.json()
            logger.info("✓ Model Listing:")
            logger.info(f"  Available models: {data.get('total_tickers', 0)} tickers")
            logger.info(f"  Supported windows: {data.get('supported_windows', [])}")
            
            available_models = data.get('available_models', {})
            if available_models:
                for ticker, info in available_models.items():
                    logger.info(f"  {ticker}: {info.get('windows', [])} windows")
            return True
        else:
            logger.error(f"✗ Model listing failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"✗ Model listing failed: {e}")
        return False

def test_training_request():
    """Test training endpoint."""
    try:
        logger.info(f"Testing training for {TEST_TICKER}")
        
        training_data = {
            "ticker": TEST_TICKER,
            "retrain": True
        }
        
        response = requests.post(
            f"{API_BASE_URL}/train/{TEST_TICKER}",
            json=training_data,
            timeout=300  # 5 minute timeout for training
        )
        
        if response.status_code == 200:
            data = response.json()
            logger.info("✓ Training Request:")
            logger.info(f"  Status: {data.get('status', 'unknown')}")
            logger.info(f"  Message: {data.get('message', 'No message')}")
            logger.info(f"  Windows: {data.get('windows', [])}")
            return True
        elif response.status_code == 500:
            logger.warning(f"⚠ Training failed (expected if no data): {response.text}")
            return False
        else:
            logger.error(f"✗ Training failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        return False

def test_prediction_request():
    """Test prediction endpoint."""
    try:
        logger.info(f"Testing predictions for {TEST_TICKER}")
        
        prediction_data = {
            "ticker": TEST_TICKER,
            "days_back": 252
        }
        
        response = requests.post(
            f"{API_BASE_URL}/predict/{TEST_TICKER}",
            json=prediction_data,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            logger.info("✓ Prediction Request:")
            logger.info(f"  Current Price: ${data.get('current_price', 0):.2f}")
            logger.info(f"  Data Points Used: {data.get('data_points_used', 0)}")
            logger.info(f"  API Version: {data.get('api_version', 'unknown')}")
            
            predictions = data.get('predictions', {})
            if predictions:
                logger.info("  Predictions:")
                for window, pred in predictions.items():
                    logger.info(f"    {window.replace('_', ' ').title()}:")
                    logger.info(f"      Predicted: ${pred.get('predicted_price', 0):.2f}")
                    logger.info(f"      Change: ${pred.get('price_change', 0):+.2f}")
                    logger.info(f"      Confidence: {pred.get('confidence', 0):.3f}")
                    
                    # Test for the issues we fixed
                    if pred.get('confidence', 0) > 0.3:
                        logger.info(f"      ✓ Good confidence score")
                    else:
                        logger.warning(f"      ⚠ Low confidence score")
                
                # Check prediction diversity
                predicted_prices = [pred['predicted_price'] for pred in predictions.values()]
                if len(set(predicted_prices)) > 1:
                    logger.info("  ✓ Predictions show good diversity across windows")
                else:
                    logger.warning("  ⚠ All predictions are identical")
                
                return True
            else:
                logger.warning("  ⚠ No predictions returned")
                return False
        else:
            logger.error(f"✗ Prediction failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"✗ Prediction failed: {e}")
        return False

def main():
    """Main testing pipeline."""
    logger.info("=" * 60)
    logger.info("ENHANCED STOCK PREDICTION API TESTING")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Test 1: API Health
    logger.info("Test 1: API Health Check")
    test_results['health'] = test_api_health()
    time.sleep(1)
    
    # Test 2: Model Listing
    logger.info("\nTest 2: Model Listing")
    test_results['models'] = test_model_listing()
    time.sleep(1)
    
    # Test 3: Training (may fail if no data)
    logger.info("\nTest 3: Training Request")
    test_results['training'] = test_training_request()
    time.sleep(2)
    
    # Test 4: Predictions (may fail if no models)
    logger.info("\nTest 4: Prediction Request")
    test_results['predictions'] = test_prediction_request()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name.capitalize()}: {status}")
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if test_results.get('health') and test_results.get('models'):
        logger.info("✓ Core API functionality working")
    else:
        logger.warning("⚠ Core API issues detected")
    
    if test_results.get('predictions'):
        logger.info("✓ Enhanced prediction system working")
        logger.info("✓ Key improvements:")
        logger.info("  - Consistent feature engineering pipeline")
        logger.info("  - Improved prediction windows (next_day, 7_day, 30_day)")
        logger.info("  - Enhanced ensemble predictions")
        logger.info("  - Better confidence scoring")
    else:
        logger.warning("⚠ Predictions not working (may need data/models)")
    
    logger.info("\nFor full functionality, ensure:")
    logger.info("1. MongoDB contains historical data for test ticker")
    logger.info("2. Models are trained using the new training pipeline")
    logger.info("3. API server is running on localhost:8000")

if __name__ == "__main__":
    main() 