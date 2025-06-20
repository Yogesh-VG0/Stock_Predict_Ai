#!/usr/bin/env python3
"""
Comprehensive Training and Testing Script for Enhanced Stock Prediction

This script tests the improved feature engineering pipeline with:
1. Consistent feature engineering between training and prediction
2. New prediction windows: next_day, 7_day, 30_day
3. Enhanced ensemble predictions
4. Backtesting for validation
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# Add ML backend to path
sys.path.append('ml_backend')

from ml_backend.models.predictor import StockPredictor
from ml_backend.utils.mongodb import MongoDBClient
from ml_backend.config.constants import MONGODB_URI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_feature_consistency(predictor, ticker, df):
    """Test that training and prediction use consistent features."""
    logger.info(f"Testing feature consistency for {ticker}")
    
    # Test each window
    for window in predictor.prediction_windows:
        logger.info(f"Testing {window} window feature consistency")
        
        try:
            # Check if pipeline file exists
            pipeline_path = f"models/{ticker}/feature_pipeline_{ticker}_{window}.json"
            if os.path.exists(pipeline_path):
                logger.info(f"✓ Feature pipeline found for {ticker}-{window}")
                
                # Test feature creation
                prediction_features = predictor.feature_engineer.create_prediction_features(
                    df=df.copy(),
                    ticker=ticker,
                    window=window,
                    mongo_client=predictor.mongo_client
                )
                
                if prediction_features is not None:
                    logger.info(f"✓ Features created successfully for {ticker}-{window}: shape {prediction_features.shape}")
                else:
                    logger.warning(f"✗ Failed to create features for {ticker}-{window}")
            else:
                logger.warning(f"✗ No feature pipeline found for {ticker}-{window}")
                
        except Exception as e:
            logger.error(f"✗ Error testing {ticker}-{window}: {e}")

def perform_backtest(predictor, ticker, test_start_date):
    """Perform backtesting to validate prediction accuracy."""
    logger.info(f"Performing backtest for {ticker} from {test_start_date}")
    
    try:
        # Get historical data for backtesting
        collection = predictor.mongo_client.db.stock_data
        
        # Get data from test start to now
        query = {
            'ticker': ticker,
            'date': {'$gte': test_start_date, '$lte': datetime.now()}
        }
        
        cursor = collection.find(query).sort('date', 1)
        test_data = list(cursor)
        
        if len(test_data) < 30:
            logger.warning(f"Insufficient test data for {ticker}: {len(test_data)} records")
            return None
        
        df_test = pd.DataFrame(test_data)
        df_test = df_test.set_index('date') if 'date' in df_test.columns else df_test
        
        # Make predictions on historical data
        predictions_history = []
        actual_prices = []
        
        # Test predictions at different points
        for i in range(0, len(df_test) - 7, 7):  # Weekly intervals
            # Get data up to this point
            df_current = df_test.iloc[:i+1]
            
            if len(df_current) < 50:  # Need enough history
                continue
                
            # Make prediction
            predictions = predictor.predict_all_windows(ticker, df_current)
            
            if predictions:
                current_price = float(df_current['Close'].iloc[-1])
                
                # Check next day prediction (if we have actual data)
                if i + 1 < len(df_test):
                    actual_next_day = float(df_test['Close'].iloc[i + 1])
                    
                    if 'next_day' in predictions:
                        pred_next_day = predictions['next_day']['predicted_price']
                        error = abs(pred_next_day - actual_next_day) / actual_next_day * 100
                        
                        predictions_history.append({
                            'date': df_current.index[-1],
                            'current_price': current_price,
                            'predicted_next_day': pred_next_day,
                            'actual_next_day': actual_next_day,
                            'error_pct': error,
                            'confidence': predictions['next_day']['confidence']
                        })
        
        if predictions_history:
            df_backtest = pd.DataFrame(predictions_history)
            
            # Calculate metrics
            mean_error = df_backtest['error_pct'].mean()
            median_error = df_backtest['error_pct'].median()
            accuracy_within_5pct = (df_backtest['error_pct'] <= 5).mean() * 100
            
            logger.info(f"Backtest Results for {ticker}:")
            logger.info(f"  Mean Error: {mean_error:.2f}%")
            logger.info(f"  Median Error: {median_error:.2f}%")
            logger.info(f"  Accuracy within 5%: {accuracy_within_5pct:.1f}%")
            logger.info(f"  Predictions tested: {len(predictions_history)}")
            
            return {
                'mean_error': mean_error,
                'median_error': median_error,
                'accuracy_5pct': accuracy_within_5pct,
                'predictions_count': len(predictions_history),
                'backtest_data': df_backtest
            }
        
    except Exception as e:
        logger.error(f"Error in backtest for {ticker}: {e}")
    
    return None

def test_prediction_diversity(predictor, ticker, df):
    """Test that different windows give different predictions."""
    logger.info(f"Testing prediction diversity for {ticker}")
    
    try:
        predictions = predictor.predict_all_windows(ticker, df)
        
        if not predictions:
            logger.warning(f"No predictions generated for {ticker}")
            return False
        
        # Check that we have predictions for different windows
        windows_with_predictions = list(predictions.keys())
        logger.info(f"Predictions available for windows: {windows_with_predictions}")
        
        # Check that predictions are different
        predicted_prices = [pred['predicted_price'] for pred in predictions.values()]
        confidence_scores = [pred['confidence'] for pred in predictions.values()]
        
        # Predictions should not all be identical
        price_diversity = len(set(predicted_prices)) > 1
        confidence_diversity = len(set(confidence_scores)) > 1
        
        logger.info(f"Price diversity: {price_diversity} (prices: {predicted_prices})")
        logger.info(f"Confidence diversity: {confidence_diversity} (confidences: {confidence_scores})")
        
        # Check confidence scores are reasonable (not all 0.1)
        reasonable_confidence = any(conf > 0.3 for conf in confidence_scores)
        logger.info(f"Reasonable confidence scores: {reasonable_confidence}")
        
        return price_diversity and reasonable_confidence
        
    except Exception as e:
        logger.error(f"Error testing diversity for {ticker}: {e}")
        return False

def main():
    """Main training and testing pipeline."""
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE STOCK PREDICTION TRAINING & TESTING")
    logger.info("=" * 60)
    
    # Initialize services
    mongo_client = MongoDBClient(MONGODB_URI)
    predictor = StockPredictor(mongo_client)
    
    # Test with AAPL (should have good data)
    ticker = "AAPL"
    
    # Step 1: Get recent data for testing
    logger.info(f"Step 1: Fetching recent data for {ticker}")
    
    collection = mongo_client.db.stock_data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year
    
    query = {
        'ticker': ticker,
        'date': {'$gte': start_date, '$lte': end_date}
    }
    
    cursor = collection.find(query).sort('date', 1)
    data = list(cursor)
    
    if len(data) < 100:
        logger.error(f"Insufficient data for {ticker}: {len(data)} records")
        return
    
    df = pd.DataFrame(data)
    df = df.set_index('date') if 'date' in df.columns else df
    
    logger.info(f"✓ Data loaded: {len(df)} records from {df.index[0]} to {df.index[-1]}")
    
    # Step 2: Train models with improved pipeline
    logger.info(f"Step 2: Training models for {ticker}")
    
    # Use 80% for training, 20% for testing
    train_split = int(0.8 * len(df))
    train_end_date = df.index[train_split].strftime('%Y-%m-%d')
    
    success = predictor.train_all_models(
        ticker=ticker,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=train_end_date
    )
    
    if not success:
        logger.error(f"Training failed for {ticker}")
        return
    
    logger.info(f"✓ Training completed for {ticker}")
    
    # Step 3: Test feature consistency
    logger.info(f"Step 3: Testing feature consistency")
    validate_feature_consistency(predictor, ticker, df)
    
    # Step 4: Test prediction diversity
    logger.info(f"Step 4: Testing prediction diversity")
    diversity_ok = test_prediction_diversity(predictor, ticker, df)
    
    if diversity_ok:
        logger.info("✓ Prediction diversity test passed")
    else:
        logger.warning("✗ Prediction diversity test failed")
    
    # Step 5: Make current predictions
    logger.info(f"Step 5: Making current predictions")
    
    current_predictions = predictor.predict_all_windows(ticker, df)
    
    if current_predictions:
        logger.info(f"Current predictions for {ticker}:")
        current_price = float(df['Close'].iloc[-1])
        logger.info(f"  Current price: ${current_price:.2f}")
        
        for window, pred in current_predictions.items():
            logger.info(f"  {window.replace('_', ' ').title()}:")
            logger.info(f"    Predicted: ${pred['predicted_price']:.2f}")
            logger.info(f"    Change: ${pred['price_change']:+.2f}")
            logger.info(f"    Confidence: {pred['confidence']:.3f}")
            logger.info(f"    Range: ${pred['price_range']['low']:.2f} - ${pred['price_range']['high']:.2f}")
    
    # Step 6: Perform backtesting
    logger.info(f"Step 6: Performing backtesting")
    
    test_start = df.index[train_split]
    backtest_results = perform_backtest(predictor, ticker, test_start)
    
    if backtest_results:
        logger.info("✓ Backtesting completed successfully")
    else:
        logger.warning("✗ Backtesting failed or insufficient data")
    
    # Step 7: Summary
    logger.info("=" * 60)
    logger.info("TESTING COMPLETE - SUMMARY")
    logger.info("=" * 60)
    
    logger.info(f"✓ Models trained for {ticker}")
    logger.info(f"✓ Feature pipelines saved for consistency")
    logger.info(f"✓ Updated windows: {predictor.prediction_windows}")
    logger.info(f"✓ Diversity test: {'PASSED' if diversity_ok else 'FAILED'}")
    
    if current_predictions:
        logger.info(f"✓ Current predictions generated for {len(current_predictions)} windows")
        
        # Check for the issues we were trying to fix
        confidences = [pred['confidence'] for pred in current_predictions.values()]
        predictions = [pred['predicted_price'] for pred in current_predictions.values()]
        
        low_confidence_count = sum(1 for c in confidences if c <= 0.15)
        identical_predictions = len(set(predictions)) == 1
        
        if low_confidence_count == 0:
            logger.info("✓ All predictions have reasonable confidence (>0.15)")
        else:
            logger.warning(f"✗ {low_confidence_count} predictions have very low confidence")
        
        if not identical_predictions:
            logger.info("✓ Predictions vary across windows (good diversity)")
        else:
            logger.warning("✗ All predictions are identical (poor diversity)")
    
    if backtest_results and backtest_results['mean_error'] < 10:
        logger.info(f"✓ Backtesting shows reasonable accuracy ({backtest_results['mean_error']:.1f}% mean error)")
    elif backtest_results:
        logger.warning(f"✗ High prediction error in backtesting ({backtest_results['mean_error']:.1f}% mean error)")
    
    logger.info("Testing complete! Check logs above for any issues.")

if __name__ == "__main__":
    main() 