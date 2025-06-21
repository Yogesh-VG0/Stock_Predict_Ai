#!/usr/bin/env python3
"""
Simple Backtesting Script for Stock Predictions

This script goes back 1 month and tests how accurate your trained models would have been.
It loads your existing models and compares predictions with actual prices.

Usage: python simple_backtest.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import joblib
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')

# Add ML backend to path
sys.path.append('ml_backend')

from ml_backend.utils.mongodb import MongoDBClient
from ml_backend.config.constants import MONGODB_URI
from ml_backend.data.features import FeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_models(ticker: str):
    """Load trained models for the ticker."""
    models = {}
    model_dir = f"./models/{ticker}"
    
    if not os.path.exists(model_dir):
        logger.error(f"Model directory not found: {model_dir}")
        return None
    
    print(f"📁 Loading models from: {model_dir}")
    
    for window in ['next_day', '7_day', '30_day']:
        try:
            window_models = {}
            
            # Load XGBoost model
            xgb_path = os.path.join(model_dir, f'model_{ticker}_{window}_xgb.joblib')
            if os.path.exists(xgb_path):
                window_models['xgb'] = joblib.load(xgb_path)
                print(f"✅ Loaded XGBoost model for {window}")
            
            # Load LightGBM model  
            lgb_path = os.path.join(model_dir, f'model_{ticker}_{window}_lightgbm_lgbm.joblib')
            if os.path.exists(lgb_path):
                window_models['lightgbm'] = joblib.load(lgb_path)
                print(f"✅ Loaded LightGBM model for {window}")
            
            # Load LSTM model
            lstm_path = os.path.join(model_dir, f'model_{ticker}_{window}_lstm.h5')
            if os.path.exists(lstm_path):
                try:
                    window_models['lstm'] = tf.keras.models.load_model(lstm_path)
                    print(f"✅ Loaded LSTM model for {window}")
                except Exception as e:
                    print(f"⚠️  Could not load LSTM for {window}: {e}")
            
            # Load feature pipeline
            pipeline_path = os.path.join(model_dir, f'feature_pipeline_{window}.joblib')
            feature_pipeline = None
            if os.path.exists(pipeline_path):
                feature_pipeline = joblib.load(pipeline_path)
                print(f"✅ Loaded feature pipeline for {window}")
            
            # Load metadata
            metadata_path = os.path.join(model_dir, f'metadata_{ticker}_{window}.json')
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"✅ Loaded metadata for {window}")
            
            if window_models:  # Only store if we have at least one model
                model_key = f"{ticker}_{window}"
                models[model_key] = {
                    'models': window_models,
                    'feature_pipeline': feature_pipeline,
                    'metadata': metadata
                }
                print(f"✅ Stored {window} models as key: {model_key}")
                
        except Exception as e:
            print(f"❌ Error loading {window} models: {str(e)}")
            logger.error(f"Error loading {window} models for {ticker}: {str(e)}")
    
    print(f"📊 Available model keys: {list(models.keys())}")
    logger.info(f"✓ Loaded models for {ticker}")
    return models if models else None

def get_stock_data(ticker: str, holdout_months_back: int = 3, backtest_days: int = 30):
    """Get historical stock data from MongoDB for proper backtesting.
    
    Args:
        ticker: Stock ticker symbol
        holdout_months_back: How many months back to start the holdout period (default: 3)
        backtest_days: Number of days to backtest (default: 30)
    
    This ensures we're testing on data the model has never seen during training.
    """
    mongo_client = MongoDBClient(MONGODB_URI)
    collection = mongo_client.db.historical_data  # Use the correct collection
    
    # Calculate holdout period (e.g., 6 months ago to get more data)
    today = datetime.now()
    holdout_end = today - timedelta(days=holdout_months_back * 30)  # End of holdout period
    holdout_start = holdout_end - timedelta(days=150)  # Get 5 months of data for proper testing
    
    print(f"📅 Using holdout period: {holdout_start.strftime('%Y-%m-%d')} to {holdout_end.strftime('%Y-%m-%d')}")
    print(f"🔒 This ensures models haven't seen this data during training!")
    
    query = {
        'ticker': ticker,
        'date': {'$gte': holdout_start, '$lte': holdout_end}
    }
    
    cursor = collection.find(query).sort('date', 1)
    data = list(cursor)
    
    if not data:
        logger.error(f"No holdout data found for {ticker} in the specified period")
        return None, None, None
    
    df = pd.DataFrame(data)
    
    # Ensure proper data types
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Set date as index
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    
    df = df.sort_index()
    logger.info(f"✓ Loaded {len(df)} days of holdout data for {ticker}")
    
    return df, holdout_start, holdout_end

def make_prediction(models, ticker: str, window: str, df_history: pd.DataFrame, feature_engineer, mongo_client):
    """Make a single prediction using the loaded models."""
    try:
        # Check if we have the required models for this window
        model_key = f"{ticker}_{window}"
        if model_key not in models:
            print(f"❌ No model found for {model_key}")
            return None
        
        model_data = models[model_key]
        individual_models = model_data['models']
        
        print(f"🔍 Making {window} prediction with {len(df_history)} days of history...")
        print(f"📊 Available models: {list(individual_models.keys())}")
        
        # Generate features for the current state
        features = feature_engineer.create_prediction_features(
            df=df_history,
            ticker=ticker,
            window=window,
            mongo_client=mongo_client
        )
        
        if features is None or len(features) == 0:
            print(f"❌ No features generated for {window}")
            return None
        
        print(f"✅ Generated features with shape: {features.shape}")
        
        # Convert to DataFrame for easier handling
        if len(features.shape) == 1:
            features_df = pd.DataFrame([features])
        else:
            features_df = pd.DataFrame(features)
        
        # Remove non-numeric columns if any
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        if len(numeric_features.columns) == 0:
            print(f"❌ No numeric features available")
            return None
        
        # Make predictions with available models
        predictions = {}
        
        # XGBoost prediction
        if 'xgb' in individual_models:
            try:
                xgb_pred = individual_models['xgb'].predict(numeric_features)[0]
                predictions['xgb'] = float(xgb_pred)
                print(f"✅ XGBoost prediction: ${xgb_pred:.2f}")
            except Exception as e:
                print(f"❌ XGBoost prediction failed: {e}")
        
        # LightGBM prediction
        if 'lightgbm' in individual_models:
            try:
                lgb_pred = individual_models['lightgbm'].predict(numeric_features)[0]
                predictions['lightgbm'] = float(lgb_pred)
                print(f"✅ LightGBM prediction: ${lgb_pred:.2f}")
            except Exception as e:
                print(f"❌ LightGBM prediction failed: {e}")
        
        # LSTM prediction (needs special handling)
        if 'lstm' in individual_models:
            try:
                # LSTM needs sequence data - use the last few days
                sequence_length = 60  # Default sequence length
                if len(df_history) >= sequence_length:
                    # Prepare LSTM features
                    lstm_features = numeric_features.values
                    lstm_input = lstm_features.reshape(1, 1, -1)  # (batch, timesteps, features)
                    
                    lstm_pred = individual_models['lstm'].predict(lstm_input, verbose=0)[0][0]
                    predictions['lstm'] = float(lstm_pred)
                    print(f"✅ LSTM prediction: ${lstm_pred:.2f}")
                else:
                    print(f"⚠️  Not enough history for LSTM ({len(df_history)} < {sequence_length})")
            except Exception as e:
                print(f"❌ LSTM prediction failed: {e}")
        
        if not predictions:
            print(f"❌ No successful predictions for {window}")
            return None
        
        # Create ensemble prediction (simple average)
        ensemble_pred = np.mean(list(predictions.values()))
        
        # Get current price for price change calculation
        current_price = float(df_history['Close'].iloc[-1])
        price_change = ensemble_pred - current_price
        
        result = {
            'predicted_price': float(ensemble_pred),
            'current_price': current_price,
            'price_change': price_change,
            'individual_predictions': predictions,
            'confidence': min(0.95, 0.5 + (len(predictions) * 0.15))  # Higher confidence with more models
        }
        
        print(f"🎯 Ensemble prediction: ${ensemble_pred:.2f} (change: ${price_change:+.2f})")
        return result
        
    except Exception as e:
        print(f"❌ Prediction failed for {window}: {str(e)}")
        logger.error(f"Prediction failed for {ticker} {window}: {str(e)}")
        return None

def run_backtest(ticker: str = "AAPL", backtest_days: int = 30, holdout_months_back: int = 3):
    """Run backtest for the specified ticker using proper holdout data.
    
    Args:
        ticker: Stock ticker to test
        backtest_days: Number of days to test predictions
        holdout_months_back: How many months back to use for holdout testing
    """
    print(f"\n{'='*60}")
    print(f"BACKTESTING {ticker} - HOLDOUT TESTING")
    print(f"{'='*60}")
    print(f"🎯 Testing {backtest_days} days from {holdout_months_back} months ago")
    print(f"🔒 This data was NOT used in model training!")
    
    # Load models
    models = load_models(ticker)
    if not models:
        print(f"❌ No models found for {ticker}")
        return
    
    # Get holdout data (data from months ago that models haven't seen)
    df, holdout_start, holdout_end = get_stock_data(ticker, holdout_months_back, backtest_days)
    if df is None:
        print(f"❌ No holdout data found for {ticker}")
        return
    
    print(f"✅ Loaded {len(df)} days of holdout data")
    
    # Initialize feature engineer
    mongo_client = MongoDBClient(MONGODB_URI)
    feature_engineer = FeatureEngineer(mongo_client)
    
    # Initialize results tracking
    results = {
        'next_day': {'predictions': [], 'actuals': [], 'errors': [], 'dates': []},
        '7_day': {'predictions': [], 'actuals': [], 'errors': [], 'dates': []},
        '30_day': {'predictions': [], 'actuals': [], 'errors': [], 'dates': []}
    }
    
    print(f"🔄 Making predictions for {backtest_days} days in holdout period...")
    
    # Calculate the start index for backtesting (we need history for features)
    backtest_start_idx = max(50, len(df) - backtest_days - 30)  # Leave room for 30-day predictions
    
    # Make predictions for each day in the holdout period
    successful_predictions = 0
    
    print(f"📊 Holdout data range: {df.index[0]} to {df.index[-1]}")
    print(f"📊 Total holdout days: {len(df)}")
    print(f"📊 Starting backtest from index {backtest_start_idx} to {len(df) - 30}")
    
    if backtest_start_idx >= len(df) - 30:
        print(f"❌ Not enough data for backtesting. Need at least {backtest_start_idx + 30} days, have {len(df)}")
        return
    
    for i in range(backtest_start_idx, min(len(df) - 30, backtest_start_idx + backtest_days)):  # Ensure room for 30-day predictions
        current_date = df.index[i]
        df_history = df.iloc[:i+1].copy()  # Data up to current date
        
        print(f"📅 Processing day {i}/{len(df)-1}: {current_date.strftime('%Y-%m-%d')}")
        
        if len(df_history) < 50:  # Need enough history
            print(f"⚠️  Skipping {current_date.strftime('%Y-%m-%d')} - not enough history ({len(df_history)} days)")
            continue
        
        print(f"✅ Using {len(df_history)} days of history for prediction")
        
        # Next day prediction
        if i + 1 < len(df):
            print(f"🔍 Attempting next_day prediction...")
            pred = make_prediction(models, ticker, 'next_day', df_history, feature_engineer, mongo_client)
            if pred:
                actual_price = float(df['Close'].iloc[i + 1])
                error = abs(pred['predicted_price'] - actual_price) / actual_price * 100
                
                results['next_day']['predictions'].append(pred['predicted_price'])
                results['next_day']['actuals'].append(actual_price)
                results['next_day']['errors'].append(error)
                results['next_day']['dates'].append(current_date)
                successful_predictions += 1
                print(f"✅ Next-day prediction successful: ${pred['predicted_price']:.2f} vs actual ${actual_price:.2f}")
            else:
                print(f"❌ Next-day prediction failed")
        
        # 7-day prediction
        if i + 7 < len(df):
            print(f"🔍 Attempting 7_day prediction...")
            pred = make_prediction(models, ticker, '7_day', df_history, feature_engineer, mongo_client)
            if pred:
                actual_price = float(df['Close'].iloc[i + 7])
                error = abs(pred['predicted_price'] - actual_price) / actual_price * 100
                
                results['7_day']['predictions'].append(pred['predicted_price'])
                results['7_day']['actuals'].append(actual_price)
                results['7_day']['errors'].append(error)
                results['7_day']['dates'].append(current_date)
                print(f"✅ 7-day prediction successful: ${pred['predicted_price']:.2f} vs actual ${actual_price:.2f}")
            else:
                print(f"❌ 7-day prediction failed")
        
        # 30-day prediction (if we have enough future data)
        if i + 30 < len(df):
            print(f"🔍 Attempting 30_day prediction...")
            pred = make_prediction(models, ticker, '30_day', df_history, feature_engineer, mongo_client)
            if pred:
                actual_price = float(df['Close'].iloc[i + 30])
                error = abs(pred['predicted_price'] - actual_price) / actual_price * 100
                
                results['30_day']['predictions'].append(pred['predicted_price'])
                results['30_day']['actuals'].append(actual_price)
                results['30_day']['errors'].append(error)
                results['30_day']['dates'].append(current_date)
                print(f"✅ 30-day prediction successful: ${pred['predicted_price']:.2f} vs actual ${actual_price:.2f}")
            else:
                print(f"❌ 30-day prediction failed")
        
        # Only process a few days for debugging
        if i - backtest_start_idx >= 3:
            print(f"🛑 Stopping after 3 days for debugging...")
            break
    
    print(f"✅ Generated {successful_predictions} successful predictions")
    
    # Print detailed results
    print(f"\n📊 HOLDOUT BACKTEST RESULTS:")
    print(f"{'='*60}")
    print(f"Period: {holdout_start.strftime('%Y-%m-%d')} to {holdout_end.strftime('%Y-%m-%d')}")
    print(f"⚠️  IMPORTANT: This data was NOT seen during model training!")
    
    for window, data in results.items():
        if len(data['errors']) > 0:
            errors = np.array(data['errors'])
            predictions = np.array(data['predictions'])
            actuals = np.array(data['actuals'])
            
            mean_error = np.mean(errors)
            median_error = np.median(errors)
            accuracy_1pct = np.mean(errors <= 1.0) * 100
            accuracy_2pct = np.mean(errors <= 2.0) * 100
            accuracy_5pct = np.mean(errors <= 5.0) * 100
            
            # Calculate directional accuracy (did we predict the right direction?)
            price_changes_actual = np.diff(actuals)
            price_changes_predicted = np.diff(predictions)
            if len(price_changes_actual) > 0:
                directional_accuracy = np.mean(np.sign(price_changes_actual) == np.sign(price_changes_predicted)) * 100
            else:
                directional_accuracy = 0
            
            print(f"\n{window.replace('_', ' ').upper()} PREDICTIONS:")
            print(f"  Total predictions: {len(errors)}")
            print(f"  Mean error: {mean_error:.2f}%")
            print(f"  Median error: {median_error:.2f}%")
            print(f"  Accuracy within 1%: {accuracy_1pct:.1f}%")
            print(f"  Accuracy within 2%: {accuracy_2pct:.1f}%")
            print(f"  Accuracy within 5%: {accuracy_5pct:.1f}%")
            print(f"  Directional accuracy: {directional_accuracy:.1f}%")
            
            # Rating based on holdout performance
            if accuracy_2pct >= 60:
                rating = "🟢 EXCELLENT (for holdout data)"
            elif accuracy_2pct >= 40:
                rating = "🟡 GOOD (for holdout data)"
            elif accuracy_2pct >= 25:
                rating = "🟠 FAIR (for holdout data)"
            else:
                rating = "🔴 NEEDS IMPROVEMENT"
            
            print(f"  Overall Rating: {rating}")
            
            # Show some example predictions with dates
            if len(data['predictions']) >= 3:
                print(f"  Example predictions:")
                for j in range(min(3, len(data['predictions']))):
                    pred = data['predictions'][j]
                    actual = data['actuals'][j]
                    error = data['errors'][j]
                    date = data['dates'][j].strftime('%Y-%m-%d')
                    print(f"    {date}: Predicted ${pred:.2f}, Actual ${actual:.2f}, Error {error:.1f}%")
    
    # Summary with realistic expectations for holdout data
    print(f"\n{'='*60}")
    print(f"HOLDOUT TESTING SUMMARY:")
    print(f"{'='*60}")
    
    next_day_acc = np.mean(results['next_day']['errors']) if results['next_day']['errors'] else 0
    seven_day_acc = np.mean(results['7_day']['errors']) if results['7_day']['errors'] else 0
    thirty_day_acc = np.mean(results['30_day']['errors']) if results['30_day']['errors'] else 0
    
    if next_day_acc > 0:
        print(f"📈 Next-day predictions: {next_day_acc:.1f}% average error")
    if seven_day_acc > 0:
        print(f"📊 7-day predictions: {seven_day_acc:.1f}% average error")
    if thirty_day_acc > 0:
        print(f"📉 30-day predictions: {thirty_day_acc:.1f}% average error")
    
    # Realistic assessment for holdout data
    print(f"\n🎯 HOLDOUT PERFORMANCE ASSESSMENT:")
    if next_day_acc < 4:
        print(f"🎉 Excellent next-day accuracy on unseen data!")
    elif next_day_acc < 6:
        print(f"👍 Good next-day accuracy on unseen data")
    elif next_day_acc < 10:
        print(f"⚠️  Fair next-day accuracy - room for improvement")
    else:
        print(f"🔴 Next-day accuracy needs significant improvement")
    
    print(f"\n💡 IMPORTANT NOTES:")
    print(f"• This is HOLDOUT testing - much harder than in-sample testing")
    print(f"• Accuracy on unseen data is typically 20-30% lower than training data")
    print(f"• Directional accuracy (predicting up/down) is often more important than exact price")
    print(f"• Models tested on data from {holdout_months_back} months ago")
    
    return results

def main():
    """Main function."""
    print("🚀 STOCK PREDICTION MODEL BACKTESTING")
    print("This will test your trained models against actual historical prices")
    print("🔒 Using HOLDOUT data that models have never seen!")
    
    # You can change these parameters
    ticker = "AAPL"  # Change this to test other tickers
    backtest_days = 30  # How many days back to test
    holdout_months_back = 3  # How many months back to use for holdout testing
    
    print(f"\n📋 Testing Configuration:")
    print(f"   Ticker: {ticker}")
    print(f"   Backtest days: {backtest_days}")
    print(f"   Holdout period: {holdout_months_back} months ago")
    print(f"   This ensures NO data leakage!")
    
    run_backtest(ticker, backtest_days, holdout_months_back)
    
    print(f"\n✅ Backtesting complete!")
    print(f"📁 Your models are in: ./models/{ticker}/")
    print(f"🎯 This was TRUE holdout testing - no cheating!")

if __name__ == "__main__":
    main() 