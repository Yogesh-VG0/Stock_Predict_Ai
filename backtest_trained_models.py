#!/usr/bin/env python3
"""
Comprehensive Backtesting Script for Trained Stock Prediction Models

This script:
1. Loads existing trained models from /models folder
2. Gets historical data from MongoDB
3. Goes back 1 month and makes predictions day by day
4. Compares predictions with actual prices
5. Calculates detailed accuracy metrics

Usage: python backtest_trained_models.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
import json
import joblib
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# Add ML backend to path
sys.path.append('ml_backend')

from ml_backend.utils.mongodb import MongoDBClient
from ml_backend.config.constants import MONGODB_URI
from ml_backend.data.features import FeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelBacktester:
    """Comprehensive backtesting for trained stock prediction models."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.mongo_client = MongoDBClient(MONGODB_URI)
        self.feature_engineer = FeatureEngineer(self.mongo_client)
        self.loaded_models = {}
        self.backtest_results = {}
        
    def load_models_for_ticker(self, ticker: str) -> bool:
        """Load all trained models for a specific ticker."""
        ticker_dir = os.path.join(self.models_dir, ticker)
        
        if not os.path.exists(ticker_dir):
            logger.error(f"No models found for ticker {ticker} in {ticker_dir}")
            return False
        
        self.loaded_models[ticker] = {}
        windows = ['next_day', '7_day', '30_day']
        model_types = ['lightgbm_lgbm', 'xgb', 'lstm']
        
        for window in windows:
            self.loaded_models[ticker][window] = {}
            
            # Load metadata
            metadata_file = os.path.join(ticker_dir, f"metadata_{ticker}_{window}.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    self.loaded_models[ticker][window]['metadata'] = json.load(f)
            
            # Load feature pipeline
            pipeline_file = os.path.join(ticker_dir, f"feature_pipeline_{window}.joblib")
            if os.path.exists(pipeline_file):
                self.loaded_models[ticker][window]['feature_pipeline'] = joblib.load(pipeline_file)
            
            # Load models
            for model_type in model_types:
                if model_type == 'lstm':
                    model_file = os.path.join(ticker_dir, f"model_{ticker}_{window}_lstm.h5")
                    if os.path.exists(model_file):
                        try:
                            model = tf.keras.models.load_model(model_file)
                            self.loaded_models[ticker][window][model_type] = model
                            logger.info(f"✓ Loaded LSTM model for {ticker}-{window}")
                        except Exception as e:
                            logger.warning(f"Failed to load LSTM model for {ticker}-{window}: {e}")
                else:
                    model_file = os.path.join(ticker_dir, f"model_{ticker}_{window}_{model_type}.joblib")
                    if os.path.exists(model_file):
                        try:
                            model = joblib.load(model_file)
                            self.loaded_models[ticker][window][model_type] = model
                            logger.info(f"✓ Loaded {model_type} model for {ticker}-{window}")
                        except Exception as e:
                            logger.warning(f"Failed to load {model_type} model for {ticker}-{window}: {e}")
        
        return len(self.loaded_models[ticker]) > 0
    
    def get_historical_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get historical stock data from MongoDB."""
        try:
            collection = self.mongo_client.db.stock_data
            
            query = {
                'ticker': ticker,
                'date': {'$gte': start_date, '$lte': end_date}
            }
            
            cursor = collection.find(query).sort('date', 1)
            data = list(cursor)
            
            if not data:
                logger.error(f"No historical data found for {ticker} between {start_date} and {end_date}")
                return None
            
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
            logger.info(f"✓ Loaded {len(df)} records for {ticker} from {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data for {ticker}: {e}")
            return None
    
    def make_prediction(self, ticker: str, window: str, df_history: pd.DataFrame) -> Optional[Dict]:
        """Make a prediction using loaded models."""
        try:
            if ticker not in self.loaded_models or window not in self.loaded_models[ticker]:
                return None
            
            models = self.loaded_models[ticker][window]
            
            # Create features using the same pipeline as training
            features = self.feature_engineer.create_prediction_features(
                df=df_history.copy(),
                ticker=ticker,
                window=window,
                mongo_client=self.mongo_client
            )
            
            if features is None:
                logger.warning(f"Could not create features for {ticker}-{window}")
                return None
            
            # Make predictions with each model
            model_predictions = {}
            
            # XGBoost prediction
            if 'xgb' in models:
                try:
                    if len(features.shape) == 3:
                        features_2d = features.reshape(features.shape[0], -1)
                    else:
                        features_2d = features
                    pred = models['xgb'].predict(features_2d)
                    model_predictions['xgb'] = float(pred[0]) if hasattr(pred, '__iter__') else float(pred)
                except Exception as e:
                    logger.warning(f"XGBoost prediction failed for {ticker}-{window}: {e}")
            
            # LightGBM prediction
            if 'lightgbm_lgbm' in models:
                try:
                    if len(features.shape) == 3:
                        features_2d = features.reshape(features.shape[0], -1)
                    else:
                        features_2d = features
                    pred = models['lightgbm_lgbm'].predict(features_2d)
                    model_predictions['lightgbm'] = float(pred[0]) if hasattr(pred, '__iter__') else float(pred)
                except Exception as e:
                    logger.warning(f"LightGBM prediction failed for {ticker}-{window}: {e}")
            
            # LSTM prediction
            if 'lstm' in models:
                try:
                    # LSTM needs proper 3D shape
                    window_size = 1
                    if window == '7_day':
                        window_size = 7
                    elif window == '30_day':
                        window_size = 30
                    
                    if len(features.shape) == 2:
                        # Reshape for LSTM: (batch_size, timesteps, features)
                        lstm_features = features.reshape(1, window_size, features.shape[1] // window_size)
                    else:
                        lstm_features = features
                    
                    pred = models['lstm'].predict(lstm_features, verbose=0)
                    model_predictions['lstm'] = float(pred[0][0]) if pred.shape[1] == 1 else float(pred[0])
                except Exception as e:
                    logger.warning(f"LSTM prediction failed for {ticker}-{window}: {e}")
            
            if not model_predictions:
                return None
            
            # Calculate ensemble prediction
            weights = {'xgb': 0.35, 'lightgbm': 0.35, 'lstm': 0.30}
            available_models = list(model_predictions.keys())
            
            # Normalize weights for available models
            total_weight = sum(weights.get(model, 0) for model in available_models)
            if total_weight > 0:
                normalized_weights = {model: weights.get(model, 0) / total_weight for model in available_models}
            else:
                normalized_weights = {model: 1.0 / len(available_models) for model in available_models}
            
            # Calculate weighted ensemble prediction
            ensemble_pred = sum(model_predictions[model] * normalized_weights[model] 
                              for model in available_models)
            
            current_price = float(df_history['Close'].iloc[-1])
            predicted_price = current_price + ensemble_pred
            
            # Calculate confidence (simplified version)
            pred_values = list(model_predictions.values())
            if len(pred_values) > 1:
                pred_std = np.std(pred_values)
                pred_mean = np.mean(pred_values)
                confidence = max(0.1, min(0.95, 1.0 - (pred_std / max(abs(pred_mean), 1))))
            else:
                confidence = 0.6
            
            return {
                'predicted_price': predicted_price,
                'price_change': ensemble_pred,
                'current_price': current_price,
                'confidence': confidence,
                'model_predictions': model_predictions,
                'ensemble_weights': normalized_weights
            }
            
        except Exception as e:
            logger.error(f"Error making prediction for {ticker}-{window}: {e}")
            return None
    
    def run_backtest(self, ticker: str, backtest_days: int = 30) -> Dict:
        """Run comprehensive backtest for a ticker."""
        logger.info(f"Starting backtest for {ticker} over {backtest_days} days")
        
        # Load models
        if not self.load_models_for_ticker(ticker):
            logger.error(f"Failed to load models for {ticker}")
            return {}
        
        # Get historical data (need extra days for feature calculation)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=backtest_days + 365)  # Extra year for features
        
        df_full = self.get_historical_data(ticker, start_date, end_date)
        if df_full is None:
            return {}
        
        # Define backtest period (last 30 days)
        backtest_start = end_date - timedelta(days=backtest_days)
        backtest_data = df_full[df_full.index >= backtest_start].copy()
        
        if len(backtest_data) < 5:
            logger.error(f"Insufficient backtest data: {len(backtest_data)} days")
            return {}
        
        results = {
            'ticker': ticker,
            'backtest_period': f"{backtest_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'predictions': [],
            'windows': {}
        }
        
        windows = ['next_day', '7_day', '30_day']
        
        # Initialize window results
        for window in windows:
            results['windows'][window] = {
                'predictions': [],
                'actual_prices': [],
                'predicted_prices': [],
                'errors': [],
                'accuracy_metrics': {}
            }
        
        # Make predictions for each day in backtest period
        for i in range(len(backtest_data) - 1):
            current_date = backtest_data.index[i]
            
            # Get data up to current date for prediction
            df_history = df_full[df_full.index <= current_date].copy()
            
            if len(df_history) < 100:  # Need sufficient history
                continue
            
            current_price = float(df_history['Close'].iloc[-1])
            
            # Make predictions for each window
            day_predictions = {
                'date': current_date,
                'current_price': current_price,
                'predictions': {}
            }
            
            for window in windows:
                prediction = self.make_prediction(ticker, window, df_history)
                
                if prediction:
                    day_predictions['predictions'][window] = prediction
                    
                    # Find actual price for validation
                    if window == 'next_day' and i + 1 < len(backtest_data):
                        actual_price = float(backtest_data['Close'].iloc[i + 1])
                        error = abs(prediction['predicted_price'] - actual_price)
                        error_pct = (error / actual_price) * 100
                        
                        results['windows'][window]['predictions'].append(prediction['predicted_price'])
                        results['windows'][window]['actual_prices'].append(actual_price)
                        results['windows'][window]['errors'].append(error_pct)
                    
                    elif window == '7_day' and i + 7 < len(backtest_data):
                        actual_price = float(backtest_data['Close'].iloc[i + 7])
                        error = abs(prediction['predicted_price'] - actual_price)
                        error_pct = (error / actual_price) * 100
                        
                        results['windows'][window]['predictions'].append(prediction['predicted_price'])
                        results['windows'][window]['actual_prices'].append(actual_price)
                        results['windows'][window]['errors'].append(error_pct)
            
            results['predictions'].append(day_predictions)
        
        # Calculate accuracy metrics for each window
        for window in windows:
            window_data = results['windows'][window]
            
            if len(window_data['errors']) > 0:
                errors = np.array(window_data['errors'])
                predictions = np.array(window_data['predictions'])
                actuals = np.array(window_data['actual_prices'])
                
                metrics = {
                    'mean_error_pct': float(np.mean(errors)),
                    'median_error_pct': float(np.median(errors)),
                    'std_error_pct': float(np.std(errors)),
                    'max_error_pct': float(np.max(errors)),
                    'accuracy_within_1pct': float(np.mean(errors <= 1.0) * 100),
                    'accuracy_within_2pct': float(np.mean(errors <= 2.0) * 100),
                    'accuracy_within_5pct': float(np.mean(errors <= 5.0) * 100),
                    'mae': float(mean_absolute_error(actuals, predictions)),
                    'rmse': float(np.sqrt(mean_squared_error(actuals, predictions))),
                    'total_predictions': len(errors)
                }
                
                window_data['accuracy_metrics'] = metrics
        
        self.backtest_results[ticker] = results
        return results
    
    def print_results(self, ticker: str):
        """Print detailed backtest results."""
        if ticker not in self.backtest_results:
            logger.error(f"No backtest results found for {ticker}")
            return
        
        results = self.backtest_results[ticker]
        
        print(f"\n{'='*80}")
        print(f"BACKTEST RESULTS FOR {ticker}")
        print(f"{'='*80}")
        print(f"Period: {results['backtest_period']}")
        print(f"Total prediction days: {len(results['predictions'])}")
        
        for window, data in results['windows'].items():
            if 'accuracy_metrics' in data and data['accuracy_metrics']:
                metrics = data['accuracy_metrics']
                
                print(f"\n{window.replace('_', ' ').upper()} WINDOW:")
                print("-" * 40)
                print(f"Total predictions: {metrics['total_predictions']}")
                print(f"Mean error: {metrics['mean_error_pct']:.2f}%")
                print(f"Median error: {metrics['median_error_pct']:.2f}%")
                print(f"Max error: {metrics['max_error_pct']:.2f}%")
                print(f"Accuracy within 1%: {metrics['accuracy_within_1pct']:.1f}%")
                print(f"Accuracy within 2%: {metrics['accuracy_within_2pct']:.1f}%")
                print(f"Accuracy within 5%: {metrics['accuracy_within_5pct']:.1f}%")
                print(f"MAE: ${metrics['mae']:.2f}")
                print(f"RMSE: ${metrics['rmse']:.2f}")
                
                # Accuracy rating
                if metrics['accuracy_within_2pct'] >= 70:
                    rating = "EXCELLENT"
                elif metrics['accuracy_within_2pct'] >= 50:
                    rating = "GOOD"
                elif metrics['accuracy_within_2pct'] >= 30:
                    rating = "FAIR"
                else:
                    rating = "POOR"
                
                print(f"Overall Rating: {rating}")
    
    def save_results(self, ticker: str, filename: str = None):
        """Save backtest results to JSON file."""
        if ticker not in self.backtest_results:
            return
        
        if filename is None:
            filename = f"backtest_results_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_json = convert_numpy(self.backtest_results[ticker])
        
        with open(filename, 'w') as f:
            json.dump(results_json, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filename}")

def main():
    """Main backtesting pipeline."""
    print("="*80)
    print("STOCK PREDICTION MODEL BACKTESTING")
    print("="*80)
    
    # Initialize backtester
    backtester = ModelBacktester()
    
    # Test with AAPL (you can add more tickers here)
    tickers = ["AAPL"]
    backtest_days = 30  # Test last 30 days
    
    for ticker in tickers:
        print(f"\nBacktesting {ticker}...")
        
        # Run backtest
        results = backtester.run_backtest(ticker, backtest_days)
        
        if results:
            # Print results
            backtester.print_results(ticker)
            
            # Save results
            backtester.save_results(ticker)
            
            print(f"\n✓ Backtest completed for {ticker}")
        else:
            print(f"✗ Backtest failed for {ticker}")
    
    print(f"\n{'='*80}")
    print("BACKTESTING COMPLETE")
    print("="*80)
    print("\nKey Insights:")
    print("• Accuracy within 2% is considered good for stock predictions")
    print("• Next-day predictions typically have highest accuracy")
    print("• 7-day predictions show medium accuracy")
    print("• 30-day predictions are most challenging")
    print("\nFiles generated:")
    print("• backtest_results_[TICKER]_[TIMESTAMP].json - Detailed results")

if __name__ == "__main__":
    main() 