"""
Machine learning model module for stock price prediction.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from ..config.constants import (
    MODEL_CONFIG,
    PREDICTION_WINDOWS,
    FEATURE_CONFIG,
    TOP_100_TICKERS
)
import time
import json
import shap
import xgboost as xgb
import shutil
import lightgbm as lgb
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from ml_backend.data.features import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoggingCallback(Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.epochs = self.params.get('epochs', 0)
        self.epoch_times = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        elapsed = time.time() - self.start_time
        self.epoch_times.append(time.time())
        avg_epoch_time = (self.epoch_times[-1] - self.epoch_times[0]) / (epoch + 1) if epoch > 0 else 0
        remaining_epochs = self.epochs - (epoch + 1)
        eta = avg_epoch_time * remaining_epochs if avg_epoch_time > 0 else 0
        logger.info(f"Epoch {epoch+1}/{self.epochs}: loss={logs.get('loss', 0):.4f}, val_loss={logs.get('val_loss', 0):.4f}, elapsed={elapsed:.1f}s, ETA={eta:.1f}s")

class StockPredictor:
    def __init__(self, mongo_client):
        self.mongo_client = mongo_client
        self.models = {}
        self.lgbm_models = {}
        self.xgb_models = {}
        self.feature_selectors = {}
        self.feature_selector = None
        self.feature_engineer = None  # Will be set via set_feature_engineer
        self.scaler = StandardScaler()
        self.model_metadata = {}
        self.model_dir = "models"  # Default models directory
        # Updated windows: next_day, 7_day (1 week), 30_day (1 month)
        self.prediction_windows = ['next_day', '7_day', '30_day']
        
    def prepare_training_data(self, ticker: str, window: str, 
                            start_date: str = None, end_date: str = None):
        """Prepare data for training with consistent feature engineering."""
        try:
            # Window size mapping
            window_size_map = {'next_day': 1, '7_day': 7, '30_day': 30}
            window_size = window_size_map.get(window, 1)
            
            logger.info(f"Preparing training data for {ticker} - {window} window (size: {window_size})")
            
            # Get historical data
            collection = self.mongo_client.get_database().stock_data
            
            # Build query
            query = {'ticker': ticker}
            if start_date or end_date:
                date_filter = {}
                if start_date:
                    date_filter['$gte'] = datetime.strptime(start_date, '%Y-%m-%d')
                if end_date:
                    date_filter['$lte'] = datetime.strptime(end_date, '%Y-%m-%d')
                query['date'] = date_filter
            
            # Fetch data
            cursor = collection.find(query).sort('date', 1)
            data = list(cursor)
            
            if len(data) < window_size + 50:  # Need enough data for windowing + minimum training
                raise ValueError(f"Insufficient data for {ticker}: {len(data)} records. Need at least {window_size + 50}")
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            if 'date' in df.columns:
                df = df.set_index('date')
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns for {ticker}: {missing_columns}")
            
            # Get sentiment and external data
            sentiment_data = self.mongo_client.get_aggregated_sentiment(ticker)
            
            # Prepare features with pipeline saving
            features, targets = self.feature_engineer.prepare_features(
                df=df,
                sentiment_dict=sentiment_data,
                window_size=window_size,
                ticker=ticker,
                mongo_client=self.mongo_client,
                handle_outliers=True,
                save_pipeline=True,  # Save pipeline for consistency
                window=window
            )
            
            if features is None or targets is None:
                raise ValueError(f"Feature engineering failed for {ticker}-{window}")
            
            logger.info(f"Training data prepared: Features shape: {features.shape}, Targets shape: {targets.shape}")
            
            return features, targets
            
        except Exception as e:
            logger.error(f"Error preparing training data for {ticker}-{window}: {e}")
            raise

    def set_feature_engineer(self, feature_engineer):
        self.feature_engineer = feature_engineer

    def custom_loss(self, y_true, y_pred):
        alpha = 0.1
        mse = tf.keras.losses.MeanSquaredError()(y_true[:, 0], y_pred[:, 0])
        penalty = alpha * tf.reduce_mean(tf.square(y_pred[:, 0] - y_true[:, 1]))
        return mse + penalty

    def build_model(self, input_shape: Tuple[int, int], hyperparams: Dict) -> tf.keras.Model:
        """Build LSTM model with given hyperparameters and custom loss."""
        # Use Input layer as first layer for best practice
        model = Sequential([
            Input(shape=input_shape),
            LSTM(
                units=hyperparams['lstm_units'],
                return_sequences=True,
                kernel_regularizer=l2(hyperparams['l2_reg']),
                recurrent_regularizer=l2(hyperparams['l2_reg'])
            ),
            Dropout(hyperparams['dropout_rate']),
            LSTM(
                units=hyperparams['lstm_units'] // 2,
                kernel_regularizer=l2(hyperparams['l2_reg']),
                recurrent_regularizer=l2(hyperparams['l2_reg'])
            ),
            Dropout(hyperparams['dropout_rate']),
            Dense(
                units=hyperparams['dense_units'],
                activation='relu',
                kernel_regularizer=l2(hyperparams['l2_reg'])
            ),
            Dense(1)
        ])
        model.compile(
            optimizer=Adam(learning_rate=hyperparams['learning_rate']),
            loss=self.custom_loss,
            metrics=['mae']
        )
        return model

    def build_rnn_model(self, input_shape: Tuple[int, int], hyperparams: Dict) -> tf.keras.Model:
        """Build a simple RNN model with given hyperparameters and custom loss."""
        model = Sequential([
            Input(shape=input_shape),
            tf.keras.layers.SimpleRNN(
                units=hyperparams['lstm_units'],
                return_sequences=True,
                kernel_regularizer=l2(hyperparams['l2_reg']),
                recurrent_regularizer=l2(hyperparams['l2_reg'])
            ),
            Dropout(hyperparams['dropout_rate']),
            tf.keras.layers.SimpleRNN(
                units=hyperparams['lstm_units'] // 2,
                kernel_regularizer=l2(hyperparams['l2_reg']),
                recurrent_regularizer=l2(hyperparams['l2_reg'])
            ),
            Dropout(hyperparams['dropout_rate']),
            Dense(
                units=hyperparams['dense_units'],
                activation='relu',
                kernel_regularizer=l2(hyperparams['l2_reg'])
            ),
            Dense(1)
        ])
        model.compile(
            optimizer=Adam(learning_rate=hyperparams['learning_rate']),
            loss=self.custom_loss,
            metrics=['mae']
        )
        return model



    def build_lgbm_model(self, **params):
        return lgb.LGBMRegressor(**params)

    def objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Objective function for hyperparameter optimization."""
        try:
            hyperparams = {
                'lstm_units': trial.suggest_int('lstm_units', 32, 64),
                'dense_units': trial.suggest_int('dense_units', 16, 32),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.3),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'l2_reg': trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [64, 128])
            }
            
            # Validate input shapes
            if len(X_train.shape) != 3 or len(X_val.shape) != 3:
                logger.warning("Invalid input shapes for hyperparameter tuning. Skipping trial.")
                return float('inf')
            
            # Use stored feature columns to find 'Close' index
            if not hasattr(self, 'feature_columns') or not self.feature_columns:
                logger.warning("No feature columns available. Using last feature column.")
                close_idx = -1
            else:
                close_idx = self.feature_columns.index('Close') if 'Close' in self.feature_columns else -1
                
            # Get mean and std from the current window's stats if available, else use y_train
            y_mean = y_train.mean() if hasattr(y_train, 'mean') else 0
            y_std = y_train.std() if hasattr(y_train, 'std') and y_train.std() != 0 else 1
            
            # Normalize targets and current price
            y_train_norm = (y_train - y_mean) / y_std
            y_val_norm = (y_val - y_mean) / y_std
            
            # Handle current price extraction safely
            try:
                current_price_train = (X_train[:, -1, close_idx] - y_mean) / y_std
                current_price_val = (X_val[:, -1, close_idx] - y_mean) / y_std
            except IndexError:
                logger.warning("Index error extracting current price. Using zeros.")
                current_price_train = np.zeros(len(X_train))
                current_price_val = np.zeros(len(X_val))
            
            # Ensure proper shapes
            if len(y_train_norm.shape) == 1:
                y_train_norm = y_train_norm.reshape(-1, 1)
            if len(y_val_norm.shape) == 1:
                y_val_norm = y_val_norm.reshape(-1, 1)
            if len(current_price_train.shape) == 1:
                current_price_train = current_price_train.reshape(-1, 1)
            if len(current_price_val.shape) == 1:
                current_price_val = current_price_val.reshape(-1, 1)
                
            # Align lengths
            min_train_len = min(len(y_train_norm), len(current_price_train), len(X_train))
            min_val_len = min(len(y_val_norm), len(current_price_val), len(X_val))
            
            y_train_combined = np.column_stack([
                y_train_norm[:min_train_len], 
                current_price_train[:min_train_len]
            ])
            y_val_combined = np.column_stack([
                y_val_norm[:min_val_len], 
                current_price_val[:min_val_len]
            ])
            
            # Align X arrays
            X_train_aligned = X_train[:min_train_len]
            X_val_aligned = X_val[:min_val_len]
            
            model = self.build_model((X_train_aligned.shape[1], X_train_aligned.shape[2]), hyperparams)
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.model_config['early_stopping_patience'],
                restore_best_weights=True
            )
            model.fit(
                X_train_aligned, y_train_combined,
                validation_data=(X_val_aligned, y_val_combined),
                epochs=self.model_config['epochs'],
                batch_size=hyperparams['batch_size'],
                callbacks=[early_stopping],
                verbose=0
            )
            val_loss = model.evaluate(X_val_aligned, y_val_combined, verbose=0)[0]
            return val_loss
        except Exception as e:
            logger.warning(f"Error in hyperparameter trial: {str(e)}")
            return float('inf')  # Return very high loss for failed trials

    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Tune hyperparameters using Optuna."""
        try:
            # Split data into train and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=0.2,
                shuffle=False
            )
            
            # Create study
            study = optuna.create_study(direction='minimize')
            study.optimize(
                lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
                n_trials=self.model_config['n_trials']
            )
            
            # Get best hyperparameters
            best_params = study.best_params
            logger.info(f"Best hyperparameters: {best_params}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error tuning hyperparameters: {str(e)}")
            return self.model_config['default_hyperparameters']



    def tune_lgbm(self, X, y):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'random_state': 42
            }
            model = self.build_lgbm_model(**params)
            model.fit(X, y)
            preds = model.predict(X)
            return mean_squared_error(y, preds)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)
        return study.best_params

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        window: str
    ) -> tf.keras.Model:
        """Train LSTM model for a specific window."""
        try:
            # Print the exact features used for training
            print(f"Features used for training {window}: {self.feature_columns}")
            
            # Validate input shapes
            logger.info(f"Input shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
            logger.info(f"Input shapes - X_val: {X_val.shape}, y_val: {y_val.shape}")
            logger.info(f"Input shapes - X_test: {X_test.shape}, y_test: {y_test.shape}")
            
            # Check for valid data
            if X_train.shape[0] == 0 or y_train.shape[0] == 0:
                logger.error(f"Empty training data for {window}")
                return None
            
            # --- Target normalization ---
            y_mean = y_train.mean()
            y_std = y_train.std() if y_train.std() != 0 else 1
            self.target_stats[window] = {"mean": float(y_mean), "std": float(y_std)}
            y_train_norm = (y_train - y_mean) / y_std
            y_val_norm = (y_val - y_mean) / y_std
            y_test_norm = (y_test - y_mean) / y_std
            
            # Handle different array shapes for different window sizes
            if len(X_train.shape) == 3:  # 3D array for multi-day windows
                # Use stored feature columns to find 'Close' index
                close_idx = self.feature_columns.index('Close') if 'Close' in self.feature_columns else -1
                if close_idx == -1:
                    logger.warning(f"'Close' column not found in features for {window}. Using last column.")
                    close_idx = -1
                # Extract and normalize current price from last time step
                current_price_train = X_train[:, -1, close_idx] 
                current_price_val = X_val[:, -1, close_idx] 
                current_price_test = X_test[:, -1, close_idx] 
            else:  # 2D array for next_day window
                # For 2D arrays, Close should be at a specific column
                close_idx = self.feature_columns.index('Close') if 'Close' in self.feature_columns else -1
                if close_idx == -1:
                    logger.warning(f"'Close' column not found in features for {window}. Using last column.")
                    close_idx = -1
                current_price_train = X_train[:, close_idx] 
                current_price_val = X_val[:, close_idx] 
                current_price_test = X_test[:, close_idx] 
            
            # Normalize current prices
            current_price_train_norm = (current_price_train - y_mean) / y_std
            current_price_val_norm = (current_price_val - y_mean) / y_std
            current_price_test_norm = (current_price_test - y_mean) / y_std
            
            # Ensure arrays have compatible shapes for concatenation
            if len(y_train_norm.shape) == 1:
                y_train_norm = y_train_norm.reshape(-1, 1)
            if len(y_val_norm.shape) == 1:
                y_val_norm = y_val_norm.reshape(-1, 1)
            if len(y_test_norm.shape) == 1:
                y_test_norm = y_test_norm.reshape(-1, 1)
            if len(current_price_train_norm.shape) == 1:
                current_price_train_norm = current_price_train_norm.reshape(-1, 1)
            if len(current_price_val_norm.shape) == 1:
                current_price_val_norm = current_price_val_norm.reshape(-1, 1)
            if len(current_price_test_norm.shape) == 1:
                current_price_test_norm = current_price_test_norm.reshape(-1, 1)
            
            # Align array lengths (take minimum length to avoid mismatch)
            min_train_len = min(len(y_train_norm), len(current_price_train_norm), len(X_train))
            min_val_len = min(len(y_val_norm), len(current_price_val_norm), len(X_val))
            min_test_len = min(len(y_test_norm), len(current_price_test_norm), len(X_test))
            
            logger.info(f"Aligning array lengths: train={min_train_len}, val={min_val_len}, test={min_test_len}")
            
            y_train_combined = np.column_stack([
                y_train_norm[:min_train_len], 
                current_price_train_norm[:min_train_len]
            ])
            y_val_combined = np.column_stack([
                y_val_norm[:min_val_len], 
                current_price_val_norm[:min_val_len]
            ])
            y_test_combined = np.column_stack([
                y_test_norm[:min_test_len], 
                current_price_test_norm[:min_test_len]
            ])
            
            # Also align X arrays with y arrays
            X_train = X_train[:min_train_len]
            X_val = X_val[:min_val_len]
            X_test = X_test[:min_test_len]
            
            logger.info(f"Target normalization for {window}: mean={y_mean}, std={y_std}")
            logger.info(f"Final array shapes - X_train: {X_train.shape}, y_train: {y_train_combined.shape}")
            
            # Skip hyperparameter tuning if insufficient data
            if min_train_len < 50:
                logger.warning(f"Insufficient data ({min_train_len} samples) for hyperparameter tuning. Using defaults.")
                hyperparams = self.model_config['default_hyperparameters']
            else:
                logger.info(f"Tuning hyperparameters for {window} window...")
                # Tune hyperparameters
                hyperparams = self.tune_hyperparameters(X_train, y_train_norm[:min_train_len])
            
            self.hyperparameters[window] = hyperparams
            logger.info(f"Building model for {window} window...")
            
            # Determine correct input shape based on array dimensions
            if len(X_train.shape) == 3:
                input_shape = (X_train.shape[1], X_train.shape[2])
            else:
                # For 2D arrays, reshape to 3D with timesteps=1
                X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
                X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
                X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
                input_shape = (X_train.shape[1], X_train.shape[2])
            
            logger.info(f"Model input shape: {input_shape}")
            
            # Build and train model
            model = self.build_model(input_shape, hyperparams)
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.model_config['early_stopping_patience'],
                restore_best_weights=True
            )
            # Note: Model saving is handled in train_all_models with proper ticker-specific naming
            
            logger.info(f"Starting model training for {window} window...")
            history = model.fit(
                X_train, y_train_combined,
                validation_data=(X_val, y_val_combined),
                epochs=self.model_config['epochs'],
                batch_size=hyperparams['batch_size'],
                callbacks=[early_stopping, LoggingCallback()],
                verbose=1
            )
            logger.info(f"Finished model training for {window} window.")
            # Evaluate on test set
            y_pred_norm = model.predict(X_test)
            # Handle prediction output shape
            if len(y_pred_norm.shape) > 1 and y_pred_norm.shape[1] > 1:
                y_pred_norm = y_pred_norm[:, 0]  # Take first column if multiple outputs
            y_pred = y_pred_norm * y_std + y_mean
            
            # Ensure y_test has correct shape for metrics
            y_test_eval = y_test[:min_test_len]
            if len(y_test_eval.shape) > 1:
                y_test_eval = y_test_eval.flatten()
            if len(y_pred.shape) > 1:
                y_pred = y_pred.flatten()
                
            metrics = {
                'mse': mean_squared_error(y_test_eval, y_pred),
                'mae': mean_absolute_error(y_test_eval, y_pred),
                'r2': r2_score(y_test_eval, y_pred)
            }
            self.metrics[window] = metrics
            logger.info(f"Test set metrics for {window}: {metrics}")
            # Note: Model will be stored with ticker-specific key in train_all_models
            return model
        except Exception as e:
            logger.error(f"Error training model for {window}: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def prepare_data(
        self,
        df: pd.DataFrame,
        window: str,
        mongo_client=None,
        ticker: str = None,
        short_interest_data: List[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for model training, including Alpha Vantage, sentiment, and short interest features."""
        try:
            if self.feature_engineer is None:
                raise ValueError("FeatureEngineer instance must be set before training or prediction.")
            
            # Ensure 'date' is a column
            if 'date' not in df.columns and getattr(df.index, 'name', None) == 'date':
                df = df.reset_index()

            # Process short interest data if provided
            if short_interest_data:
                # Let the feature engineer handle short interest data
                df = self.feature_engineer.add_short_interest_features(df, ticker, mongo_client)

            # Fetch latest Alpha Vantage and sentiment data from MongoDB if available
            alpha_vantage_dict = None
            sentiment_dict = None
            if mongo_client is not None and ticker is not None:
                # Fetch all relevant Alpha Vantage endpoints for this ticker
                alpha_vantage_dict = {
                    'alpha_earnings': mongo_client.get_alpha_vantage_data(ticker, 'earnings'),
                    'alpha_earnings_call': mongo_client.get_alpha_vantage_data(ticker, 'earnings_call_transcript'),
                    'alpha_insider_transactions': mongo_client.get_alpha_vantage_data(ticker, 'insider_transactions'),
                    'alpha_dividends': mongo_client.get_alpha_vantage_data(ticker, 'dividends'),
                    'alpha_quote': mongo_client.get_alpha_vantage_data(ticker, 'quote'),
                    'alphavantage_sentiment': mongo_client.get_alpha_vantage_data(ticker, 'alphavantage_sentiment'),
                }
                # Fetch latest sentiment dict (date->score or dict)
                latest_sentiment = mongo_client.get_latest_sentiment(ticker)
                if latest_sentiment and 'sentiment_timeseries' in latest_sentiment:
                    sentiment_dict = latest_sentiment['sentiment_timeseries']
                else:
                    sentiment_dict = None

            # Map window name to window_size
            window_size_map = {
                'next_day': 1,
                '30_day': 30,
                '90_day': 90
            }
            window_size = window_size_map.get(window, 1)

            # Prepare features using feature engineer
            features, targets = self.feature_engineer.prepare_features(
                df,
                sentiment_dict=sentiment_dict,
                alpha_vantage_dict=alpha_vantage_dict,
                window_size=window_size,
                mongo_client=mongo_client,
                ticker=ticker
            )

            # Store feature columns for later use
            if isinstance(features, np.ndarray) and features.shape[0] > 0:
                self.feature_columns = list(df.select_dtypes(include=[np.number]).columns)

            # Split data: 70% train, 15% val, 15% test
            n = len(features)
            if n < 10:
                logger.warning(f"Not enough samples ({n}) for proper train/val/test split.")
                return None, None, None, None, None, None

            train_end = int(n * 0.7)
            val_end = int(n * 0.85)
            X_train = features[:train_end]
            y_train = targets[:train_end]
            X_val = features[train_end:val_end]
            y_val = targets[train_end:val_end]
            X_test = features[val_end:]
            y_test = targets[val_end:]

            return X_train, y_train, X_val, y_val, X_test, y_test

        except Exception as e:
            logger.error(f"Error preparing data for {window} window: {str(e)}")
            return None, None, None, None, None, None

    def train_models(self, ticker: str, features: np.ndarray, targets: np.ndarray, window: str):
        """Train multiple models for ensemble prediction."""
        try:
            logger.info(f"Training models for {ticker}-{window}")
            
            # Split data for training and validation
            split_index = int(0.8 * len(features))
            X_train, X_test = features[:split_index], features[split_index:]
            y_train, y_test = targets[:split_index], targets[split_index:]
            
            # Handle different shapes for different model types
            window_size_map = {'next_day': 1, '7_day': 7, '30_day': 30}
            window_size = window_size_map.get(window, 1)
            
            models = {}
            
            # 1. LSTM Model (handles 3D input)
            try:
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
                from tensorflow.keras.optimizers import Adam
                from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
                
                if len(X_train.shape) == 3:  # Already windowed for LSTM
                    lstm_X_train, lstm_X_test = X_train, X_test
                else:  # Need to create windows
                    lstm_X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
                    lstm_X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
                
                # Build LSTM model
                lstm_model = Sequential([
                    LSTM(128, return_sequences=True, input_shape=(lstm_X_train.shape[1], lstm_X_train.shape[2])),
                    Dropout(0.2),
                    LSTM(64, return_sequences=False),
                    Dropout(0.2),
                    BatchNormalization(),
                    Dense(32, activation='relu'),
                    Dropout(0.1),
                    Dense(1)
                ])
                
                lstm_model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='mse',
                    metrics=['mae']
                )
                
                # Train LSTM
                early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
                reduce_lr = ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
                
                history = lstm_model.fit(
                    lstm_X_train, y_train,
                    validation_data=(lstm_X_test, y_test),
                    epochs=100,
                    batch_size=32,
                    callbacks=[early_stopping, reduce_lr],
                    verbose=0
                )
                
                models['lstm'] = lstm_model
                logger.info(f"LSTM model trained for {ticker}-{window}")
                
            except Exception as e:
                logger.warning(f"LSTM training failed for {ticker}-{window}: {e}")
            
            # 2. Flatten features for sklearn models
            if len(X_train.shape) == 3:
                flat_X_train = X_train.reshape(X_train.shape[0], -1)
                flat_X_test = X_test.reshape(X_test.shape[0], -1)
            else:
                flat_X_train, flat_X_test = X_train, X_test
            
            # 3. XGBoost
            try:
                import xgboost as xgb
                
                xgb_model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
                
                xgb_model.fit(
                    flat_X_train, y_train,
                    eval_set=[(flat_X_test, y_test)],
                    early_stopping_rounds=20,
                    verbose=False
                )
                
                models['xgboost'] = xgb_model
                logger.info(f"XGBoost model trained for {ticker}-{window}")
                
            except Exception as e:
                logger.warning(f"XGBoost training failed for {ticker}-{window}: {e}")
            
            # 4. LightGBM
            try:
                import lightgbm as lgb
                
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
                
                lgb_model.fit(
                    flat_X_train, y_train,
                    eval_set=[(flat_X_test, y_test)],
                    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
                )
                
                models['lightgbm'] = lgb_model
                logger.info(f"LightGBM model trained for {ticker}-{window}")
                
            except Exception as e:
                logger.warning(f"LightGBM training failed for {ticker}-{window}: {e}")
            

            
            # Store models and metadata
            if not ticker in self.models:
                self.models[ticker] = {}
            self.models[ticker][window] = models
            
            # Store metadata for evaluation
            self.model_metadata[f"{ticker}_{window}"] = {
                'feature_shape': X_train.shape,
                'target_mean': np.mean(y_train),
                'target_std': np.std(y_train),
                'n_samples': len(X_train),
                'window_size': window_size
            }
            
            logger.info(f"Trained {len(models)} models for {ticker}-{window}")
            return models
            
        except Exception as e:
            logger.error(f"Error training models for {ticker}-{window}: {e}")
            raise
    
    def train_all_models(self, ticker: str, start_date: str = None, end_date: str = None):
        """Train models for all prediction windows."""
        try:
            logger.info(f"Training all models for {ticker}")
            
            for window in self.prediction_windows:
                logger.info(f"Training {window} models for {ticker}")
                
                # Prepare training data
                features, targets = self.prepare_training_data(
                    ticker=ticker,
                    window=window,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Train models
                models = self.train_models(ticker, features, targets, window)
                
                # Save models
                self.save_models(ticker, window, models)
            
            logger.info(f"All models trained for {ticker}")
            return True
            
        except Exception as e:
            logger.error(f"Error training all models for {ticker}: {e}")
            return False

    def load_models(self):
        """Load trained models from disk or download from cloud storage if needed."""
        # Check if models directory exists and has models
        if not os.path.exists(self.model_dir) or not any(Path(self.model_dir).rglob('*.h5')):
            logger.info("No local models found, attempting to download from cloud storage...")
            try:
                # Try to download models from cloud storage
                import subprocess
                script_path = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'download_models.py')
                result = subprocess.run(['python', script_path], 
                                      capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    logger.info("Successfully downloaded models from cloud storage")
                else:
                    logger.warning(f"Model download failed: {result.stderr}")
            except Exception as e:
                logger.warning(f"Could not download models from cloud storage: {e}")
        
        if not os.path.exists(self.model_dir):
            logger.warning(f"Model directory {self.model_dir} does not exist")
            return
            
        # Load feature columns
        feature_columns_path = os.path.join(self.model_dir, 'feature_columns.json')
        if os.path.exists(feature_columns_path):
            with open(feature_columns_path, 'r') as f:
                self.feature_columns = json.load(f)
            logger.info(f"Loaded feature columns from {feature_columns_path}")
        
        # Initialize feature selectors dict
        self.feature_selectors = {}
        
        # Look for ticker directories
        for ticker_dir in os.listdir(self.model_dir):
            ticker_path = os.path.join(self.model_dir, ticker_dir)
            if not os.path.isdir(ticker_path):
                continue
                
            ticker = ticker_dir
            logger.info(f"Loading models for {ticker}")
            
            # Load models for each window
            for window in self.prediction_windows:
                # Load LSTM models
                lstm_model_path = os.path.join(ticker_path, f'model_{ticker}_{window}_lstm.h5')
                if os.path.exists(lstm_model_path):
                    try:
                        model = load_model(lstm_model_path, custom_objects={'custom_loss': self.custom_loss})
                        self.models[(ticker, window)] = model
                        logger.info(f"Loaded LSTM model for {ticker}-{window}")
                    except Exception as e:
                        logger.error(f"Error loading LSTM model for {ticker}-{window}: {e}")
                
                # Load LightGBM models
                for model_type in ['classic', 'quantile']:
                    lgbm_model_path = os.path.join(ticker_path, f'model_{ticker}_{window}_{model_type}_lgbm.joblib')
                    if os.path.exists(lgbm_model_path):
                        try:
                            if model_type == 'quantile':
                                # For quantile models, load the dictionary of quantile models
                                quantile_models = joblib.load(lgbm_model_path)
                                self.lgbm_models[(ticker, window, model_type)] = quantile_models
                                logger.info(f"Loaded LightGBM {model_type} models for {ticker}-{window}")
                            else:
                                model = joblib.load(lgbm_model_path)
                                self.lgbm_models[(ticker, window, model_type)] = model
                                logger.info(f"Loaded LightGBM {model_type} model for {ticker}-{window}")
                        except Exception as e:
                            logger.error(f"Error loading LightGBM {model_type} model for {ticker}-{window}: {e}")
                
                # Load XGBoost models
                xgb_model_path = os.path.join(ticker_path, f'model_{ticker}_{window}_xgb.joblib')
                if os.path.exists(xgb_model_path):
                    try:
                        model = joblib.load(xgb_model_path)
                        self.xgb_models[(ticker, window)] = model
                        logger.info(f"Loaded XGBoost model for {ticker}-{window}")
                    except Exception as e:
                        logger.error(f"Error loading XGBoost model for {ticker}-{window}: {e}")
                

                
                # Load feature selectors
                selector_path = os.path.join(ticker_path, f"feature_selector_{ticker}_{window}.joblib")
                if os.path.exists(selector_path):
                    try:
                        selector = joblib.load(selector_path)
                        self.feature_selectors[(ticker, window)] = selector
                        logger.info(f"Loaded feature selector for {ticker}-{window}")
                    except Exception as e:
                        logger.error(f"Error loading feature selector for {ticker}-{window}: {e}")
                
                # Load feature info (for debugging)
                feature_info_path = os.path.join(ticker_path, f"feature_info_{ticker}_{window}.json")
                if os.path.exists(feature_info_path):
                    try:
                        with open(feature_info_path, 'r') as f:
                            feature_info = json.load(f)
                        logger.info(f"Feature info for {ticker}-{window}: {feature_info}")
                    except Exception as e:
                        logger.error(f"Error loading feature info for {ticker}-{window}: {e}")
        
        logger.info(f"Model loading completed. Loaded {len(self.models)} LSTM models, {len(self.lgbm_models)} LightGBM models, {len(self.xgb_models)} XGBoost models, {len(self.feature_selectors)} feature selectors.")

    def predict(
        self,
        df: pd.DataFrame,
        window: str,
        ticker: str,
        raw_current_price: float = None,
        short_interest_data: List[Dict] = None
    ) -> Dict[str, Any]:
        """Make quantile and point predictions for a given ticker and window, including short interest data."""
        try:
            if isinstance(df, pd.DataFrame):
                # Use proper path to feature_columns.json
                feature_columns_path = os.path.join(self.model_dir, 'feature_columns.json')
                with open(feature_columns_path, 'r') as f:
                    feature_columns = json.load(f)
                df = df.reindex(columns=feature_columns, fill_value=0)
                
                # Add short interest data if provided
                if short_interest_data:
                    short_interest_df = pd.DataFrame(short_interest_data)
                    short_interest_df['settlement_date'] = pd.to_datetime(short_interest_df['settlement_date'])
                    short_interest_df.set_index('settlement_date', inplace=True)
                    df = df.merge(short_interest_df, left_index=True, right_index=True, how='left')
                    df.fillna(method='ffill', inplace=True)
                
                features, _ = self.feature_engineer.prepare_features(df)
                if raw_current_price is None:
                    raw_current_price = float(df['Close'].iloc[-1])
            else:
                features = df
                if raw_current_price is None:
                    raise ValueError("raw_current_price must be provided when input is not a DataFrame.")

            logger.info(f"Input features for prediction (ticker={ticker}, window={window}): {features}")
            if features.ndim == 2:
                features = features[np.newaxis, ...]
            X_flat = features.reshape(1, -1)
            
            # Apply the same feature selection used during training
            if hasattr(self, 'feature_selectors') and (ticker, window) in self.feature_selectors:
                selector = self.feature_selectors[(ticker, window)]
                X_flat = selector.transform(X_flat)
                logger.info(f"Applied feature selection: {X_flat.shape[1]} features for {ticker}-{window}")

            # Quantile regression (LightGBM)
            lgbm_quantile_models = self.lgbm_models.get((ticker, window, 'quantile'), {})
            preds = {}
            for q in [0.1, 0.5, 0.9]:
                model = lgbm_quantile_models.get(q)
                if model:
                    preds[q] = model.predict(X_flat)[0]
                else:
                    preds[q] = None

            # Classic regression point predictions
            lgbm_model = self.lgbm_models.get((ticker, window, 'classic'))
            lgbm_pred = lgbm_model.predict(X_flat)[0] if lgbm_model else None
            xgb_model = self.xgb_models.get((ticker, window))
            xgb_pred = xgb_model.predict(X_flat)[0] if xgb_model else None


            # LSTM (if available)
            lstm_pred = None
            if (ticker, window) in self.models:
                try:
                    # Prepare features for LSTM prediction
                    if window == 'next_day':
                        # For next_day, features should be 2D -> reshape to 3D
                        if len(features.shape) == 2:
                            lstm_features = features.reshape(1, 1, features.shape[1])
                        else:
                            lstm_features = features
                    else:
                        # For 30_day and 90_day, features should already be 3D
                        lstm_features = features
                    
                    # Ensure we have the right shape
                    logger.info(f"LSTM input shape for {window}: {lstm_features.shape}")
                    
                    # Make prediction
                    pred_norm = self.models[(ticker, window)].predict(lstm_features, verbose=0)[0]
                    
                    # Handle prediction output - take first column if multiple outputs
                    if len(pred_norm.shape) > 0 and len(pred_norm) > 1:
                        pred_norm = pred_norm[0]
                    elif len(pred_norm.shape) == 0:
                        pred_norm = float(pred_norm)
                    else:
                        pred_norm = float(pred_norm)
                    
                    # Denormalize prediction
                    stats = self.target_stats.get(window, {"mean": 0, "std": 1})
                    predicted_delta = float(pred_norm * stats["std"] + stats["mean"])
                    lstm_pred = raw_current_price + predicted_delta
                    
                    logger.info(f"LSTM prediction successful for {ticker}-{window}: {lstm_pred}")
                    
                except Exception as e:
                    logger.error(f"Error making LSTM prediction for {ticker} - {window}: {str(e)}")
                    lstm_pred = None

            # Calculate main prediction (use median)
            median_pred = raw_current_price + preds[0.5] if preds[0.5] is not None else None

            # Calculate confidence (1 - normalized range width)
            if preds[0.1] is not None and preds[0.9] is not None:
                range_width = abs(preds[0.9] - preds[0.1])
                # Normalize by current price to get a value between 0 and 1
                confidence = max(0.0, min(1.0, 1.0 - (range_width / (abs(raw_current_price) + 1e-6))))
            else:
                confidence = 0.5  # fallback

            result = {
                "prediction": median_pred,
                "confidence": confidence,
                "range": [raw_current_price + preds[0.1] if preds[0.1] is not None else None,
                          raw_current_price + preds[0.9] if preds[0.9] is not None else None],
                "median": median_pred,
                "lstm": lstm_pred,
                "xgb": raw_current_price + xgb_pred if xgb_pred is not None else None,

                "lgbm": raw_current_price + lgbm_pred if lgbm_pred is not None else None,
                "quantiles": preds
            }

            logger.info(f"Unified prediction (ticker={ticker}, window={window}): {result}")

            # Get feature importance for explanation
            feature_importance = self.explain_prediction(features, window)
            
            # Generate LLM explanation
            explanation_data = {
                'prediction_range': result['range'],
                'confidence_score': result['confidence'],
                'key_factors': self._extract_key_factors(feature_importance),
                'technical_analysis': self._analyze_technical_indicators(df, feature_importance),
                'sentiment_analysis': self._analyze_sentiment_impact(df, feature_importance),
                'market_context': self._analyze_market_context(df, feature_importance),
                'llm_explanation': self._generate_llm_explanation(
                    ticker,
                    result,
                    feature_importance,
                    df
                ),
                'feature_importance': feature_importance,
                'timestamp': datetime.utcnow()
            }
            
            # Store explanation in MongoDB
            if self.feature_engineer:
                self.feature_engineer.store_llm_explanation(
                    ticker,
                    df['date'].iloc[-1] if isinstance(df, pd.DataFrame) else datetime.utcnow(),
                    explanation_data
                )
            
            # Add explanation to result
            result['explanation'] = explanation_data
            
            return result

        except Exception as e:
            logger.error(f"Error making unified prediction for {ticker} - {window} window: {str(e)}")
            return None

    def _extract_key_factors(self, feature_importance: Dict) -> List[str]:
        """Extract key factors from feature importance."""
        try:
            # Sort features by importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            # Get top 5 most important features
            top_features = sorted_features[:5]
            
            # Convert to readable factors
            factors = []
            for feature, importance in top_features:
                if 'technical' in feature.lower():
                    factors.append(f"Technical indicator {feature} ({importance:.2f})")
                elif 'sentiment' in feature.lower():
                    factors.append(f"Sentiment factor {feature} ({importance:.2f})")
                elif 'market' in feature.lower():
                    factors.append(f"Market condition {feature} ({importance:.2f})")
                else:
                    factors.append(f"{feature} ({importance:.2f})")
                    
            return factors
            
        except Exception as e:
            logger.error(f"Error extracting key factors: {e}")
            return []
            
    def _analyze_technical_indicators(self, df: pd.DataFrame, feature_importance: Dict) -> Dict:
        """Analyze technical indicators that contributed to the prediction."""
        try:
            technical_analysis = {}
            
            # Get technical indicators from feature importance
            tech_features = {
                k: v for k, v in feature_importance.items()
                if any(x in k.lower() for x in ['macd', 'rsi', 'bollinger', 'sma', 'ema'])
            }
            
            for feature, importance in tech_features.items():
                if feature in df.columns:
                    current_value = df[feature].iloc[-1]
                    technical_analysis[feature] = {
                        'importance': importance,
                        'current_value': current_value,
                        'trend': 'bullish' if current_value > df[feature].mean() else 'bearish'
                    }
                    
            return technical_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing technical indicators: {e}")
            return {}
            
    def _analyze_sentiment_impact(self, df: pd.DataFrame, feature_importance: Dict) -> Dict:
        """Analyze sentiment factors that influenced the prediction."""
        try:
            sentiment_analysis = {}
            
            # Get sentiment features from feature importance
            sentiment_features = {
                k: v for k, v in feature_importance.items()
                if any(x in k.lower() for x in ['sentiment', 'news', 'social'])
            }
            
            for feature, importance in sentiment_features.items():
                if feature in df.columns:
                    current_value = df[feature].iloc[-1]
                    sentiment_analysis[feature] = {
                        'importance': importance,
                        'current_value': current_value,
                        'impact': 'positive' if current_value > 0 else 'negative'
                    }
                    
            return sentiment_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment impact: {e}")
            return {}
            
    def _analyze_market_context(self, df: pd.DataFrame, feature_importance: Dict) -> Dict:
        """Analyze broader market conditions."""
        try:
            market_context = {}
            
            # Get market-related features from feature importance
            market_features = {
                k: v for k, v in feature_importance.items()
                if any(x in k.lower() for x in ['market', 'sector', 'macro'])
            }
            
            for feature, importance in market_features.items():
                if feature in df.columns:
                    current_value = df[feature].iloc[-1]
                    market_context[feature] = {
                        'importance': importance,
                        'current_value': current_value,
                        'trend': 'favorable' if current_value > df[feature].mean() else 'unfavorable'
                    }
                    
            return market_context
            
        except Exception as e:
            logger.error(f"Error analyzing market context: {e}")
            return {}
            
    def _generate_llm_explanation(
        self,
        ticker: str,
        prediction: Dict[str, float],
        feature_importance: Dict,
        df: pd.DataFrame
    ) -> str:
        """Generate LLM explanation for the prediction."""
        try:
            # Extract key information
            price_range = f"${prediction['lower']:.2f} - ${prediction['upper']:.2f}"
            confidence = prediction.get('confidence', 0.0)
            
            # Get key factors
            key_factors = self._extract_key_factors(feature_importance)
            
            # Get technical analysis
            tech_analysis = self._analyze_technical_indicators(df, feature_importance)
            
            # Get sentiment analysis
            sentiment_analysis = self._analyze_sentiment_impact(df, feature_importance)
            
            # Get market context
            market_context = self._analyze_market_context(df, feature_importance)
            
            # Generate explanation
            explanation = f"""
            Based on our analysis of {ticker}, we predict a price range of {price_range} with {confidence:.1%} confidence.
            
            Key factors influencing this prediction:
            {chr(10).join(f'- {factor}' for factor in key_factors)}
            
            Technical Analysis:
            {chr(10).join(f'- {k}: {v["trend"]} (importance: {v["importance"]:.2f})' for k, v in tech_analysis.items())}
            
            Sentiment Analysis:
            {chr(10).join(f'- {k}: {v["impact"]} impact (importance: {v["importance"]:.2f})' for k, v in sentiment_analysis.items())}
            
            Market Context:
            {chr(10).join(f'- {k}: {v["trend"]} conditions (importance: {v["importance"]:.2f})' for k, v in market_context.items())}
            """
            
            return explanation.strip()
            
        except Exception as e:
            logger.error(f"Error generating LLM explanation: {e}")
            return "Unable to generate detailed explanation at this time."

    def predict_all_windows(self, ticker: str, df: pd.DataFrame) -> Dict[str, Dict]:
        """Make predictions for all windows using consistent feature engineering."""
        try:
            logger.info(f"Making predictions for {ticker} across all windows")
            
            predictions = {}
            
            for window in self.prediction_windows:
                try:
                    # Load models for this window
                    if ticker not in self.models or window not in self.models[ticker]:
                        # Try to load from disk
                        self.load_models_for_ticker_window(ticker, window)
                    
                    if ticker not in self.models or window not in self.models[ticker]:
                        logger.warning(f"No models found for {ticker}-{window}")
                        continue
                    
                    models = self.models[ticker][window]
                    
                    # Create prediction features using consistent pipeline
                    prediction_features = self.feature_engineer.create_prediction_features(
                        df=df.copy(),
                        ticker=ticker,
                        window=window,
                        mongo_client=self.mongo_client
                    )
                    
                    if prediction_features is None:
                        logger.warning(f"Could not create features for {ticker}-{window}")
                        continue
                    
                    # Make ensemble predictions
                    model_predictions = {}
                    
                    # LSTM prediction
                    if 'lstm' in models:
                        try:
                            lstm_features = prediction_features
                            if len(lstm_features.shape) == 2:
                                lstm_features = lstm_features.reshape(1, lstm_features.shape[0], lstm_features.shape[1])
                            elif len(lstm_features.shape) == 1:
                                lstm_features = lstm_features.reshape(1, 1, -1)
                            
                            lstm_pred = models['lstm'].predict(lstm_features, verbose=0)
                            model_predictions['lstm'] = float(lstm_pred[0][0])
                            
                        except Exception as e:
                            logger.warning(f"LSTM prediction failed for {ticker}-{window}: {e}")
                    
                    # Flatten features for sklearn models
                    if len(prediction_features.shape) == 3:
                        flat_features = prediction_features.reshape(prediction_features.shape[0], -1)
                    else:
                        flat_features = prediction_features
                    
                    # XGBoost prediction
                    if 'xgboost' in models:
                        try:
                            xgb_pred = models['xgboost'].predict(flat_features)
                            model_predictions['xgboost'] = float(xgb_pred[0])
                        except Exception as e:
                            logger.warning(f"XGBoost prediction failed for {ticker}-{window}: {e}")
                    
                    # LightGBM prediction
                    if 'lightgbm' in models:
                        try:
                            lgb_pred = models['lightgbm'].predict(flat_features)
                            model_predictions['lightgbm'] = float(lgb_pred[0])
                        except Exception as e:
                            logger.warning(f"LightGBM prediction failed for {ticker}-{window}: {e}")
                    

                    
                    if not model_predictions:
                        logger.warning(f"No successful predictions for {ticker}-{window}")
                        continue
                    
                    # Ensemble prediction with dynamic weights based on recent performance
                    # Default weights for new models or when no performance data available
                    default_weights = {
                        'lstm': 0.35,
                        'xgboost': 0.35,
                        'lightgbm': 0.30,
                        # Removed random_forest for better focus
                    }
                    
                    # Try to get performance-based weights from MongoDB
                    try:
                        performance_weights = self._get_performance_weights(ticker, window)
                        weights = performance_weights if performance_weights else default_weights
                    except:
                        weights = default_weights
                    
                    # Normalize weights for available models
                    available_models = list(model_predictions.keys())
                    total_weight = sum(weights.get(model, 0.2) for model in available_models)
                    normalized_weights = {model: weights.get(model, 0.2) / total_weight for model in available_models}
                    
                    # Calculate ensemble prediction
                    ensemble_pred = sum(pred * normalized_weights[model] for model, pred in model_predictions.items())
                    
                    # Calculate confidence based on prediction variance
                    pred_values = list(model_predictions.values())
                    if len(pred_values) > 1:
                        pred_std = np.std(pred_values)
                        pred_mean = np.mean(pred_values)
                        confidence = max(0.1, min(0.95, 1.0 - (pred_std / max(abs(pred_mean), 1))))
                    else:
                        confidence = 0.6  # Medium confidence for single model
                    
                    # Get current price
                    current_price = float(df['Close'].iloc[-1])
                    
                    # Calculate predicted price
                    predicted_price = current_price + ensemble_pred
                    
                    # Calculate prediction range based on window
                    window_ranges = {'next_day': 0.05, '7_day': 0.08, '30_day': 0.12}
                    range_factor = window_ranges.get(window, 0.1)
                    price_range = abs(ensemble_pred) * range_factor
                    
                    predictions[window] = {
                        'predicted_price': round(predicted_price, 2),
                        'price_change': round(ensemble_pred, 2),
                        'confidence': round(confidence, 3),
                        'current_price': round(current_price, 2),
                        'model_predictions': model_predictions,
                        'ensemble_weight': normalized_weights,
                        'price_range': {
                            'low': round(predicted_price - price_range, 2),
                            'high': round(predicted_price + price_range, 2)
                        }
                    }
                    
                    logger.info(f"{ticker}-{window}: ${predicted_price:.2f} (change: ${ensemble_pred:+.2f}, confidence: {confidence:.3f})")
                    
                except Exception as e:
                    logger.error(f"Error predicting {ticker}-{window}: {e}")
                    continue
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in predict_all_windows for {ticker}: {e}")
            return {}
    
    def load_models_for_ticker_window(self, ticker: str, window: str):
        """Load models from disk for specific ticker and window."""
        try:
            ticker_dir = os.path.join("models", ticker)
            
            if not os.path.exists(ticker_dir):
                return False
            
            models = {}
            
            # Load LSTM model
            lstm_path = os.path.join(ticker_dir, f'model_{ticker}_{window}_lstm.h5')
            if os.path.exists(lstm_path):
                models['lstm'] = load_model(lstm_path)
                logger.info(f"Loaded LSTM model for {ticker}-{window}")
            
            # Load XGBoost model
            xgb_path = os.path.join(ticker_dir, f'model_{ticker}_{window}_xgb.joblib')
            if os.path.exists(xgb_path):
                models['xgboost'] = joblib.load(xgb_path)
                logger.info(f"Loaded XGBoost model for {ticker}-{window}")
            
            # Load LightGBM model
            lgb_path = os.path.join(ticker_dir, f'model_{ticker}_{window}_lightgbm_lgbm.joblib')
            if os.path.exists(lgb_path):
                models['lightgbm'] = joblib.load(lgb_path)
                logger.info(f"Loaded LightGBM model for {ticker}-{window}")
            

            
            if models:
                if ticker not in self.models:
                    self.models[ticker] = {}
                self.models[ticker][window] = models
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error loading models for {ticker}-{window}: {e}")
            return False

    def _get_performance_weights(self, ticker: str, window: str) -> dict:
        """Get ensemble weights based on recent model performance."""
        try:
            # Query recent predictions and their accuracy
            collection = self.mongo_client.db.stock_predictions
            recent_predictions = list(collection.find({
                'ticker': ticker,
                'window': window,
                'timestamp': {'$gte': datetime.now() - timedelta(days=30)},
                'model_predictions': {'$exists': True},
                'actual_price': {'$exists': True}
            }).sort('timestamp', -1).limit(20))
            
            if len(recent_predictions) < 5:
                return None  # Not enough data for performance-based weighting
            
            # Calculate accuracy for each model
            model_accuracies = {'lstm': [], 'xgboost': [], 'lightgbm': []}
            
            for pred in recent_predictions:
                if 'model_predictions' in pred and 'actual_price' in pred:
                    actual = pred['actual_price']
                    for model_name, model_pred in pred['model_predictions'].items():
                        if model_name in model_accuracies:
                            error = abs(model_pred - actual) / actual if actual != 0 else 0
                            accuracy = max(0, 1 - error)  # Convert error to accuracy
                            model_accuracies[model_name].append(accuracy)
            
            # Calculate average accuracy and convert to weights
            weights = {}
            total_accuracy = 0
            
            for model_name, accuracies in model_accuracies.items():
                if accuracies:
                    avg_accuracy = sum(accuracies) / len(accuracies)
                    weights[model_name] = avg_accuracy
                    total_accuracy += avg_accuracy
            
            # Normalize weights
            if total_accuracy > 0:
                normalized_weights = {k: v/total_accuracy for k, v in weights.items()}
                # Ensure weights are reasonable (no single model > 60%)
                for model, weight in normalized_weights.items():
                    normalized_weights[model] = min(weight, 0.6)
                
                # Re-normalize after capping
                total_capped = sum(normalized_weights.values())
                if total_capped > 0:
                    return {k: v/total_capped for k, v in normalized_weights.items()}
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting performance weights for {ticker}-{window}: {e}")
            return None

if __name__ == "__main__":
    import argparse
    import os
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description="Train stock prediction models from MongoDB data.")
    parser.add_argument('--ticker', type=str, default=None, help='Single ticker to train (default: all)')
    args = parser.parse_args()
    from ml_backend.utils.mongodb import MongoDBClient
    from ml_backend.data.features import FeatureEngineer
    mongo_uri = os.getenv("MONGODB_URI")
    mongo_client = MongoDBClient(mongo_uri)
    feature_engineer = FeatureEngineer(mongo_client=mongo_client)
    predictor = StockPredictor(mongo_client=mongo_client)
    predictor.set_feature_engineer(feature_engineer)
    if args.ticker:
        print(f"Training model for {args.ticker}...")
        # Fetch historical data
        df = mongo_client.get_historical_data(args.ticker)
        if df is not None and not df.empty:
            predictor.train_all_models({args.ticker: df})
            print(f"Trained model for {args.ticker}.")
        else:
            print(f"No data for {args.ticker}, skipping.")
    else:
        print("Training models for all tickers...")
        all_data = {ticker: mongo_client.get_historical_data(ticker) for ticker in TOP_100_TICKERS}
        predictor.train_all_models(all_data)
        print("Trained models for all tickers.")
    mongo_client.close() 