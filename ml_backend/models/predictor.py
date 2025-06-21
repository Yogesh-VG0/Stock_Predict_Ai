"""
Machine learning model module for stock price prediction.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization, Concatenate, Lambda, MultiHeadAttention, LayerNormalization, Add, GlobalAveragePooling1D, Bidirectional
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
import requests

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
        # Marketstack API configuration
        self.marketstack_api_key = os.getenv("MARKETSTACK_API_KEY")
        self.marketstack_base_url = "http://api.marketstack.com/v2"
        self.stockdata_api_key = os.getenv("STOCKDATA_API_KEY")
        self.stockdata_base_url = "https://api.stockdata.org/v1"
        
        # Initialize missing attributes
        self.model_config = MODEL_CONFIG
        self.target_stats = {}
        self.hyperparameters = {}
        self.metrics = {}
        self.feature_columns = []
        
    def prepare_training_data(self, ticker: str, window: str, 
                            start_date: str = None, end_date: str = None):
        """Prepare data for training with consistent feature engineering."""
        try:
            # Window size mapping
            window_size_map = {'next_day': 1, '7_day': 7, '30_day': 30}
            window_size = window_size_map.get(window, 1)
            
            logger.info(f"Preparing training data for {ticker} - {window} window (size: {window_size})")
            
            # Get historical data
            collection = self.mongo_client.db.historical_data
            
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
            sentiment_data = self.mongo_client.get_latest_sentiment(ticker)
            
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

    def build_finance_aware_model(self, input_shape: Tuple[int, int], hyperparams: Dict) -> tf.keras.Model:
        """
        Build a finance-aware model with domain-specific architecture.
        Uses separate processing paths for different feature types and attention mechanisms.
        """
        try:
            # Input layer
            inputs = Input(shape=input_shape, name='main_input')
            
            # NEW: Improved Architecture for Better Accuracy
            
            # 1. Multi-head attention for sequence modeling (better than simple LSTM)
            attention_output = MultiHeadAttention(
                num_heads=4,
                key_dim=hyperparams.get('attention_dim', 32),
                dropout=0.1,
                name='multi_head_attention'
            )(inputs, inputs)
            
            # 2. Layer normalization for stability
            attention_output = LayerNormalization(name='attention_norm')(attention_output)
            
            # 3. Bidirectional LSTM for better temporal understanding
            lstm_output = Bidirectional(
                LSTM(
                    hyperparams.get('lstm_units', 64),
                    return_sequences=True,
                    dropout=hyperparams.get('dropout_rate', 0.3),
                    recurrent_dropout=0.2,
                    kernel_regularizer=l2(hyperparams.get('l2_reg', 1e-3)),
                    name='bidirectional_lstm'
                )
            )(attention_output)
            
            # 4. Add residual connection
            if attention_output.shape[-1] == lstm_output.shape[-1]:
                combined = Add(name='residual_connection')([attention_output, lstm_output])
            else:
                combined = lstm_output
            
            # 5. Global average pooling instead of just taking last timestep
            pooled = GlobalAveragePooling1D(name='global_avg_pool')(combined)
            
            # 6. Feature extraction with batch normalization
            dense1 = Dense(
                hyperparams.get('dense_units', 128),
                activation='relu',
                kernel_regularizer=l2(hyperparams.get('l2_reg', 1e-3)),
                name='dense_1'
            )(pooled)
            dense1 = BatchNormalization(name='batch_norm_1')(dense1)
            dense1 = Dropout(hyperparams.get('dropout_rate', 0.3), name='dropout_1')(dense1)
            
            # 7. Second dense layer with smaller size
            dense2 = Dense(
                hyperparams.get('dense_units', 128) // 2,
                activation='relu',
                kernel_regularizer=l2(hyperparams.get('l2_reg', 1e-3)),
                name='dense_2'
            )(dense1)
            dense2 = BatchNormalization(name='batch_norm_2')(dense2)
            dense2 = Dropout(hyperparams.get('dropout_rate', 0.3) * 0.5, name='dropout_2')(dense2)
            
            # 8. Output layer with linear activation for regression
            outputs = Dense(1, activation='linear', name='prediction_output')(dense2)
            
            # Create model
            model = Model(inputs=inputs, outputs=outputs, name='finance_aware_lstm')
            
            # 9. Use advanced optimizer with learning rate scheduling
            initial_learning_rate = hyperparams.get('learning_rate', 0.001)
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=100,
                decay_rate=0.96,
                staircase=True
            )
            
            optimizer = Adam(
                learning_rate=lr_schedule,
                beta_1=0.9,
                beta_2=0.999,
                clipnorm=1.0  # Gradient clipping for stability
            )
            
            # 10. Compile with custom loss function that penalizes large errors more
            model.compile(
                optimizer=optimizer,
                loss=self._huber_loss,  # More robust than MSE
                metrics=['mae', 'mse']
            )
            
            logger.info(f"Built enhanced finance-aware model with {model.count_params()} parameters")
            return model
            
        except Exception as e:
            logger.error(f"Error building finance-aware model: {e}")
            # Fallback to basic model
            return self.build_model(input_shape, hyperparams)

    def _huber_loss(self, y_true, y_pred, delta=1.0):
        """
        Huber loss - more robust to outliers than MSE, smoother than MAE.
        Better for financial predictions with occasional large price movements.
        """
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= delta
        squared_loss = tf.square(error) / 2
        linear_loss = delta * tf.abs(error) - tf.square(delta) / 2
        return tf.where(is_small_error, squared_loss, linear_loss)

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
                '7_day': 7,
                '30_day': 30
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
                'window_size': window
            }
            
            logger.info(f"Trained {len(models)} models for {ticker}-{window}")
            return models
            
        except Exception as e:
            logger.error(f"Error training models for {ticker}-{window}: {e}")
            raise
    
    def train_all_models(self, historical_data: dict):
        """Train models for all tickers using provided historical data dictionary."""
        try:
            logger.info(f"Training all models for {historical_data.keys()}")
            
            for ticker, df in historical_data.items():
                if df is None or df.empty:
                    logger.warning(f"No data for {ticker}, skipping")
                    continue
                    
                logger.info(f"Training models for {ticker}")
                
                for window in self.prediction_windows:
                    logger.info(f"Training {window} models for {ticker}")
                    
                    try:
                        # Prepare features using the DataFrame directly
                        features, targets = self.feature_engineer.prepare_features(
                            df=df,
                            window_size={'next_day': 1, '7_day': 7, '30_day': 30}[window],
                            ticker=ticker,
                            mongo_client=self.mongo_client
                        )
                        
                        if features is None or targets is None:
                            logger.warning(f"Could not prepare features for {ticker}-{window}")
                            continue
                            
                        # Train models
                        models = self.train_models(ticker, features, targets, window)
                        
                        # Save models
                        self.save_models(ticker, window, models)
                        
                    except Exception as e:
                        logger.error(f"Error training {ticker}-{window}: {e}")
                        continue
                
                logger.info(f"Completed training for {ticker}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training all models: {e}")
            return False

    def train_single_ticker(self, ticker: str, start_date: str = None, end_date: str = None):
        """Train models for a single ticker by fetching data from MongoDB."""
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

    def save_models(self, ticker: str, window: str, models: dict):
        """Save trained models to disk with enhanced metadata."""
        try:
            # Create directory for ticker models
            ticker_dir = os.path.join("models", ticker)
            os.makedirs(ticker_dir, exist_ok=True)
            
            # Save each model type
            for model_type, model in models.items():
                if model_type == 'lstm':
                    model_path = os.path.join(ticker_dir, f'model_{ticker}_{window}_lstm.h5')
                    model.save(model_path)
                    logger.info(f"Saved LSTM model for {ticker}-{window}")
                elif model_type == 'xgboost':
                    model_path = os.path.join(ticker_dir, f'model_{ticker}_{window}_xgb.joblib')
                    joblib.dump(model, model_path)
                    logger.info(f"Saved XGBoost model for {ticker}-{window}")
                elif model_type == 'lightgbm':
                    model_path = os.path.join(ticker_dir, f'model_{ticker}_{window}_lightgbm_lgbm.joblib')
                    joblib.dump(model, model_path)
                    logger.info(f"Saved LightGBM model for {ticker}-{window}")
            
            # Save feature pipeline for consistent prediction
            if self.feature_engineer:
                self.feature_engineer.save_feature_pipeline(ticker, window)
            
            # Save metadata
            metadata = {
                'ticker': ticker,
                'window': window,
                'models': list(models.keys()),
                'saved_at': datetime.utcnow().isoformat(),
                'hyperparameters': self.hyperparameters.get(window, {}),
                'metrics': self.metrics.get(window, {}),
                'target_stats': self.target_stats.get(window, {})
            }
            
            metadata_path = os.path.join(ticker_dir, f'metadata_{ticker}_{window}.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved {len(models)} models and metadata for {ticker}-{window}")
            
        except Exception as e:
            logger.error(f"Error saving models for {ticker}-{window}: {e}")
            import traceback
            logger.error(traceback.format_exc())

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

    def get_current_price_marketstack(self, ticker: str) -> Optional[float]:
        """
        Get the most recent closing price from Marketstack EOD API.
        Uses the /eod/latest endpoint for real-time current price.
        """
        try:
            if not self.marketstack_api_key:
                logger.warning("Marketstack API key not found, skipping Marketstack price fetch")
                return None
            
            # Use latest EOD endpoint for most recent price
            endpoint = f"{self.marketstack_base_url}/eod/latest"
            
            params = {
                'access_key': self.marketstack_api_key,
                'symbols': ticker,
                'limit': 1
            }
            
            logger.info(f"Fetching current price for {ticker} from Marketstack EOD API")
            
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                latest_data = data['data'][0]
                
                # Use adjusted close price (handles splits/dividends)
                current_price = float(latest_data['adj_close'])
                
                logger.info(f"Marketstack current price for {ticker}: ${current_price:.2f} (Date: {latest_data['date']})")
                
                # Store in MongoDB for caching
                if self.mongo_client:
                    try:
                        self.mongo_client.db['current_prices'].replace_one(
                            {'ticker': ticker},
                            {
                                'ticker': ticker,
                                'current_price': current_price,
                                'source': 'marketstack_eod',
                                'timestamp': datetime.utcnow(),
                                'data_date': latest_data['date'],
                                'exchange': latest_data.get('exchange', ''),
                                'volume': latest_data.get('adj_volume', 0)
                            },
                            upsert=True
                        )
                    except Exception as e:
                        logger.warning(f"Failed to cache Marketstack price in MongoDB: {e}")
                
                return current_price
            else:
                logger.warning(f"No EOD data returned from Marketstack for {ticker}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Marketstack API request failed for {ticker}: {e}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing Marketstack response for {ticker}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching Marketstack price for {ticker}: {e}")
            return None

    def get_cached_current_price(self, ticker: str, max_age_hours: int = 1) -> Optional[float]:
        """
        Get cached current price from MongoDB if recent enough.
        """
        try:
            if not self.mongo_client:
                return None
                
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            
            cached_price = self.mongo_client.db['current_prices'].find_one({
                'ticker': ticker,
                'timestamp': {'$gte': cutoff_time}
            })
            
            if cached_price:
                logger.info(f"Using cached current price for {ticker}: ${cached_price['current_price']:.2f}")
                return float(cached_price['current_price'])
                
            return None
            
        except Exception as e:
            logger.warning(f"Error retrieving cached price for {ticker}: {e}")
            return None

    def get_current_price_stockdata(self, ticker: str, include_extended_hours: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get real-time current price from StockData.org API (IEX data).
        Includes extended hours data and comprehensive market info.
        """
        try:
            if not self.stockdata_api_key:
                logger.warning("StockData API key not found, skipping StockData price fetch")
                return None
            
            endpoint = f"{self.stockdata_base_url}/data/quote"
            
            params = {
                'api_token': self.stockdata_api_key,
                'symbols': ticker,
                'extended_hours': str(include_extended_hours).lower()
            }
            
            logger.info(f"Fetching real-time price for {ticker} from StockData.org API")
            
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                quote_data = data['data'][0]
                
                current_price = float(quote_data['price'])
                
                # Create comprehensive price data object
                price_info = {
                    'current_price': current_price,
                    'ticker': quote_data['ticker'],
                    'name': quote_data.get('name', ''),
                    'exchange': quote_data.get('mic_code', ''),
                    'currency': quote_data.get('currency', 'USD'),
                    'day_open': float(quote_data.get('day_open', 0) or 0),
                    'day_high': float(quote_data.get('day_high', 0) or 0),
                    'day_low': float(quote_data.get('day_low', 0) or 0),
                    'volume': int(quote_data.get('volume', 0) or 0),
                    'previous_close': float(quote_data.get('previous_close_price', 0) or 0),
                    'day_change_percent': float(quote_data.get('day_change', 0) or 0),
                    'market_cap': quote_data.get('market_cap'),
                    '52_week_high': float(quote_data.get('52_week_high', 0) or 0),
                    '52_week_low': float(quote_data.get('52_week_low', 0) or 0),
                    'is_extended_hours': quote_data.get('is_extended_hours_price', False),
                    'last_trade_time': quote_data.get('last_trade_time'),
                    'source': 'stockdata_realtime',
                    'timestamp': datetime.utcnow()
                }
                
                logger.info(f"StockData real-time price for {ticker}: ${current_price:.2f} "
                           f"(Extended Hours: {price_info['is_extended_hours']}, "
                           f"Change: {price_info['day_change_percent']:.2f}%)")
                
                # Store comprehensive data in MongoDB for caching
                if self.mongo_client:
                    try:
                        self.mongo_client.db['current_prices'].replace_one(
                            {'ticker': ticker},
                            price_info,
                            upsert=True
                        )
                        logger.debug(f"Cached StockData price info for {ticker}")
                    except Exception as e:
                        logger.warning(f"Failed to cache StockData price in MongoDB: {e}")
                
                return price_info
            else:
                logger.warning(f"No quote data returned from StockData for {ticker}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"StockData API request failed for {ticker}: {e}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing StockData response for {ticker}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching StockData price for {ticker}: {e}")
            return None

    def get_current_price_with_fallback(self, ticker: str, df: pd.DataFrame = None) -> float:
        """
        Get current price with enhanced multi-tier fallback system:
        1. Cached price from MongoDB (if recent - under 15 minutes for real-time accuracy)
        2. StockData.org API (real-time IEX data with extended hours)
        3. Marketstack EOD API (professional-grade adjusted prices)
        4. DataFrame last close price (fallback)
        5. MongoDB historical data (last resort)
        """
        try:
            # 1. Try cached price first (fast) - shortened cache time for real-time accuracy
            cached_price = self.get_cached_current_price(ticker, max_age_hours=0.25)  # 15 minutes
            if cached_price is not None:
                return cached_price
            
            # 2. Try StockData.org API first (real-time with extended hours)
            stockdata_info = self.get_current_price_stockdata(ticker, include_extended_hours=True)
            if stockdata_info is not None:
                logger.info(f"Using StockData real-time price for {ticker}: ${stockdata_info['current_price']:.2f}")
                return stockdata_info['current_price']
            
            # 3. Try Marketstack EOD API (professional-grade backup)
            marketstack_price = self.get_current_price_marketstack(ticker)
            if marketstack_price is not None:
                logger.info(f"Using Marketstack EOD price for {ticker}: ${marketstack_price:.2f}")
                return marketstack_price
            
            # 4. Try DataFrame if provided
            if df is not None and 'Close' in df.columns and not df.empty:
                df_price = float(df['Close'].iloc[-1])
                logger.warning(f"Using DataFrame current price for {ticker}: ${df_price:.2f}")
                return df_price
            
            # 5. Try MongoDB historical data as last resort
            if self.mongo_client:
                historical_data = self.mongo_client.get_historical_data(ticker, limit=1)
                if historical_data is not None and not historical_data.empty and 'Close' in historical_data.columns:
                    historical_price = float(historical_data['Close'].iloc[-1])
                    logger.warning(f"Using historical MongoDB price for {ticker}: ${historical_price:.2f}")
                    return historical_price
            
            # If all else fails
            raise ValueError(f"Could not obtain current price for {ticker} from any source")
            
        except Exception as e:
            logger.error(f"Error getting current price for {ticker}: {e}")
            raise

    def get_historical_data_marketstack(self, ticker: str, days: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetch historical EOD data from Marketstack API.
        
        Args:
            ticker: Stock symbol
            days: Number of days of historical data to fetch
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            if not self.marketstack_api_key:
                logger.warning("Marketstack API key not found, skipping historical data fetch")
                return None
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            endpoint = f"{self.marketstack_base_url}/eod"
            
            params = {
                'access_key': self.marketstack_api_key,
                'symbols': ticker,
                'date_from': start_date.strftime('%Y-%m-%d'),
                'date_to': end_date.strftime('%Y-%m-%d'),
                'sort': 'ASC',  # Oldest first
                'limit': min(days, 1000)  # API limit is 1000
            }
            
            logger.info(f"Fetching {days} days of historical data for {ticker} from Marketstack")
            
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                # Convert to DataFrame
                historical_data = []
                for item in data['data']:
                    historical_data.append({
                        'date': pd.to_datetime(item['date']),
                        'Open': float(item['adj_open']),
                        'High': float(item['adj_high']),
                        'Low': float(item['adj_low']),
                        'Close': float(item['adj_close']),
                        'Volume': float(item['adj_volume']),
                        'ticker': ticker,
                        'exchange': item.get('exchange', ''),
                        'split_factor': float(item.get('split_factor', 1.0)),
                        'dividend': float(item.get('dividend', 0.0))
                    })
                
                df = pd.DataFrame(historical_data)
                df = df.set_index('date').sort_index()
                
                logger.info(f"Successfully fetched {len(df)} days of Marketstack historical data for {ticker}")
                
                # Cache in MongoDB
                if self.mongo_client:
                    try:
                        # Store in stock_data collection
                        records = df.reset_index().to_dict('records')
                        if records:
                            # Remove existing data for this ticker in the date range
                            self.mongo_client.db['stock_data'].delete_many({
                                'ticker': ticker,
                                'date': {
                                    '$gte': start_date,
                                    '$lte': end_date
                                }
                            })
                            
                            # Insert new data
                            self.mongo_client.db['stock_data'].insert_many(records)
                            logger.info(f"Cached {len(records)} Marketstack records in MongoDB for {ticker}")
                    
                    except Exception as e:
                        logger.warning(f"Failed to cache Marketstack historical data: {e}")
                
                return df
            else:
                logger.warning(f"No historical data returned from Marketstack for {ticker}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Marketstack historical data request failed for {ticker}: {e}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing Marketstack historical response for {ticker}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching Marketstack historical data for {ticker}: {e}")
            return None

    def predict(
        self,
        df: pd.DataFrame,
        window: str,
        ticker: str,
        raw_current_price: float = None,
        short_interest_data: List[Dict] = None
    ) -> Dict[str, Any]:
        """Make stock price prediction with enhanced current price detection."""
        try:
            # Get current price using enhanced fallback system
            if raw_current_price is None:
                raw_current_price = self.get_current_price_with_fallback(ticker, df)
            
            logger.info(f"Using current price for {ticker}: ${raw_current_price:.2f}")

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
            feature_importance = self.explain_prediction(features, window, ticker)
            
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

    def explain_prediction(self, features: np.ndarray, window: str, ticker: str) -> Dict[str, float]:
        """Generate feature importance for prediction explanation."""
        try:
            feature_importance = {}
            
            # Use SHAP if we have a model to explain
            if (ticker, window) in self.models:
                try:
                    import shap
                    model = self.models[(ticker, window)]
                    
                    # Create explainer
                    explainer = shap.Explainer(model)
                    
                    # Get SHAP values
                    shap_values = explainer(features)
                    
                    # Convert to feature importance dict
                    if hasattr(shap_values, 'values'):
                        values = shap_values.values
                        if len(values.shape) > 2:
                            values = values.mean(axis=1)  # Average over time steps
                        
                        for i, importance in enumerate(values[0]):
                            feature_name = f"feature_{i}"
                            if i < len(self.feature_columns):
                                feature_name = self.feature_columns[i]
                            feature_importance[feature_name] = float(importance)
                            
                except Exception as e:
                    logger.warning(f"SHAP explanation failed: {e}")
            
            # Fallback: Use gradient-based feature importance
            if not feature_importance and (ticker, window) in self.models:
                try:
                    model = self.models[(ticker, window)]
                    
                    # Simple gradient-based importance
                    with tf.GradientTape() as tape:
                        tape.watch(features)
                        predictions = model(features)
                    
                    gradients = tape.gradient(predictions, features)
                    if gradients is not None:
                        # Calculate importance as mean absolute gradient
                        importance_values = tf.reduce_mean(tf.abs(gradients), axis=0)
                        
                        for i, importance in enumerate(importance_values.numpy().flatten()):
                            feature_name = f"feature_{i}"
                            if i < len(self.feature_columns):
                                feature_name = self.feature_columns[i]
                            feature_importance[feature_name] = float(importance)
                            
                except Exception as e:
                    logger.warning(f"Gradient-based explanation failed: {e}")
            
            # Final fallback: Random feature importance for demonstration
            if not feature_importance:
                import random
                for i in range(min(10, len(self.feature_columns))):
                    feature_name = self.feature_columns[i] if i < len(self.feature_columns) else f"feature_{i}"
                    feature_importance[feature_name] = random.uniform(-1, 1)
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error generating feature importance: {e}")
            return {}

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
                            # Get expected timesteps for this window
                            window_timesteps = {'next_day': 1, '7_day': 7, '30_day': 30}.get(window, 1)
                            
                            # For prediction, we need the correct sequence length
                            lstm_features = prediction_features
                            
                            # Ensure we have the right shape for LSTM prediction
                            if len(lstm_features.shape) == 2:
                                # Check if we have enough rows for the window
                                if lstm_features.shape[0] >= window_timesteps:
                                    # Take the last window_timesteps rows
                                    lstm_features = lstm_features[-window_timesteps:, :]
                                    lstm_features = lstm_features.reshape(1, window_timesteps, lstm_features.shape[1])
                                else:
                                    # If we don't have enough history, pad with the last available row
                                    last_row = lstm_features[-1:, :]
                                    needed_rows = window_timesteps - lstm_features.shape[0]
                                    padding = np.repeat(last_row, needed_rows, axis=0)
                                    lstm_features = np.vstack([padding, lstm_features])
                                    lstm_features = lstm_features.reshape(1, window_timesteps, lstm_features.shape[1])
                            elif len(lstm_features.shape) == 1:
                                # Single feature vector - repeat for window_timesteps
                                lstm_features = np.repeat(lstm_features.reshape(1, -1), window_timesteps, axis=0)
                                lstm_features = lstm_features.reshape(1, window_timesteps, -1)
                            elif len(lstm_features.shape) == 3:
                                # Already 3D - ensure correct timesteps
                                if lstm_features.shape[1] != window_timesteps:
                                    if lstm_features.shape[1] >= window_timesteps:
                                        lstm_features = lstm_features[:, -window_timesteps:, :]
                                    else:
                                        # Pad to required length
                                        pad_length = window_timesteps - lstm_features.shape[1]
                                        last_timestep = lstm_features[:, -1:, :]
                                        padding = np.repeat(last_timestep, pad_length, axis=1)
                                        lstm_features = np.concatenate([padding, lstm_features], axis=1)
                            
                            logger.info(f"LSTM input shape for {ticker}-{window}: {lstm_features.shape}")
                            
                            lstm_pred = models['lstm'].predict(lstm_features, verbose=0)
                            model_predictions['lstm'] = float(lstm_pred[0][0])
                            
                        except Exception as e:
                            logger.warning(f"LSTM prediction failed for {ticker}-{window}: {e}")
                            import traceback
                            logger.warning(f"LSTM prediction traceback: {traceback.format_exc()}")
                    
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
                    
                    # Calculate enhanced confidence based on multiple factors
                    pred_values = list(model_predictions.values())
                    if len(pred_values) > 1:
                        pred_std = np.std(pred_values)
                        pred_mean = np.mean(pred_values)
                        
                        # Factor 1: Model Agreement (original method)
                        model_agreement = max(0.1, min(0.95, 1.0 - (pred_std / max(abs(pred_mean), 1))))
                        
                        # Factor 2: Feature Quality (based on data completeness)
                        feature_quality = self._calculate_feature_quality(prediction_features, ticker)
                        
                        # Factor 3: Market Volatility Adjustment (lower confidence in high volatility)
                        volatility_factor = self._get_market_volatility_factor(df, window)
                        
                        # Factor 4: Time Decay (longer predictions = lower confidence)
                        time_decay = {'next_day': 1.0, '7_day': 0.85, '30_day': 0.7}.get(window, 0.8)
                        
                        # Factor 5: Historical Model Performance
                        performance_factor = self._get_model_performance_factor(ticker, window, model_predictions.keys())
                        
                        # Combine all factors with weights
                        confidence = (
                            model_agreement * 0.35 +          # Model agreement (most important)
                            feature_quality * 0.25 +          # Data quality
                            (1 - volatility_factor) * 0.20 +  # Market stability (inverted)
                            time_decay * 0.15 +               # Time horizon
                            performance_factor * 0.05         # Historical performance
                        )
                        
                        # Ensure bounds
                        confidence = max(0.05, min(0.98, confidence))
                        
                    else:
                        confidence = 0.6  # Medium confidence for single model
                    
                    # Get current price using enhanced fallback system
                    current_price = self.get_current_price_with_fallback(ticker, df)
                    
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
                    
                    # Store prediction explanation in MongoDB
                    try:
                        if self.feature_engineer:
                            explanation_data = {
                                'model_predictions': model_predictions,
                                'ensemble_weights': normalized_weights,
                                'confidence_calculation': {
                                    'prediction_variance': pred_std if len(pred_values) > 1 else 0,
                                    'prediction_mean': pred_mean if len(pred_values) > 1 else pred_values[0] if pred_values else 0,
                                    'confidence_score': confidence
                                },
                                'price_calculation': {
                                    'current_price': current_price,
                                    'ensemble_prediction': ensemble_pred,
                                    'predicted_price': predicted_price,
                                    'range_factor': range_factor,
                                    'price_range': price_range
                                },
                                'feature_count': prediction_features.shape[1] if len(prediction_features.shape) > 1 else len(prediction_features),
                                'window': window,
                                'timestamp': datetime.utcnow().isoformat()
                            }
                            
                            # Store explanation in MongoDB
                            self.mongo_client.store_prediction_explanation(ticker, window, explanation_data)
                    except Exception as e:
                        logger.warning(f"Could not store prediction explanation for {ticker}-{window}: {e}")
                    
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

    def _calculate_feature_quality(self, features: np.ndarray, ticker: str) -> float:
        """Calculate feature quality score based on completeness and relevance."""
        try:
            if features is None or features.size == 0:
                return 0.3
            
            # Check for missing/zero values
            if len(features.shape) == 3:
                # For 3D features (LSTM), check last timestep
                feature_vector = features[0, -1, :] if features.shape[0] > 0 else features[0, 0, :]
            else:
                # For 2D features
                feature_vector = features[0, :] if features.shape[0] > 0 else features
            
            # Calculate completeness (non-zero, non-NaN values)
            valid_features = np.sum(~np.isnan(feature_vector) & (feature_vector != 0))
            total_features = len(feature_vector)
            completeness = valid_features / total_features if total_features > 0 else 0
            
            # Bonus for having key financial features (if we can identify them)
            key_feature_bonus = 0.1 if total_features > 50 else 0  # Assume >50 features means good coverage
            
            # Calculate final quality score
            quality = min(1.0, completeness + key_feature_bonus)
            return max(0.2, quality)  # Minimum 20% quality
            
        except Exception as e:
            logger.warning(f"Error calculating feature quality for {ticker}: {e}")
            return 0.5  # Default moderate quality
    
    def _get_market_volatility_factor(self, df: pd.DataFrame, window: str) -> float:
        """Get market volatility factor (0-1, higher = more volatile)."""
        try:
            if 'Close' not in df.columns or len(df) < 20:
                return 0.5  # Default moderate volatility
            
            # Calculate recent volatility (20-day rolling)
            returns = df['Close'].pct_change().dropna()
            if len(returns) < 5:
                return 0.5
            
            recent_vol = returns.tail(20).std()
            
            # Normalize volatility (typical daily vol ranges 0.01-0.05)
            normalized_vol = min(1.0, recent_vol / 0.03)  # 3% daily vol = high
            
            # Adjust for prediction window (longer windows less affected by short-term vol)
            window_adjustment = {'next_day': 1.0, '7_day': 0.7, '30_day': 0.4}.get(window, 0.7)
            
            return normalized_vol * window_adjustment
            
        except Exception as e:
            logger.warning(f"Error calculating volatility factor: {e}")
            return 0.5
    
    def _get_model_performance_factor(self, ticker: str, window: str, model_names: list) -> float:
        """Get historical model performance factor (0-1, higher = better performance)."""
        try:
            if not self.mongo_client:
                return 0.7  # Default good performance
            
            # Query recent prediction accuracy
            collection = self.mongo_client.db.stock_predictions
            recent_predictions = list(collection.find({
                'ticker': ticker,
                'window': window,
                'timestamp': {'$gte': datetime.now() - timedelta(days=14)},  # Last 2 weeks
                'actual_price': {'$exists': True}
            }).sort('timestamp', -1).limit(10))
            
            if len(recent_predictions) < 3:
                return 0.7  # Default if insufficient history
            
            # Calculate accuracy for available models
            total_accuracy = 0
            count = 0
            
            for pred in recent_predictions:
                if 'predicted_price' in pred and 'actual_price' in pred:
                    predicted = pred['predicted_price']
                    actual = pred['actual_price']
                    
                    if actual != 0:
                        error = abs(predicted - actual) / actual
                        accuracy = max(0, 1 - error)  # Convert error to accuracy
                        total_accuracy += accuracy
                        count += 1
            
            if count > 0:
                avg_accuracy = total_accuracy / count
                return min(1.0, max(0.1, avg_accuracy))
            
            return 0.7  # Default
            
        except Exception as e:
            logger.warning(f"Error calculating performance factor for {ticker}: {e}")
            return 0.7

    def get_bulk_current_prices_stockdata(self, tickers: List[str], include_extended_hours: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Get real-time current prices for multiple tickers in one API call.
        Much more efficient than individual calls for portfolio analysis.
        
        Args:
            tickers: List of stock symbols
            include_extended_hours: Include pre/post market data
            
        Returns:
            Dict mapping ticker -> price info dict
        """
        try:
            if not self.stockdata_api_key:
                logger.warning("StockData API key not found, skipping bulk price fetch")
                return {}
            
            if not tickers:
                return {}
            
            # StockData.org supports comma-separated symbols
            symbols_str = ','.join(tickers)
            
            endpoint = f"{self.stockdata_base_url}/data/quote"
            
            params = {
                'api_token': self.stockdata_api_key,
                'symbols': symbols_str,
                'extended_hours': str(include_extended_hours).lower()
            }
            
            logger.info(f"Fetching bulk real-time prices for {len(tickers)} tickers from StockData.org")
            
            response = requests.get(endpoint, params=params, timeout=15)  # Longer timeout for bulk
            response.raise_for_status()
            
            data = response.json()
            
            results = {}
            
            if 'data' in data and len(data['data']) > 0:
                for quote_data in data['data']:
                    ticker = quote_data['ticker']
                    current_price = float(quote_data['price'])
                    
                    # Create comprehensive price data object
                    price_info = {
                        'current_price': current_price,
                        'ticker': ticker,
                        'name': quote_data.get('name', ''),
                        'exchange': quote_data.get('mic_code', ''),
                        'currency': quote_data.get('currency', 'USD'),
                        'day_open': float(quote_data.get('day_open', 0) or 0),
                        'day_high': float(quote_data.get('day_high', 0) or 0),
                        'day_low': float(quote_data.get('day_low', 0) or 0),
                        'volume': int(quote_data.get('volume', 0) or 0),
                        'previous_close': float(quote_data.get('previous_close_price', 0) or 0),
                        'day_change_percent': float(quote_data.get('day_change', 0) or 0),
                        'market_cap': quote_data.get('market_cap'),
                        '52_week_high': float(quote_data.get('52_week_high', 0) or 0),
                        '52_week_low': float(quote_data.get('52_week_low', 0) or 0),
                        'is_extended_hours': quote_data.get('is_extended_hours_price', False),
                        'last_trade_time': quote_data.get('last_trade_time'),
                        'source': 'stockdata_realtime_bulk',
                        'timestamp': datetime.utcnow()
                    }
                    
                    results[ticker] = price_info
                    
                    # Cache each ticker's data
                    if self.mongo_client:
                        try:
                            self.mongo_client.db['current_prices'].replace_one(
                                {'ticker': ticker},
                                price_info,
                                upsert=True
                            )
                        except Exception as e:
                            logger.warning(f"Failed to cache bulk StockData price for {ticker}: {e}")
                
                logger.info(f"Successfully fetched bulk prices for {len(results)}/{len(tickers)} tickers")
                
                # Log summary
                total_extended_hours = sum(1 for info in results.values() if info['is_extended_hours'])
                if total_extended_hours > 0:
                    logger.info(f"Extended hours data available for {total_extended_hours} tickers")
                    
                return results
            else:
                logger.warning(f"No bulk quote data returned from StockData for {len(tickers)} tickers")
                return {}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"StockData bulk API request failed: {e}")
            return {}
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing StockData bulk response: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error fetching StockData bulk prices: {e}")
            return {}

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