"""
Machine learning model module for stock price prediction.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import os
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
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge

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
    def __init__(self, feature_engineer=None):
        self.model_config = MODEL_CONFIG
        self.prediction_windows = PREDICTION_WINDOWS
        self.models = {}
        self.xgb_models = {}
        self.rf_models = {}
        self.lgbm_models = {}
        self.feature_engineer = feature_engineer
        self.hyperparameters = {}
        self.metrics = {}
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        self.target_stats = {}  # Store mean/std for each window
        self.feature_columns = []  # Store feature columns for later use
        self.recent_residuals = {w: [] for w in self.prediction_windows}  # For price range
        self.feature_importances_ = {}

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

    def build_rf_model(self, **params):
        return RandomForestRegressor(**params)

    def build_lgbm_model(self, **params):
        return lgb.LGBMRegressor(**params)

    def objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Objective function for hyperparameter optimization."""
        hyperparams = {
            'lstm_units': trial.suggest_int('lstm_units', 32, 64),
            'dense_units': trial.suggest_int('dense_units', 16, 32),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.3),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'l2_reg': trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128])
        }
        # Use stored feature columns to find 'Close' index
        close_idx = self.feature_columns.index('Close')
        # Get mean and std from the current window's stats if available, else use y_train
        y_mean = y_train.mean() if hasattr(y_train, 'mean') else 0
        y_std = y_train.std() if hasattr(y_train, 'std') and y_train.std() != 0 else 1
        # Normalize targets and current price
        y_train_norm = (y_train - y_mean) / y_std
        y_val_norm = (y_val - y_mean) / y_std
        current_price_train = (X_train[:, -1, close_idx] - y_mean) / y_std
        current_price_val = (X_val[:, -1, close_idx] - y_mean) / y_std
        y_train_combined = np.column_stack([y_train_norm, current_price_train])
        y_val_combined = np.column_stack([y_val_norm, current_price_val])
        model = self.build_model((X_train.shape[1], X_train.shape[2]), hyperparams)
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.model_config['early_stopping_patience'],
            restore_best_weights=True
        )
        model.fit(
            X_train, y_train_combined,
            validation_data=(X_val, y_val_combined),
            epochs=self.model_config['epochs'],
            batch_size=hyperparams['batch_size'],
            callbacks=[early_stopping],
            verbose=0
        )
        val_loss = model.evaluate(X_val, y_val_combined, verbose=0)[0]
        return val_loss

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

    def tune_rf(self, X, y):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'random_state': 42
            }
            model = self.build_rf_model(**params)
            model.fit(X, y)
            preds = model.predict(X)
            return mean_squared_error(y, preds)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)
        return study.best_params

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
        """Train the model for a specific prediction window."""
        try:
            # Save feature columns after feature engineering
            if hasattr(self, 'feature_columns') and self.feature_columns:
                with open('models/feature_columns.json', 'w') as f:
                    json.dump(self.feature_columns, f)
            # Print the exact features used for training
            print(f"Features used for training {window}: {self.feature_columns}")
            # --- Target normalization ---
            y_mean = y_train.mean()
            y_std = y_train.std() if y_train.std() != 0 else 1
            self.target_stats[window] = {"mean": float(y_mean), "std": float(y_std)}
            y_train_norm = (y_train - y_mean) / y_std
            y_val_norm = (y_val - y_mean) / y_std
            y_test_norm = (y_test - y_mean) / y_std
            # Use stored feature columns to find 'Close' index
            close_idx = self.feature_columns.index('Close')
            # Extract and normalize current price
            current_price_train = (X_train[:, -1, close_idx] - y_mean) / y_std
            current_price_val = (X_val[:, -1, close_idx] - y_mean) / y_std
            current_price_test = (X_test[:, -1, close_idx] - y_mean) / y_std
            y_train_combined = np.column_stack([y_train_norm, current_price_train])
            y_val_combined = np.column_stack([y_val_norm, current_price_val])
            y_test_combined = np.column_stack([y_test_norm, current_price_test])
            logger.info(f"Target normalization for {window}: mean={y_mean}, std={y_std}")
            logger.info(f"Tuning hyperparameters for {window} window...")
            # Tune hyperparameters
            hyperparams = self.tune_hyperparameters(X_train, y_train_norm)
            self.hyperparameters[window] = hyperparams
            logger.info(f"Building model for {window} window...")
            # Build and train model
            model = self.build_model((X_train.shape[1], X_train.shape[2]), hyperparams)
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.model_config['early_stopping_patience'],
                restore_best_weights=True
            )
            model_checkpoint = ModelCheckpoint(
                os.path.join(self.model_dir, f'model_{window}.h5'),
                monitor='val_loss',
                save_best_only=True
            )
            logger.info(f"Starting model training for {window} window...")
            history = model.fit(
                X_train, y_train_combined,
                validation_data=(X_val, y_val_combined),
                epochs=self.model_config['epochs'],
                batch_size=hyperparams['batch_size'],
                callbacks=[early_stopping, model_checkpoint, LoggingCallback()],
                verbose=1
            )
            logger.info(f"Finished model training for {window} window.")
            # Evaluate on test set
            y_pred_norm = model.predict(X_test)
            y_pred = y_pred_norm * y_std + y_mean
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            self.metrics[window] = metrics
            logger.info(f"Test set metrics for {window}: {metrics}")
            # Save model
            self.models[window] = model
            return model
        except Exception as e:
            logger.error(f"Error training model for {window}: {str(e)}")
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

    def train_all_models(self, historical_data: Dict[str, pd.DataFrame], model_type: str = 'lstm'):
        feature_columns_saved = False
        for ticker, df in historical_data.items():
            if df is None or df.empty:
                logger.warning(f"No historical data for {ticker}, skipping.")
                continue
            logger.info(f"Training models for ticker: {ticker}")
            ticker_dir = os.path.join(self.model_dir, ticker)
            os.makedirs(ticker_dir, exist_ok=True)
            for window in self.prediction_windows:
                logger.info(f"Training model for {ticker} - {window} window")
                X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(df, window)
                if X_train is not None and X_train.size > 0:
                    # Quantile regression with LightGBM
                    X_flat = X_train.reshape(X_train.shape[0], -1)
                    quantiles = [0.1, 0.5, 0.9]
                    lgbm_quantile_models = {}
                    for q in quantiles:
                        lgbm = lgb.LGBMRegressor(objective='quantile', alpha=q, n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)
                        lgbm.fit(X_flat, y_train)
                        lgbm_quantile_models[q] = lgbm
                    self.lgbm_models[(ticker, window, 'quantile')] = lgbm_quantile_models
                    logger.info(f"Trained LightGBM quantile models for {ticker} - {window} window.")
                    # Classic regression models for point prediction
                    # LightGBM (classic)
                    lgbm_model = lgb.LGBMRegressor(objective='regression', n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)
                    lgbm_model.fit(X_flat, y_train)
                    self.lgbm_models[(ticker, window, 'classic')] = lgbm_model
                    # XGBoost
                    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)
                    xgb_model.fit(X_flat, y_train)
                    self.xgb_models[(ticker, window)] = xgb_model
                    # Random Forest
                    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
                    rf_model.fit(X_flat, y_train)
                    self.rf_models[(ticker, window)] = rf_model
                    # LSTM (optional, only if model_type == 'lstm')
                    if model_type == 'lstm':
                        model = self.train_model(X_train, y_train, X_val, y_val, X_test, y_test, window)
                        if model is not None:
                            self.models[(ticker, window)] = model
                            model_path = os.path.join(ticker_dir, f"model_{ticker}_{window}_lstm.h5")
                            model.save(model_path)
                    # Save feature columns for later use
                    if not feature_columns_saved and hasattr(self, 'feature_columns') and self.feature_columns:
                        with open(os.path.join(self.model_dir, 'feature_columns.json'), 'w') as f:
                            json.dump(self.feature_columns, f)
                        feature_columns_saved = True
                else:
                    logger.warning(f"No training data for {ticker} - {window} window. Skipping.")

    def load_models(self):
        """Load trained models for all tickers and windows from disk, using only the new naming convention."""
        for ticker in os.listdir(self.model_dir):
            ticker_dir = os.path.join(self.model_dir, ticker)
            if not os.path.isdir(ticker_dir):
                continue
            for filename in os.listdir(ticker_dir):
                if filename.startswith(f"model_{ticker}_") and filename.endswith(".h5"):
                    parts = filename.replace(f"model_{ticker}_", "").replace(".h5", "").split("_")
                    if len(parts) >= 2:
                        window = "_".join(parts[:-1])
                        model_type = parts[-1]
                        model_path = os.path.join(ticker_dir, filename)
                        self.models[(ticker, window)] = load_model(model_path, custom_objects={'custom_loss': self.custom_loss})
                        logger.info(f"Loaded model for {ticker} - {window} ({model_type}) from {model_path}")

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
                with open('models/feature_columns.json', 'r') as f:
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
            rf_model = self.rf_models.get((ticker, window))
            rf_pred = rf_model.predict(X_flat)[0] if rf_model else None

            # LSTM (if available)
            lstm_pred = None
            if (ticker, window) in self.models:
                close_idx = self.feature_columns.index('Close')
                pred_norm = self.models[(ticker, window)].predict(features)[0][0]
                stats = self.target_stats.get(window, {"mean": 0, "std": 1})
                predicted_delta = float(pred_norm * stats["std"] + stats["mean"])
                lstm_pred = raw_current_price + predicted_delta

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
                "rf": raw_current_price + rf_pred if rf_pred is not None else None,
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

    def predict_all_windows(self, features: np.ndarray, ticker: str, raw_current_price: float = None, model_type: str = 'lstm') -> Dict[str, Dict[str, float]]:
        logger.info(f"Predicting all windows for ticker={ticker} with features: {features}")
        predictions = {}
        for window in self.prediction_windows:
            if (ticker, window) not in self.models:
                logger.warning(f"No model trained for {ticker} - {window} window. Skipping prediction.")
                continue
            # LSTM/RNN
            lstm_result = self.predict(features, window, ticker, raw_current_price=raw_current_price)
            # Ensure confidence is present
            confidence = lstm_result.get('confidence', 0.5) if lstm_result else 0.5
            # Stacking Ensemble
            stacking_model = self.models.get((ticker, window, 'stacking'))
            stacking_pred = None
            if stacking_model is not None:
                stacking_pred_delta = stacking_model.predict(features.reshape(1, -1))[0]
                stacking_pred = raw_current_price + stacking_pred_delta if raw_current_price is not None else stacking_pred_delta
            # XGBoost
            xgb_model = self.xgb_models.get((ticker, window))
            xgb_pred_delta = xgb_model.predict(features.reshape(1, -1))[0] if xgb_model else 0
            xgb_pred_price = raw_current_price + xgb_pred_delta if raw_current_price is not None else xgb_pred_delta
            # RandomForest
            rf_model = self.rf_models.get((ticker, window))
            rf_pred_delta = rf_model.predict(features.reshape(1, -1))[0] if rf_model else 0
            rf_pred_price = raw_current_price + rf_pred_delta if raw_current_price is not None else rf_pred_delta
            # LightGBM
            lgbm_model = self.lgbm_models.get((ticker, window))
            lgbm_pred_delta = lgbm_model.predict(features.reshape(1, -1))[0] if lgbm_model else 0
            lgbm_pred_price = raw_current_price + lgbm_pred_delta if raw_current_price is not None else lgbm_pred_delta
            # Ensemble: average all
            preds = [lstm_result['prediction'], xgb_pred_price, rf_pred_price, lgbm_pred_price]
            ensemble_pred = np.mean(preds)
            # Super-ensemble: average LSTM and stacking ensemble
            if stacking_pred is not None:
                super_ensemble_pred = np.mean([lstm_result['prediction'], stacking_pred])
                logger.info(f"Super-ensemble: LSTM={lstm_result['prediction']}, Stacking={stacking_pred}, Blended={super_ensemble_pred}")
            else:
                super_ensemble_pred = lstm_result['prediction']
            # --- Price range logic for ensemble ---
            residuals = self.recent_residuals.get(window, [])
            if len(residuals) >= 20:
                rolling_std = np.std(residuals[-100:])
            else:
                stats = self.target_stats.get(window, {"std": 1})
                rolling_std = stats["std"]
            k = 1.28
            lower = super_ensemble_pred - k * rolling_std
            upper = super_ensemble_pred + k * rolling_std
            logger.info(f"Super-ensemble: {super_ensemble_pred}, range=({lower}, {upper})")
            predictions[window] = {
                'prediction': super_ensemble_pred,
                'confidence': confidence,
                'range': [lower, upper],
                'model_preds': {
                    'lstm': lstm_result['prediction'] if lstm_result else None,
                    'stacking': stacking_pred,
                    'xgb': xgb_pred_price,
                    'rf': rf_pred_price,
                    'lgbm': lgbm_pred_price
                }
            }
        return predictions

    def save_models(self):
        """Save trained models to disk."""
        os.makedirs("models", exist_ok=True)
        for window, model in self.models.items():
            model_path = os.path.join(self.model_dir, f'model_{window}.h5')
            model.save(model_path)
            logger.info(f"Saved model for {window} to {model_path}")

    def walk_forward_validation(self, df: pd.DataFrame, window: str, n_splits: int = 5) -> Dict[str, float]:
        """Perform walk-forward validation for realistic backtesting."""
        try:
            features, targets = self.feature_engineer.prepare_features(df)
            n_samples = len(features)
            split_size = n_samples // (n_splits + 1)
            results = []
            for i in range(n_splits):
                train_end = split_size * (i + 1)
                test_start = train_end
                test_end = test_start + split_size
                if test_end > n_samples:
                    break
                X_train, y_train = features[:train_end], targets[:train_end]
                X_test, y_test = features[test_start:test_end], targets[test_start:test_end]
                model = self.train_model(X_train, y_train, X_train[-split_size:], y_train[-split_size:], X_test, y_test, window)
                y_pred = model.predict(X_test).flatten()
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                logger.info(f"Walk-forward split {i+1}/{n_splits}: MSE={mse}, MAE={mae}, R2={r2}")
                results.append({'mse': mse, 'mae': mae, 'r2': r2})
            # Aggregate results
            avg_mse = np.mean([r['mse'] for r in results])
            avg_mae = np.mean([r['mae'] for r in results])
            avg_r2 = np.mean([r['r2'] for r in results])
            logger.info(f"Walk-forward validation (avg over {n_splits} splits): MSE={avg_mse}, MAE={avg_mae}, R2={avg_r2}")
            return {'mse': avg_mse, 'mae': avg_mae, 'r2': avg_r2}
        except Exception as e:
            logger.error(f"Error in walk-forward validation: {str(e)}")
            return {}

    def explain_prediction(self, X: np.ndarray, window: str, model_type: str = 'lstm') -> dict:
        """Return SHAP values for a given input and model type (lstm or xgboost)."""
        try:
            if model_type == 'lstm':
                # Use DeepExplainer for LSTM
                explainer = shap.DeepExplainer(self.models[window], X)
                shap_values = explainer.shap_values(X)
                logger.info(f"Computed SHAP values for LSTM model, window={window}")
                return {'shap_values': shap_values}
            elif model_type == 'xgboost':
                # Use TreeExplainer for XGBoost
                from .ensemble import XGBoostPredictor
                xgb_model = XGBoostPredictor().model
                explainer = shap.TreeExplainer(xgb_model)
                shap_values = explainer.shap_values(X)
                logger.info(f"Computed SHAP values for XGBoost model, window={window}")
                return {'shap_values': shap_values}
            else:
                logger.warning(f"Unknown model_type for SHAP: {model_type}")
                return {}
        except Exception as e:
            logger.error(f"Error computing SHAP values: {str(e)}")
            return {}

    def train_stacking_ensemble(self, X_train, y_train):
        """Train stacking ensemble with XGBoost, LightGBM, Ridge, and MLP as base models, Ridge as meta-learner."""
        base_models = [
            ("xgb", xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42)),
            ("lgb", lgb.LGBMRegressor(n_estimators=100, max_depth=3, random_state=42)),
            ("ridge", Ridge()),
            ("mlp", MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=200, random_state=42))
        ]
        meta_learner = Ridge()
        stack = StackingRegressor(estimators=base_models, final_estimator=meta_learner, passthrough=True, n_jobs=-1)
        stack.fit(X_train, y_train)
        logger.info("Trained stacking ensemble with base models: XGBoost, LightGBM, Ridge, MLP; meta-learner: Ridge.")
        return stack

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
    feature_engineer = FeatureEngineer()
    predictor = StockPredictor(feature_engineer=feature_engineer)
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