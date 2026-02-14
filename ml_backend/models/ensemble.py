# ensemble.py
import numpy as np
from .predictor import StockPredictor
import xgboost as xgb
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, LayerNormalization, MultiHeadAttention, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GRUPredictor:
    def __init__(self, input_shape=None, hyperparams=None):
        self.model = None
        self.input_shape = input_shape
        self.hyperparams = hyperparams or {
            'gru_units': 48,
            'dense_units': 24,
            'dropout_rate': 0.3,
            'learning_rate': 1e-3,
            'l2_reg': 1e-3,
            'batch_size': 64,
            'epochs': 20
        }

    def build_model(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),
            GRU(self.hyperparams['gru_units'], return_sequences=False, kernel_regularizer=l2(self.hyperparams['l2_reg'])),
            Dropout(self.hyperparams['dropout_rate']),
            Dense(self.hyperparams['dense_units'], activation='relu', kernel_regularizer=l2(self.hyperparams['l2_reg'])),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=self.hyperparams['learning_rate']), loss='mse', metrics=['mae'])
        return model

    def fit(self, X, y):
        if self.model is None:
            self.model = self.build_model(X.shape[1:])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.model.fit(X, y, validation_split=0.2, epochs=self.hyperparams['epochs'], batch_size=self.hyperparams['batch_size'], callbacks=[early_stopping], verbose=0)

    def predict(self, X):
        if self.model is None:
            raise ValueError("GRU model is not trained.")
        return self.model.predict(X).flatten()

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TransformerPredictor:
    def __init__(self, input_shape=None, hyperparams=None):
        self.model = None
        self.input_shape = input_shape
        self.hyperparams = hyperparams or {
            'embed_dim': 24,
            'num_heads': 2,
            'ff_dim': 24,
            'dropout_rate': 0.2,
            'dense_units': 24,
            'learning_rate': 1e-3,
            'batch_size': 64,
            'epochs': 20
        }

    def build_model(self, input_shape):
        inputs = Input(shape=input_shape)
        x = TransformerBlock(self.hyperparams['embed_dim'], self.hyperparams['num_heads'], self.hyperparams['ff_dim'], self.hyperparams['dropout_rate'])(inputs)
        x = Flatten()(x)
        x = Dense(self.hyperparams['dense_units'], activation='relu')(x)
        outputs = Dense(1)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.hyperparams['learning_rate']), loss='mse', metrics=['mae'])
        return model

    def fit(self, X, y):
        if self.model is None:
            self.model = self.build_model(X.shape[1:])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.model.fit(X, y, validation_split=0.2, epochs=self.hyperparams['epochs'], batch_size=self.hyperparams['batch_size'], callbacks=[early_stopping], verbose=0)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Transformer model is not trained.")
        return self.model.predict(X).flatten()

class XGBoostPredictor:
    def __init__(self):
        self.model = xgb.XGBRegressor(objective='reg:squarederror')
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)

class EnsemblePredictor:
    def __init__(self, feature_engineer=None):
        self.lstm = None  # Will be initialized properly
        self.transformer = None
        self.xgb = XGBoostPredictor()
        self.feature_engineer = feature_engineer
        self.trained = False
        # NEW: Dynamic weighting system
        self.model_performance_history = {}
        self.base_weights = {'lstm': 0.35, 'transformer': 0.35, 'xgb': 0.30}
        self.adaptive_weights = self.base_weights.copy()

    def fit(self, X, y, ticker=None, window=None, mongo_client=None):
        """
        FIXED: Properly train all ensemble models with correct feature integration.
        """
        try:
            logger.info("Training ensemble models with proper integration...")
            
            # Initialize LSTM properly if we have the required components
            if self.feature_engineer and mongo_client:
                from .predictor import StockPredictor
                self.lstm = StockPredictor(mongo_client)
                self.lstm.set_feature_engineer(self.feature_engineer)
                
                # Train LSTM with proper data format
                if ticker and window:
                    # Use the proper training method
                    self.lstm.train_models(ticker, X, y, window)
                else:
                    logger.warning("LSTM training skipped: ticker and window required")
            else:
                logger.warning("LSTM training skipped: feature_engineer and mongo_client required")
            
            # Train Transformer
            self.transformer = TransformerPredictor(input_shape=X.shape[1:])
            self.transformer.fit(X, y)
            
            # Train XGBoost
            X_flat = X.reshape((X.shape[0], -1))
            self.xgb.fit(X_flat, y)
            
            self.trained = True
            logger.info("Ensemble training completed successfully")
            
        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            raise

    def predict(self, X, ticker=None, window=None, market_regime=None, raw_current_price=None):
        """
        FIXED: Properly handle predictions from all models with correct data formats.
        """
        if not self.trained:
            raise ValueError("EnsemblePredictor is not trained. Call fit() first.")
            
        model_predictions = {}
        
        # LSTM prediction (FIXED: Use proper predict method)
        if self.lstm and hasattr(self.lstm, 'models') and ticker and window:
            try:
                # Create a dummy DataFrame for LSTM prediction
                if raw_current_price:
                    # Use the proper predict method from StockPredictor
                    lstm_result = self.lstm.predict(
                        df=X,  # X should be a DataFrame or convert it
                        window=window,
                        ticker=ticker,
                        raw_current_price=raw_current_price
                    )
                    if lstm_result and 'prediction' in lstm_result:
                        model_predictions['lstm'] = lstm_result['prediction']
                else:
                    logger.warning("LSTM prediction skipped: raw_current_price required")
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")
        
        # Transformer prediction
        if self.transformer:
            try:
                transformer_pred = self.transformer.predict(X)
                model_predictions['transformer'] = (
                    np.mean(transformer_pred) if hasattr(transformer_pred, '__len__') 
                    else transformer_pred
                )
            except Exception as e:
                logger.warning(f"Transformer prediction failed: {e}")
        
        # XGBoost prediction
        try:
            X_flat = X.reshape((X.shape[0], -1))
            xgb_pred = self.xgb.predict(X_flat)
            model_predictions['xgb'] = (
                np.mean(xgb_pred) if hasattr(xgb_pred, '__len__') 
                else xgb_pred
            )
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}")
        
        if not model_predictions:
            raise ValueError("All models failed to make predictions")
        
        # Calculate dynamic weights
        if ticker and window:
            weights = self._calculate_dynamic_weights(model_predictions, ticker, window, market_regime)
        else:
            weights = {model: self.base_weights.get(model, 1.0/len(model_predictions)) 
                      for model in model_predictions.keys()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        else:
            weights = {k: 1.0/len(model_predictions) for k in model_predictions.keys()}
        
        # Calculate weighted ensemble prediction
        ensemble_pred = sum(pred * weights[model] for model, pred in model_predictions.items())
        
        logger.info(f"Ensemble prediction: {ensemble_pred:.4f} with weights: {weights}")
        
        return {
            'prediction': ensemble_pred,
            'model_predictions': model_predictions,
            'weights': weights,
            'confidence': self._calculate_ensemble_confidence(model_predictions, weights)
        }

    def _calculate_dynamic_weights(self, model_predictions: dict, ticker: str, window: str, market_regime: str = None) -> dict:
        """Calculate dynamic weights based on recent performance and market conditions."""
        try:
            # Start with base weights
            weights = self.base_weights.copy()
            
            # Factor 1: Recent Performance (last 10 predictions)
            performance_weights = self._get_performance_based_weights(ticker, window)
            
            # Factor 2: Market Regime Adaptation
            regime_weights = self._get_regime_based_weights(market_regime)
            
            # Factor 3: Prediction Confidence (model agreement)
            confidence_weights = self._get_confidence_based_weights(model_predictions)
            
            # Combine all factors with weights
            final_weights = {}
            for model in model_predictions.keys():
                base_w = weights.get(model, 0.33)
                perf_w = performance_weights.get(model, 0.33)
                regime_w = regime_weights.get(model, 0.33)
                conf_w = confidence_weights.get(model, 0.33)
                
                # Weighted combination
                final_weights[model] = (
                    base_w * 0.25 +         # Base weight
                    perf_w * 0.40 +         # Performance (most important)
                    regime_w * 0.25 +       # Market regime
                    conf_w * 0.10           # Confidence
                )
            
            # Ensure all weights are positive and sum to 1
            min_weight = 0.05  # Minimum weight for any model
            for model in final_weights:
                final_weights[model] = max(min_weight, final_weights[model])
            
            return final_weights
            
        except Exception as e:
            logger.warning(f"Error calculating dynamic weights: {e}")
            return {model: 1.0/len(model_predictions) for model in model_predictions.keys()}
    
    def _get_performance_based_weights(self, ticker: str, window: str) -> dict:
        """
        FIXED: Get weights based on recent prediction performance using MongoDB.
        """
        try:
            # Default weights as fallback
            default_weights = {'lstm': 0.33, 'transformer': 0.33, 'xgb': 0.34}
            
            # Check if we have a feature engineer with MongoDB access
            if not self.feature_engineer or not hasattr(self.feature_engineer, 'mongo_client'):
                return default_weights
                
            mongo_client = self.feature_engineer.mongo_client
            if not mongo_client:
                return default_weights
            
            # Query recent prediction performance from MongoDB
            collection = mongo_client.db['model_performance']
            
            # Get performance data for the last 30 days
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            
            performance_data = list(collection.find({
                'ticker': ticker,
                'window': window,
                'timestamp': {'$gte': thirty_days_ago}
            }).sort('timestamp', -1).limit(50))
            
            if not performance_data:
                logger.info(f"No performance history found for {ticker}-{window}, using default weights")
                return default_weights
            
            # Calculate performance metrics for each model
            model_scores = {'lstm': [], 'transformer': [], 'xgb': []}
            
            for record in performance_data:
                if 'model_accuracies' in record:
                    accuracies = record['model_accuracies']
                    for model in model_scores.keys():
                        if model in accuracies and accuracies[model] is not None:
                            model_scores[model].append(accuracies[model])
            
            # Calculate average performance and convert to weights
            performance_weights = {}
            total_performance = 0
            
            for model, scores in model_scores.items():
                if scores:
                    avg_score = np.mean(scores)
                    # Convert accuracy to weight (higher accuracy = higher weight)
                    performance_weights[model] = max(0.1, avg_score)  # Minimum weight of 0.1
                    total_performance += performance_weights[model]
                else:
                    performance_weights[model] = 0.33  # Default if no history
                    total_performance += 0.33
            
            # Normalize weights
            if total_performance > 0:
                performance_weights = {
                    model: weight / total_performance 
                    for model, weight in performance_weights.items()
                }
            else:
                performance_weights = default_weights
            
            logger.info(f"Performance-based weights for {ticker}-{window}: {performance_weights}")
            return performance_weights
            
        except Exception as e:
            logger.warning(f"Error calculating performance-based weights: {e}")
            return {'lstm': 0.33, 'transformer': 0.33, 'xgb': 0.34}
    
    def _get_regime_based_weights(self, market_regime: str = None) -> dict:
        """Adjust weights based on market regime."""
        try:
            if market_regime == 'high_volatility':
                # In high volatility, favor more stable models
                return {'lstm': 0.25, 'transformer': 0.25, 'xgb': 0.50}
            elif market_regime == 'trending':
                # In trending markets, favor sequence models
                return {'lstm': 0.45, 'transformer': 0.45, 'xgb': 0.10}
            elif market_regime == 'sideways':
                # In sideways markets, favor traditional ML
                return {'lstm': 0.20, 'transformer': 0.30, 'xgb': 0.50}
            else:
                # Default balanced weights
                return {'lstm': 0.35, 'transformer': 0.35, 'xgb': 0.30}
        except:
            return {'lstm': 0.35, 'transformer': 0.35, 'xgb': 0.30}
    
    def _get_confidence_based_weights(self, model_predictions: dict) -> dict:
        """Adjust weights based on prediction confidence (model agreement)."""
        try:
            if len(model_predictions) < 2:
                return {model: 1.0 for model in model_predictions.keys()}
            
            pred_values = list(model_predictions.values())
            pred_std = np.std(pred_values)
            pred_mean = np.mean(pred_values)
            
            # If models agree (low std), use equal weights
            # If models disagree (high std), favor the median prediction
            if pred_std / abs(pred_mean) < 0.1:  # Low disagreement
                return {model: 1.0/len(model_predictions) for model in model_predictions.keys()}
            else:
                # High disagreement - favor models closer to median
                median_pred = np.median(pred_values)
                weights = {}
                for model, pred in model_predictions.items():
                    # Closer to median = higher weight
                    distance = abs(pred - median_pred)
                    weights[model] = 1.0 / (1.0 + distance)
                
                return weights
        except:
            return {model: 1.0/len(model_predictions) for model in model_predictions.keys()}

    def _calculate_ensemble_confidence(self, model_predictions: dict, weights: dict) -> float:
        """
        Calculate ensemble confidence based on model agreement and individual model confidence.
        """
        try:
            if len(model_predictions) < 2:
                return 0.5  # Low confidence with only one model
            
            pred_values = list(model_predictions.values())
            
            # Factor 1: Model agreement (lower std = higher confidence)
            pred_std = np.std(pred_values)
            pred_mean = np.mean(pred_values)
            
            if abs(pred_mean) > 1e-6:
                agreement_factor = max(0.0, 1.0 - (pred_std / abs(pred_mean)))
            else:
                agreement_factor = 0.5
            
            # Factor 2: Weight distribution (more balanced = higher confidence)
            weight_values = list(weights.values())
            weight_entropy = -sum(w * np.log(w + 1e-10) for w in weight_values)
            max_entropy = np.log(len(weight_values))
            balance_factor = weight_entropy / max_entropy if max_entropy > 0 else 0.5
            
            # Factor 3: Number of successful models
            model_count_factor = min(len(model_predictions) / 3.0, 1.0)  # Max confidence with 3+ models
            
            # Combine factors
            confidence = (
                agreement_factor * 0.5 +    # Model agreement is most important
                balance_factor * 0.3 +      # Weight balance
                model_count_factor * 0.2    # Number of models
            )
            
            return max(0.1, min(0.95, confidence))  # Clamp between 0.1 and 0.95
            
        except Exception as e:
            logger.warning(f"Error calculating ensemble confidence: {e}")
            return 0.5

    def _to_dataframe(self, X, y):
        # Helper to convert X (3D) and y to DataFrame for LSTM
        # Assumes feature_engineer has feature columns
        if self.feature_engineer and hasattr(self.feature_engineer, 'feature_columns'):
            columns = self.feature_engineer.feature_columns
        else:
            columns = [f'feature_{i}' for i in range(X.shape[2])]
        # Flatten X to 2D for DataFrame
        X_flat = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
        df = np.concatenate([X_flat, y.reshape(-1, 1)], axis=1)
        col_names = [f'{c}_{i}' for c in columns for i in range(X.shape[1])] + ['Close']
        return pd.DataFrame(df, columns=col_names)

    # NOTE: Legacy train_ensemble() and _calculate_and_store_feature_importance()
    # were removed — they referenced non-existent rf_model/gb_model/nn_model
    # attributes (never initialized in __init__) and would crash with AttributeError.
    # Use the working fit() method instead for training.

    def store_model_performance(self, ticker: str, window: str, actual_price: float, 
                              predicted_price: float, model_predictions: dict, mongo_client=None):
        """
        Store model performance for future weight calculations.
        """
        try:
            if not mongo_client:
                if self.feature_engineer and hasattr(self.feature_engineer, 'mongo_client'):
                    mongo_client = self.feature_engineer.mongo_client
                else:
                    return False
            
            # Calculate accuracy for each model
            model_accuracies = {}
            for model, pred in model_predictions.items():
                if pred is not None:
                    # Calculate accuracy as 1 - |relative_error|
                    if actual_price != 0:
                        relative_error = abs(pred - actual_price) / abs(actual_price)
                        accuracy = max(0.0, 1.0 - relative_error)
                    else:
                        accuracy = 0.5  # Neutral score if actual price is 0
                    model_accuracies[model] = accuracy
                else:
                    model_accuracies[model] = None
            
            # Store in MongoDB
            collection = mongo_client.db['model_performance']
            
            performance_doc = {
                'ticker': ticker,
                'window': window,
                'actual_price': actual_price,
                'predicted_price': predicted_price,
                'model_predictions': model_predictions,
                'model_accuracies': model_accuracies,
                'timestamp': datetime.utcnow(),
                'ensemble_accuracy': model_accuracies.get('ensemble', 0.0)
            }
            
            collection.insert_one(performance_doc)
            logger.info(f"Stored performance data for {ticker}-{window}: accuracies {model_accuracies}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing model performance: {e}")
            return False 