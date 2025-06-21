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
        self.lstm = StockPredictor(feature_engineer=feature_engineer)
        self.transformer = None
        self.xgb = XGBoostPredictor()
        self.feature_engineer = feature_engineer
        self.trained = False
        # NEW: Dynamic weighting system
        self.model_performance_history = {}
        self.base_weights = {'lstm': 0.35, 'transformer': 0.35, 'xgb': 0.30}
        self.adaptive_weights = self.base_weights.copy()

    def fit(self, X, y):
        # LSTM expects 3D, XGBoost expects 2D
        # Train LSTM
        self.lstm.train_all_models({'main': self._to_dataframe(X, y)})
        # Train Transformer
        self.transformer = TransformerPredictor(input_shape=X.shape[1:])
        self.transformer.fit(X, y)
        # Train XGBoost
        X_flat = X.reshape((X.shape[0], -1))
        self.xgb.fit(X_flat, y)
        self.trained = True

    def predict(self, X, ticker=None, window=None, market_regime=None):
        if not self.trained:
            raise ValueError("EnsemblePredictor is not trained. Call fit() first.")
        preds = []
        model_names = []
        
        # Extract the true, unnormalized current price from the last time step in the sequence
        if self.feature_engineer and hasattr(self.feature_engineer, 'feature_columns') and 'Close' in self.feature_engineer.feature_columns:
            close_idx = self.feature_engineer.feature_columns.index('Close')
        else:
            close_idx = -1  # fallback to last column
        raw_current_price = float(X[-1, -1, close_idx])
        
        # Get predictions from each model
        model_predictions = {}
        
        # LSTM
        try:
            lstm_pred = self.lstm.predict_all_windows(X, raw_current_price=raw_current_price)
            if lstm_pred:
                lstm_vals = [v['prediction'] for v in lstm_pred.values() if v and 'prediction' in v]
                if lstm_vals:
                    model_predictions['lstm'] = np.mean(lstm_vals)
                    model_names.append('lstm')
        except Exception as e:
            logger.warning(f"LSTM prediction failed: {e}")
        
        # Transformer
        try:
            transformer_pred = self.transformer.predict(X)
            model_predictions['transformer'] = np.mean(transformer_pred) if hasattr(transformer_pred, '__len__') else transformer_pred
            model_names.append('transformer')
        except Exception as e:
            logger.warning(f"Transformer prediction failed: {e}")
        
        # XGBoost
        try:
            X_flat = X.reshape((X.shape[0], -1))
            xgb_pred = self.xgb.predict(X_flat)
            model_predictions['xgb'] = np.mean(xgb_pred) if hasattr(xgb_pred, '__len__') else xgb_pred
            model_names.append('xgb')
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}")
        
        if not model_predictions:
            raise ValueError("All models failed to make predictions")
        
        # NEW: Dynamic ensemble weighting based on performance and market conditions
        if ticker and window:
            weights = self._calculate_dynamic_weights(model_predictions, ticker, window, market_regime)
        else:
            # Fallback to base weights
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
        
        return ensemble_pred

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
        """Get weights based on recent prediction performance."""
        try:
            # This would query MongoDB for recent prediction accuracy
            # For now, return equal weights as placeholder
            return {'lstm': 0.33, 'transformer': 0.33, 'xgb': 0.34}
        except:
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

    def train_ensemble(self, X_train, y_train, feature_names=None, ticker=None, mongo_client=None):
        """Train all models in the ensemble and store feature importance."""
        try:
            logger.info("Training ensemble models...")
            
            # Convert to numpy array if needed
            if hasattr(X_train, 'values'):
                X_train = X_train.values
            if hasattr(y_train, 'values'):
                y_train = y_train.values
            
            # Store feature names
            if feature_names is not None:
                self.feature_names = feature_names
            
            # Train individual models
            self.rf_model.fit(X_train, y_train)
            self.gb_model.fit(X_train, y_train)
            self.xgb_model.fit(X_train, y_train)
            
            # Train neural network
            if self.nn_model:
                self.nn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            
            logger.info("Ensemble training completed successfully")
            
            # Calculate and store feature importance
            if ticker and mongo_client and feature_names is not None:
                self._calculate_and_store_feature_importance(ticker, mongo_client)
                
            return True
            
        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            return False
    
    def _calculate_and_store_feature_importance(self, ticker: str, mongo_client):
        """Calculate combined feature importance from all models and store in MongoDB."""
        try:
            feature_importance = {}
            
            # Get Random Forest importance
            if hasattr(self.rf_model, 'feature_importances_'):
                rf_importance = self.rf_model.feature_importances_
                for i, importance in enumerate(rf_importance):
                    if i < len(self.feature_names):
                        feature_name = self.feature_names[i]
                        feature_importance[feature_name] = feature_importance.get(feature_name, 0) + importance * 0.3
            
            # Get Gradient Boosting importance
            if hasattr(self.gb_model, 'feature_importances_'):
                gb_importance = self.gb_model.feature_importances_
                for i, importance in enumerate(gb_importance):
                    if i < len(self.feature_names):
                        feature_name = self.feature_names[i]
                        feature_importance[feature_name] = feature_importance.get(feature_name, 0) + importance * 0.3
            
            # Get XGBoost importance
            if hasattr(self.xgb_model, 'feature_importances_'):
                xgb_importance = self.xgb_model.feature_importances_
                for i, importance in enumerate(xgb_importance):
                    if i < len(self.feature_names):
                        feature_name = self.feature_names[i]
                        feature_importance[feature_name] = feature_importance.get(feature_name, 0) + importance * 0.4
            
            # Store in MongoDB using FeatureEngineer's method
            if feature_importance:
                from ml_backend.data.features import FeatureEngineer
                feature_engineer = FeatureEngineer()
                feature_engineer.store_feature_importance(
                    ticker=ticker,
                    feature_scores=feature_importance,
                    window="ensemble_training",
                    mongo_client=mongo_client
                )
                
                # Log top 5 features
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                logger.info(f"Top 5 features for {ticker}: {[f'{name}: {score:.4f}' for name, score in top_features]}")
                
        except Exception as e:
            logger.error(f"Error calculating feature importance for {ticker}: {e}") 