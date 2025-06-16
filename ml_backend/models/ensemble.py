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
            'gru_units': 64,
            'dense_units': 32,
            'dropout_rate': 0.2,
            'learning_rate': 1e-3,
            'l2_reg': 1e-4,
            'batch_size': 64,
            'epochs': 30
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
            'embed_dim': 32,
            'num_heads': 2,
            'ff_dim': 32,
            'dropout_rate': 0.1,
            'dense_units': 32,
            'learning_rate': 1e-3,
            'batch_size': 64,
            'epochs': 30
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

    def predict(self, X):
        if not self.trained:
            raise ValueError("EnsemblePredictor is not trained. Call fit() first.")
        preds = []
        # Extract the true, unnormalized current price from the last time step in the sequence (assume 'Close' is the last feature in the last time step)
        # If X is 3D: (samples, sequence_length, num_features_per_step)
        # Use the last sample's last time step's 'Close' value
        # If feature_engineer has feature_columns, use its index
        if self.feature_engineer and hasattr(self.feature_engineer, 'feature_columns') and 'Close' in self.feature_engineer.feature_columns:
            close_idx = self.feature_engineer.feature_columns.index('Close')
        else:
            close_idx = -1  # fallback to last column
        raw_current_price = float(X[-1, -1, close_idx])
        # LSTM
        lstm_pred = self.lstm.predict_all_windows(X, raw_current_price=raw_current_price)
        # Use the first window's prediction (or average if multiple)
        if lstm_pred:
            lstm_vals = [v['prediction'] for v in lstm_pred.values() if v and 'prediction' in v]
            if lstm_vals:
                preds.append(np.array(lstm_vals))
        # Transformer
        preds.append(self.transformer.predict(X))
        # XGBoost
        X_flat = X.reshape((X.shape[0], -1))
        preds.append(self.xgb.predict(X_flat))
        # Average/blend predictions
        # Ensure all preds are the same shape
        min_len = min(len(p) for p in preds)
        preds = [p[:min_len] for p in preds]
        ensemble_pred = np.mean(preds, axis=0)
        return ensemble_pred

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