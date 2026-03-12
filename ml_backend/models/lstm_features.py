"""
LSTM Temporal Feature Extractor — generates sequence-aware features for LightGBM.

Architecture: Lightweight LSTM that processes rolling windows of price/volume data
and outputs learned temporal embeddings. These embeddings capture sequential patterns
(momentum persistence, regime transitions, volume dynamics) that LightGBM cannot
learn from independent rows.

The LSTM is trained as a regression model to predict future alpha, then the hidden
state from the trained LSTM is extracted as features for LightGBM. This is a
"representation learning" approach — the LSTM learns useful temporal representations,
and LightGBM uses them alongside tabular features.

Features produced (per horizon):
  - lstm_pred_{horizon}    : LSTM's own alpha prediction (weak but directional)
  - lstm_hidden_{horizon}_0..3 : 4 hidden state components (learned temporal patterns)
  - lstm_trend_{horizon}   : LSTM prediction trend (current vs 5-day-ago prediction)

Total: 6 features per horizon × 3 horizons = 18 LSTM features added to LightGBM.

Training: Uses the same walk-forward split as LightGBM to prevent leakage.
Inference: Runs on CPU (no GPU needed), ~2 seconds per ticker.

PIT Safety: Uses shift(1) on all inputs — the LSTM sees data up to day t-1 only.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# LSTM config
LSTM_CONFIG = {
    "enabled": False,          # v9.0: disabled until full integration into LightGBM training loop
    "seq_len": 20,           # 20 trading days lookback (1 month)
    "hidden_size": 16,       # Small hidden state — we only need a few temporal features
    "num_layers": 1,         # Single layer — fast, less overfitting
    "dropout": 0.0,          # No dropout for single layer
    "epochs": 30,            # Quick training — we're extracting features, not maximizing accuracy
    "batch_size": 256,       # Large batches for speed
    "learning_rate": 0.001,
    "n_output_features": 4,  # Number of hidden state components to extract
    "input_features": [      # Raw price/volume features fed to LSTM (PIT-safe, pre-shift)
        "log_return_1d",
        "log_return_5d",
        "volatility_20d",
        "volume_ratio",
        "rsi",
        "momentum_5d",
        "price_vs_sma20",
        "atr_norm",
    ],
}


def _try_import_torch():
    """Lazy import torch — returns None if not available."""
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        logger.warning("PyTorch not available — LSTM features disabled")
        return None, None


class LSTMFeatureExtractor:
    """Trains a lightweight LSTM and extracts temporal features for LightGBM."""

    def __init__(self):
        self.models = {}       # {horizon: trained LSTM model}
        self.scalers = {}      # {horizon: (mean, std) for input normalization}
        self._torch = None
        self._nn = None

    def is_available(self) -> bool:
        """Check if PyTorch is available."""
        if self._torch is None:
            self._torch, self._nn = _try_import_torch()
        return self._torch is not None

    def _build_model(self, n_input_features: int, hidden_size: int, num_layers: int):
        """Build a small LSTM model."""
        nn = self._nn
        torch = self._torch

        class SmallLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.0,
                )
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                # x: (batch, seq_len, features)
                lstm_out, (h_n, _) = self.lstm(x)
                # Use last hidden state
                last_hidden = h_n[-1]  # (batch, hidden_size)
                pred = self.fc(last_hidden)  # (batch, 1)
                return pred.squeeze(-1), last_hidden

        return SmallLSTM(n_input_features, hidden_size, num_layers)

    def _prepare_sequences(
        self, feature_matrix: np.ndarray, targets: np.ndarray, seq_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create rolling sequences from feature matrix.

        Returns X_seq (n_samples, seq_len, n_features) and y_seq (n_samples,).
        """
        n_rows, n_feats = feature_matrix.shape
        n_seq = n_rows - seq_len
        if n_seq <= 0:
            return np.array([]), np.array([])

        X_seq = np.zeros((n_seq, seq_len, n_feats), dtype=np.float32)
        y_seq = np.zeros(n_seq, dtype=np.float32)

        for i in range(n_seq):
            X_seq[i] = feature_matrix[i : i + seq_len]
            y_seq[i] = targets[i + seq_len]

        return X_seq, y_seq

    def train_for_horizon(
        self,
        feature_matrices: List[np.ndarray],
        targets_list: List[np.ndarray],
        horizon_name: str,
        train_ratio: float = 0.85,
    ) -> bool:
        """Train LSTM for one horizon using pooled data from all tickers.

        Args:
            feature_matrices: List of (n_rows, n_input_features) arrays, one per ticker
            targets_list: List of target arrays, one per ticker
            horizon_name: "next_day", "7_day", or "30_day"
            train_ratio: fraction of each ticker's data used for training

        Returns:
            True if training succeeded.
        """
        if not self.is_available():
            return False

        torch = self._torch
        nn = self._nn
        cfg = LSTM_CONFIG
        seq_len = cfg["seq_len"]

        # Normalize inputs across all tickers
        all_features = np.vstack(feature_matrices)
        mean = np.nanmean(all_features, axis=0)
        std = np.nanstd(all_features, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        self.scalers[horizon_name] = (mean, std)

        # Build sequences from each ticker, using only the training portion
        train_X, train_y = [], []
        val_X, val_y = [], []

        for feat_mat, targets in zip(feature_matrices, targets_list):
            if len(feat_mat) < seq_len + 20:
                continue

            # Normalize
            feat_norm = (feat_mat - mean) / std
            feat_norm = np.nan_to_num(feat_norm, nan=0.0, posinf=0.0, neginf=0.0)

            # Time-based split
            n = len(feat_mat)
            split_idx = int(n * train_ratio)

            # Create sequences
            X_all, y_all = self._prepare_sequences(feat_norm, targets, seq_len)
            if len(X_all) == 0:
                continue

            # Split: sequences ending before split_idx are train, rest are val
            split_seq_idx = max(0, split_idx - seq_len)
            train_X.append(X_all[:split_seq_idx])
            train_y.append(y_all[:split_seq_idx])
            if split_seq_idx < len(X_all):
                val_X.append(X_all[split_seq_idx:])
                val_y.append(y_all[split_seq_idx:])

        if not train_X:
            logger.warning("LSTM-%s: no training data", horizon_name)
            return False

        X_train = np.concatenate(train_X)
        y_train = np.concatenate(train_y)
        X_val = np.concatenate(val_X) if val_X else X_train[-500:]
        y_val = np.concatenate(val_y) if val_y else y_train[-500:]

        if len(X_train) < 200:
            logger.warning("LSTM-%s: too few samples (%d)", horizon_name, len(X_train))
            return False

        logger.info(
            "LSTM-%s: training on %d sequences (val=%d), seq_len=%d, features=%d",
            horizon_name, len(X_train), len(X_val), seq_len, X_train.shape[2],
        )

        # Build and train model
        model = self._build_model(
            n_input_features=X_train.shape[2],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        loss_fn = nn.MSELoss()

        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)

        dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg["batch_size"], shuffle=True
        )

        best_val_loss = float("inf")
        best_state = None
        patience = 5
        patience_counter = 0

        model.train()
        for epoch in range(cfg["epochs"]):
            epoch_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred, _ = model(X_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred, _ = model(X_val_t)
                val_loss = loss_fn(val_pred, y_val_t).item()
            model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Evaluate on val set
        model.eval()
        with torch.no_grad():
            val_pred, val_hidden = model(X_val_t)
            val_pred_np = val_pred.numpy()
            y_val_np = y_val

            if len(y_val_np) > 10 and np.std(val_pred_np) > 1e-12:
                corr = np.corrcoef(y_val_np, val_pred_np)[0, 1]
                hit = float(np.mean((y_val_np > 0) == (val_pred_np > 0)))
            else:
                corr = 0.0
                hit = 0.5

        logger.info(
            "LSTM-%s: val_loss=%.6f, corr=%.3f, hit=%.1f%%, epochs=%d",
            horizon_name, best_val_loss, corr, hit * 100,
            min(epoch + 1, cfg["epochs"]),
        )

        self.models[horizon_name] = model
        return True

    def extract_features(
        self,
        feature_matrix: np.ndarray,
        horizon_name: str,
    ) -> Optional[np.ndarray]:
        """Extract LSTM temporal features for a single ticker's feature matrix.

        Args:
            feature_matrix: (n_rows, n_input_features) — raw input features
            horizon_name: which horizon's LSTM to use

        Returns:
            (n_rows, n_output_features) array of LSTM features, or None.
            First seq_len rows are NaN (insufficient lookback).
        """
        if not self.is_available() or horizon_name not in self.models:
            return None

        torch = self._torch
        cfg = LSTM_CONFIG
        seq_len = cfg["seq_len"]
        n_output = cfg["n_output_features"]
        model = self.models[horizon_name]
        scaler = self.scalers.get(horizon_name)

        if scaler is None or len(feature_matrix) < seq_len + 1:
            return None

        mean, std = scaler
        feat_norm = (feature_matrix - mean) / std
        feat_norm = np.nan_to_num(feat_norm, nan=0.0, posinf=0.0, neginf=0.0)

        n_rows = len(feature_matrix)
        # Output: lstm_pred + n_output hidden features + lstm_trend = n_output + 2
        n_total_features = n_output + 2
        output = np.full((n_rows, n_total_features), np.nan, dtype=np.float32)

        model.eval()
        with torch.no_grad():
            for i in range(seq_len, n_rows):
                seq = feat_norm[i - seq_len : i]
                X = torch.FloatTensor(seq).unsqueeze(0)  # (1, seq_len, features)
                pred, hidden = model(X)
                pred_val = pred.item()

                # Feature 0: LSTM prediction
                output[i, 0] = pred_val

                # Features 1..n_output: hidden state components
                hidden_np = hidden[0].numpy()
                output[i, 1 : 1 + n_output] = hidden_np[:n_output]

                # Feature n_output+1: trend (current pred vs 5-step-ago pred)
                if i >= seq_len + 5 and not np.isnan(output[i - 5, 0]):
                    output[i, n_output + 1] = pred_val - output[i - 5, 0]

        return output

    def get_feature_names(self, horizon_name: str) -> List[str]:
        """Return feature column names for a given horizon."""
        n_output = LSTM_CONFIG["n_output_features"]
        names = [f"lstm_pred_{horizon_name}"]
        for i in range(n_output):
            names.append(f"lstm_hidden_{horizon_name}_{i}")
        names.append(f"lstm_trend_{horizon_name}")
        return names

    def save(self, base_dir: str) -> None:
        """Save trained LSTM models to disk."""
        if not self.is_available():
            return
        torch = self._torch
        lstm_dir = os.path.join(base_dir, "_lstm")
        os.makedirs(lstm_dir, exist_ok=True)

        for horizon_name, model in self.models.items():
            path = os.path.join(lstm_dir, f"lstm_{horizon_name}.pt")
            torch.save(model.state_dict(), path)

            # Save scaler
            if horizon_name in self.scalers:
                mean, std = self.scalers[horizon_name]
                np.savez(
                    os.path.join(lstm_dir, f"scaler_{horizon_name}.npz"),
                    mean=mean, std=std,
                )

        # Save config for reproducibility
        import json
        with open(os.path.join(lstm_dir, "config.json"), "w") as f:
            json.dump(LSTM_CONFIG, f, indent=2)

        logger.info("Saved LSTM models for %d horizons", len(self.models))

    def load(self, base_dir: str) -> bool:
        """Load trained LSTM models from disk."""
        if not self.is_available():
            return False
        torch = self._torch
        lstm_dir = os.path.join(base_dir, "_lstm")
        if not os.path.isdir(lstm_dir):
            return False

        import json
        config_path = os.path.join(lstm_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                saved_cfg = json.load(f)
        else:
            saved_cfg = LSTM_CONFIG

        n_input = len(saved_cfg.get("input_features", LSTM_CONFIG["input_features"]))
        hidden_size = saved_cfg.get("hidden_size", LSTM_CONFIG["hidden_size"])
        num_layers = saved_cfg.get("num_layers", LSTM_CONFIG["num_layers"])

        loaded = 0
        for horizon_name in ["next_day", "7_day", "30_day"]:
            model_path = os.path.join(lstm_dir, f"lstm_{horizon_name}.pt")
            scaler_path = os.path.join(lstm_dir, f"scaler_{horizon_name}.npz")

            if not os.path.exists(model_path):
                continue

            try:
                model = self._build_model(n_input, hidden_size, num_layers)
                model.load_state_dict(torch.load(model_path, weights_only=True))
                model.eval()
                self.models[horizon_name] = model

                if os.path.exists(scaler_path):
                    data = np.load(scaler_path)
                    self.scalers[horizon_name] = (data["mean"], data["std"])

                loaded += 1
            except Exception as e:
                logger.warning("Could not load LSTM-%s: %s", horizon_name, e)

        if loaded > 0:
            logger.info("Loaded %d LSTM models", loaded)
        return loaded > 0
