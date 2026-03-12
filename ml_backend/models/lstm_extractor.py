"""
LSTM-based temporal feature extractor for StockPredict.

v10.0: Trains a lightweight LSTM on rolling windows of features to capture
sequential momentum/mean-reversion patterns that LightGBM cannot learn
(LightGBM treats each row as independent — no sequence awareness).

The LSTM hidden state is extracted as a 32-dimensional embedding and appended
to the existing feature set.  LightGBM then uses these temporal embeddings
alongside the original features, keeping all existing quality gates, kill-switch,
and shrinkage logic intact.

Architecture:
    Raw Features (N) × 30 timesteps → LSTM Encoder → Hidden State (32-dim)
    → Appended to LightGBM feature vector as lstm_0 .. lstm_31

Training:
    - Trained on first 50% of each ticker's chronological data only
    - Target: 30_day market-neutral alpha (strongest signal horizon)
    - Frozen after training — used purely as feature extractor
    - If training fails, pipeline falls back to standard features gracefully
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy torch import — only load when LSTM is actually used
_torch = None
_nn = None


def _import_torch():
    """Lazy-import torch to avoid startup cost when LSTM is disabled."""
    global _torch, _nn
    if _torch is None:
        import torch
        import torch.nn as nn
        _torch = torch
        _nn = nn
    return _torch, _nn


class _LSTMEncoder:
    """PyTorch LSTM encoder module (created lazily)."""

    def __init__(self, input_size: int, hidden_size: int = 32,
                 num_layers: int = 2, dropout: float = 0.1):
        torch, nn = _import_torch()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

        # Combine into a single module for save/load
        self.model = nn.Sequential()
        self.model.add_module("lstm", self.lstm)
        self.model.add_module("head", self.head)

        # Move to CPU explicitly (GH Actions has no GPU)
        self.device = torch.device("cpu")
        self.lstm.to(self.device)
        self.head.to(self.device)

    def forward(self, x):
        """Forward pass: x is (batch, seq_len, input_size)."""
        torch, _ = _import_torch()
        # LSTM output: (batch, seq_len, hidden_size), (h_n, c_n)
        output, (h_n, c_n) = self.lstm(x)
        # Use last hidden state from the top layer
        last_hidden = h_n[-1]  # (batch, hidden_size)
        pred = self.head(last_hidden)  # (batch, 1)
        return pred.squeeze(-1), last_hidden

    def parameters(self):
        torch, _ = _import_torch()
        params = list(self.lstm.parameters()) + list(self.head.parameters())
        return params

    def train_mode(self):
        self.lstm.train()
        self.head.train()

    def eval_mode(self):
        self.lstm.eval()
        self.head.eval()


class LSTMFeatureExtractor:
    """Train and use an LSTM as a frozen feature extractor for LightGBM."""

    def __init__(self, input_size: int, hidden_size: int = 32,
                 num_layers: int = 2, window_size: int = 30,
                 dropout: float = 0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.window_size = window_size
        self.dropout = dropout

        # Feature normalization stats (computed during training)
        self._feat_mean: Optional[np.ndarray] = None
        self._feat_std: Optional[np.ndarray] = None

        self._encoder: Optional[_LSTMEncoder] = None
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Z-score normalize features using training statistics."""
        if self._feat_mean is None or self._feat_std is None:
            return X
        return (X - self._feat_mean) / (self._feat_std + 1e-8)

    def _build_sequences(
        self, feats: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Build rolling-window sequences from a single ticker's features.

        Args:
            feats: (n_rows, n_features) — one ticker, chronologically sorted
            y: (n_rows,) targets, or None for inference

        Returns:
            X_seq: (n_valid, window_size, n_features)
            y_seq: (n_valid,) or None
        """
        n = len(feats)
        w = self.window_size
        if n < w:
            if y is not None:
                return np.empty((0, w, feats.shape[1])), np.empty(0)
            return np.empty((0, w, feats.shape[1])), None

        n_seq = n - w + 1
        # Use stride tricks for memory-efficient windowing (axis=0 only)
        X_seq = np.lib.stride_tricks.sliding_window_view(
            feats, window_shape=w, axis=0
        )  # shape: (n_seq, w, n_features)

        if y is not None:
            # Target for sequence ending at index t is y[t]
            y_seq = y[w - 1:]
            assert len(y_seq) == n_seq, f"Sequence/target mismatch: {n_seq} vs {len(y_seq)}"
            return X_seq, y_seq
        return X_seq, None

    def train_extractor(
        self,
        ticker_features: dict,
        target_horizon: int = 21,
        train_pct: float = 0.50,
        epochs: int = 15,
        batch_size: int = 256,
        lr: float = 1e-3,
    ) -> bool:
        """Train the LSTM feature extractor on pooled per-ticker data.

        Args:
            ticker_features: {ticker: (feats_array, close_array, spy_close_or_None)}
                feats_array: (n_rows, n_features) — chronologically sorted
                close_array: (n_rows,) — closing prices
                spy_close_or_None: (n_rows,) SPY closes or None
            target_horizon: horizon in trading days for the target return
            train_pct: fraction of each ticker's data to use for training
            epochs: number of training epochs
            batch_size: mini-batch size
            lr: learning rate

        Returns:
            True if training succeeded, False otherwise.
        """
        try:
            torch, nn = _import_torch()
        except ImportError:
            logger.warning("PyTorch not available — LSTM features disabled")
            return False

        all_sequences = []
        all_targets = []

        # Collect training sequences from each ticker
        for ticker, (feats, close, spy_close) in ticker_features.items():
            if feats is None or len(feats) < self.window_size + target_horizon + 10:
                continue

            # Build target: market-neutral log return
            y = np.log(close[target_horizon:] / close[:-target_horizon])
            if spy_close is not None and len(spy_close) >= target_horizon:
                spy_ret = np.log(spy_close[target_horizon:] / spy_close[:-target_horizon])
                if y.shape == spy_ret.shape:
                    y = y - spy_ret

            # Align features with target
            feats_aligned = feats[:len(y)]

            # Only use first train_pct of this ticker's data
            n_train = int(len(feats_aligned) * train_pct)
            if n_train < self.window_size + 10:
                continue

            feats_train = feats_aligned[:n_train]
            y_train = y[:n_train]

            # Filter NaN/inf targets
            valid = np.isfinite(y_train)
            feats_train = feats_train[valid]
            y_train = y_train[valid]

            # Build sequences
            X_seq, y_seq = self._build_sequences(feats_train, y_train)
            if len(X_seq) > 0:
                all_sequences.append(X_seq)
                all_targets.append(y_seq)

        if not all_sequences:
            logger.warning("LSTM: No valid training sequences collected")
            return False

        X_all = np.concatenate(all_sequences, axis=0).astype(np.float32)
        y_all = np.concatenate(all_targets, axis=0).astype(np.float32)

        logger.info(
            "LSTM training: %d sequences, window=%d, features=%d",
            len(X_all), self.window_size, X_all.shape[2],
        )

        # Compute and store normalization statistics
        # Reshape to (n_samples * window, n_features) for per-feature stats
        flat = X_all.reshape(-1, X_all.shape[2])
        self._feat_mean = np.nanmean(flat, axis=0).astype(np.float32)
        self._feat_std = np.nanstd(flat, axis=0).astype(np.float32)
        self._feat_std[self._feat_std < 1e-8] = 1.0  # avoid division by zero

        # Normalize
        X_all = (X_all - self._feat_mean) / (self._feat_std + 1e-8)

        # Replace NaN/inf with 0 after normalization
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

        # Create encoder
        self._encoder = _LSTMEncoder(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

        # Training
        self._encoder.train_mode()
        optimizer = torch.optim.Adam(self._encoder.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        n = len(X_all)
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(n)
            X_shuffled = X_all[perm]
            y_shuffled = y_all[perm]

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n, batch_size):
                X_batch = torch.from_numpy(X_shuffled[i:i + batch_size])
                y_batch = torch.from_numpy(y_shuffled[i:i + batch_size])

                optimizer.zero_grad()
                pred, _ = self._encoder.forward(X_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                # Gradient clipping for LSTM stability
                torch.nn.utils.clip_grad_norm_(self._encoder.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            # Simple early stopping
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 4:
                    logger.info("LSTM: early stopping at epoch %d (loss=%.6f)", epoch + 1, avg_loss)
                    break

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info("LSTM epoch %d/%d: loss=%.6f", epoch + 1, epochs, avg_loss)

        self._encoder.eval_mode()
        self._trained = True
        logger.info("LSTM training complete: %d sequences, final loss=%.6f", n, best_loss)
        return True

    def extract_embeddings(self, feats: np.ndarray) -> np.ndarray:
        """Extract LSTM temporal embeddings for a single ticker's features.

        Args:
            feats: (n_rows, n_features) — full feature matrix for one ticker

        Returns:
            embeddings: (n_rows, hidden_size) — temporal embedding per row.
            Rows with insufficient history (< window_size) are zero-padded.
        """
        if not self._trained or self._encoder is None:
            return np.zeros((len(feats), self.hidden_size), dtype=np.float32)

        torch, _ = _import_torch()
        n = len(feats)
        embeddings = np.zeros((n, self.hidden_size), dtype=np.float32)

        if n < self.window_size:
            return embeddings

        # Normalize features
        feats_norm = self._normalize(feats.astype(np.float32))
        feats_norm = np.nan_to_num(feats_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # Build sequences for all valid rows
        X_seq, _ = self._build_sequences(feats_norm, None)  # (n_valid, window, features)

        if len(X_seq) == 0:
            return embeddings

        # Extract embeddings in batches to manage memory
        # .copy() needed: sliding_window_view returns read-only view, torch requires writable
        batch_size = 512
        all_embeds = []
        with torch.no_grad():
            for i in range(0, len(X_seq), batch_size):
                batch = torch.from_numpy(np.ascontiguousarray(X_seq[i:i + batch_size]))
                _, hidden = self._encoder.forward(batch)
                all_embeds.append(hidden.numpy())

        all_embeds = np.concatenate(all_embeds, axis=0)

        # Place embeddings: row index (window_size - 1) maps to first valid sequence
        start_idx = self.window_size - 1
        embeddings[start_idx:start_idx + len(all_embeds)] = all_embeds

        return embeddings

    def save(self, path: str) -> None:
        """Save LSTM model and normalization stats to disk."""
        if not self._trained or self._encoder is None:
            return

        torch, _ = _import_torch()
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        state = {
            "lstm_state": self._encoder.lstm.state_dict(),
            "head_state": self._encoder.head.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "window_size": self.window_size,
            "dropout": self.dropout,
            "feat_mean": self._feat_mean,
            "feat_std": self._feat_std,
        }
        torch.save(state, path)
        logger.info("Saved LSTM extractor to %s", path)

    def load(self, path: str) -> bool:
        """Load LSTM model from disk."""
        if not os.path.exists(path):
            return False

        try:
            torch, _ = _import_torch()
            state = torch.load(path, map_location="cpu", weights_only=False)

            self.input_size = state["input_size"]
            self.hidden_size = state["hidden_size"]
            self.num_layers = state["num_layers"]
            self.window_size = state["window_size"]
            self.dropout = state["dropout"]
            self._feat_mean = state["feat_mean"]
            self._feat_std = state["feat_std"]

            self._encoder = _LSTMEncoder(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
            self._encoder.lstm.load_state_dict(state["lstm_state"])
            self._encoder.head.load_state_dict(state["head_state"])
            self._encoder.eval_mode()
            self._trained = True
            logger.info("Loaded LSTM extractor from %s (hidden=%d, window=%d)",
                        path, self.hidden_size, self.window_size)
            return True
        except Exception as e:
            logger.warning("Failed to load LSTM extractor: %s", e)
            return False
