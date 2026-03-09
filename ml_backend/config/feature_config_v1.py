"""
Minimal V1 Feature Configuration - Leakage-proof, hard to overfit.
"""

# Feature config - strict time-based evaluation
FEATURE_CONFIG_V1 = {
    "lookback_days": 60,           # Warmup for rolling features
    "min_rows": 65,                # Minimum rows after warmup
    "train_ratio": 0.7,            # First 70% for train
    "val_ratio": 0.15,             # Next 15% for validation
    "holdout_ratio": 0.15,         # Last 15% true holdout (never touch until final eval)
    "purge_days": 7,               # Purge between train/val/test to prevent leakage (increased from 5)
    "embargo_days": 3,             # Days after target before next train sample (increased from 2)
}

# Prediction targets - log returns (better behaved than raw price)
TARGET_CONFIG = {
    "next_day": {"horizon": 1, "target": "log_return_1d"},
    "7_day": {"horizon": 5, "target": "log_return_5d"},   # 5 trading days
    "30_day": {"horizon": 21, "target": "log_return_21d"},  # 21 trading days
}

# Market-neutral target: predict alpha (stock return - SPY return) instead of absolute return
# Focuses model on cross-sectional ranking; removes market direction; random pick ~ net-0
USE_MARKET_NEUTRAL_TARGET = True

# Sentiment features toggle: set False to run A/B comparison without sentiment.
# When False, sentiment columns are filled with zeros (preserves feature dimensions).
USE_SENTIMENT_FEATURES = True

# Insider features toggle: set False to exclude insider-transaction features.
# Decoupled from sentiment so each alternative-data source can be A/B-tested independently.
USE_INSIDER_FEATURES = True

# v2.0 feature toggles — new data sources from stored MongoDB data
USE_EARNINGS_FEATURES = True       # Post-earnings drift (FMP earnings data)
USE_FUNDAMENTAL_FEATURES = True    # Valuation ratios (Finnhub basic financials)
USE_SHORT_INTEREST_FEATURES = True # Crowding signal (short interest data)

# Walk-forward folds: 0 = single split; 3-4 = rolling folds, report median metrics (credibility upgrade)
WALK_FORWARD_FOLDS = 4

# Trade filters: only recommend when model is both optimistic and confident
TRADE_MIN_ALPHA = 0.0002  # Minimum predicted alpha (0.02%) — lowered to allow more trades in market-neutral regime
TRADE_MIN_PROB_POSITIVE = 0.52  # P(return > 0) must exceed this — 0.50 was a no-op (any positive pred passes); 0.52 filters out noise
# Per-horizon probability thresholds: next_day signal is much weaker so use
# a lower bar (0.505) to allow *some* trades rather than zero.  7_day/30_day
# keep the standard 0.52 threshold.
TRADE_MIN_PROB_BY_HORIZON = {
    "next_day": 0.505,
    "7_day": 0.52,
    "30_day": 0.52,
}
ROUND_TRIP_COST_BPS = 10  # Round-trip transaction cost in basis points
TRADE_SIGMA_MULT = 0.3  # Regime-adaptive threshold multiplier — lowered from 0.5 (pred_std was making threshold too strict)

# Per-horizon caps for adaptive trade_threshold.
# Without caps, pred_mean + sigma*pred_std can be unreasonably high (killing
# all trades) or negative (allowing garbage trades).  Caps are in log-return
# units and scale with horizon length.
TRADE_THRESHOLD_CAP = {
    "next_day": {"min": 0.0001, "max": 0.004},   # 0.01% – 0.4%  (lowered min for 1d alpha)
    "7_day":    {"min": 0.0003, "max": 0.012},    # 0.03% – 1.2%  (lowered min)
    "30_day":   {"min": 0.0008, "max": 0.035},    # 0.08% – 3.5%  (lowered min)
}

# Pooled model config (one model per horizon across all tickers)
POOL_CONFIG = {
    "enabled": True,
    "min_samples_per_ticker": 120,
    "min_total_samples": 500,
    "use_sector_feature": True,
}

# LightGBM params — production-grade, tuned for 75-ticker pooled model (~91K samples)
# v3.0: switched from Huber to MSE (regression). Huber with alpha=0.9 was clipping
# gradients on market-neutral targets (±2-5% range), crushing all predictions toward
# zero — causing 0.000 correlations and coin-flip hit rates across all horizons.
# MSE lets the model actually capture return magnitude while early stopping + L2
# regularization prevent overfitting to outliers.
LIGHTGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 500,       # More rounds with slower learning for gradual convergence
    "max_depth": 5,            # Depth-5 is appropriate for 91K samples
    "learning_rate": 0.01,     # Slightly faster than v2 — MSE converges better than Huber
    "num_leaves": 31,          # Standard 2^5-1 for depth-5 — more capacity than v2's 24
    "min_child_samples": 25,   # 91K samples support finer splits
    "min_split_gain": 0.005,   # Lower than v2 — let genuine splits through with MSE
    "reg_alpha": 0.2,          # Moderate L1 — sparsity without crushing weak signals
    "reg_lambda": 1.0,         # L2 replaces Huber's outlier robustness
    "subsample": 0.75,         # Row sampling for diversity
    "subsample_freq": 1,       # Apply row sampling every boosting round
    "colsample_bytree": 0.7,   # Feature diversity per tree
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

# Next-day-specific LightGBM overrides — 1-day alpha is much noisier than 7d/30d,
# so we use heavier regularization and shallower trees.
# v3.0: relaxed from v2's over-regularization. With MSE loss the model can
# actually learn short-term patterns; we just need enough regularization to
# prevent overfitting on daily noise.
LIGHTGBM_PARAMS_NEXT_DAY = {
    **LIGHTGBM_PARAMS,
    "n_estimators": 500,       # Adequate rounds with early stopping
    "max_depth": 4,            # Shallower than base — limits overfit on daily noise
    "learning_rate": 0.01,     # Same as base
    "num_leaves": 20,          # Relaxed from 14 — depth-4 can handle 20 leaves with MSE
    "min_child_samples": 30,   # Relaxed from 40 — let the model find patterns
    "min_split_gain": 0.01,    # Relaxed from 0.02
    "reg_alpha": 0.5,          # Moderate L1
    "reg_lambda": 2.0,         # Stronger L2 than base for noisy target
    "subsample": 0.7,          # Closer to base
    "colsample_bytree": 0.6,   # Slightly more restricted for next_day
}

# Feature pruning: remove noisy features based on pooled model importance
# Phase 1: train pooled with all features → extract top-k by gain
# Phase 2: retrain pooled + per-ticker with shortlisted features only
# Per-horizon top_k: next_day gets fewer features (noisier target → simpler model)
# v3.2: increased top_k to accommodate new MACD/ATR/52w/BB features
FEATURE_PRUNING_TOP_K_BY_HORIZON = {
    "next_day": 45,   # v3.0: increased from 35 — preserving more signal with MSE
    "7_day": 55,      # v3.0: increased from 50
    "30_day": 60,     # v3.0: increased from 50
}
FEATURE_PRUNING = {
    "enabled": True,
    "top_k": 50,                       # v3.2: 45→50 for new features
    "protected_features": [            # Core stability features — never prune
        "log_return_1d", "log_return_5d", "log_return_21d",
        "volatility_20d", "volume_ratio", "rsi",
        "sector_id", "ticker_id",
        "vix_return_1d", "vix_level",
        "sent_mean_7d", "sent_available",
        "insider_available",
        "regime_score", "vol_ratio_5_20",
        # Short-term microstructure (critical for next_day)
        "clv", "ret_zscore_5d", "volume_spike_z",
        "volume_return_interaction", "consecutive_days",
        "candle_body_ratio", "obv_slope_10d",
        # v3.2: new technical features — proven alpha signals
        "macd_norm", "macd_hist_norm",
        "atr_norm", "dist_52w_high", "bb_width",
    ],
    "min_features": 22,                # Raised from 20 for additional new features
}

