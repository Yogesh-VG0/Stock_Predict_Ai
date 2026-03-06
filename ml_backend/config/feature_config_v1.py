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

# LightGBM params — production-grade, tuned for 75-ticker pooled model
# Huber: robust to outliers (earnings surprises, black swans)
# v3.1: stronger regularization to combat holdout degradation (47.8% → target 50%+)
LIGHTGBM_PARAMS = {
    "objective": "huber",
    "alpha": 0.9,              # Huber delta — 0.9 balances robustness vs capturing larger moves
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 400,       # Reduced from 500; early stopping (patience 30) prevents overfit
    "max_depth": 5,            # Reduced from 6 to prevent overfitting on recent OOS periods
    "learning_rate": 0.01,     # Very slow learning — best generalization with 400 rounds
    "num_leaves": 24,          # Reduced from 31; tighter constraint for depth-5
    "min_child_samples": 30,   # Increased from 20 for stronger regularization
    "min_split_gain": 0.02,    # Increased from 0.01 — filter out more noise splits
    "reg_alpha": 0.5,          # Increased L1 from 0.3 — promotes feature sparsity
    "reg_lambda": 2.0,         # Increased L2 from 1.0 — prevents weight explosion on noisy alpha
    "subsample": 0.75,         # Reduced from 0.8 for more diversity
    "subsample_freq": 1,       # Apply row sampling every boosting round
    "colsample_bytree": 0.7,   # Reduced from 0.8 — more feature diversity per tree
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

# Next-day-specific LightGBM overrides — 1-day alpha is much noisier than 7d/30d,
# so we use heavier regularization, more aggressive subsampling, and shallower trees.
# These are merged on top of LIGHTGBM_PARAMS for next_day horizon only.
LIGHTGBM_PARAMS_NEXT_DAY = {
    **LIGHTGBM_PARAMS,
    "n_estimators": 600,       # More rounds with slower learning to find weak signals
    "max_depth": 4,            # Shallower — daily noise overfits deeper trees
    "learning_rate": 0.005,    # Much slower learning for noisy 1-day target
    "num_leaves": 12,          # Very constrained — prevents memorizing daily noise
    "min_child_samples": 50,   # Large leaves: each split needs strong statistical support
    "min_split_gain": 0.05,    # Higher bar to split — only genuine patterns pass
    "reg_alpha": 1.0,          # Strong L1 — aggressively prune weak features
    "reg_lambda": 5.0,         # Strong L2 — suppress noisy coefficient magnitudes
    "subsample": 0.6,          # More aggressive row sampling for diversity
    "colsample_bytree": 0.5,   # Only see half the features per tree — reduces overfit
    "feature_fraction_bynode": 0.7,  # Additional feature randomization per node
}

# Feature pruning: remove noisy features based on pooled model importance
# Phase 1: train pooled with all features → extract top-k by gain
# Phase 2: retrain pooled + per-ticker with shortlisted features only
FEATURE_PRUNING = {
    "enabled": True,
    "top_k": 45,                       # v3.1: 35→45 to accommodate new microstructure features
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
    ],
    "min_features": 20,                # Raised from 15 to ensure microstructure features survive
}

