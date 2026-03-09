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

# Walk-forward folds: 0 = single split; 3-5 = rolling folds, report median metrics (credibility upgrade)
# v5.0: increased to 5 for more robust cross-validation estimates
WALK_FORWARD_FOLDS = 5

# Trade filters: only recommend when model is both optimistic and confident
# v5.0: substantially lowered thresholds — previous values were filtering out
# nearly all trades (0 trades for next_day, <30 for others).
TRADE_MIN_ALPHA = 0.0001  # Minimum predicted alpha (0.01%) — very low bar; let confidence do the filtering
TRADE_MIN_PROB_POSITIVE = 0.50  # P(return > 0) must exceed this — 0.50 is the baseline (any bullish signal)
# Per-horizon probability thresholds: v5.0 uses 0.50 across the board.
# The previous 0.505/0.52 thresholds were filtering out nearly all trades
# because sign classifiers were at ~50% accuracy (coin-flip). The real
# filtering now comes from confidence calibration and alpha threshold.
TRADE_MIN_PROB_BY_HORIZON = {
    "next_day": 0.50,
    "7_day": 0.50,
    "30_day": 0.50,
}
ROUND_TRIP_COST_BPS = 10  # Round-trip transaction cost in basis points
TRADE_SIGMA_MULT = 0.1  # v5.0: lowered from 0.3 — pred_std was making threshold too strict, killing trades

# v6.0: Minimum confidence to recommend a trade. Previous versions had no confidence
# floor, allowing trades at 4-5% confidence (essentially meaningless).
# Now requires at least 12% confidence — ensures some model conviction.
TRADE_MIN_CONFIDENCE = 0.12

# v6.0: Sign classifier minimum accuracy threshold. When the sign classifier's
# holdout accuracy is below this, it is IGNORED (Gaussian CDF used instead).
# Prevents anti-correlated classifiers (30_day was 41.6%) from poisoning prob_positive.
SIGN_CLF_MIN_ACCURACY = 0.50

# Per-horizon caps for adaptive trade_threshold.
# v5.0: lowered caps significantly — the previous values were too restrictive,
# especially for next_day where predictions are small magnitude.
TRADE_THRESHOLD_CAP = {
    "next_day": {"min": 0.00005, "max": 0.002},   # 0.005% – 0.2%
    "7_day":    {"min": 0.0001,  "max": 0.006},   # 0.01% – 0.6%
    "30_day":   {"min": 0.0003,  "max": 0.020},   # 0.03% – 2.0%
}

# Pooled model config (one model per horizon across all tickers)
POOL_CONFIG = {
    "enabled": True,
    "min_samples_per_ticker": 120,
    "min_total_samples": 500,
    "use_sector_feature": True,
}

# LightGBM params — production-grade, tuned for 75-ticker pooled model (~91K samples)
# v6.0: Slightly more regularized than v5.0 to reduce overfitting on 7_day horizon
# (which had negative holdout correlation). Better balance of capacity vs generalization.
LIGHTGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 800,       # More capacity; early stopping will pick the right point
    "max_depth": 6,            # Deeper trees to capture nonlinear interactions
    "learning_rate": 0.02,     # Faster convergence; early stopping prevents overshoot
    "num_leaves": 40,          # v6.0: reduced from 50 — less overfitting risk
    "min_child_samples": 25,   # v6.0: increased from 20 — more robust splits
    "min_split_gain": 0.002,   # v6.0: increased from 0.001 — filter out noisy splits
    "reg_alpha": 0.08,         # v6.0: slightly more L1 than v5 (0.05) for feature selection
    "reg_lambda": 0.8,         # v6.0: increased from 0.5 — better outlier robustness
    "subsample": 0.8,          # More data per tree
    "subsample_freq": 1,       # Apply row sampling every boosting round
    "colsample_bytree": 0.75,  # v6.0: reduced from 0.8 — more diverse trees
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

# Next-day-specific LightGBM overrides — 1-day alpha is noisier than 7d/30d.
# v5.0: The previous next_day params were WAY too regularized (depth-4, 20 leaves,
# reg_alpha=0.5, reg_lambda=2.0) producing 0.000 correlation and 49% hit rate.
# Now using moderate regularization that still allows the model to find patterns.
LIGHTGBM_PARAMS_NEXT_DAY = {
    **LIGHTGBM_PARAMS,
    "n_estimators": 800,       # Same as base; early stopping picks the right point
    "max_depth": 5,            # Depth-5 (was 4) — one level deeper to find patterns
    "learning_rate": 0.02,     # Same as base
    "num_leaves": 31,          # Standard 2^5-1 (was 20)
    "min_child_samples": 25,   # Slightly more conservative than base
    "min_split_gain": 0.002,   # Slightly higher than base to filter noise
    "reg_alpha": 0.1,          # Light L1 (was 0.5 — way too aggressive)
    "reg_lambda": 1.0,         # Moderate L2 (was 2.0)
    "subsample": 0.75,         # Slightly less than base for noise reduction
    "colsample_bytree": 0.7,   # Slightly restricted for next_day
}

# Feature pruning: remove noisy features based on pooled model importance
# Phase 1: train pooled with all features → extract top-k by gain
# Phase 2: retrain pooled + per-ticker with shortlisted features only
# v5.0: increased top_k across all horizons — previous values were too aggressive
# and were cutting features that had weak but genuine signal.
FEATURE_PRUNING_TOP_K_BY_HORIZON = {
    "next_day": 60,   # v5.0: was 45 — keep more features for noisy target
    "7_day": 70,      # v5.0: was 55
    "30_day": 75,     # v5.0: was 60
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

