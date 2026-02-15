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
    "purge_days": 5,               # Purge between train/val/test to prevent leakage
    "embargo_days": 2,            # Days after target before next train sample
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

# Walk-forward folds: 0 = single split; 3-4 = rolling folds, report median metrics (credibility upgrade)
WALK_FORWARD_FOLDS = 3

# Trade filters: only recommend when model is both optimistic and confident
TRADE_MIN_ALPHA = 0.001  # Minimum predicted alpha (0.1%)
TRADE_MIN_PROB_POSITIVE = 0.52  # P(return > 0) must exceed this
ROUND_TRIP_COST_BPS = 10  # Round-trip transaction cost in basis points
TRADE_SIGMA_MULT = 1.0  # Regime-adaptive threshold multiplier

# Pooled model config (one model per horizon across all tickers)
POOL_CONFIG = {
    "enabled": True,
    "min_samples_per_ticker": 120,
    "min_total_samples": 500,
    "use_sector_feature": True,
}

# LightGBM params - conservative to avoid overfitting
# Huber: robust to outliers (earnings surprises, black swans)
LIGHTGBM_PARAMS = {
    "objective": "huber",
    "alpha": 0.9,  # Huber delta (moderately robust)
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 150,
    "max_depth": 4,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 25,  # Allow smaller leaves for limited per-ticker data
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

# Feature pruning: remove noisy features based on pooled model importance
# Phase 1: train pooled with all features → extract top-k by gain
# Phase 2: retrain pooled + per-ticker with shortlisted features only
FEATURE_PRUNING = {
    "enabled": True,
    "top_k": 30,                       # Keep top-k features by gain per horizon
    "protected_features": [            # Core stability features — never prune
        "log_return_1d", "log_return_5d", "log_return_21d",
        "volatility_20d", "volume_ratio", "rsi",
        "sector_id", "ticker_id",
        "vix_return_1d", "vix_level",             # cross-asset regime
        "sector_etf_return_1d",                    # sector rotation
        # Sector regime (v2)
        "sector_etf_return_20d",                   # sector momentum
        "excess_vs_sector_5d",                     # stock vs sector excess
        "sector_etf_vol_20d",                      # sector volatility
        "sector_momentum_rank",                    # cross-sector rotation rank (v3)
        # Sentiment (v2)
        "sent_mean_7d",                            # rolling sentiment
        "sent_momentum",                           # sentiment regime change
        "news_count_7d",                           # news flow intensity
        "news_spike_1d",                           # unusual coverage burst detector
        # Insider (v4) — direct transaction features
        "insider_net_value_30d",                   # dollar flow direction
        "insider_buy_ratio_30d",                   # buy/sell balance
        "insider_cluster_buying",                  # cluster-buying alpha signal
        "insider_activity_z_90d",                  # abnormal activity detector
        # Technical (v4)
        "rsi_divergence",                          # bullish/bearish RSI divergence
    ],
    "min_features": 15,                # Don't prune below this many features
}
