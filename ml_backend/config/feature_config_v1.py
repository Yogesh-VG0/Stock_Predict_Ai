"""
Minimal V1 Feature Configuration - Leakage-proof, hard to overfit.
Data source ranking: KEEP / REDUCE / REMOVE
"""

# =============================================================================
# DATA SOURCE RANKING (for reference - implemented in features_minimal.py)
# =============================================================================
# KEEP: OHLCV, returns, volatility, relative strength (SPY/sector), volume
# KEEP: 1 news sentiment source, earnings proximity (days to/since, NOT outcomes)
# REDUCE: Short interest (ratio + days_to_cover only, use change)
# REDUCE: Macro (Fed Funds, 2Y-10Y spread, CPI YoY, Unemployment - lagged)
# REMOVE: Seeking Alpha comments, Finviz, VADER, RoBERTa Large, redundant sentiment
# REMOVE: Event actuals (use existence/density only), most macro indicators
# =============================================================================

# Minimal macro indicators (V1 engine uses only spread + Fed Funds; CPI/unemployment excluded)
MINIMAL_MACRO_INDICATORS = [
    "FEDERAL_FUNDS_RATE",
    "TREASURY_10Y",
    "TREASURY_2Y",
]

# Sector ETFs for relative strength (keep simple)
MINIMAL_SECTOR_ETFS = ["XLK", "XLF", "SPY"]  # Tech, Financial, Market

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
    ],
    "min_features": 15,                # Don't prune below this many features
}
