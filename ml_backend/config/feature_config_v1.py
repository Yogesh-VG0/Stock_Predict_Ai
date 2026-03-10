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
# v7.0: reduced from 5 to 3 — more training data per fold reduces overfitting.
# v6.0 showed classic overfit: fold correlation positive but holdout correlation negative.
WALK_FORWARD_FOLDS = 3

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

# v7.0: Minimum confidence to recommend a trade.
# v6.0 used 12% but confidence was inflated (65% for models with 0.05 correlation).
# v7.0 uses honest confidence formula, so 15% is meaningful.
TRADE_MIN_CONFIDENCE = 0.15

# v6.0: Sign classifier minimum accuracy threshold. When the sign classifier's
# holdout accuracy is below this, it is IGNORED (Gaussian CDF used instead).
# Prevents anti-correlated classifiers (30_day was 41.6%) from poisoning prob_positive.
SIGN_CLF_MIN_ACCURACY = 0.52

# v7.0: Per-horizon confidence caps — prevents inflated confidence when model has
# limited genuine edge. next_day alpha is essentially random; 30_day has the most signal.
CONFIDENCE_CAP_BY_HORIZON = {
    "next_day": 0.25,
    "7_day":    0.45,
    "30_day":   0.65,
}

# v7.0: Prediction shrinkage — scale predictions toward 0 based on model quality.
# When the model has no edge (50% hit, 0 corr), predictions are fully shrunk to 0.
# This prevents low-quality models from generating confidently wrong predictions.
# Dramatically improves backtest by eliminating noise trades.
PREDICTION_SHRINKAGE_ENABLED = True

# v7.1: Per-ticker model quality gate.
# Per-ticker models with correlation below this threshold are EXCLUDED from
# the ensemble. This prevents anti-correlated models (e.g. QCOM-30d corr=-0.63)
# from poisoning predictions. The pooled model is used alone instead.
# v7.0 used INVERT-SIGNAL to flip anti-correlated models, but that assumes
# negative correlation persists out-of-sample (it doesn't).
PER_TICKER_MIN_CORRELATION = 0.02  # Must show at least slight positive correlation
PER_TICKER_MIN_HIT_RATE = 0.48    # Must not be significantly worse than random

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
# v7.0: MAJOR regularization increase. v6.0 showed classic overfitting:
#   - Fold correlation positive but holdout correlation negative across all horizons
#   - Per-ticker win_rate 38.7% (worse than naive baseline)
#   - Pooled 30_day holdout corr = -0.098 (anti-correlated)
# Fix: much stronger regularization to prevent fitting noise.
LIGHTGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 500,       # v7: reduced from 800 — less capacity to memorize noise
    "max_depth": 5,            # v7: reduced from 6 — shallower trees
    "learning_rate": 0.015,    # v7: slower learning for better generalization
    "num_leaves": 25,          # v7: reduced from 40 — simpler tree structure
    "min_child_samples": 40,   # v7: increased from 25 — requires more support per split
    "min_split_gain": 0.005,   # v7: 2.5x increase — only keep meaningful splits
    "reg_alpha": 0.15,         # v7: increased from 0.08 — more L1 sparsity
    "reg_lambda": 1.5,         # v7: nearly 2x increase — stronger L2 ridge penalty
    "subsample": 0.7,          # v7: reduced from 0.8 — more randomness per tree
    "subsample_freq": 1,       # Apply row sampling every boosting round
    "colsample_bytree": 0.65,  # v7: reduced from 0.75 — more feature dropout
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

# Next-day-specific LightGBM overrides — 1-day alpha is noisier than 7d/30d.
# v7.1: v7.0 was TOO regularized — min_split_gain=0.008 + min_child_samples=50
# + max_depth=4 + num_leaves=15 produced a model with ZERO splits (constant output).
# All 75 tickers showed corr=0.000 and 0% feature gain. The model learned nothing.
# v7.1 relaxes just enough to allow learning while still preventing overfitting.
# Prediction shrinkage will scale down output if the model has no real edge.
LIGHTGBM_PARAMS_NEXT_DAY = {
    **LIGHTGBM_PARAMS,
    "n_estimators": 400,       # v7.1: was 300 — need more trees to find weak signal
    "max_depth": 5,            # v7.1: was 4 — match base depth
    "learning_rate": 0.01,     # v7.1: slower than base to prevent overfit
    "num_leaves": 20,          # v7.1: was 15 — allow more tree complexity
    "min_child_samples": 40,   # v7.1: was 50 — match base (still conservative)
    "min_split_gain": 0.003,   # v7.1: was 0.008 — allow weaker splits through
    "reg_alpha": 0.20,         # v7.1: slightly reduced from 0.25
    "reg_lambda": 1.8,         # v7.1: reduced from 2.0
    "subsample": 0.65,         # v7.1: same — more randomness
    "colsample_bytree": 0.60,  # v7.1: was 0.55 — slightly more features per tree
}

# Feature pruning: remove noisy features based on pooled model importance
# Phase 1: train pooled with all features → extract top-k by gain
# Phase 2: retrain pooled + per-ticker with shortlisted features only
# v7.0: Reduced top_k. v6.0 kept 45-46 features (from 113), but many were noise.
# Fewer features = simpler model = less overfitting = better holdout performance.
FEATURE_PRUNING_TOP_K_BY_HORIZON = {
    "next_day": 30,   # v7: aggressive pruning for noisiest target
    "7_day": 35,      # v7: moderate pruning
    "30_day": 40,     # v7: keep more for longer horizon (more signal)
}
FEATURE_PRUNING = {
    "enabled": True,
    "top_k": 50,                       # v3.2: 45→50 for new features
    "protected_features": [            # Core stability features — never prune
        # v7.1: Reduced from 27 to 15 protected features. v7.0 had top_k=30
        # but 27 protected features, so pruning only removed 3 features —
        # effectively useless. Now pruning can actually remove noisy features.
        "log_return_1d", "log_return_5d", "log_return_21d",
        "volatility_20d", "volume_ratio", "rsi",
        "sector_id", "ticker_id",
        "vix_level",
        "sent_mean_7d", "sent_available",
        "regime_score",
        "dist_52w_high", "dist_52w_low",
        "atr_norm",
    ],
    "min_features": 18,                # v7.1: lowered from 22 — allow more aggressive pruning
}

