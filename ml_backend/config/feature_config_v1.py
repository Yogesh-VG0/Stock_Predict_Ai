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
# v8.1: 4 folds starting at 50% with 10% steps: 50%→60%→70%→80%.
# v8.0 started at 40% but that gave too little training data for the first fold.
# 50% start with 92K samples = ~46K training samples in fold 0 — plenty.
WALK_FORWARD_FOLDS = 4

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
TRADE_SIGMA_MULT = 0.05  # v8.2: back to v8.0 value — v8.1's 0.03 let through too many weak trades

# v7.0: Minimum confidence to recommend a trade.
# v6.0 used 12% but confidence was inflated (65% for models with 0.05 correlation).
# v7.0 uses honest confidence formula, so 15% is meaningful.
# v7.1.1: Per-horizon minimum confidence. 30_day backtest lost -7.30% because
# weak pooled-only predictions (holdout corr=0.013) passed the 0.15 gate and
# created 41 noise trades. Higher thresholds for longer horizons where the
# model has weaker per-ticker signal and larger potential losses.
TRADE_MIN_CONFIDENCE = 0.15  # Global fallback
TRADE_MIN_CONFIDENCE_BY_HORIZON = {
    "next_day": 0.10,  # Low bar — shrinkage + confidence cap already limits next_day
    "7_day":    0.10,  # v8.1: lowered from 0.18 — production backtest was positive (+3.3%) even with weak pooled signal
    "30_day":   0.15,  # v8.1: lowered from 0.25 — production backtest was positive (+3.3%); 0.25 filtered good trades
}

# v6.0: Sign classifier minimum accuracy threshold. When the sign classifier's
# holdout accuracy is below this, it is IGNORED (Gaussian CDF used instead).
# Prevents anti-correlated classifiers (30_day was 41.6%) from poisoning prob_positive.
SIGN_CLF_MIN_ACCURACY = 0.52

# v7.0: Per-horizon confidence caps — prevents inflated confidence when model has
# limited genuine edge. next_day alpha is essentially random; 30_day has the most signal.
# v7.2: Raised caps for 7_day and 30_day. v7.0/v7.1 caps were too restrictive:
# AAPL-30d (68.6% hit, corr=0.506) was capped at 65% — should express its edge.
# 7_day had the best backtest (Sharpe 0.657, +9.89%) — deserves higher cap.
# next_day stays low since even the best per-ticker models have ~0 correlation.
CONFIDENCE_CAP_BY_HORIZON = {
    "next_day": 0.25,  # next_day holdout corr ≈ 0 — keep cap low
    "7_day":    0.65,  # v8.1: raised from 0.55 — per-ticker models with genuine edge deserve higher confidence
    "30_day":   0.80,  # v8.1: raised from 0.75 — 30_day has strongest per-ticker signal (some stocks 70%+ hit rate)
}

# v7.0: Prediction shrinkage — scale predictions toward 0 based on model quality.
# When the model has no edge (50% hit, 0 corr), predictions are fully shrunk to 0.
# This prevents low-quality models from generating confidently wrong predictions.
# Dramatically improves backtest by eliminating noise trades.
PREDICTION_SHRINKAGE_ENABLED = True

# v10.2: Shrinkage floor — minimum fraction of raw prediction that passes through.
# v8.2 used 0.08 (8%) but production backtests proved genuine edge exists
# (30_day: +11.24% Sharpe 1.793, 7_day: +5.41% Sharpe 1.088). The 8% floor
# crushed predictions to near-zero for users (AAPL 30_day: +5.7% raw → +0.46% shown).
# 15% lets ~2x more signal through while still being very conservative.
# Cross-sectional ranking is unaffected (relative ranking unchanged by uniform scaling).
PREDICTION_SHRINKAGE_FLOOR = 0.15

# v7.1: Per-ticker model quality gate.
# Per-ticker models with correlation below this threshold are EXCLUDED from
# the ensemble. This prevents anti-correlated models (e.g. QCOM-30d corr=-0.63)
# from poisoning predictions. The pooled model is used alone instead.
# v7.0 used INVERT-SIGNAL to flip anti-correlated models, but that assumes
# negative correlation persists out-of-sample (it doesn't).
PER_TICKER_MIN_CORRELATION = 0.03  # v8.1: lowered from 0.05 — with 92K samples even 0.03 corr is statistically meaningful
PER_TICKER_MIN_HIT_RATE = 0.48    # v8.1: lowered from 0.49 — accept slightly below random if correlation is positive

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
    "n_estimators": 500,       # v8.2: between v8.0(400) and v8.1(600)
    "max_depth": 5,            # v8.2: keep v8.1 depth — 92K samples can support it
    "learning_rate": 0.015,    # v8.2: between v8.0(0.01) and v8.1(0.02)
    "num_leaves": 25,          # v8.2: between v8.0(20) and v8.1(31)
    "min_child_samples": 40,   # v8.2: between v8.0(50) and v8.1(35)
    "min_split_gain": 0.005,   # v8.2: between v8.0(0.008) and v8.1(0.003)
    "reg_alpha": 0.15,         # v8.2: between v8.0(0.20) and v8.1(0.10)
    "reg_lambda": 1.5,         # v8.2: between v8.0(2.0) and v8.1(1.0)
    "subsample": 0.70,         # v8.2: between v8.0(0.65) and v8.1(0.75)
    "subsample_freq": 1,       # Apply row sampling every boosting round
    "colsample_bytree": 0.65,  # v8.2: between v8.0(0.60) and v8.1(0.70)
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
    "n_estimators": 350,       # v8.2: conservative for noisy 1-day alpha
    "max_depth": 4,            # v8.2: keep v8.1 — slightly deeper than v8.0's 3
    "learning_rate": 0.01,     # v8.2: slow learning for noisy target
    "num_leaves": 15,          # v8.2: keep v8.1 value
    "min_child_samples": 50,   # v8.2: conservative
    "min_split_gain": 0.008,   # v8.2: stricter than v8.1(0.005) — next_day needs heavy filtering
    "reg_alpha": 0.25,         # v8.2: strong L1
    "reg_lambda": 2.0,         # v8.2: strong L2
    "subsample": 0.65,         # v8.2: moderate randomness
    "colsample_bytree": 0.60,  # v8.2: moderate feature dropout
}

# Feature pruning: remove noisy features based on pooled model importance
# Phase 1: train pooled with all features → extract top-k by gain
# Phase 2: retrain pooled + per-ticker with shortlisted features only
# v7.0: Reduced top_k. v6.0 kept 45-46 features (from 113), but many were noise.
# Fewer features = simpler model = less overfitting = better holdout performance.
FEATURE_PRUNING_TOP_K_BY_HORIZON = {
    "next_day": 25,   # v8.2: keep v8.1 (was 20 in v8.0)
    "7_day": 35,      # v8.2: between v8.0(25) and v8.1(40) — avoid too many noisy features
    "30_day": 40,     # v8.2: between v8.0(30) and v8.1(50) — v8.1's 50 caused overfitting
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

# ── v10.0: Cross-Sectional Ranking ──
# Instead of predicting each ticker in isolation, rank all tickers by predicted
# alpha and only go long the top quintile. Even a pooled model with corr=0.019
# becomes useful when you only need it to RANK tickers correctly.
USE_CROSS_SECTIONAL_RANKING = True
RANKING_CONFIG = {
    "top_pct": 0.20,       # Go long top 20% of tickers (15 out of 75)
    "min_tickers": 5,      # Need at least 5 tickers with predictions to rank
    "confidence_boost": 0.10,  # Boost confidence for top-ranked tickers
    "disabled_horizons": ["next_day"],  # v10.1: disable ranking for horizons with zero signal
                                        # next_day corr=0.000 → ranking on noise amplifies losses
}

# ── v10.0: LSTM Temporal Feature Extractor ──
# Train a lightweight LSTM on 30-day rolling windows to capture temporal patterns
# that LightGBM misses (momentum regimes, mean-reversion cycles).
# The LSTM hidden state (32-dim) is appended as additional features for LightGBM.
# If LSTM training fails, gracefully falls back to standard features.
USE_LSTM_FEATURES = True
LSTM_CONFIG = {
    "hidden_size": 32,     # Embedding dimension
    "num_layers": 2,       # LSTM depth
    "window_size": 30,     # 30 trading days lookback (~6 weeks)
    "dropout": 0.1,        # Between LSTM layers
    "epochs": 15,          # Training epochs
    "batch_size": 256,     # Mini-batch size
    "lr": 1e-3,            # Adam learning rate
    "train_pct": 0.50,     # Use first 50% of each ticker for LSTM training
    "target_horizon": "30_day",  # Train on strongest-signal horizon
    "min_tickers": 20,     # Need ≥20 tickers for meaningful LSTM training
                           # (10 tickers = ~5K sequences → overfits; 75 tickers = ~44K → useful)
}
