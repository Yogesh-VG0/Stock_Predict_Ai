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
TRADE_MIN_PROB_POSITIVE = 0.50  # P(return > 0) must exceed this — lowered from 0.51 (sign classifiers ~50% accuracy)
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

# Feature pruning: remove noisy features based on pooled model importance
# Phase 1: train pooled with all features → extract top-k by gain
# Phase 2: retrain pooled + per-ticker with shortlisted features only
FEATURE_PRUNING = {
    "enabled": True,
    "top_k": 35,                       # v3.0: 30→35 (retain more signal with improved regularization)
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
        "sent_available",                          # missingness indicator (0=no data, 1=data)
        # Insider (v4) — direct transaction features
        "insider_net_value_30d",                   # dollar flow direction
        "insider_buy_ratio_30d",                   # buy/sell balance
        "insider_cluster_buying",                  # cluster-buying alpha signal
        "insider_activity_z_90d",                  # abnormal activity detector
        "insider_available",                       # missingness indicator (0=no insider data)
        # Technical (v4)
        "rsi_divergence",                          # bullish/bearish RSI divergence
        # FMP analyst features (v1.5) — fundamental signals from analyst consensus
        "analyst_sentiment_7d",                    # analyst estimates consensus (7d rolling)
        "analyst_rating_7d",                       # overall analyst rating score (7d rolling)
        # Earnings features (v2.0) — post-earnings drift signals
        "earnings_surprise",                       # EPS actual - estimated
        "earnings_beat",                           # beat/miss binary signal
        "earnings_recency",                        # decay weight since last earnings
        # Fundamental features (v2.0) — valuation signals
        "fund_pe_ratio",                           # price-to-earnings ratio
        "fund_roe",                                # return on equity
        "fund_beta",                               # market sensitivity
        # Short interest features (v2.0) — crowding signals
        "si_short_float_pct",                      # short interest % of float
        "si_days_to_cover",                        # squeeze pressure metric
    ],
    "min_features": 15,                # Don't prune below this many features
}

