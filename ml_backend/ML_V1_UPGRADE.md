# ML V1 Upgrade - Leakage-Proof, Accurate Predictions

## V1.2.2 Fixes (Production Robustness)

- **Empty val/test**: `logger.warning` + `continue` instead of assert — one bad ticker won't crash full training
- **Warnings include**: `n`, `horizon`, `gap`, `train_idx_end`, `val_idx_start`, `val_idx_end`, `test_idx_start` for debugging
- **Date-order assert**: Message includes `n`, `horizon`, `gap` when violated
- **Metadata**: `meta_out` pattern — always `n_test` (0 when no holdout), `val_rmse`, `test_rmse` (when holdout), `has_holdout`, `eval_rmse` (avoids KeyError downstream)
- **30_day min history**: Skip if `horizon >= 21` and `n < 300` (gap in log); window-agnostic
- **Metadata splits**: `splits` dict with train_end, val_start, val_end, test_start, test_end, gap
- **Baseline RMSE**: Store `baseline_rmse` (naive=0) on exact same `y_test` when holdout exists
- **beats_baseline**: `bool(test_rmse < baseline_rmse)` in metadata (holdout only)
- **print_training_summary()**: `has_holdout_count/total`, median eval/baseline (holdout only), `win_rate` = mean(beats_baseline)
- **splits.test_end**: Always present (n when no holdout); no missing keys

## V1.2.1 Fixes (Production Correctness)

- **Holdout**: test_idx_end = n (remainder of series, not truncated to 20)
- **val_idx_end**: capped at n
- **test_idx_start >= n**: no holdout when gaps consume all samples
- **Mask asserts**: train/val/test never overlap
- **Date logging**: train max, val range, test min + n_test
- **Leakage test**: dates_x / dates_y method (cleaner indexing)

## V1.2 Fixes (Alignment / Split / Prediction)

- **df_aligned**: `df_clean.loc[feature_df.index]` - exact same rows as features
- **as_of_date**: df_aligned also filtered when features filtered
- **volume_ratio**: inf→NaN (let dropna remove warmup), not 1
- **SPY**: normalize both indices for alignment
- **Macro**: `ffill().shift(1).fillna(0)` in one line
- **Predictor**: assert len(df_aligned)==len(features); gap=max(purge,horizon)+embargo
- **Split**: train/val/holdout with proper gaps; no double-purge
- **Scaling**: removed (LightGBM doesn't need it)
- **current_price**: from df_aligned in predict
- **confidence**: from val_rmse when available
- **Leakage test**: `scripts/test_leakage.py` - feature date < label date

## V1.1 Fixes (Leakage/Quality)

- **volume_pct_rank**: Fixed rolling percentile (rank within 60-day window, not global)
- **vol_regime**: Quantile shifted by 1 day (no self-referential threshold)
- **SPY relative strength**: Join on exact date index (no normalize) to avoid timezone/session mismatch
- **Macro**: Shift macro features by 1 day (release timing)
- **Earnings**: Removed days_to_earnings (hindsight bias); kept days_since_earnings only
- **Warmup**: Stricter dropna on core features (log_return_1d, volatility_20d, volume_ratio, price_vs_sma20, rsi)
- **Returns as features**: Kept log_return_1d, log_return_5d, log_return_21d (safe for predicting r_{t+1})

## What Changed

### 1. **Minimal Feature Engine** (`data/features_minimal.py`)
- **Point-in-time features**: Each row uses only data available at market close that day
- **No lookahead**: Returns, volatility, indicators use past values only
- **Reduced feature set** (~25 features instead of 100+):
  - Returns (1d, 5d, 21d log returns)
  - Volatility, intraday range, overnight gap
  - Volume features (ratio, percentile rank)
  - Trend (SMA, price vs SMA, momentum)
  - RSI
  - Relative strength vs SPY
  - Minimal macro (2Y-10Y spread, Fed Funds - lagged)
  - Regime flags (volatility regime)
  - Earnings proximity (days to/since - NO actual outcomes)

### 2. **LightGBM Predictor** (`models/predictor.py`)
- **Tree model** instead of LSTM (better for tabular features)
- **Time-based splits**: 70% train, 15% val, 15% holdout (no random split)
- **Purged walk-forward**: 5-day purge between splits to prevent leakage
- **Log-return targets**: Predicts returns, not raw price
- **Conservative hyperparameters**: max_depth=4, strong regularization

### 3. **Configuration** (`config/feature_config_v1.py`)
- Data source ranking (keep/reduce/remove)
- Minimal macro indicators
- Walk-forward CV settings
- LightGBM params

### 4. **API Updates**
- Uses `MinimalFeatureEngineer` instead of `FeatureEngineer`
- Uses new `StockPredictor` (LightGBM)
- Models saved to `models/v1/{ticker}/`
- Fixed `get_historical_data` to pass date range
- Fixed `/predict/{ticker}` to use `historical_data` collection

## How to Use

### Train
```bash
# Train all tickers (uses existing historical data in MongoDB)
POST /api/v1/train

# Train single ticker
POST /train/{ticker}?retrain=true
```

### Predict
```bash
GET /api/v1/predictions/{ticker}
POST /predict/{ticker}
```

## Baseline Comparison

Before training, compare against:
- **Naive**: prediction = 0 (no change)
- **Last return**: use yesterday's return
- **SMA crossover**: simple rule

If your model doesn't beat these after costs, it's not adding value.

## Data Sources (V1)

| Source | Status |
|--------|--------|
| OHLCV (Yahoo) | KEEP |
| Returns, volatility | KEEP |
| SPY relative strength | KEEP |
| Sector ETF (minimal) | KEEP |
| Fed Funds, 2Y-10Y spread | KEEP (lagged) |
| Earnings proximity | KEEP (no outcomes) |
| 5 sentiment models | REMOVED |
| Seeking Alpha comments | REMOVED |
| Finviz, event actuals | REMOVED |
| Most macro indicators | REMOVED |

## Next Steps

1. **Evaluate**: Run walk-forward backtest, compare to baselines
2. **Add sentiment**: If model beats baseline, add 1 news sentiment source with timestamp discipline
3. **Trading metrics**: Track hit rate, Sharpe, drawdown, turnover, costs
