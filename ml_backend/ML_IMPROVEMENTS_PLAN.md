# ML Accuracy & Production-Safety Improvements Plan

File-by-file patch-style plan based on code review. Prioritized by impact and effort.

---

## IMPLEMENTED (Version 3 - Feb 2026)

| Change | Status | Files |
|--------|--------|-------|
| Market-neutral target (alpha vs SPY) | Done | `feature_config_v1.py`, `features_minimal.py`, `predictor.py` |
| MAE + last-return baseline | Done | `predictor.py` |
| Walk-forward evaluation (pooled) | Done | `feature_config_v1.py`, `predictor.py` |
| Regime features (SPY vol) | Done | `features_minimal.py` |
| Production_ready = beats both baselines + hit_rate > 50% | Done | `predictor.py` |
| Skip-trade rule (prob_positive, trade_recommended) | Done | `predictor.py`, `feature_config_v1.py` |
| Transaction cost metadata (min_return, covers_cost) | Done | `predictor.py` |
| Normalized return + best_window for horizon selection | Done | `predictor.py`, `api/utils.py` |
| Constants: TICKER_SUBREDDITS before TICKER_TO_ID | Done | `constants.py` |
| Backtest module (Sharpe, max drawdown, vs SPY) | Done | `backtest.py`, `scripts/run_backtest.py` |
| Regime-adaptive threshold (mean + 2σ) | Done | `predictor.py`, `feature_config_v1.py` |
| prob_above_threshold (P(return > threshold)) | Done | `predictor.py` |

**Config flags:** `USE_MARKET_NEUTRAL_TARGET=True`, `WALK_FORWARD_FOLDS=3`, `TRADE_MIN_ALPHA=0.001`, `TRADE_MIN_PROB_POSITIVE=0.70`, `TRADE_SIGMA_MULT=2.0`, `ROUND_TRIP_COST_BPS=20`

---

## LOSING LOONIES V4 VIDEO TAKEAWAYS (Applied)

| Video suggestion | Our implementation |
|------------------|-------------------|
| **Normalized return (compound)** | We use log returns: `pred_return/horizon_days` is the correct compound-equivalent daily log return. No change needed. |
| **Overlapping windows** | We use time-based splits + purge/embargo, not sliding LSTM windows. No overlap issue. |
| **Classification target** | Instead of training a classifier, we output `prob_above_threshold` = P(return > mean+2σ) from normal approx. Same idea: focus on tradable moves. |
| **Threshold = mean + 2σ** | Implemented: `trade_threshold` stored in metadata per horizon; `trade_recommended` uses it. Config: `TRADE_SIGMA_MULT=2.0`. |
| **Slippage/fees** | Already have `ROUND_TRIP_COST_BPS=20`. Comments suggest adding slippage bps; optional future add. |

---

## 1. BIGGEST ACCURACY WINS

### 1A. Pooled Model (Recommended: Default)

**Why:** Per-ticker history is small. One model per horizon across all tickers → more samples, better generalization.

**Files:** `models/predictor.py`, `config/feature_config_v1.py`, `data/features_minimal.py`

**Changes:**
- Add `ticker_id` (categorical) or `sector_etf` as feature
- Train one LightGBM per horizon on concatenated ticker data
- Predictor key: `(None, window)` for pooled; optionally keep `(ticker, window)` for per-ticker fine-tune when n ≥ 300
- **Recommendation:** Make pooled the **default**; use per-ticker only when explicitly requested or when pooled unavailable

### 1B. Rolling Walk-Forward CV

**Files:** `models/predictor.py`

**Changes:**
- Replace single split with multiple rolling folds:
  - Fold 1: train 2018–2021, val 2022, test 2023
  - Fold 2: train 2018–2022, val 2023, test 2024
- Use purge/embargo in each fold
- Store `median(test_rmse)`, `median(baseline_rmse)` across folds

### 1C. Robust Loss

**Files:** `config/feature_config_v1.py`, `models/predictor.py`

**Changes:**
- LightGBM `objective`: try `huber` or `quantile` (median)
- Or keep MSE but winsorize targets (e.g. clip to ±10%)

### 1D. Real Price Intervals (Replace ±2% Constant)

**Files:** `models/predictor.py` in `predict_all_windows`

**Changes:**
- Compute residual std on val/test per horizon
- Return `pred ± 1.0σ` and `pred ± 2.0σ` in price_range
- Or use conformal prediction for valid intervals

---

## 2. FEATURE ENGINE IMPROVEMENTS

### 2A. Cache SPY + Macro in Mongo (CRITICAL)

**Problem:** `features_minimal.py` downloads SPY via yfinance and hits FRED during feature build → slow, fragile, alignment drift.

**Files:** `data/features_minimal.py`, `data/ingestion.py`

**Changes:**
1. **SPY:** Ingest SPY into `historical_data` (same as tickers). In `_add_relative_strength`:
   - Try `mongo_client.get_historical_data("SPY", start, end)` first
   - Fall back to yfinance only if Mongo empty (log warning)
   - Add in-memory cache: `(start_str, end_str) -> df` per process
2. **Macro:** In `_add_macro_features`:
   - Try `mongo_client.get_macro_data(indicator, start, end)` first
   - Only call `fetch_and_store_all_fred_indicators` on cache miss
   - Cache in memory: `(start, end) -> macro_df`
3. **Ingestion:** Ensure SPY is ingested alongside tickers (add to ingestion pipeline or cron).

### 2B. Vectorize Earnings Proximity

**Problem:** `_add_earnings_proximity` uses `iterrows()` + per-row loop → very slow.

**File:** `data/features_minimal.py`

**Change:** Replace with `np.searchsorted`:
```python
earnings_dates = np.array(sorted(earnings_dates))
df_dates = pd.to_datetime(df["date"]).values.astype("datetime64[D]")
# For each row: find largest earning date <= row date
idx = np.searchsorted(earnings_dates, df_dates, side="right") - 1
mask = idx >= 0
days_since = np.full(len(df), 90)
days_since[mask] = (df_dates[mask] - earnings_dates[idx[mask]]).astype("timedelta64[D]").astype(int)
df["days_since_earnings"] = np.minimum(days_since, 90)
```

### 2C. volume_ratio inf Fix

**File:** `data/features_minimal.py` line 119

**Change:** Divide by `volume_ma20.replace(0, np.nan)` to avoid inf:
```python
df["volume_ratio"] = df["Volume"] / df["volume_ma20"].replace(0, np.nan)
```

### 2D. Optimize volume_pct_rank (Optional)

**Problem:** `rolling(60).apply(lambda x: Series(x).rank(pct=True).iloc[-1])` is expensive.

**Options:**
- Use `scipy.stats.percentileofscore` approximation for last value
- Or compute only during training and cache; skip in prediction if stored with features

---

## 3. PREDICTOR/TRAINING IMPROVEMENTS

### 3A. Recency Weighting

**File:** `models/predictor.py`

**Change:** Pass `sample_weight` to LightGBM: exponentially increasing toward recent (e.g. `weight = exp(λ * (t - t_min))`).

### 3B. Direction Metrics

**File:** `models/predictor.py`

**Change:** In metadata, add:
- `hit_rate = mean(sign(y) == sign(pred))`
- `corr(pred, y)`
- Optional: top-k concentration

### 3C. Additional Baselines

**File:** `models/predictor.py`

**Change:** Add:
- "Last return" baseline: `y_hat = log_return_1d(t)` for next_day
- Optional: SMA crossover signal baseline
- Compare model RMSE to both; don't ship if model doesn't beat baselines.

---

## 4. REMOVE / QUARANTINE (Dead Weight)

| Module | Action | Reason |
|--------|--------|--------|
| `data/features.py` | Move to `legacy/` or delete | Legacy, huge, likely unused by V1 |
| `data/economic_calendar.py` | Quarantine | Selenium/cloudscraper, heavy |
| `data/seeking_alpha.py` | Quarantine | Fragile scraping |
| `data/sec_filings.py` | Keep if used by explain | Else move to `explain_sources/` |

**Check:** Grep for imports of these in `features_minimal.py`, `predictor.py`, `sentiment.py`. If unused in training path → quarantine.

---

## 5. UNUSED IMPORTS (Quick Wins)

| File | Remove |
|------|--------|
| `data/features_minimal.py` | `List`, `os`, `MINIMAL_MACRO_INDICATORS`, `MINIMAL_SECTOR_ETFS`, `TICKER_YFINANCE_MAP` |
| `models/predictor.py` | `Tuple`, `Any`, `TOP_100_TICKERS` (if unused) |
| `scripts/test_leakage.py` | `timedelta` |

---

## 6. CRITICAL: api/utils.py

**Status:** ✅ Verified clean. No routes, no app references. (Bug mentioned was from pasted message, not actual codebase.)

---

## 7. IMPLEMENTATION ORDER

### Phase A: Quick Wins (Today)
1. Unused imports cleanup
2. volume_ratio inf fix
3. Vectorize earnings proximity

### Phase B: Production Safety (This Week)
4. SPY + macro Mongo cache
5. Ensure SPY in ingestion

### Phase C: Accuracy (Next Sprint)
6. Pooled model (default)
7. Rolling walk-forward CV
8. Robust loss + real intervals
9. Recency weighting + direction metrics + baselines

### Phase D: Cleanup (When Ready)
10. Quarantine legacy modules

---

## 8. POOLED MODEL: Default vs Optional

**Recommendation: Default**

- Pooled model as primary
- Per-ticker fine-tune only when: (a) user explicitly requests, or (b) pooled not trained yet
- Fallback: when per-ticker n < 300 for a horizon, use pooled only

---

## 9. RUFF ON WINDOWS

All commands from **repo root** (parent of `ml_backend`):

```powershell
py -V
py -m pip install -U ruff
py -m ruff check ml_backend --select F401,F841
py -m ruff check ml_backend --select F401 --fix
```

Or use the Windows helper (run from repo root, i.e. parent of `ml_backend`):

```powershell
.\ml_backend\scripts\run_ruff.ps1
.\ml_backend\scripts\run_ruff.ps1 -Fix
.\ml_backend\scripts\run_ruff.ps1 -Select F401,F841 -Fix
```

---

## 10. FILES TO CREATE

- `ml_backend/legacy/` – move `features.py`, etc.
- `ml_backend/data/spy_macro_cache.py` – optional helper for Mongo-first SPY/macro fetch with in-memory cache

---

## 11. SURVIVORSHIP BIAS (Documented – Future)

**Problem:** `TOP_100_TICKERS` uses current S&P 100 constituents. Stocks that were delisted (bankruptcy, acquisition, dropped from index) are excluded. Backtesting on current constituents overstates historical returns because we only include survivors.

**Impact:** If you add proper historical backtesting, use historical index constituents (e.g. S&P 100 membership by date) instead of current tickers. For live predictions, survivorship bias is less critical since we predict current names.

**Mitigation (future):**
- Source historical S&P 100 constituents (e.g. from S&P Dow Jones, Compustat, or free sources like Wikipedia snapshots)
- When backtesting date D, only include tickers that were in the index on date D
- Document this limitation in backtest output

---

## 12. BACKTEST MODULE (Implemented – Feb 2026)

- `ml_backend/backtest.py` – `run_backtest()` simulates buy when `trade_recommended`, hold for horizon, sell
- `ml_backend/scripts/run_backtest.py` – CLI entry point
- Reports: total return, Sharpe ratio, max drawdown, vs SPY benchmark
- Run: `python -m ml_backend.scripts.run_backtest --tickers AAPL MSFT --horizon next_day`
