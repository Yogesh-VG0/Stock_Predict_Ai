# StockPredict-AI — Complete Technical Audit

**Run analyzed:** `59478963797` (2026-03-05)  
**Auditor:** Automated Code Audit  
**Date:** 2026-03-05  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [CI/CD Pipeline](#2-cicd-pipeline)
3. [Data Pipelines & APIs](#3-data-pipelines--apis)
4. [Sentiment Engine](#4-sentiment-engine)
5. [Feature Engineering](#5-feature-engineering)
6. [ML Models & Training](#6-ml-models--training)
7. [Ensemble & Prediction Logic](#7-ensemble--prediction-logic)
8. [Backtesting](#8-backtesting)
9. [Prediction Confidence & Calibration](#9-prediction-confidence--calibration)
10. [SHAP & Explainability](#10-shap--explainability)
11. [Deployment & Infrastructure](#11-deployment--infrastructure)
12. [Log Analysis](#12-log-analysis)
13. [Security](#13-security)
14. [Code Quality](#14-code-quality)
15. [Recommended Fixes (Priority-Ordered)](#15-recommended-fixes-priority-ordered)

---

## 1. Executive Summary

### Verdict: The system is architecturally sophisticated but the models have near-zero predictive edge.

**What works well:**
- Leakage-proof feature engineering with `shift(1)` and purge/embargo gaps
- Walk-forward validation with cross-sectional purge (industry best-practice)
- Market-neutral alpha target (removes market-direction noise)
- SHA-256 model integrity verification
- Conformal prediction intervals (calibrated uncertainty)
- Feature coverage guardrails (logs per-group missingness)
- Robust sentiment pipeline (75/75 tickers succeeded, multi-source fallbacks)
- CI pipeline with retry logic, batched predictions, and consecutive-failure abort

**What is critically broken:**

| Issue | Severity | Evidence |
|-------|----------|----------|
| Pooled model holdout: hit_rate < 50%, negative correlation | **CRITICAL** | POOLED-30_day corr=-0.140, hit=40.8% |
| next_day backtest: ZERO trades | **CRITICAL** | Log line 4513 |
| Median per-ticker win_rate 42-45% (worse than baseline) | **CRITICAL** | Training summary |
| Sign classifiers collapse to 1 tree (no learnable signal) | **HIGH** | POOLED-next_day, POOLED-30_day |
| Probability clipping inconsistency: [0.15,0.85] vs [0.20,0.80] | **HIGH** | `predict_all_windows` vs `predict_batch` |
| Node.js backend has zero authentication | **HIGH** | `backend/src/app.js` |
| CORS wide open on Node.js backend | **HIGH** | `cors()` with no config |
| Several ML API endpoints lack auth | **HIGH** | `/train/{ticker}`, `/predict/{ticker}` |
| Finnhub recommendation_trends timeout (10s, should be 30s) | **MODERATE** | Repeated ReadTimeout errors |
| SEC/FMP features zero-filled (`SKIP_SEC_FMP=true`) | **MODERATE** | Log warning |

---

## 2. CI/CD Pipeline

**File:** `.github/workflows/daily-predictions.yml` (~345 lines)

### Architecture
- Cron: Mon-Fri 22:15 UTC
- Python 3.11 on `ubuntu-latest`, pip cache enabled
- 10-step pipeline: sentiment → train → predict (8 batches) → verify → SHAP → explain → evaluate → drift → cleanup

### Issues Found

**2.1 Prediction batches run sequentially (slow)**  
8 batches × 3 retries × ~5min each = potential 2+ hour prediction window. Model files are loaded 8 times (225 models + 225 sign classifiers per batch — confirmed in logs). Consider loading once and predicting all tickers in a single process.

**2.2 No failure notification**  
Pipeline failures are silent. No Slack/email notification on workflow failure. Individual batch failures are logged but not aggregated.

**2.3 SHAP and AI explanations have generous timeouts**  
SHAP: 80 minutes total, 15min/batch. Explanations: 90 minutes. These are fine for reliability but extend the total workflow to 4+ hours.

**2.4 `--no-predict` then `--predict-only` separation is correct**  
Training outputs models to disk, prediction batches load from disk. This is safe — no model-loading race condition exists. The models are fully serialized before any prediction batch runs.

### Verdict: Pipeline structure is well-designed. Main inefficiency is 8× model loading.

---

## 3. Data Pipelines & APIs

### 3.1 Finnhub — MODERATE issue
**Actual error:** `ReadTimeout` on `recommendation_trends` endpoint (10-second timeout).  
**Your pre-identified issue:** "Client.__init__ got unexpected argument" — **NOT found in this run's logs.** The `finnhub.Client(api_key=api_key, timeout=30)` calls work fine (insider transactions succeed). The timeout issues are specific to the `recommendation_trends` endpoint using a different code path with a 10-second timeout.

**Evidence (log line 721):**
```
ReadTimeout(ReadTimeoutError("HTTPSConnectionPool(host='api.finnhub.io', port=443): 
Read timed out. (read timeout=10)"))
```

**Impact:** Finnhub recommendation/analyst sentiment is intermittently missing. However, insider transactions DO work (hundreds stored per ticker). The `finnhub_error` field in sentiment data captures the failure, and the feature pipeline falls back gracefully.

**Fix:** Increase the recommendation_trends timeout from 10s to 30s to match the client-level timeout.

### 3.2 Marketaux — Expected (user-acknowledged)
HTTP 402 Payment Required starting from batch 3 (NKE onwards). Rate limit exhausted. Works normally with fresh quota.

### 3.3 Short Interest — Partial failures, fallback works
- **Nasdaq API:** BRK-B returns `rCode 400: Symbol not exists` (Berkshire uses `.B` suffix on Nasdaq, not `-B`).
- Non-Nasdaq stocks (AMT, etc.): "Short interest is not available" → Finviz scraping fallback succeeds.
- Feature coverage shows 100% for short_interest group, confirming fallbacks populate data.

### 3.4 Yahoo Finance — Working
Historical data via `yfinance` backfill functions as expected. SPY data available for market-neutral calculations.

### 3.5 SEC/FMP — Explicitly skipped
`SKIP_SEC_FMP=true` is set in the CI environment. The code logs a warning:
```
[SEC/FMP] Training with SKIP_SEC_FMP=true — SEC/FMP features will be zero-filled.
```
The code also has a consistency check at inference time. If you ever want these features, remove `SKIP_SEC_FMP=true` from CI secrets.

---

## 4. Sentiment Engine

**File:** `ml_backend/data/sentiment.py` (~2200 lines)

### Status: WORKING (75/75 tickers)

| Source | Status | Notes |
|--------|--------|-------|
| RSS News | ✅ Working | Primary news source |
| FinBERT | ✅ Working | NLP sentiment via `ProsusAI/finbert` |
| VADER | ✅ Working | Fallback NLP |
| Finviz | ✅ Working | Scraping sentiment |
| Reddit (PRAW) | ✅ Working | 8 subreddits |
| Finnhub Insider | ✅ Working | Hundreds of transactions stored |
| Finnhub Recommendations | ⚠️ Intermittent | ReadTimeout (10s), marked down 2min, recovers |
| FMP Analyst | ✅ Working | Analyst estimates |
| Marketaux | ❌ Quota exhausted | Expected. HTTP 402 after batch 2 |
| Economic Calendar | ✅ Working | Event-driven features |

**Health check (log line ~4090):** 75/75 successful, 0 failed, 1758.8 seconds.

### Your pre-identified issue: "Sentiment cron failed"
**CONTRADICTED by logs.** `SENTIMENT_FRESH=true` is set throughout the pipeline. The sentiment cron completed successfully for all 75 tickers. The HuggingFace `BertForSequenceClassification` weight warning is cosmetic — the model loads and runs correctly.

### Issue: Multiple Finnhub Client instantiations
`finnhub.Client(api_key=api_key, timeout=30)` is created at lines 630, 690, 1479, and 2333 — a new client for every function call. This is wasteful. Should create once per `SentimentAnalyzer` instance.

---

## 5. Feature Engineering

**File:** `ml_backend/data/features_minimal.py`

### Architecture: Solid
- 86 features before pruning → 60-63 after
- All features use `.shift(1)` for point-in-time safety
- Groups: technical (returns, vol, RSI, SMA), sector (ETF returns, excess), sentiment (7d rolling, momentum, spike), insider (net value, buy ratio, cluster buying), fundamental (PE, ROE, beta), short interest (float %, days to cover), market regime (regime score, vol cluster, term slope)

### Feature Coverage (from logs):
```
short_interest: 100.0% (3 features)
sentiment:      100.0% (8 features)  
insider:        100.0% (12 features)
fundamental:    100.0% (5 features)
sector_etf:     100.0% (7 features)
```

All groups show 100% coverage, which is good — the fallback/zero-fill logic is working. However, 100% for sentiment is suspicious: it likely means the `sent_available` flag is set even when individual API sources fail (which is correct behavior — any one source triggers available=1).

### Issue: Protected features list is too large
`FEATURE_PRUNING.protected_features` contains 45 features, and `top_k=35`. Since protected + top_k overlap is merged with `set union`, you end up keeping 60-63 of 86 features. This defeats the purpose of pruning — you're only removing 23-26 features. Either reduce protected list or increase pruning aggression.

---

## 6. ML Models & Training

### Architecture
- **Pooled model:** One LightGBM regressor per horizon across all 75 tickers (cross-sectional)
- **Per-ticker models:** Individual LightGBM per ticker per horizon
- **Sign classifiers:** Binary LightGBM for P(return > 0)
- **Target:** Market-neutral alpha (log-return minus SPY log-return)
- **Validation:** Walk-forward with 4 folds, purge/embargo gaps

### CRITICAL: Models show near-zero predictive edge

**Pooled model holdout (Phase 2 — after feature pruning):**

| Horizon | RMSE | Hit Rate | Correlation | Verdict |
|---------|------|----------|-------------|---------|
| next_day | 0.0211 | 48.7% | 0.000 | **Coin flip** |
| 7_day | 0.0479 | 50.0% | -0.072 | **Negative correlation** |
| 30_day | 0.1042 | 42.5% | -0.143 | **Actively harmful** |

A hit rate below 50% with negative correlation means the model's predictions are worse than random. The 30-day pooled model is anti-correlated with reality — you'd do better inverting its predictions.

**Per-ticker training summary (75 tickers, holdout metrics):**

| Horizon | Median eval_rmse | Baseline RMSE | Median hit_rate | Win rate | Prod ready |
|---------|-----------------|---------------|-----------------|----------|------------|
| next_day | 0.0166 | 0.0166 | 50.6% | 44.0% | 10.7% |
| 7_day | 0.0372 | 0.0370 | 49.7% | 45.3% | 25.3% |
| 30_day | 0.0762 | 0.0754 | 50.3% | 42.7% | 32.0% |

- **next_day median eval_rmse = baseline RMSE exactly** — zero improvement over predicting 0.
- **Win rate < 50% for all horizons** — most per-ticker models are worse than the naive baseline.
- **Prod ready: 10.7% to 32.0%** — only 8-24 of 75 tickers pass the production gate.

### Sign classifier collapse
```
POOLED-next_day sign clf stopped at 1 trees → forced to 30 rounds → accuracy=50.8%
POOLED-30_day sign clf stopped at 1 trees → forced to 30 rounds → accuracy=51.0-53.2%  
POOLED-7_day: accuracy=52.0-52.8% (best, at 107-135 trees)
```

Early stopping kills the sign classifier at 1 tree because there is no learnable signal in the features for directional prediction at the next_day horizon. The forced 30-tree safety net produces a near-constant classifier.

### Root cause analysis

1. **Market-neutral alpha is extremely noisy at short horizons.** Daily alpha (stock - SPY) is dominated by idiosyncratic noise. The signal-to-noise ratio for 1-day alpha prediction using fundamental/sentiment features is essentially zero.

2. **Feature set is too broad relative to signal.** 60-63 features for a ~92,000 sample pooled model sounds reasonable, but when the target has near-zero predictability, more features = more noise dimensions for the model to overfit.

3. **Huber loss with alpha=0.9** is appropriate for robustness, but if the target is unpredictable, the model simply learns to predict near-zero (regression to mean). This is exactly what we see: predictions cluster near zero, producing threshold-gated trade filters that block everything.

4. **Feature pruning doesn't help enough** because it keeps 60+ features due to the oversized protected list.

### Recommendations

1. **Consider dropping next_day predictions entirely.** The data shows zero predictive power at 1-day alpha horizon. This is consistent with efficient market hypothesis for large-cap S&P stocks.

2. **Focus on 7_day and 30_day** where there's marginal evidence of signal (some per-ticker models pass production gate at 25-32%).

3. **Reduce protected features aggressively** — cut from 45 to 15-20. Let pruning do its job.

4. **Try ensemble of alpha models only for prod-ready tickers.** Don't serve predictions for tickers where `production_ready=False`.

5. **Add a global kill switch:** If pooled holdout hit_rate < 50%, don't serve any predictions for that horizon. The pipeline should detect and halt rather than serving anti-correlated predictions.

---

## 7. Ensemble & Prediction Logic

**File:** `ml_backend/models/predictor.py`, `predict_all_windows()` (lines 1086-1440)

### Architecture
- Pooled model is default
- Per-ticker model used only if `production_ready=True` AND `n_train >= 300`
- Ensemble: inverse-RMSE weighting when both qualify
- Sign classifier for `prob_positive`, with Gaussian CDF fallback
- Regime-adaptive confidence: bear/high-vol → reduced confidence, widened threshold

### BUG: Probability clipping inconsistency

**`predict_all_windows`** (single prediction, line ~1265):
```python
prob_positive = max(0.15, min(0.85, prob_positive))
```

**`predict_batch`** (vectorized backtest path, line ~1485):
```python
probs = np.clip(probs, 0.20, 0.80)
```

The backtest clips to [0.20, 0.80] while live predictions clip to [0.15, 0.85]. This means:
- A live prediction can show `prob_positive=0.16` and recommend a trade
- The same prediction in backtest would be clipped to 0.20, potentially changing `trade_recommended`
- **Backtest performance doesn't match live behavior**

### BUG: `predict_batch` doesn't apply ensemble logic

`predict_batch` (used by backtest) takes a shortcut:
```python
if ticker_meta is not None and ticker_meta.get("production_ready", False):
    model = self.models.get(key, model)  # replace pooled with ticker model
```

This is a **hard switch** — it uses either pooled OR ticker model. But `predict_all_windows` (live path) does **inverse-RMSE weighted blending** when both are available. This means the backtest doesn't test the actual ensemble logic used in production.

### Issue: Transaction cost check is asymmetric
`covers_transaction_cost` checks `abs(pred_return) >= min_return_for_profit`, but `trade_recommended` only triggers when `pred_return >= min_alpha` (positive direction only). Short positions are never recommended even if `pred_return` is strongly negative.

---

## 8. Backtesting

**File:** `ml_backend/backtest.py` (~320 lines)

### Results (this run):

| Horizon | Trades | Return | Sharpe | Max DD | Avg Trade Return | Win Rate |
|---------|--------|--------|--------|--------|------------------|----------|
| next_day | **0** ⚠️ | 0.00% | 0.000 | 0.00% | — | — |
| 7_day | 22 | 2.92% | 1.247 | -4.15% | -0.0001 | 45.5% |
| 30_day | 7 | 4.66% | 1.754 | -4.49% | 0.0149 | 57.1% |

SPY returned 1.58% over the same period (Dec 19, 2025 → Mar 4, 2026).

### Analysis

**next_day: ZERO trades** — The trade threshold (even capped to min 0.0001) was too high relative to the model's near-zero predictions. When the model predicts ~0 alpha for everything, no trade passes the filter. This is actually correct behavior (the model should not trade when it has no signal), but it means the next_day prediction product is useless.

**7_day: Marginal** — 22 trades, avg return -0.0001, 45.5% win rate. Essentially breakeven after costs. The 2.92% portfolio return came from a few winning trades offsetting many small losses. Not statistically significant with n=22.

**30_day: Promising but tiny sample** — 7 trades, 57.1% win rate, 1.49% avg return per trade. The 4.66% portfolio return vs SPY's 1.58% looks good, but n=7 trades is far too few for statistical significance.

### Issue: OOS window is too short
OOS period is Dec 19, 2025 → Mar 4, 2026 (~2.5 months, ~50 trading days). For 7_day and 30_day horizons, this yields very few independent trade opportunities. Need 6-12 months of OOS data for meaningful evaluation.

### Issue: Backtest uses `predict_batch` (inconsistent with live)
As noted in Section 7, `predict_batch` doesn't use the full ensemble logic. This makes backtest results not representative of live prediction quality.

---

## 9. Prediction Confidence & Calibration

### Conformal Prediction Intervals
- q90 and q95 quantiles computed from absolute residuals on holdout
- Used to build `price_range.low` / `price_range.high`
- Overrides fold-based conformal with holdout-based (good — more honest)

### Calibration concerns

1. **Sign classifier accuracy is ~50%** — The `prob_positive` output is essentially random. Any confidence display to users based on this is misleading.

2. **Confidence is a renamed probability** — `confidence = prob_positive` after clipping and regime adjustment. This is not a true calibration — a prediction that says "65% confidence" should be correct 65% of the time. With ~50% hit rates, the confidence values are uncalibrated.

3. **Regime-adaptive adjustments are multiplicative** — Bear regime reduces confidence by up to 20%, vol expansion by up to 15%. These are ad-hoc multipliers, not empirically calibrated.

4. **SENTIMENT_FRESH penalty is reasonable** — 15% penalty for stale sentiment. This is a good guardrail.

### Recommendation
Add a Platt scaling or isotonic regression calibration step using the holdout predictions. Train a simple sigmoid mapping from raw `prob_positive` → calibrated probability using holdout actual outcomes.

---

## 10. SHAP & Explainability

**Files:** `ml_backend/explain/shap_analysis.py`, `ml_backend/scripts/generate_explanations.py`

### SHAP: Well-implemented
- Uses LightGBM's native TreeSHAP via `pred_contrib=True` (no external `shap` package)
- Per-ticker timeout of 180 seconds
- Strict alignment check between SHAP values and feature names
- Feature list SHA-256 hashing for cache invalidation

### AI Explanations: Working, smart dedup
- Groq preferred (higher rate limit), Gemini fallback
- All 75 tickers skipped as "already exists" — dedup logic works
- Rich multi-section prompt with sentiment, technicals, fundamentals, insider data, SHAP

### Issue: Explanations are based on unreliable predictions
Given the near-random model performance, the SHAP explanations (which explain what the model thinks matters) and AI-generated text are rationalizing noise. Users reading "The model is bullish because insider buying was strong" are getting plausible-sounding but statistically unfounded narratives.

### Recommendation
Add a disclaimer to explanations when the ticker's model is not `production_ready`. Something like: "Note: The model for this ticker has not passed statistical significance tests for this horizon."

---

## 11. Deployment & Infrastructure

### MongoDB Atlas
- Connection pooling: `maxPoolSize=50`, `minPoolSize=10`
- Timeouts: server=15s, connect=20s, socket=90s
- `retryWrites=True`, `retryReads=True`
- gzip compression for cached data
- 14 collections with proper indexing
- Data retention cleanup for >12 months

### Redis
- Rate limiting with Lua script for atomicity
- In-memory fallback when Redis unavailable → graceful degradation
- Prediction caching with 5-minute TTL

### Model Storage
- `models/v1/` directory with joblib serialization
- SHA-256 hashes stored in `model_metadata.json`
- Integrity verification on load (corrupted models are skipped)
- Model version: `v2.0.0`

### Node.js Backend
- Express.js with compression, JSON body limit (1MB)
- Cache-control headers (`max-age=30`)
- Proxy routes to ML backend
- No clustering or PM2 configuration for production

---

## 12. Log Analysis

### Errors Found
| Error | Count | Impact |
|-------|-------|--------|
| Finnhub ReadTimeout | ~8 occurrences | Recommendation data intermittently missing |
| Marketaux HTTP 402 | Many (quota) | Expected, user-acknowledged |
| BRK-B short interest "Symbol not exists" | 1 | Finviz fallback handles |
| `[SEC/FMP] SKIP_SEC_FMP=true` warning | 2 (phase 1 + 2) | Expected configuration |

### Warnings Found
| Warning | Category | Impact |
|---------|----------|--------|
| Finnhub API marked as down for 2 minutes | Sentinel | Recovers automatically |
| HuggingFace model weight warning | Cosmetic | FinBERT loads fine |
| Backtest produced ZERO trades | Critical | next_day horizon unusable |
| Sign classifier stopped at 1 tree | Signal quality | No learnable short-term signal |

### Pipeline Health Summary (from log)
```
MongoDB connected:        YES
Historical data fetched:  75 tickers
Training status:          completed (75 tickers)
Backtest status:          completed (all horizons)
Predictions stored:       -- (--no-predict mode in training step)
```

### Your Pre-identified Issues — Verification

| # | Your Issue | Actual Finding |
|---|-----------|---------------|
| 1 | "Finnhub Client.__init__ unexpected argument" | **NOT found** in this run. Actual issue is ReadTimeout on recommendation_trends |
| 2 | "Marketaux rate limit" | **CONFIRMED.** HTTP 402. You said to ignore it ✓ |
| 3 | "Short Interest 404" | **Not exactly.** Nasdaq API returns `rCode 400` for BRK-B, "not available" for others. Finviz fallback works |
| 4 | "Sentiment cron failed" | **CONTRADICTED.** 75/75 tickers completed, SENTIMENT_FRESH=true |
| 5 | "No saved models found" | **Not triggered.** This is a CI guard check — models loaded fine (225 models + 225 sign classifiers) |
| 6 | "HuggingFace LOAD REPORT" | **Confirmed.** Cosmetic weight warning, model functions correctly |
| 7 | "Too Many warnings (327)" | **Confirmed.** Mostly Finnhub timeouts + Marketaux 402s + per-ticker training messages |

---

## 13. Security

### CRITICAL Findings

**13.1 Node.js backend: No authentication whatsoever**
All endpoints are public. Watchlist endpoints use `userId` from URL params — anyone can enumerate and modify any user's watchlist.

**13.2 CORS wide open on Node.js backend**
```js
app.use(cors());  // backend/src/app.js line 47
```
Any website can make cross-origin requests.

**13.3 ML backend auth disabled when env var unset**
`_require_api_key` returns early (allowing all requests) when `ML_API_KEY` is not set. Forgetting this env var in production exposes all protected endpoints.

**13.4 Unprotected ML endpoints**
- `POST /train/{ticker}` — Can trigger model training (compute cost)
- `POST /predict/{ticker}` — Can generate predictions
- `POST /optimization/trigger/{ticker}` — Can trigger optimization
- `POST /api/v1/explain/batch` — Can trigger Gemini API calls (money)
- `GET /debug/sec-filing-extraction/{ticker}` — Debug endpoint in production

### MODERATE Findings

**13.5 No rate limiting on Node.js backend**  
No `express-rate-limit` or similar. Backend can be DOS'd.

**13.6 No `helmet` security headers**  
Missing HSTS, X-Frame-Options, X-Content-Type-Options, etc.

**13.7 Weak input validation on Node.js stock endpoints**  
`req.params.symbol` gets `.toUpperCase()` but no regex validation in most controller paths.

### Good Practices Found
- All secrets via environment variables
- CI uses `${{ secrets.* }}`
- MongoDB credentials masked in logs
- ML backend has `validate_ticker()` function (`^[A-Z0-9.\-]{1,6}$`)
- ML backend has rate limiting (100 req/hr per IP with Redis)
- No hardcoded credentials found
- `.gitignore` properly excludes `.env`, models, venvs

---

## 14. Code Quality

### Strengths
- Code is well-organized with clear separation: `data/`, `models/`, `config/`, `scripts/`, `explain/`, `api/`, `utils/`
- Comprehensive logging throughout the pipeline
- Assertion-based data integrity checks (date ordering, alignment)
- Feature coverage guardrails with per-group logging
- Pipeline health summary class with quality gate checks
- Model versioning (`v2.0.0`) and metadata tracking
- Conformal prediction intervals (statistically principled)

### Issues

**14.1 `predictor.py` is 1550 lines**  
Contains training, prediction, ensemble, model I/O, feature selection, and sign classification. Should be split into:
- `trainer.py` — training logic
- `inference.py` — prediction logic  
- `model_io.py` — save/load/verify

**14.2 `sentiment.py` is 2200 lines**  
Monolithic file handling 8+ API integrations. Each API source should be a separate module.

**14.3 Redundant Finnhub client creation**  
`finnhub.Client()` instantiated at 4 different locations in sentiment.py instead of once.

**14.4 `import pandas as pd` inside method body**  
`predict_all_windows()` (line ~1203) has `import pandas as pd` inside the function. This is a leftover from debugging — pandas is already imported at the module top.

**14.5 `constants.py` says `TOP_100_TICKERS` but contains 75**  
Misleading name. The variable should be renamed to `TICKERS` or `TOP_75_TICKERS`.

**14.6 `PREDICTION_WINDOWS` horizon mismatch**  
```python
PREDICTION_WINDOWS = {"next_day": 1, "7_day": 7, "30_day": 30}
```
But `TARGET_CONFIG` uses trading days:
```python
"7_day": {"horizon": 5},   # 5 trading days
"30_day": {"horizon": 21},  # 21 trading days
```
The naming is confusing — "7_day" means 5 trading days, "30_day" means 21 trading days. The `PREDICTION_WINDOWS` dict with values 7 and 30 is never used for actual horizon calculations (TARGET_CONFIG is used instead), but it's misleading.

---

## 15. Recommended Fixes (Priority-Ordered)

### P0 — CRITICAL (Fix immediately)

#### Fix 1: Add global kill switch for anti-correlated predictions
When pooled holdout shows negative correlation, don't serve predictions for that horizon.

**File:** `ml_backend/models/predictor.py`, `predict_all_windows()`

Add after pooled metadata loading:
```python
# Kill switch: refuse to predict if pooled model is anti-correlated
pooled_corr = pooled_meta.get("holdout_correlation", 0.0)
if pooled_corr < -0.05:
    logger.warning(
        "KILL-SWITCH: Pooled %s has negative holdout correlation (%.3f). "
        "Predictions suppressed.", window_name, pooled_corr
    )
    results[window_name] = {
        "prediction": 0.0, "alpha": 0.0, "confidence": 0.0,
        "trade_recommended": False, "reason": "model_anti_correlated",
        # ... fill remaining fields
    }
    continue
```

#### Fix 2: Unify probability clipping
**File:** `ml_backend/models/predictor.py`

In `predict_batch()`, change:
```python
probs = np.clip(probs, 0.20, 0.80)
```
to:
```python
probs = np.clip(probs, 0.15, 0.85)
```

#### Fix 3: Add authentication to Node.js backend
**File:** `backend/src/app.js`

- Add `express-rate-limit` and `helmet`
- Configure CORS with specific origins
- Add API key middleware for mutative endpoints

#### Fix 4: Protect ML backend endpoints
**File:** `ml_backend/api/main.py`

Add `dependencies=[Depends(_require_api_key)]` to:
- `POST /predict/{ticker}`
- `POST /train/{ticker}`
- `POST /optimization/trigger/{ticker}`
- `POST /api/v1/explain/batch`
- Remove or auth-gate `GET /debug/sec-filing-extraction/{ticker}`

#### Fix 5: Fail closed when ML_API_KEY unset
```python
def _require_api_key(api_key: str = Header(None, alias="X-API-Key"), ...):
    expected = os.getenv("ML_API_KEY")
    if not expected:
        raise HTTPException(503, "Server misconfigured: API key not set")
    # ... existing comparison logic
```

### P1 — HIGH (Fix this week)

#### Fix 6: Make backtest use ensemble logic
Create a shared `_predict_single_row()` method that both `predict_all_windows` and `predict_batch` call, ensuring identical model selection and ensemble weighting.

#### Fix 7: Increase Finnhub recommendation_trends timeout
In `sentiment.py`, the `recommendation_trends` fetch uses a different timeout than the client-level 30s. Find and increase it to 30s.

#### Fix 8: Reduce protected features list
Cut `FEATURE_PRUNING.protected_features` from 45 to ~15 core features:
```python
"protected_features": [
    "log_return_1d", "log_return_5d", "log_return_21d",
    "volatility_20d", "volume_ratio", "rsi",
    "sector_id", "ticker_id",
    "vix_return_1d", "vix_level",
    "sent_mean_7d", "sent_available",
    "insider_available",
    "regime_score", "vol_ratio_5_20",
],
```

#### Fix 9: Only serve predictions for `production_ready` tickers
In the prediction batch storage step, skip tickers where no horizon has `production_ready=True`.

### P2 — MODERATE (Fix this sprint)

#### Fix 10: Create shared Finnhub client
```python
class SentimentAnalyzer:
    def __init__(self, ...):
        self._finnhub_client = None
    
    @property
    def finnhub_client(self):
        if self._finnhub_client is None:
            api_key = os.getenv("FINNHUB_API_KEY")
            if api_key:
                self._finnhub_client = finnhub.Client(api_key=api_key, timeout=30)
        return self._finnhub_client
```

#### Fix 11: Add calibration to sign classifier
After training, apply Platt scaling:
```python
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(sign_clf, cv="prefit", method="sigmoid")
calibrated.fit(X_val_df, sign_y_val)
```

#### Fix 12: Add prediction quality disclaimer to explanations
When generating AI explanations, prepend: "Model confidence: [LOW/MODERATE/HIGH] based on holdout performance" depending on `production_ready` status and holdout metrics.

#### Fix 13: Consolidate model loading in CI
Instead of 8 separate `--predict-only` batches, add a `--predict-all` mode that loads models once and iterates through all tickers internally.

### P3 — LOW (Backlog)

- Rename `TOP_100_TICKERS` → `TICKERS`
- Fix `PREDICTION_WINDOWS` values to match actual trading-day horizons
- Remove inline `import pandas as pd` in `predict_all_windows()`
- Split `predictor.py` into trainer/inference/model_io
- Split `sentiment.py` into per-source modules
- Create `.env.example`
- Add Slack/email notification on CI pipeline failure

---

*End of audit.*
