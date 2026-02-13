# API Refactor Summary

## PowerShell + Ruff

```powershell
# Check Python
py -V

# Install ruff
py -m pip install -U ruff

# Run from stockpredict-ai/
py -m ruff check ml_backend/api/main.py --select F401,F811,F821,F841

# Auto-fix unused imports
py -m ruff check ml_backend/api/main.py --select F401 --fix
```

Or run `.\scripts\run_ruff.ps1` from project root.

## Completed

### 1. Clean up imports
- Removed unused: `jwt`, `OAuth2PasswordBearer`, `JSONResponse`, `time`, `httpx`, `sys`, `genai`, `RateLimiter`, `Depends`, `Request`, `Query`, `functools.wraps`
- Removed unused constants: `RATE_LIMIT`, `JWT_ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES`, `API_PREFIX`, `API_VERSION`, `API_CONFIG`, `PREDICTION_WINDOWS`, etc.
- Added: `BackgroundTasks`, `uuid`

### 2. Remove globals
- Removed `mongo_client = None` and `predictor = None`
- `_training_jobs` moved to `app.state.training_jobs` (initialized in startup)
- Startup uses local variables; all state in `app.state.*`

### 3. Background job for training
- `POST /api/v1/train` returns `job_id` immediately, runs training in background
- `GET /api/v1/train/status/{job_id}` returns `{status, progress, message}`
- Job status stored in `_training_jobs` (use Redis in production for persistence)

### 4. normalize_prediction_dict()
- New `ml_backend/api/utils.py` with `normalize_prediction_dict(predictions)`
- Used in: train endpoint, get_predictions endpoint
- Ensures consistent shape: `prediction`, `confidence`, `price_change`, etc.

### 5. Standardized responses
- Historical: `{ticker, start_date, end_date, data: [...]}` with `data.to_dict(orient="records")`
- CORS: `CORS_ORIGINS` env (comma-separated) for prod; `*` for dev

### 6. Caching + indexes
- Redis cache: `predictions:v1:{ticker}` with 60s TTL (versioned for future invalidation)
- Mongo indexes: (ticker, timestamp) on predictions, (ticker, date) on historical, (ticker, last_updated) on sentiment

### 7. Standardized predictions response
- `GET /predictions/{ticker}` returns `{ ticker, as_of, windows: { next_day: {...}, ... } }`

### 8. normalize_prediction_dict improvements
- Preserves existing `predicted_price`, `current_price`; never wipes
- Computes `predicted_price` from `prediction` (log return) + `current_price` when missing
- Computes `price_change` when missing

### 9. Mongo indexes
- Named indexes: `idx_ticker_timestamp`, `idx_ticker_date`, `idx_ticker_last_updated`

### 10. Rate limiting
- Removed custom `rate_limit` decorator (used FastAPILimiter only)
- FastAPILimiter middleware handles rate limiting when Redis is configured

## Explain module (structure created)

- `ml_backend/explain/budget.py` — `SECTION_BUDGETS`, `truncate_section`, `take_top_n`
- `ml_backend/explain/renderers/` — news.py, social.py, fundamentals.py, technicals.py (stubs)
- TODO: migrate `build_comprehensive_explanation_prompt` and helpers from main.py

## Deferred

### Explanation builder full migration
- Prompt builder logic remains in `api/main.py`
- Renderers are stubs; integrate budget into sections when migrating

### Production improvements
- Use Celery/RQ/Arq for training jobs (persist to Redis)
- Add `response_model` to all routes
- Auth: implement or remove unused OAuth2PasswordBearer
- Structured error responses: `{error: {code, message, request_id}}`
