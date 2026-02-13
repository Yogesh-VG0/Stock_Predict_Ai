"""
Prediction routes. Uses Request for app.state (no circular import).
"""

from datetime import datetime, timedelta
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Request

from ml_backend.api.utils import normalize_prediction_dict, validate_ticker, validate_days_back
from ml_backend.api.cache import get_predictions_cached, set_predictions_cache, PREDICTIONS_CACHE_TTL, PREDICTIONS_CACHE_VERSION

router = APIRouter(tags=["Predictions"])


def _validate_tickers(tickers: str) -> list:
    """Validate comma-separated tickers. Raises HTTPException on invalid input."""
    parts = [t.strip() for t in tickers.split(",") if t.strip()]
    out = []
    for p in parts:
        try:
            out.append(validate_ticker(p))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return out


@router.get("/api/v1/predictions/bulk/complete")
async def get_bulk_complete_predictions(request: Request, tickers: str) -> Dict[str, Any]:
    """More specific route - must be before /api/v1/predictions/{ticker}."""
    ticker_list = _validate_tickers(tickers)
    all_predictions = {}
    for t in ticker_list:
        p = request.app.state.mongo_client.get_latest_predictions(t)
        if p:
            all_predictions[t] = p
    return {"tickers": ticker_list, "predictions": all_predictions, "timestamp": datetime.utcnow().isoformat()}


@router.get(
    "/api/v1/predictions/{ticker}",
    summary="Get Predictions",
    description="Get model predictions for a given ticker.",
)
async def get_predictions(request: Request, ticker: str) -> Dict[str, Any]:
    try:
        ticker = validate_ticker(ticker)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    from ml_backend.config.constants import TOP_100_TICKERS

    if ticker not in TOP_100_TICKERS:
        raise HTTPException(status_code=404, detail="Ticker not found in S&P 100")

    redis_client = getattr(request.app.state, "redis_client", None)
    cached = await get_predictions_cached(ticker, redis_client, PREDICTIONS_CACHE_VERSION)
    if isinstance(cached, dict) and "windows" in cached:
        return cached

    if isinstance(cached, dict) and "windows" not in cached:
        predictions = cached
    else:
        predictions = request.app.state.mongo_client.get_latest_predictions(ticker)

    if not predictions:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=365 * 2)
        df = request.app.state.mongo_client.get_historical_data(ticker, start_date, end_date)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No historical data available")

        preds = request.app.state.stock_predictor.predict_all_windows(ticker, df)
        windows = normalize_prediction_dict(preds)
        request.app.state.mongo_client.store_predictions(ticker, windows)
        predictions = windows

    windows = normalize_prediction_dict(predictions)
    result = {"ticker": ticker, "as_of": datetime.utcnow().isoformat(), "windows": windows}
    await set_predictions_cache(ticker, result, redis_client, PREDICTIONS_CACHE_VERSION, PREDICTIONS_CACHE_TTL)
    return result


@router.get("/api/v1/predictions/{ticker}/complete")
async def get_complete_predictions(request: Request, ticker: str) -> Dict[str, Any]:
    try:
        ticker = validate_ticker(ticker)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    predictions = request.app.state.mongo_client.get_latest_predictions(ticker)
    if not predictions:
        raise HTTPException(status_code=404, detail=f"No predictions found for {ticker}")
    return {"ticker": ticker, "predictions": predictions, "timestamp": datetime.utcnow().isoformat()}


@router.get("/api/v1/predictions/{ticker}/explanation")
async def get_prediction_explanation(request: Request, ticker: str, window: str = "next_day") -> Dict[str, Any]:
    try:
        ticker = validate_ticker(ticker)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    window = window.strip()
    explanation = request.app.state.mongo_client.get_prediction_explanation(ticker, window)
    if not explanation:
        raise HTTPException(status_code=404, detail=f"No explanation found for {ticker}-{window}")
    return {
        "ticker": ticker,
        "window": window,
        "explanation": explanation,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/api/v1/predictions/{ticker}/history")
async def get_prediction_history(request: Request, ticker: str, days: int = 30) -> Dict[str, Any]:
    try:
        ticker = validate_ticker(ticker)
        days = validate_days_back(days, min_days=1, max_days=365)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    history = request.app.state.mongo_client.get_prediction_history(ticker, days) or []
    return {
        "ticker": ticker,
        "days": days,
        "history": history,
        "count": len(history),
        "timestamp": datetime.utcnow().isoformat(),
    }
