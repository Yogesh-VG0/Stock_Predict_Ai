"""
Batch prediction endpoints for processing multiple tickers efficiently.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import asyncio
import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, validator

from ml_backend.api.utils import normalize_prediction_dict, validate_ticker
from ml_backend.api.cache import (
    get_predictions_cached,
    set_predictions_cache,
    PREDICTIONS_CACHE_TTL,
    PREDICTIONS_CACHE_VERSION,
)
from ml_backend.config.constants import TOP_100_TICKERS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/batch", tags=["Batch Predictions"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""

    tickers: List[str] = Field(..., description="List of ticker symbols (1-10)")
    days_back: Optional[int] = Field(
        252,
        ge=1,
        le=2520,
        description="Historical days to use for prediction",
    )
    use_cache: Optional[bool] = Field(
        True,
        description="Whether to use cached predictions",
    )

    @validator("tickers")
    @classmethod
    def validate_tickers(cls, tickers: List[str]) -> List[str]:
        """Validate and normalize all tickers (1-10, no duplicates)."""
        if not tickers:
            raise ValueError("At least one ticker required")
        if len(tickers) > 10:
            raise ValueError("Maximum 10 tickers per batch")
        validated = []
        for ticker in tickers:
            try:
                validated.append(validate_ticker(ticker))
            except ValueError as e:
                raise ValueError(f"Invalid ticker '{ticker}': {str(e)}")
        if len(validated) != len(set(validated)):
            raise ValueError("Duplicate tickers not allowed")
        return validated


class BatchPredictionResult(BaseModel):
    """Result for a single ticker in batch."""

    ticker: str
    status: str  # "success" or "error"
    predictions: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cached: bool = False
    processing_time_ms: Optional[float] = None


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""

    total: int
    successful: int
    failed: int
    results: List[BatchPredictionResult]
    total_processing_time_ms: float
    as_of: datetime


# ============================================================================
# BATCH PROCESSOR
# ============================================================================


async def _predict_single_ticker(
    ticker: str,
    app_state,
    days_back: int,
    use_cache: bool,
    redis_client,
) -> BatchPredictionResult:
    """
    Generate prediction for a single ticker using the same flow as get_predictions.
    """
    start_time = datetime.now()

    try:
        # Must be in S&P 100
        if ticker not in TOP_100_TICKERS:
            return BatchPredictionResult(
                ticker=ticker,
                status="error",
                error="Ticker not found in S&P 100",
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

        # Check cache first
        if use_cache and redis_client:
            cached = await get_predictions_cached(ticker, redis_client, PREDICTIONS_CACHE_VERSION)
            if isinstance(cached, dict) and "windows" in cached:
                elapsed = (datetime.now() - start_time).total_seconds() * 1000
                return BatchPredictionResult(
                    ticker=ticker,
                    status="success",
                    predictions=cached,
                    cached=True,
                    processing_time_ms=elapsed,
                )

        # Get from mongo or generate (cached was already checked above for "windows")
        predictions = app_state.mongo_client.get_latest_predictions(ticker)

        if not predictions:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=365 * 2)
            df = app_state.mongo_client.get_historical_data(ticker, start_date, end_date)
            if df is None or df.empty:
                return BatchPredictionResult(
                    ticker=ticker,
                    status="error",
                    error="No historical data available",
                    processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                )
            preds = await asyncio.to_thread(
                app_state.stock_predictor.predict_all_windows,
                ticker,
                df,
            )
            windows = normalize_prediction_dict(preds)
            app_state.mongo_client.store_predictions(ticker, windows)
            predictions = windows

        windows = normalize_prediction_dict(predictions)
        result = {
            "ticker": ticker,
            "as_of": datetime.utcnow().isoformat(),
            "windows": windows,
        }

        if use_cache and redis_client:
            await set_predictions_cache(
                ticker, result, redis_client, PREDICTIONS_CACHE_VERSION, PREDICTIONS_CACHE_TTL
            )

        elapsed = (datetime.now() - start_time).total_seconds() * 1000
        return BatchPredictionResult(
            ticker=ticker,
            status="success",
            predictions=result,
            cached=False,
            processing_time_ms=elapsed,
        )

    except Exception as e:
        logger.error("Error processing %s: %s", ticker, e)
        return BatchPredictionResult(
            ticker=ticker,
            status="error",
            error=str(e),
            processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
        )


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.post("", response_model=BatchPredictionResponse, summary="Batch Predictions")
async def batch_predictions(
    request: Request,
    batch_request: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """
    Get predictions for multiple tickers in a single request.

    - **tickers**: List of 1-10 ticker symbols
    - **days_back**: Historical days for prediction (default: 252)
    - **use_cache**: Use cached predictions if available (default: true)

    Returns predictions for all tickers, marking failures individually.
    Processing is done concurrently for better performance.
    """
    start_time = datetime.now()
    redis_client = getattr(request.app.state, "redis_client", None)

    # Process up to 5 concurrently
    semaphore = asyncio.Semaphore(5)

    async def process_one(ticker: str) -> BatchPredictionResult:
        async with semaphore:
            return await _predict_single_ticker(
                ticker=ticker,
                app_state=request.app.state,
                days_back=batch_request.days_back,
                use_cache=batch_request.use_cache,
                redis_client=redis_client,
            )

    tasks = [process_one(t) for t in batch_request.tickers]
    results = await asyncio.gather(*tasks)

    successful = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "error")
    total_time = (datetime.now() - start_time).total_seconds() * 1000

    logger.info(
        "Batch prediction completed: %d/%d successful, %.2fms total",
        successful,
        len(results),
        total_time,
    )

    return BatchPredictionResponse(
        total=len(results),
        successful=successful,
        failed=failed,
        results=results,
        total_processing_time_ms=total_time,
        as_of=datetime.utcnow(),
    )


@router.get("/status", summary="Batch Processing Status")
async def batch_status() -> Dict[str, Any]:
    """
    Get current batch processing capacity and limits.
    """
    return {
        "max_tickers_per_batch": 10,
        "max_concurrent_processing": 5,
        "cache_enabled": True,
        "recommended_patterns": {
            "small_batch": "1-3 tickers for quick updates",
            "medium_batch": "4-7 tickers for dashboard updates",
            "large_batch": "8-10 tickers for portfolio analysis",
        },
        "tips": [
            "Enable caching for frequently requested tickers",
            "Use days_back=252 for best performance (1 year)",
            "Failed predictions don't block successful ones",
            "Results are returned in the same order as requested",
        ],
    }
