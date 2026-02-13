"""
API utilities: prediction normalization, error responses, etc.
No FastAPI routes or app references.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import logging
import math

logger = logging.getLogger(__name__)


def _safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if v is None:
            return default
        x = float(v)
        if math.isfinite(x):
            return x
        return default
    except Exception:
        return default


def normalize_prediction_dict(predictions: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Normalize prediction dict to consistent shape. Never wipes existing fields.
    Keys per window: prediction, predicted_price, price_change, confidence, current_price, price_range

    Prefer "prediction" (log return). If missing, derive from price_change/current_price.
    Preserve provided current_price, predicted_price, price_change. Only compute missing ones.
    """
    if not predictions or not isinstance(predictions, dict):
        return {}

    normalized: Dict[str, Dict[str, Any]] = {}

    for window, vals in predictions.items():
        if window == "_meta":
            normalized["_meta"] = vals if isinstance(vals, dict) else {}
            continue
        if not isinstance(vals, dict):
            logger.warning("Invalid prediction structure for %s: %r", window, vals)
            normalized[str(window)] = {
                "prediction": 0.0,
                "confidence": 0.0,
                "price_change": 0.0,
                "predicted_price": 0.0,
                "current_price": 0.0,
                "price_range": {},
            }
            continue

        out = dict(vals)

        # prediction (log return)
        pred_val = _safe_float(vals.get("prediction"), 0.0) or 0.0
        if pred_val == 0.0 and vals.get("price_change") is not None and vals.get("current_price") not in (None, 0, "0"):
            chg = _safe_float(vals.get("price_change"))
            cur = _safe_float(vals.get("current_price"))
            if chg is not None and cur is not None and cur != 0 and (cur + chg) > 0:
                pred_val = math.log((cur + chg) / cur)
        out["prediction"] = float(pred_val)

        # confidence
        out["confidence"] = float(_safe_float(vals.get("confidence"), 0.0) or 0.0)

        # prices
        current = float(_safe_float(vals.get("current_price"), 0.0) or 0.0)
        predicted = _safe_float(vals.get("predicted_price"))
        price_change = _safe_float(vals.get("price_change"))

        if predicted is None:
            if price_change is not None and current:
                predicted = current + price_change
            elif current and pred_val != 0.0:
                predicted = current * math.exp(pred_val)
            else:
                predicted = current

        if price_change is None:
            price_change = (float(predicted) - current) if current else 0.0

        out["current_price"] = float(current)
        out["predicted_price"] = float(predicted if predicted is not None else 0.0)
        out["price_change"] = float(price_change)

        # price_range must be dict
        pr = out.get("price_range")
        out["price_range"] = pr if isinstance(pr, dict) else {}

        normalized[str(window)] = out

    return normalized


def error_response(code: str, message: str, request_id: Optional[str] = None) -> Dict[str, Any]:
    """Structured error response."""
    return {"error": {"code": code, "message": message, "request_id": request_id}}


def validate_ticker(ticker: str) -> str:
    """
    Sanitize and validate ticker symbol. Raises ValueError on invalid input.
    Returns uppercase ticker.
    """
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Ticker cannot be empty")
    t = ticker.strip().upper()
    if not t.isalnum():
        raise ValueError("Ticker must contain only alphanumeric characters")
    if len(t) < 1 or len(t) > 5:
        raise ValueError("Ticker must be 1-5 characters long")
    return t


def validate_days_back(days: int, min_days: int = 1, max_days: int = 2520) -> int:
    """
    Validate days_back parameter. Raises ValueError on invalid input.
    max_days=2520 is ~10 years of trading days.
    """
    if not isinstance(days, int):
        raise ValueError("days_back must be an integer")
    if days < min_days or days > max_days:
        raise ValueError(f"days_back must be between {min_days} and {max_days}")
    return days
