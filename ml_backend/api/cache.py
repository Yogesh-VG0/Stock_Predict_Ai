"""
Prediction cache helpers. No FastAPI app or circular imports.
"""

import json
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

PREDICTIONS_CACHE_TTL = 60
PREDICTIONS_CACHE_VERSION = "v1"


async def get_predictions_cached(
    ticker: str,
    redis_client: Any,
    version: str = PREDICTIONS_CACHE_VERSION,
) -> Optional[Dict]:
    """Get predictions from Redis cache if available."""
    if redis_client is None:
        return None
    try:
        key = f"predictions:{version}:{ticker}"
        cached = await redis_client.get(key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        logger.debug("Cache read failed for %s: %s", ticker, e)
    return None


async def set_predictions_cache(
    ticker: str,
    data: Dict,
    redis_client: Any,
    version: str = PREDICTIONS_CACHE_VERSION,
    ttl: int = PREDICTIONS_CACHE_TTL,
) -> None:
    """Store predictions in Redis cache."""
    if redis_client is None or not data:
        return
    try:
        key = f"predictions:{version}:{ticker}"
        await redis_client.setex(key, ttl, json.dumps(data, default=str))
    except Exception as e:
        logger.debug("Cache write failed for %s: %s", ticker, e)
