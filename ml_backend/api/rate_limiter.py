"""
In-memory rate limiter fallback for when Redis is unavailable.
"""

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class InMemoryRateLimiter:
    """
    Thread-safe in-memory rate limiter with sliding window.
    Used as fallback when Redis is unavailable.
    """

    def __init__(self):
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()
        logger.info("Initialized in-memory rate limiter fallback")

    async def is_allowed(
        self,
        client_ip: str,
        limit: int = 100,
        window_seconds: int = 3600,
    ) -> Tuple[bool, Dict]:
        """
        Check if request is allowed under rate limit.

        Args:
            client_ip: Client IP address
            limit: Maximum requests allowed in window
            window_seconds: Time window in seconds

        Returns:
            Tuple of (allowed: bool, info: dict)
        """
        async with self._lock:
            now = datetime.now().timestamp()
            cutoff = now - window_seconds

            # Clean old requests (sliding window)
            self.requests[client_ip] = [
                timestamp for timestamp in self.requests[client_ip]
                if timestamp > cutoff
            ]

            request_count = len(self.requests[client_ip])
            allowed = request_count < limit

            if allowed:
                self.requests[client_ip].append(now)

            reset_time = (
                int(min(self.requests[client_ip]) + window_seconds)
                if self.requests[client_ip]
                else int(now + window_seconds)
            )

            return allowed, {
                "limit": limit,
                "remaining": max(0, limit - request_count - (1 if allowed else 0)),
                "reset": reset_time,
            }

    async def cleanup_old_entries(self, max_age_seconds: int = 7200):
        """
        Periodic cleanup of old entries to prevent memory buildup.
        Call this from a background task.
        """
        async with self._lock:
            now = datetime.now().timestamp()
            cutoff = now - max_age_seconds

            ips_to_remove = []
            for ip, timestamps in self.requests.items():
                recent = [ts for ts in timestamps if ts > cutoff]
                if recent:
                    self.requests[ip] = recent
                else:
                    ips_to_remove.append(ip)

            for ip in ips_to_remove:
                del self.requests[ip]

            if ips_to_remove:
                logger.info("Cleaned up %d inactive IPs from rate limiter", len(ips_to_remove))

    def get_stats(self) -> Dict:
        """Get rate limiter statistics."""
        return {
            "active_ips": len(self.requests),
            "total_tracked_requests": sum(len(reqs) for reqs in self.requests.values()),
        }


# ============================================================================
# MIDDLEWARE FOR RATE LIMITING WITH FALLBACK
# ============================================================================

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

try:
    import redis.asyncio as redis
except ImportError:
    redis = None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with automatic Redis/in-memory fallback.
    """

    def __init__(
        self,
        app,
        redis_client: Optional["redis.Redis"] = None,
        fallback_limiter: Optional[InMemoryRateLimiter] = None,
        limit: int = 100,
        window: int = 3600,
        exclude_paths: Optional[List[str]] = None,
    ):
        super().__init__(app)
        self.redis = redis_client
        self.fallback = fallback_limiter or InMemoryRateLimiter()
        self.limit = limit
        self.window = window
        self.exclude_paths = exclude_paths or [
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
        ]
        self.using_fallback = False

        logger.info(
            "Rate limiting enabled: %d requests per %ds (Redis: %s)",
            limit,
            window,
            redis_client is not None,
        )

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""

        # Skip rate limiting for excluded paths
        if any(request.url.path.startswith(p) for p in self.exclude_paths):
            return await call_next(request)

        client_ip = self._get_client_ip(request)

        # Try Redis first, fallback to in-memory
        if self.redis and not self.using_fallback:
            allowed, info = await self._check_redis(client_ip)
            if allowed is None:
                self.using_fallback = True
                logger.warning("Redis unavailable, switching to in-memory rate limiter")
                allowed, info = await self.fallback.is_allowed(
                    client_ip, self.limit, self.window
                )
        else:
            allowed, info = await self.fallback.is_allowed(
                client_ip, self.limit, self.window
            )

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Please try again later.",
                    "limit": info["limit"],
                    "reset": info["reset"],
                },
                headers={
                    "X-RateLimit-Limit": str(info["limit"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(info["reset"]),
                    "Retry-After": str(
                        max(0, info["reset"] - int(datetime.now().timestamp()))
                    ),
                },
            )

        response = await call_next(request)

        response.headers["X-RateLimit-Limit"] = str(info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(info["reset"])

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request (handles proxies)."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        return request.client.host if request.client else "unknown"

    async def _check_redis(self, client_ip: str) -> Tuple[Optional[bool], Dict]:
        """
        Check rate limit using Redis.
        Returns (None, {}) if Redis fails.
        """
        try:
            key = f"rate_limit:{client_ip}"
            current = await self.redis.get(key)

            if current is None:
                await self.redis.setex(key, self.window, 1)
                return True, {
                    "limit": self.limit,
                    "remaining": self.limit - 1,
                    "reset": int(datetime.now().timestamp() + self.window),
                }

            current = int(current)
            if current >= self.limit:
                ttl = await self.redis.ttl(key)
                return False, {
                    "limit": self.limit,
                    "remaining": 0,
                    "reset": int(datetime.now().timestamp() + ttl),
                }

            await self.redis.incr(key)
            ttl = await self.redis.ttl(key)

            return True, {
                "limit": self.limit,
                "remaining": self.limit - current - 1,
                "reset": int(datetime.now().timestamp() + ttl),
            }

        except Exception as e:
            logger.error("Redis rate limit check failed: %s", e)
            return None, {}


async def cleanup_rate_limiter_task(
    limiter: InMemoryRateLimiter, interval: int = 3600
):
    """
    Background task to periodically clean up old rate limiter entries.
    """
    while True:
        try:
            await asyncio.sleep(interval)
            await limiter.cleanup_old_entries()
            stats = limiter.get_stats()
            logger.info("Rate limiter stats: %s", stats)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Error in rate limiter cleanup: %s", e)
