"""
In-memory rate limiter fallback for when Redis is unavailable.
"""

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import os
import time

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

    # Trusted reverse-proxy CIDRs. Only trust X-Forwarded-For when the direct
    # client is one of these.  Override via TRUSTED_PROXY_IPS env var (comma-
    # separated).  When running behind Vercel/Cloudflare the edge proxy IP
    # will be the direct client, so add it here.
    _TRUSTED_PROXIES: set = set()

    # How often (seconds) to retry Redis after a fallback switch
    _REDIS_RETRY_INTERVAL = 60

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
        self._last_redis_retry: float = 0

        # Populate trusted proxies from env
        raw = os.getenv("TRUSTED_PROXY_IPS", "")
        if raw:
            self._TRUSTED_PROXIES = {ip.strip() for ip in raw.split(",") if ip.strip()}

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

        # Periodically retry Redis when in fallback mode
        if self.using_fallback and self.redis:
            now = time.monotonic()
            if now - self._last_redis_retry > self._REDIS_RETRY_INTERVAL:
                self._last_redis_retry = now
                try:
                    await self.redis.ping()
                    self.using_fallback = False
                    logger.info("Redis recovered — switching back from in-memory fallback")
                except Exception:
                    pass  # still down

        # Try Redis first, fallback to in-memory
        if self.redis and not self.using_fallback:
            allowed, info = await self._check_redis(client_ip)
            if allowed is None:
                self.using_fallback = True
                self._last_redis_retry = time.monotonic()
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
        """Extract client IP, only trusting proxy headers from known proxies."""
        direct_ip = request.client.host if request.client else "unknown"

        # Only trust forwarded headers when the direct connection comes from
        # a known reverse proxy (Vercel, Cloudflare, etc.)
        if self._TRUSTED_PROXIES and direct_ip in self._TRUSTED_PROXIES:
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                # First entry is the original client
                return forwarded.split(",")[0].strip()
            real_ip = request.headers.get("X-Real-IP")
            if real_ip:
                return real_ip

        return direct_ip

    async def _check_redis(self, client_ip: str) -> Tuple[Optional[bool], Dict]:
        """
        Check rate limit using Redis with atomic Lua script.
        Returns (None, {}) if Redis fails.
        """
        try:
            key = f"rate_limit:{client_ip}"
            # Atomic increment-and-check via Lua script:
            #   KEYS[1] = rate limit key
            #   ARGV[1] = limit, ARGV[2] = window (seconds)
            # Returns: {current_count, ttl}
            lua = """
            local key = KEYS[1]
            local limit = tonumber(ARGV[1])
            local window = tonumber(ARGV[2])
            local current = redis.call('INCR', key)
            if current == 1 then
                redis.call('EXPIRE', key, window)
            end
            local ttl = redis.call('TTL', key)
            return {current, ttl}
            """
            result = await self.redis.eval(lua, 1, key, self.limit, self.window)
            current = int(result[0])
            ttl = max(int(result[1]), 0)

            allowed = current <= self.limit
            remaining = max(0, self.limit - current)
            reset_time = int(datetime.now().timestamp()) + ttl

            return allowed, {
                "limit": self.limit,
                "remaining": remaining,
                "reset": reset_time,
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
