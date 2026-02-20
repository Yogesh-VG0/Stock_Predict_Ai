"""
Centralized rate-limiter utilities for external API calls.

Provides two classes:
  - AsyncRateLimiter  : token-bucket with optional burst cap (async)
  - DailyBudgetLimiter: hard daily cap that resets at midnight UTC (async)

Usage:
    from ml_backend.utils.rate_limiter import finnhub_limiter, marketaux_limiter
    await finnhub_limiter.acquire()      # blocks until a token is available
    await marketaux_limiter.acquire()    # raises BudgetExhausted when daily cap hit
"""

import asyncio
import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token-bucket rate limiter (async)
# ---------------------------------------------------------------------------

class AsyncRateLimiter:
    """Sliding-window token-bucket rate limiter.

    Parameters
    ----------
    max_calls : int
        Maximum calls allowed within *period_seconds*.
    period_seconds : float
        Length of the rolling window in seconds.
    burst_limit : int | None
        If set, also enforces a per-second burst cap.
    name : str
        Label used in log messages.
    """

    def __init__(
        self,
        max_calls: int,
        period_seconds: float,
        burst_limit: int | None = None,
        name: str = "RateLimiter",
    ):
        self.max_calls = max_calls
        self.period = period_seconds
        self.burst_limit = burst_limit
        self.name = name
        self._lock = asyncio.Lock()
        self._timestamps: list[float] = []
        self._burst_timestamps: list[float] = []

    async def acquire(self) -> None:
        """Wait until a call is allowed, then record it."""
        while True:
            async with self._lock:
                now = time.monotonic()

                # Evict expired timestamps from rolling window
                cutoff = now - self.period
                self._timestamps = [t for t in self._timestamps if t > cutoff]

                # Also evict burst-window timestamps (last 1 second)
                if self.burst_limit is not None:
                    self._burst_timestamps = [
                        t for t in self._burst_timestamps if t > now - 1.0
                    ]

                # Check rolling-window capacity
                if len(self._timestamps) >= self.max_calls:
                    wait = self._timestamps[0] - cutoff + 0.05
                    logger.debug(
                        "[%s] rate limit reached (%d/%d), sleeping %.2fs",
                        self.name, len(self._timestamps), self.max_calls, wait,
                    )
                    # Release lock while sleeping
                    await asyncio.sleep(0)
                    # Will retry from the top of while loop
                    continue

                # Check burst capacity
                if (
                    self.burst_limit is not None
                    and len(self._burst_timestamps) >= self.burst_limit
                ):
                    wait = self._burst_timestamps[0] - (now - 1.0) + 0.05
                    logger.debug(
                        "[%s] burst limit reached (%d/%d/sec), sleeping %.2fs",
                        self.name,
                        len(self._burst_timestamps),
                        self.burst_limit,
                        wait,
                    )
                    await asyncio.sleep(0)
                    continue

                # Token available — record and return
                self._timestamps.append(now)
                if self.burst_limit is not None:
                    self._burst_timestamps.append(now)
                return

            # Small sleep between retries (outside lock)
            await asyncio.sleep(0.1)

    def acquire_sync(self) -> None:
        """Blocking version for synchronous callers (e.g. FRED)."""
        while True:
            now = time.monotonic()
            cutoff = now - self.period
            self._timestamps = [t for t in self._timestamps if t > cutoff]

            if len(self._timestamps) >= self.max_calls:
                wait = self._timestamps[0] - cutoff + 0.05
                logger.debug(
                    "[%s] sync rate limit, sleeping %.2fs", self.name, wait,
                )
                time.sleep(wait)
                continue

            self._timestamps.append(now)
            return


# ---------------------------------------------------------------------------
# Daily budget limiter (async) — hard cap that resets at midnight UTC
# ---------------------------------------------------------------------------

class BudgetExhausted(Exception):
    """Raised when a DailyBudgetLimiter has no remaining budget."""
    pass


class DailyBudgetLimiter:
    """Hard daily call budget that resets at midnight UTC.

    Parameters
    ----------
    daily_limit : int
        Maximum calls per UTC day.
    name : str
        Label used in log messages.
    """

    def __init__(self, daily_limit: int, name: str = "DailyBudget"):
        self.daily_limit = daily_limit
        self.name = name
        self._lock = asyncio.Lock()
        self._count = 0
        self._day: str = ""

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    async def acquire(self) -> None:
        """Consume one unit of daily budget.  Raises BudgetExhausted if none left."""
        async with self._lock:
            today = self._today()
            if today != self._day:
                # New day — reset counter
                self._day = today
                self._count = 0

            if self._count >= self.daily_limit:
                logger.warning(
                    "[%s] daily budget exhausted (%d/%d) — skipping",
                    self.name, self._count, self.daily_limit,
                )
                raise BudgetExhausted(
                    f"{self.name}: {self._count}/{self.daily_limit} calls used today"
                )

            self._count += 1
            remaining = self.daily_limit - self._count
            if remaining <= 10:
                logger.warning(
                    "[%s-BUDGET] %d/%d used, %d remaining",
                    self.name, self._count, self.daily_limit, remaining,
                )
            return

    @property
    def remaining(self) -> int:
        today = self._today()
        if today != self._day:
            return self.daily_limit
        return max(0, self.daily_limit - self._count)


# ---------------------------------------------------------------------------
# Global singleton instances — import these in calling modules
# ---------------------------------------------------------------------------

# Finnhub: 60 calls/min overall, 30 calls/sec global cap
# Use 55/min and 25/sec for safety margin
finnhub_limiter = AsyncRateLimiter(
    max_calls=55, period_seconds=60, burst_limit=25, name="Finnhub"
)

# FMP: 4 requests/sec on free tier
# Use 3/sec for margin
fmp_limiter = AsyncRateLimiter(
    max_calls=3, period_seconds=1, name="FMP"
)

# Marketaux: 100 requests/day — use 95 for 5% margin
marketaux_limiter = DailyBudgetLimiter(daily_limit=95, name="Marketaux")

# Reddit: 100 QPM per OAuth client (10-min rolling window)
# Use 90/min for margin
reddit_limiter = AsyncRateLimiter(
    max_calls=90, period_seconds=60, name="Reddit"
)

# FRED: ~120 requests/min — use 100/min for margin
fred_limiter = AsyncRateLimiter(
    max_calls=100, period_seconds=60, name="FRED"
)
