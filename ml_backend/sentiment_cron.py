#!/usr/bin/env python3
"""
Sentiment Pipeline Cron Job
Runs every 4 hours to fetch and store sentiment data for top stocks.

Uses SentimentAnalyzer.get_combined_sentiment() which:
  1. Fetches from all sources (RSS, Reddit, FinViz, FMP, Finnhub, etc.)
  2. Blends scores via weighted average
  3. Stores to MongoDB `sentiment` collection keyed on {ticker, date}

The ML feature engine reads from `sentiment` via get_sentiment_timeseries().

Architecture: asyncio.run() + Semaphore(CONCURRENCY) for bounded parallelism.
"""

import asyncio
import os
import sys
import logging
import time
from datetime import datetime, timezone

# Always run as a module: python -m ml_backend.sentiment_cron
from ml_backend.data.sentiment import SentimentAnalyzer
from ml_backend.utils.mongodb import MongoDBClient
from ml_backend.config.constants import TOP_100_TICKERS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Max concurrent tickers to avoid API rate-limits
CONCURRENCY = 3
# Per-ticker timeout (seconds) — prevents one hanging API from blocking the whole run
TICKER_TIMEOUT = 300


async def _process_ticker(
    sem: asyncio.Semaphore,
    analyzer: SentimentAnalyzer,
    ticker: str,
) -> bool:
    """Process sentiment for a single ticker (bounded by semaphore + timeout)."""
    async with sem:
        for attempt in range(1, 4):
            try:
                result = await asyncio.wait_for(
                    analyzer.get_combined_sentiment(ticker),
                    timeout=TICKER_TIMEOUT,
                )
                if result and result.get("ticker") == ticker:
                    blended = result.get("blended_sentiment", result.get("composite_sentiment", None))
                    logger.info(f"  {ticker}: blended_sentiment={blended}, date={result.get('date')} (Attempt {attempt})")
                    return True
                else:
                    logger.warning(f"  {ticker}: get_combined_sentiment returned empty/invalid (Attempt {attempt})")
                    if attempt == 3:
                        return False
            except asyncio.TimeoutError:
                logger.warning(f"  {ticker}: TIMED OUT after {TICKER_TIMEOUT}s (Attempt {attempt})")
            except Exception as e:
                logger.warning(f"  {ticker}: FAILED — {e} (Attempt {attempt})")
            
            if attempt < 3:
                await asyncio.sleep(5 * attempt)
        
        logger.error(f"  {ticker}: Failed all 3 attempts.")
        return False


async def _run_pipeline(tickers: list[str]) -> tuple[int, int]:
    """Run the full pipeline with bounded concurrency."""
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        raise ValueError("MONGODB_URI environment variable not set")

    mongo_client = MongoDBClient(mongodb_uri)
    analyzer = SentimentAnalyzer(mongo_client)
    sem = asyncio.Semaphore(CONCURRENCY)

    logger.info("Launching %d tickers with concurrency=%d", len(tickers), CONCURRENCY)

    results = await asyncio.gather(
        *[_process_ticker(sem, analyzer, t) for t in tickers],
        return_exceptions=True,
    )

    ok = sum(1 for r in results if r is True)
    fail = len(results) - ok

    # ── Health summary: query MongoDB by the trading-day date this run targeted ──
    try:
        from ml_backend.data.sentiment import get_previous_trading_day
        from ml_backend.config.constants import ARTICLE_COUNT_VOLUME_KEYS

        utc_now = datetime.now(timezone.utc)
        utc_today_str = utc_now.strftime("%Y-%m-%d")
        target_day_str = get_previous_trading_day(utc_today_str) or utc_today_str
        # Construct naive UTC midnight — exact same type/value as the upsert key
        _y, _m, _d = (int(x) for x in target_day_str.split("-"))
        target_day_midnight = datetime(_y, _m, _d, 0, 0, 0)

        coll = mongo_client.db["sentiment"]
        # Projection: pull article-count keys so we can inspect per-source
        proj = {"ticker": 1, "news_count": 1, "composite_sentiment": 1, "date": 1}
        for k in ARTICLE_COUNT_VOLUME_KEYS:
            proj[k] = 1
        today_docs = list(coll.find({"date": target_day_midnight}, proj))

        upserted = len(today_docs)
        zero_news = sum(1 for d in today_docs if d.get("news_count", 0) == 0)

        # Per-source breakdown: missing / zero / nonzero
        # Mirror the live ARTICLE_COUNT_VOLUME_KEYS (alphavantage removed)
        _src_labels = {
            "rss_news_volume": "RSS",
            "marketaux_volume": "Marketaux",
            "finviz_volume": "FinViz",
        }
        src_stats: dict[str, dict[str, int]] = {}
        for k, label in _src_labels.items():
            missing = sum(1 for d in today_docs if k not in d)
            zero = sum(1 for d in today_docs if k in d and d[k] == 0)
            nonzero = sum(1 for d in today_docs if k in d and isinstance(d[k], (int, float)) and d[k] > 0)
            src_stats[label] = {"missing": missing, "zero": zero, "nonzero": nonzero}

        logger.info("─── HEALTH CHECK ───")
        logger.info("  Target trading day : %s", target_day_str)
        logger.info("  Tickers processed  : %d", len(tickers))
        logger.info("  Docs upserted      : %d", upserted)
        logger.info("  news_count == 0    : %d  (possible API outage if high)", zero_news)
        for label, counts in src_stats.items():
            if counts["missing"] > 0 or counts["zero"] > 0:
                logger.info("    %-14s  nonzero=%d  zero=%d  missing=%d",
                            label, counts["nonzero"], counts["zero"], counts["missing"])

        # ── Coverage density metrics ──
        news_counts = [d.get("news_count", 0) for d in today_docs]
        comp_sents = [d.get("composite_sentiment", 0.0) for d in today_docs]
        if today_docs:
            import statistics
            tickers_with_news = sum(1 for nc in news_counts if nc >= 3)
            tickers_with_sentiment = sum(1 for cs in comp_sents if cs != 0.0)
            sentiment_coverage_pct = round(100 * tickers_with_sentiment / max(len(today_docs), 1), 1)
            news_coverage_pct = round(100 * tickers_with_news / max(len(today_docs), 1), 1)
            median_news = statistics.median(news_counts) if news_counts else 0
            p90_news = sorted(news_counts)[int(len(news_counts) * 0.9)] if news_counts else 0
            logger.info("  sentiment_coverage : %.1f%% tickers with non-zero composite", sentiment_coverage_pct)
            logger.info("  news_coverage      : %.1f%% tickers with news_count >= 3", news_coverage_pct)
            logger.info("  news_count         : median=%d  p90=%d", median_news, p90_news)

        # ── Insider transaction coverage ──
        try:
            insider_coll = mongo_client.db["insider_transactions"]
            insider_tickers_today = len(insider_coll.distinct("symbol"))
            total_insider_docs = insider_coll.count_documents({})
            logger.info("  insider_coverage   : %d tickers, %d total docs in collection",
                        insider_tickers_today, total_insider_docs)
        except Exception:
            pass

        if zero_news > len(tickers) * 0.5:
            logger.warning("  ⚠ >50%% of tickers have zero news — check API keys")
    except Exception as e:
        logger.warning("Health check query failed: %s", e)

    return ok, fail


def run_sentiment_pipeline() -> bool:
    """Main entry point — wraps the async pipeline."""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Starting sentiment pipeline at %s", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 60)

    tickers = list(TOP_100_TICKERS)
    logger.info("Processing %d tickers from TOP_100_TICKERS", len(tickers))

    try:
        ok, fail = asyncio.run(_run_pipeline(tickers))

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("SENTIMENT PIPELINE COMPLETED")
        logger.info("Successful: %d/%d tickers", ok, len(tickers))
        logger.info("Failed: %d/%d tickers", fail, len(tickers))
        logger.info("Total time: %.1f seconds", elapsed)
        logger.info("Completed at: %s", datetime.now(timezone.utc).isoformat())
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error("Pipeline failed with critical error: %s", e)
        return False


if __name__ == "__main__":
    success = run_sentiment_pipeline()
    sys.exit(0 if success else 1)