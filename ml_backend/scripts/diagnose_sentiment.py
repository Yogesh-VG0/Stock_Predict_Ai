#!/usr/bin/env python
"""
Diagnose sentiment feature pipeline  -  verify non-zero values end-to-end.

Checks:
  1. MongoDB sentiment collection has docs for test tickers
  2. get_sentiment_timeseries() returns rows with non-zero composite scores
  3. make_sentiment_features() produces non-zero rolling features
  4. prepare_features() includes sentiment columns in the final feature matrix

Usage:
  python -m ml_backend.scripts.diagnose_sentiment
"""

import logging
import sys
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(__file__).rsplit("scripts", 1)[0].rsplit("ml_backend", 1)[0])

from ml_backend.utils.mongodb import MongoDBClient
from ml_backend.data.features_minimal import MinimalFeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TEST_TICKERS = ["AAPL", "NVDA", "TSLA"]
LOOKBACK_DAYS = 180


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _get_price_df(mongo_client, ticker: str) -> pd.DataFrame:
    """Get price data  -  try MongoDB first, fall back to yfinance."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=LOOKBACK_DAYS + 60)
    end = now

    df = mongo_client.get_historical_data(ticker, start, end)
    if df is not None and not df.empty:
        return df

    # Fallback to yfinance
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "date"})
            print(f"  (price data from yfinance, {len(df)} rows)")
            return df
    except Exception as e:
        print(f"  yfinance fallback failed: {e}")
    return None


def diagnose_mongo_raw(mongo_client, ticker: str) -> dict:
    """Stage 1: raw Mongo sentiment docs."""
    _section(f"Stage 1  -  Raw MongoDB docs for {ticker}")

    # Check both possible collection names
    for coll_name in ["sentiment", "sentiment_data"]:
        coll = mongo_client.db[coll_name]
        total = coll.count_documents({"ticker": ticker})
        if total > 0:
            print(f"  Collection '{coll_name}': {total} docs for {ticker}")
        else:
            print(f"  Collection '{coll_name}': 0 docs")

    # Use the one that get_sentiment_timeseries uses
    coll = mongo_client.db["sentiment"]
    total = coll.count_documents({"ticker": ticker})
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=LOOKBACK_DAYS)
    recent = coll.count_documents({"ticker": ticker, "date": {"$gte": cutoff}})
    print(f"  Docs in last {LOOKBACK_DAYS}d (used by ML): {recent}")

    # Sample one doc
    sample = coll.find_one({"ticker": ticker}, sort=[("date", -1)])
    if sample:
        print(f"  Latest doc date: {sample.get('date')}")
        sent_fields = {k: v for k, v in sample.items()
                       if k.endswith("_sentiment") and isinstance(v, (int, float))}
        print(f"  Sentiment fields: {sent_fields}")
        count_fields = {k: v for k, v in sample.items()
                        if ("count" in k.lower() or "volume" in k.lower() or "articles" in k.lower())
                        and isinstance(v, (int, float))}
        print(f"  Count fields: {count_fields}")
    else:
        print("  [WARN] NO docs found in 'sentiment' collection!")
        print("  -> Run sentiment_cron.py first to populate sentiment data.")

    return {"total": total, "recent": recent, "has_sample": sample is not None}


def diagnose_timeseries(mongo_client, ticker: str) -> dict:
    """Stage 2: get_sentiment_timeseries() output."""
    _section(f"Stage 2  -  get_sentiment_timeseries({ticker})")
    now = datetime.now(timezone.utc)
    end = now
    start = end - timedelta(days=LOOKBACK_DAYS)
    df = mongo_client.get_sentiment_timeseries(ticker, start, end)

    if df is None or df.empty:
        print("  [WARN] EMPTY DataFrame returned  -  sentiment features will be all zeros!")
        print("  -> This is expected if sentiment_cron.py has never been run.")
        return {"rows": 0, "pct_nonzero_sent": 0, "pct_nonzero_count": 0}

    print(f"  Rows returned: {len(df)}")
    print(f"  Date range: {df['date'].min()} -> {df['date'].max()}")
    print(f"  composite_sentiment  -  mean={df['composite_sentiment'].mean():.4f}, "
          f"std={df['composite_sentiment'].std():.4f}, "
          f"nonzero={(df['composite_sentiment'] != 0).mean():.1%}")
    print(f"  news_count  -  mean={df['news_count'].mean():.1f}, "
          f"max={df['news_count'].max()}, "
          f"nonzero={(df['news_count'] > 0).mean():.1%}")
    return {
        "rows": len(df),
        "pct_nonzero_sent": float((df["composite_sentiment"] != 0).mean()),
        "pct_nonzero_count": float((df["news_count"] > 0).mean()),
    }


def diagnose_features(mongo_client, ticker: str) -> dict:
    """Stage 3: make_sentiment_features() output."""
    _section(f"Stage 3  -  make_sentiment_features({ticker})")

    price_df = _get_price_df(mongo_client, ticker)
    if price_df is None or price_df.empty:
        print("  [WARN] No price data  -  cannot test feature alignment.")
        return {"sentiment_cols_nonzero": {}}

    from ml_backend.data.sentiment_features import make_sentiment_features

    sent_df = make_sentiment_features(price_df, ticker, mongo_client)
    print(f"  Shape: {sent_df.shape}")

    stats = {}
    for col in sent_df.columns:
        nz = (sent_df[col] != 0).sum()
        pct = nz / len(sent_df) if len(sent_df) > 0 else 0
        mn = sent_df[col].mean()
        stats[col] = {"nonzero_pct": pct, "mean": mn}
        marker = "[OK]" if pct > 0 else "[WARN] ALL ZEROS"
        print(f"  {col:20s}  nonzero={pct:.1%}  mean={mn:.6f}  {marker}")
    return {"sentiment_cols_nonzero": stats}


def diagnose_full_pipeline(mongo_client, ticker: str) -> dict:
    """Stage 4: prepare_features()  -  sentiment present in final matrix."""
    _section(f"Stage 4  -  Full prepare_features({ticker})")

    price_df = _get_price_df(mongo_client, ticker)
    if price_df is None or price_df.empty:
        print("  [WARN] No price data.")
        return {}

    fe = MinimalFeatureEngineer(mongo_client)
    features, meta = fe.prepare_features(price_df, ticker=ticker, mongo_client=mongo_client)
    if features is None:
        print("  [WARN] prepare_features returned None!")
        return {}

    cols = meta.get("feature_columns", [])
    sent_cols = [c for c in cols if c.startswith("sent_") or c.startswith("news_count") or c.startswith("news_spike")]
    print(f"  Total features: {len(cols)}")
    print(f"  Sentiment features in final matrix: {sent_cols}")

    if not sent_cols:
        print("  [WARN] NO sentiment columns in feature matrix!")
        return {"sent_in_matrix": False, "sent_cols": []}

    # Check values
    for col in sent_cols:
        idx = cols.index(col)
        vals = features[:, idx]
        nz = (vals != 0).sum()
        pct = nz / len(vals) if len(vals) > 0 else 0
        marker = "[OK]" if pct > 0 else "[WARN] ALL ZEROS (expected if no sentiment data yet)"
        print(f"  {col:20s}  nonzero={pct:.1%}  mean={vals.mean():.6f}  {marker}")

    return {"sent_in_matrix": True, "sent_cols": sent_cols}


def main():
    print("=" * 60)
    print("  SENTIMENT FEATURE DIAGNOSTIC")
    print("=" * 60)

    try:
        mongo_client = MongoDBClient()
        if mongo_client.db is None:
            print("\n[WARN] Cannot connect to MongoDB. Exiting.")
            sys.exit(1)
    except Exception as e:
        print(f"\n[WARN] MongoDB connection failed: {e}")
        sys.exit(1)

    summary = {}
    for ticker in TEST_TICKERS:
        print(f"\n{'#'*60}")
        print(f"  TICKER: {ticker}")
        print(f"{'#'*60}")
        s1 = diagnose_mongo_raw(mongo_client, ticker)
        s2 = diagnose_timeseries(mongo_client, ticker)
        s3 = diagnose_features(mongo_client, ticker)
        s4 = diagnose_full_pipeline(mongo_client, ticker)
        summary[ticker] = {"raw": s1, "timeseries": s2, "features": s3, "pipeline": s4}

    # Final verdict
    _section("VERDICT")
    # Minimum days of sentiment data required before rolling features can
    # produce non-zero values (due to shift(1) + min_periods in rolling windows).
    WARMUP_THRESHOLD = 3  # sent_mean_7d needs min_periods=3; _1d needs 2
    for ticker, s in summary.items():
        raw_ok = s["raw"]["recent"] > 0
        ts_ok = s["timeseries"]["rows"] > 0 and s["timeseries"]["pct_nonzero_sent"] > 0
        ts_rows = s["timeseries"]["rows"]
        feat_ok = any(
            v.get("nonzero_pct", 0) > 0
            for v in s["features"].get("sentiment_cols_nonzero", {}).values()
        )
        pipe_ok = s["pipeline"].get("sent_in_matrix", False)

        # Distinguish: PASS / WARMUP / NO DATA / FAIL
        if raw_ok and ts_ok and feat_ok and pipe_ok:
            status = "[OK] PASS"
        elif not raw_ok:
            status = "[FAIL] NO DATA (run sentiment_cron.py first)"
        elif raw_ok and ts_ok and not feat_ok and ts_rows < WARMUP_THRESHOLD:
            status = (f"[WARMUP] Data flowing ({ts_rows} day(s) stored) — "
                      f"features warm after {WARMUP_THRESHOLD}+ trading days")
        elif raw_ok and ts_ok and not feat_ok:
            status = "[FAIL] FAIL — data exists but features still zero (check alignment)"
        else:
            status = "[FAIL] FAIL"
        print(f"  {ticker}: {status}  (raw={raw_ok}, ts={ts_ok}, feat={feat_ok}, pipe={pipe_ok})")

    # Summary guidance
    any_raw = any(s["raw"]["recent"] > 0 for s in summary.values())
    if not any_raw:
        print("\n  ----------------------------------------------")
        print("  No sentiment data found in MongoDB.")
        print("  Action: run `python -m ml_backend.sentiment_cron` to populate data.")
        print("  Then re-run this diagnostic.")

    print()


if __name__ == "__main__":
    main()
