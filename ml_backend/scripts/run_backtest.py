#!/usr/bin/env python
"""
Run backtest for StockPredict strategy.

Usage (from repo root):
  python -m ml_backend.scripts.run_backtest
  python -m ml_backend.scripts.run_backtest --tickers AAPL MSFT NVDA --horizon 7_day

Requires: MongoDB with historical_data, or pass --no-mongo to use yfinance (slower).
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta

import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(__file__).rsplit("scripts", 1)[0].rsplit("ml_backend", 1)[0])

from ml_backend.backtest import run_backtest, DEFAULT_BACKTEST_TICKERS
from ml_backend.models.predictor import StockPredictor
from ml_backend.data.features_minimal import MinimalFeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run StockPredict backtest")
    parser.add_argument("--tickers", nargs="+", default=None, help="Tickers to backtest")
    parser.add_argument("--horizon", default="next_day", choices=["next_day", "7_day", "30_day"])
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--no-mongo", action="store_true", help="Use yfinance instead of MongoDB")
    args = parser.parse_args()

    tickers = args.tickers or DEFAULT_BACKTEST_TICKERS
    mongo_client = None
    if not args.no_mongo:
        try:
            from ml_backend.utils.mongodb import MongoDBClient
            mongo_client = MongoDBClient()
        except Exception as e:
            logger.warning("MongoDB unavailable: %s. Use --no-mongo for yfinance.", e)
            mongo_client = None

    predictor = StockPredictor(mongo_client)
    predictor.set_feature_engineer(MinimalFeatureEngineer(mongo_client))
    predictor.load_models()

    if not predictor.pooled_models and not predictor.models:
        logger.error("No models loaded. Train models first.")
        sys.exit(1)

    # Fetch historical data
    end = datetime.utcnow()
    start = end - timedelta(days=365 * 2)
    historical_data = {}
    spy_data = None

    if mongo_client:
        for t in tickers:
            try:
                df = mongo_client.get_historical_data(t, start, end)
                if df is not None and not df.empty:
                    historical_data[t] = df
            except Exception as e:
                logger.warning("Could not fetch %s: %s", t, e)
        try:
            spy_data = mongo_client.get_historical_data("SPY", start, end)
        except Exception:
            pass
    else:
        import yfinance as yf
        for t in tickers:
            try:
                df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
                if df is not None and not df.empty:
                    df = df.reset_index()
                    if "Date" in df.columns and "date" not in df.columns:
                        df = df.rename(columns={"Date": "date"})
                    historical_data[t] = df
            except Exception as e:
                logger.warning("Could not fetch %s: %s", t, e)
        try:
            spy_df = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
            if spy_df is not None and not spy_df.empty:
                spy_df = spy_df.reset_index()
                if "Date" in spy_df.columns:
                    spy_df = spy_df.rename(columns={"Date": "date"})
                spy_data = spy_df
        except Exception as e:
            logger.warning("Could not fetch SPY: %s", e)

    if not historical_data:
        logger.error("No historical data. Check MongoDB or use --no-mongo.")
        sys.exit(1)

    result = run_backtest(
        predictor=predictor,
        historical_data=historical_data,
        spy_data=spy_data,
        tickers=tickers,
        start_date=args.start,
        end_date=args.end,
        max_positions=args.max_positions,
        horizon=args.horizon,
        oos_start_date=predictor.get_oos_start_date(),
    )

    if "error" in result:
        logger.error("Backtest failed: %s", result["error"])
        sys.exit(1)

    print("\n=== Backtest Results ===")
    print(f"Period: {result['start_date']} to {result['end_date']}")
    print(f"Horizon: {args.horizon}")
    print(f"Strategy return: {result['total_return']:.2%}")
    print(f"Strategy Sharpe: {result['sharpe_ratio']:.3f}")
    print(f"Max drawdown: {result['max_drawdown']:.2%}")
    print(f"Trades: {result['n_trades']}")
    if result.get("spy_return") is not None:
        print(f"SPY return: {result['spy_return']:.2%}")
        print(f"SPY Sharpe: {result['spy_sharpe']:.3f}")
    print()


if __name__ == "__main__":
    main()
