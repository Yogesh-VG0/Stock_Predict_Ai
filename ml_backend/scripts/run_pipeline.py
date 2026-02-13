#!/usr/bin/env python
"""
Run the full ML pipeline:
1. Fetch historical data (MongoDB or yfinance).
2. Train models (StockPredictor).
3. Run backtest (backtest.py).

Usage:
  python -m ml_backend.scripts.run_pipeline --tickers AAPL MSFT --no-mongo
"""

import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(__file__).rsplit("scripts", 1)[0].rsplit("ml_backend", 1)[0])

from ml_backend.backtest import run_backtest, DEFAULT_BACKTEST_TICKERS
from ml_backend.models.predictor import StockPredictor
from ml_backend.data.features_minimal import MinimalFeatureEngineer
from ml_backend.utils.mongodb import MongoDBClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run StockPredict ML Pipeline")
    parser.add_argument("--tickers", nargs="+", default=None, help="Tickers to process")
    parser.add_argument("--horizon", default="next_day", choices=["next_day", "7_day", "30_day"])
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--no-mongo", action="store_true", help="Use yfinance instead of MongoDB")
    parser.add_argument("--retrain", action="store_true", help="Force retrain models")
    args = parser.parse_args()

    tickers = args.tickers or DEFAULT_BACKTEST_TICKERS
    mongo_client = None
    
    # 1. Setup MongoDB or Fallback
    if not args.no_mongo:
        try:
            mongo_client = MongoDBClient()
            if mongo_client.db is None:
                 logger.warning("MongoDB connection failed. Falling back to yfinance (use --no-mongo to suppress this).")
                 args.no_mongo = True
        except Exception as e:
            logger.warning("MongoDB unavailable: %s. Falling back to yfinance.", e)
            args.no_mongo = True
            
    # 2. Initialize Components
    predictor = StockPredictor(mongo_client)
    feature_engineer = MinimalFeatureEngineer(mongo_client)
    predictor.set_feature_engineer(feature_engineer)

    # 3. Fetch Data
    logger.info("Fetching historical data...")
    end_date = datetime.utcnow()
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
        
    start_date = end_date - timedelta(days=365 * 2) # Default 2 years
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")

    historical_data = {}
    spy_data = None

    if not args.no_mongo and mongo_client:
        for t in tickers:
            try:
                df = mongo_client.get_historical_data(t, start_date, end_date)
                if df is not None and not df.empty:
                    historical_data[t] = df
            except Exception as e:
                logger.warning("Could not fetch %s from Mongo: %s", t, e)
        try:
            spy_data = mongo_client.get_historical_data("SPY", start_date, end_date)
        except Exception:
            pass
    
    if not historical_data: # Fallback to yfinance if Mongo empty or disabled
        logger.info("Using yfinance to fetch data...")
        import yfinance as yf
        
        # Download tickers
        for t in tickers:
            try:
                logger.info(f"Downloading {t}...")
                df = yf.download(t, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if df is not None and not df.empty:
                    # Fix columns for yfinance v0.2+
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                        
                    df = df.reset_index()
                    if "Date" in df.columns:
                        df = df.rename(columns={"Date": "date"})
                    
                    # Ensure required columns exist
                    required_cols = ["Open", "High", "Low", "Close", "Volume"]
                    if all(col in df.columns for col in required_cols):
                         historical_data[t] = df
                    else:
                        logger.warning(f"Missing columns for {t}: {df.columns}")
            except Exception as e:
                logger.warning("Could not download %s: %s", t, e)

        # Download SPY
        try:
             spy_df = yf.download("SPY", start=start_date, end=end_date, progress=False, auto_adjust=True)
             if spy_df is not None and not spy_df.empty:
                if isinstance(spy_df.columns, pd.MultiIndex):
                    spy_df.columns = spy_df.columns.get_level_values(0)
                spy_df = spy_df.reset_index()
                if "Date" in spy_df.columns:
                    spy_df = spy_df.rename(columns={"Date": "date"})
                spy_data = spy_df
        except Exception as e:
            logger.warning("Could not download SPY: %s", e)

    if not historical_data:
        logger.error("No historical data available. Exiting.")
        sys.exit(1)

    if spy_data is not None and not spy_data.empty:
         # Inject SPY data into feature engineer cache to ensure it's used
         # Use a wide date range key to ensure it matches specific lookups
         # Warning: this relies on the cache key format in cache_fetch.py
         try:
             # Manually populate cache if possible, or just rely on global cache if implemented
             # MinimalFeatureEngineer instance has its own cache
             if hasattr(feature_engineer, 'price_cache'):
                 # We need to know the start/end date used by _add_relative_strength
                 # But since we don't know exact dates, we can't easily pre-populate the key-based cache 
                 # unless we overwrite the fetch method or use a very broad key?
                 # Actually, better to just let cache_fetch handle it now that it's fixed.
                 pass
         except Exception as e:
            logger.warning(f"Could not inject SPY data: {e}")

    # 4. Train Models
    logger.info("Training models...")
    # We can pass the historical data directly to train_all_models which might expect a slightly different format 
    # or we can train one by one. StockPredictor.train_all_models takes a dict of DataFrames.
    try:
        # StockPredictor.train_all_models will try to fetch SPY data internally via feature_engineer
        # Since we fixed cache_fetch.py, it should fallback to yfinance correctly now.
        predictor.train_all_models(historical_data)
        logger.info("Training completed.")
    except Exception as e:
        logger.error("Training failed: %s", e)
        # Continue if some models trained? 
        # If train_all_models fails completely, we can't backtest.
        if not predictor.models and not predictor.pooled_models:
             sys.exit(1)

    # 5. Run Backtest
    logger.info("Running backtest...")
    result = run_backtest(
        predictor=predictor,
        historical_data=historical_data,
        spy_data=spy_data,
        tickers=list(historical_data.keys()),
        start_date=args.start,
        end_date=args.end,
        max_positions=5,
        horizon=args.horizon,
    )

    if "error" in result:
        logger.error("Backtest failed: %s", result["error"])
        sys.exit(1)

    print("\n=== Pipeline Results ===")
    print(f"Period: {result.get('start_date')} to {result.get('end_date')}")
    print(f"Horizon: {args.horizon}")
    print(f"Strategy return: {result.get('total_return', 0):.2%}")
    print(f"Strategy Sharpe: {result.get('sharpe_ratio', 0):.3f}")
    if result.get("spy_return") is not None:
        print(f"SPY return: {result['spy_return']:.2%}")
    print("========================\n")

    # 6. Save Models
    if args.retrain:
        logger.info("Saving models...")
        predictor.save_models()

    # 7. Generate and Store Live Predictions
    if not args.no_mongo and mongo_client:
        logger.info("Generating and storing live predictions...")
        for ticker in tickers:
            try:
                df = historical_data.get(ticker)
                if df is None or df.empty:
                    continue
                
                if len(df) < 100:
                    logger.warning(f"Not enough data for {ticker} predictions (n={len(df)})")
                    continue

                preds = predictor.predict_all_windows(ticker, df)
                
                if preds:
                    success = mongo_client.store_predictions(ticker, preds)
                    if success:
                        logger.info(f"Stored predictions for {ticker}")
                    else:
                        logger.error(f"Failed to store predictions for {ticker}")
            except Exception as e:
                logger.error(f"Error generating/storing predictions for {ticker}: {e}")

    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
