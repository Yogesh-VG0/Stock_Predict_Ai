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
from ml_backend.config.constants import TOP_100_TICKERS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class PipelineHealthSummary:
    """Tracks what happened during the pipeline run so we can print a
    clear summary at the end — turning 'it didn't crash' into
    'we know exactly what happened.'"""

    def __init__(self):
        self.data_fetched: dict = {}          # ticker -> row_count
        self.data_failed: list = []
        self.training_tickers: int = 0
        self.training_status: str = "skipped"
        self.backtest_status: str = "skipped"
        self.predictions_stored: int = 0
        self.predictions_failed: int = 0
        self.predictions_skipped: int = 0
        self.seeking_alpha_status: str = "SKIPPED (not in pipeline)"
        self.gemini_explanations: str = "not run (separate script)"
        self.evaluation_samples: int = 0
        self.mongo_connected: bool = False
        self.horizons_used: list = []

    def print_summary(self):
        print("\n" + "=" * 56)
        print("  PIPELINE HEALTH SUMMARY")
        print("=" * 56)
        print(f"  MongoDB connected:        {'YES' if self.mongo_connected else 'NO (yfinance fallback)'}")
        print(f"  Historical data fetched:  {len(self.data_fetched)} tickers")
        if self.data_failed:
            print(f"  Data fetch failures:      {len(self.data_failed)} ({', '.join(self.data_failed[:5])}{'...' if len(self.data_failed) > 5 else ''})")
        print(f"  Training status:          {self.training_status} ({self.training_tickers} tickers)")
        print(f"  Backtest status:          {self.backtest_status}")
        stored_horizons = f" x {len(self.horizons_used)} horizons" if self.horizons_used else ""
        print(f"  Predictions stored:       {self.predictions_stored} tickers{stored_horizons}")
        if self.predictions_failed:
            print(f"  Predictions failed:       {self.predictions_failed}")
        if self.predictions_skipped:
            print(f"  Predictions skipped:      {self.predictions_skipped} (insufficient data)")
        print(f"  SeekingAlpha scraped:     {self.seeking_alpha_status}")
        print(f"  Gemini explanations:      {self.gemini_explanations}")
        print(f"  Evaluation samples found: {self.evaluation_samples} (expected 0 early)")
        print("=" * 56 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run StockPredict ML Pipeline")
    parser.add_argument("--tickers", nargs="+", default=None, help="Tickers to process (for prediction)")
    parser.add_argument("--horizon", default="next_day", choices=["next_day", "7_day", "30_day"])
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--no-mongo", action="store_true", help="Use yfinance instead of MongoDB")
    parser.add_argument("--retrain", action="store_true", help="Force retrain models")
    parser.add_argument("--predict-only", action="store_true",
                        help="Skip training; load saved models and only generate predictions")
    parser.add_argument("--all-tickers", action="store_true",
                        help="Train on ALL 100 tickers (pooled model uses full cross-section)")
    parser.add_argument("--no-predict", action="store_true",
                        help="Train only — skip prediction storage (used for training step)")
    args = parser.parse_args()

    health = PipelineHealthSummary()
    tickers = args.tickers or DEFAULT_BACKTEST_TICKERS
    # When --all-tickers is set, training uses all 100 tickers but prediction
    # uses only --tickers (or defaults).  This ensures the pooled model is
    # trained on the full S&P 100 cross-section instead of a small batch.
    train_tickers = list(TOP_100_TICKERS) if args.all_tickers else tickers
    mongo_client = None
    
    # 1. Setup MongoDB or Fallback
    if not args.no_mongo:
        try:
            mongo_client = MongoDBClient()
            if mongo_client.db is None:
                 logger.warning("MongoDB connection failed. Falling back to yfinance (use --no-mongo to suppress this).")
                 args.no_mongo = True
            else:
                 health.mongo_connected = True
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

    # Determine which tickers to fetch data for.
    # For training we need ALL train_tickers; for predict-only we only need --tickers.
    fetch_tickers = list(set(train_tickers) | set(tickers)) if not args.predict_only else tickers

    historical_data = {}
    spy_data = None

    if not args.no_mongo and mongo_client:
        for t in fetch_tickers:
            try:
                df = mongo_client.get_historical_data(t, start_date, end_date)
                if df is not None and not df.empty:
                    historical_data[t] = df
                    health.data_fetched[t] = len(df)
            except Exception as e:
                logger.warning("Could not fetch %s from Mongo: %s", t, e)
                health.data_failed.append(t)
        try:
            spy_data = mongo_client.get_historical_data("SPY", start_date, end_date)
        except Exception:
            pass
    
    # Backfill any missing tickers via yfinance (covers --no-mongo, partial
    # MongoDB coverage, and the common case where only a few tickers have
    # been ingested but --all-tickers requests the full 100 cross-section).
    missing_tickers = [t for t in fetch_tickers if t not in historical_data]
    if missing_tickers or not historical_data:
        import yfinance as yf
        logger.info("Fetching %d tickers from yfinance...", len(missing_tickers) if missing_tickers else len(fetch_tickers))
        dl_tickers = missing_tickers if missing_tickers else fetch_tickers
        for t in dl_tickers:
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
                         health.data_fetched[t] = len(df)
                    else:
                        logger.warning(f"Missing columns for {t}: {df.columns}")
                        health.data_failed.append(t)
            except Exception as e:
                logger.warning("Could not download %s: %s", t, e)
                health.data_failed.append(t)

    # Ensure SPY is available (for relative-strength features)
    if spy_data is None or (hasattr(spy_data, 'empty') and spy_data.empty):
        import yfinance as yf
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

    # 4. Train or Load Models
    if args.predict_only:
        # --predict-only: skip training, load models from disk
        logger.info("Predict-only mode: loading saved models...")
        predictor.load_models()
        if not predictor.pooled_models and not predictor.models:
            logger.error("No saved models found. Run training first (without --predict-only).")
            health.training_status = "FAILED (no saved models)"
            health.print_summary()
            sys.exit(1)
        logger.info("Loaded %d pooled + %d per-ticker models.",
                     len(predictor.pooled_models), len(predictor.models))
        health.training_status = "loaded from disk"
        health.training_tickers = len(predictor.pooled_models) + len(predictor.models)
    else:
        # Build training data dict — may be larger than prediction tickers
        # when --all-tickers is used.
        training_data = {t: historical_data[t] for t in train_tickers if t in historical_data}
        if not training_data:
            logger.error("No training data available. Exiting.")
            health.training_status = "FAILED (no data)"
            health.print_summary()
            sys.exit(1)
        health.training_tickers = len(training_data)
        logger.info("Training on %d tickers (pooled cross-section)...", len(training_data))
        try:
            predictor.train_all_models(training_data)
            logger.info("Training completed.")
            health.training_status = "completed"
        except Exception as e:
            logger.error("Training failed: %s", e)
            health.training_status = f"FAILED ({e})"
            if not predictor.models and not predictor.pooled_models:
                health.print_summary()
                sys.exit(1)

        # 5. Run Backtest (OOS only — restricted to dates after training cutoff)
        # Use only the prediction tickers for backtest (not all 100).
        backtest_data = {t: historical_data[t] for t in tickers if t in historical_data}
        logger.info("Running backtest...")
        oos_start = predictor.get_oos_start_date()
        if oos_start:
            logger.info("OOS backtest start date: %s", oos_start.date())
        else:
            logger.warning("No OOS start date available — backtest will use last 20%% fallback.")
        result = run_backtest(
            predictor=predictor,
            historical_data=backtest_data,
            spy_data=spy_data,
            tickers=list(backtest_data.keys()),
            start_date=args.start,
            end_date=args.end,
            max_positions=5,
            horizon=args.horizon,
            oos_start_date=oos_start,
        )

        if "error" in result:
            logger.error("Backtest failed: %s", result["error"])
            health.backtest_status = f"FAILED ({result['error']})"
            health.print_summary()
            sys.exit(1)

        health.backtest_status = "completed"
        health.evaluation_samples = result.get("trade_stats", {}).get("n_trades", 0)
        print("\n=== Pipeline Results (OOS) ===")
        print(f"OOS start: {result.get('oos_start', 'N/A')}")
        print(f"Period: {result.get('start_date')} to {result.get('end_date')}")
        print(f"Horizon: {args.horizon}")
        print(f"Strategy return: {result.get('total_return', 0):.2%}")
        print(f"Strategy Sharpe: {result.get('sharpe_ratio', 0):.3f}")
        print(f"Max drawdown: {result.get('max_drawdown', 0):.2%}")
        if result.get("spy_return") is not None:
            print(f"SPY return: {result['spy_return']:.2%}")
        ts = result.get("trade_stats", {})
        if ts.get("n_trades", 0) > 0:
            print(f"Trades: {ts['n_trades']} ({ts.get('trades_per_year', 0):.0f}/yr)")
            print(f"Avg return/trade: {ts.get('avg_return_per_trade', 0):.4f}")
            print(f"Win rate: {ts.get('win_rate', 0):.1%}")
            print(f"Avg holding: {ts.get('avg_holding_bars', 0):.1f} bars")
        else:
            print("Trades: 0 (check filters)")
        print("==============================\n")

    # 6. Generate and Store Live Predictions (skip with --no-predict)
    if args.no_predict:
        logger.info("--no-predict: skipping prediction storage.")
    elif not args.no_mongo and mongo_client:
        logger.info("Generating and storing live predictions...")
        horizons_seen = set()
        for ticker in tickers:
            try:
                df = historical_data.get(ticker)
                if df is None or df.empty:
                    health.predictions_skipped += 1
                    continue
                
                if len(df) < 100:
                    logger.warning(f"Not enough data for {ticker} predictions (n={len(df)})")
                    health.predictions_skipped += 1
                    continue

                preds = predictor.predict_all_windows(ticker, df)
                
                if preds:
                    horizons_seen.update(preds.keys())
                    success = mongo_client.store_predictions(ticker, preds)
                    if success:
                        health.predictions_stored += 1
                        logger.info(f"Stored predictions for {ticker}")
                    else:
                        health.predictions_failed += 1
                        logger.error(f"Failed to store predictions for {ticker}")
                else:
                    health.predictions_failed += 1
            except Exception as e:
                health.predictions_failed += 1
                logger.error(f"Error generating/storing predictions for {ticker}: {e}")
        health.horizons_used = sorted(horizons_seen)

    # Check for SeekingAlpha deps
    try:
        import playwright  # noqa: F401
        health.seeking_alpha_status = "available (deps installed)"
    except ImportError:
        health.seeking_alpha_status = "SKIPPED (playwright not installed)"

    health.print_summary()
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
