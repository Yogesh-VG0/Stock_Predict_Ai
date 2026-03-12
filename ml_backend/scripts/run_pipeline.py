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
from datetime import datetime, timedelta, timezone
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(__file__).rsplit("scripts", 1)[0].rsplit("ml_backend", 1)[0])

from ml_backend.backtest import run_backtest, DEFAULT_BACKTEST_TICKERS
from ml_backend.models.predictor import StockPredictor
from ml_backend.data.features_minimal import MinimalFeatureEngineer
from ml_backend.utils.mongodb import MongoDBClient
from ml_backend.config.constants import TOP_100_TICKERS
from ml_backend.config.feature_config_v1 import TARGET_CONFIG, USE_CROSS_SECTIONAL_RANKING, RANKING_CONFIG

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
        if self.predictions_stored == 0 and self.predictions_failed == 0 and self.predictions_skipped == 0:
            print(f"  Predictions stored:       -- (--no-predict mode)")
        else:
            print(f"  Predictions stored:       {self.predictions_stored} tickers{stored_horizons}")
        if self.predictions_failed:
            print(f"  Predictions failed:       {self.predictions_failed}")
        if self.predictions_skipped:
            print(f"  Predictions skipped:      {self.predictions_skipped} (insufficient data)")

        print(f"  Gemini explanations:      {self.gemini_explanations}")
        eval_note = " (none yet -- predictions need 1+ day to become evaluable)" if self.evaluation_samples == 0 else ""
        print(f"  Evaluation samples found: {self.evaluation_samples}{eval_note}")
        print("=" * 56 + "\n")

    def check_quality_gate(self) -> bool:
        """Return True if the pipeline run meets minimum quality thresholds.

        Thresholds are configurable via environment variables.
        """
        min_pred_rate = float(os.environ.get("QG_MIN_PREDICTION_RATE", "0.80"))
        max_data_fail_rate = float(os.environ.get("QG_MAX_DATA_FAILURE_RATE", "0.20"))

        total_pred = self.predictions_stored + self.predictions_failed + self.predictions_skipped
        total_data = len(self.data_fetched) + len(self.data_failed)
        reasons = []

        # Gate 1: prediction rate
        if total_pred > 0:
            pred_rate = self.predictions_stored / total_pred
            if pred_rate < min_pred_rate:
                reasons.append(
                    f"Prediction rate {pred_rate:.0%} < {min_pred_rate:.0%} "
                    f"({self.predictions_stored}/{total_pred} stored)"
                )
        elif total_data > 0:
            # Predictions were expected but none attempted
            reasons.append("No predictions attempted despite data being fetched")

        # Gate 2: data fetch failure rate
        if total_data > 0:
            data_fail_rate = len(self.data_failed) / total_data
            if data_fail_rate > max_data_fail_rate:
                reasons.append(
                    f"Data failure rate {data_fail_rate:.0%} > {max_data_fail_rate:.0%} "
                    f"({len(self.data_failed)}/{total_data} failed)"
                )

        if reasons:
            print("\n" + "!" * 56)
            print("  QUALITY GATE: FAILED")
            for r in reasons:
                print(f"  - {r}")
            print("!" * 56 + "\n")
            return False

        print("  QUALITY GATE: PASSED")
        return True


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
                        help="Train on ALL tickers (pooled model uses full cross-section)")
    parser.add_argument("--no-predict", action="store_true",
                        help="Train only — skip prediction storage (used for training step)")
    args = parser.parse_args()

    health = PipelineHealthSummary()
    tickers = args.tickers or DEFAULT_BACKTEST_TICKERS
    # When --all-tickers is set, training uses all tickers but prediction
    # uses only --tickers (or defaults).  This ensures the pooled model is
    # trained on the full cross-section instead of a small batch.
    train_tickers = list(TOP_100_TICKERS) if args.all_tickers else tickers
    # --predict-only --all-tickers: predict all tickers in one process
    # (loads models once instead of once per batch).
    if args.predict_only and args.all_tickers and not args.tickers:
        tickers = list(TOP_100_TICKERS)
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

    # --- Quota pre-check: run data retention BEFORE any heavy writes ----
    if not args.no_mongo and mongo_client and mongo_client.db is not None:
        try:
            stats = mongo_client.db.command("dbStats")
            data_mb = stats.get("dataSize", 0) / (1024 * 1024)
            storage_mb = stats.get("storageSize", 0) / (1024 * 1024)
            logger.info(f"MongoDB storage: dataSize={data_mb:.1f} MB, storageSize={storage_mb:.1f} MB")
            if data_mb > 400:  # >400 MB → proactively run retention cleanup
                logger.warning("Storage above 400 MB — running data retention cleanup before pipeline continues...")
                try:
                    from ml_backend.scripts.data_retention import run_retention
                    retention_results = run_retention(dry_run=False)
                    total_deleted = sum(v for v in retention_results.values() if v > 0)
                    logger.info(f"Data retention cleanup deleted {total_deleted} documents")
                except Exception as ret_exc:
                    logger.warning(f"Data retention cleanup failed: {ret_exc}")
        except Exception as e:
            logger.warning(f"Could not check MongoDB storage stats: {e}")
            
    # 2. Initialize Components
    predictor = StockPredictor(mongo_client)
    feature_engineer = MinimalFeatureEngineer(mongo_client)
    predictor.set_feature_engineer(feature_engineer)

    # 3. Fetch Data
    logger.info("Fetching historical data...")
    end_date = datetime.now(timezone.utc)
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
        
    start_date = end_date - timedelta(days=365 * 5) # 5 years of data for better model accuracy
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
            success = predictor.train_all_models(training_data)
            if not success:
                raise RuntimeError("Training failed internally (see logs)")
            logger.info("Training completed.")
            health.training_status = "completed"
        except Exception as e:
            logger.error("Training failed: %s", e)
            health.training_status = f"FAILED ({e})"
            if not predictor.models and not predictor.pooled_models:
                health.print_summary()
                sys.exit(1)

        # 5. Run Backtest (OOS only — restricted to dates after training cutoff)
        # Run for ALL 3 horizons so we see which horizon actually has signal.
        backtest_data = {t: historical_data[t] for t in tickers if t in historical_data}
        logger.info("Running backtest for all horizons...")
        oos_start = predictor.get_oos_start_date()
        if oos_start:
            logger.info("OOS backtest start date: %s", oos_start.date())
        else:
            logger.warning("No OOS start date available — backtest will use last 20%% fallback.")

        all_horizons = list(TARGET_CONFIG.keys())  # ["next_day", "7_day", "30_day"]
        total_eval_trades = 0
        any_backtest_ok = False
        for hz in all_horizons:
            logger.info("--- Backtest horizon: %s ---", hz)
            result = run_backtest(
                predictor=predictor,
                historical_data=backtest_data,
                spy_data=spy_data,
                tickers=list(backtest_data.keys()),
                start_date=args.start,
                end_date=args.end,
                max_positions=5,
                horizon=hz,
                oos_start_date=oos_start,
            )

            if "error" in result:
                logger.warning("Backtest %s: %s", hz, result["error"])
                continue

            any_backtest_ok = True
            ts = result.get("trade_stats", {})
            n_trades = ts.get("n_trades", 0)
            total_eval_trades += n_trades
            mn_label = " (market-neutral)" if result.get("market_neutral") else ""
            print(f"\n=== Backtest: {hz}{mn_label} ===")
            print(f"  OOS start: {result.get('oos_start', 'N/A')}")
            print(f"  Period: {result.get('start_date')} to {result.get('end_date')}")
            print(f"  Strategy return: {result.get('total_return', 0):.2%}")
            print(f"  Strategy Sharpe: {result.get('sharpe_ratio', 0):.3f}")
            print(f"  Max drawdown: {result.get('max_drawdown', 0):.2%}")
            if result.get("spy_return") is not None:
                print(f"  SPY return: {result['spy_return']:.2%}")
            if n_trades > 0:
                print(f"  Trades: {n_trades} ({ts.get('trades_per_year', 0):.0f}/yr)")
                print(f"  Avg return/trade: {ts.get('avg_return_per_trade', 0):.4f}")
                print(f"  Win rate: {ts.get('win_rate', 0):.1%}")
                print(f"  Avg holding: {ts.get('avg_holding_bars', 0):.1f} bars")
            else:
                print("  Trades: 0 (check filters)")

            # Performance sanity guard: warn if backtest is catastrophically bad
            sharpe = result.get("sharpe_ratio", 0)
            win_rate = ts.get("win_rate", 0.5)
            if n_trades >= 10 and (sharpe < -1.0 or win_rate < 0.40):
                logger.warning(
                    "SANITY CHECK FAILED for %s: Sharpe=%.2f win_rate=%.1f%% — "
                    "model may be broken or data degraded",
                    hz, sharpe, win_rate * 100,
                )
                print(f"  [!] SANITY WARNING: poor performance (Sharpe={sharpe:.2f}, win_rate={win_rate:.1%})")
            print("=" * 40)

        if not any_backtest_ok:
            logger.error("All backtests failed.")
            health.backtest_status = "FAILED (all horizons)"
            health.print_summary()
            sys.exit(1)

        health.backtest_status = "completed (all horizons)"
        health.evaluation_samples = total_eval_trades

    # 6. Generate and Store Live Predictions (skip with --no-predict)
    if args.no_predict:
        logger.info("--no-predict: skipping prediction storage.")
    elif not args.no_mongo and mongo_client:
        logger.info("Generating and storing live predictions...")
        horizons_seen = set()
        consecutive_failures = 0

        # v10.0: Collect all predictions first, then apply cross-sectional ranking
        all_ticker_preds = {}
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
                    real_horizons = {k: v for k, v in preds.items() if k != "_meta" and isinstance(v, dict)}
                    all_suppressed = all(
                        v.get("reason") in ("model_anti_correlated", "no_model", "feature_mismatch")
                        for v in real_horizons.values()
                    )
                    if all_suppressed and real_horizons:
                        logger.info(f"Skipping {ticker}: all horizons suppressed (kill-switch / no model)")
                        health.predictions_skipped += 1
                        continue
                    all_ticker_preds[ticker] = preds
                else:
                    health.predictions_failed += 1
            except Exception as e:
                health.predictions_failed += 1
                logger.error(f"Error generating predictions for {ticker}: {e}")

        # v10.0: Apply cross-sectional ranking across all tickers
        if USE_CROSS_SECTIONAL_RANKING and len(all_ticker_preds) >= RANKING_CONFIG.get("min_tickers", 5):
            try:
                from ml_backend.models.cross_sectional import CrossSectionalRanker
                ranker = CrossSectionalRanker(
                    top_pct=RANKING_CONFIG.get("top_pct", 0.20),
                    min_tickers=RANKING_CONFIG.get("min_tickers", 5),
                    confidence_boost=RANKING_CONFIG.get("confidence_boost", 0.10),
                )
                _disabled_hz = RANKING_CONFIG.get("disabled_horizons", [])
                for hz in TARGET_CONFIG.keys():
                    if hz in _disabled_hz:
                        logger.info("Cross-sectional ranking skipped for %s (disabled)", hz)
                        continue
                    all_ticker_preds = ranker.apply_ranking(all_ticker_preds, hz)
            except Exception as e:
                logger.warning("Cross-sectional ranking failed: %s — storing without ranking", e)

        # Store predictions
        for ticker, preds in all_ticker_preds.items():
            try:
                horizons_seen.update(k for k in preds.keys() if k != "_meta")
                success = mongo_client.store_predictions(ticker, preds)
                if success:
                    health.predictions_stored += 1
                    consecutive_failures = 0
                    logger.info(f"Stored predictions for {ticker}")
                else:
                    health.predictions_failed += 1
                    consecutive_failures += 1
                    logger.error(f"Failed to store predictions for {ticker}")
                    if consecutive_failures >= 3:
                        logger.error("3 consecutive storage failures (likely quota exceeded) — aborting prediction loop to save CI time")
                        break
            except Exception as e:
                health.predictions_failed += 1
                consecutive_failures += 1
                logger.error(f"Error storing predictions for {ticker}: {e}")
                if consecutive_failures >= 3:
                    logger.error("3 consecutive storage failures — aborting prediction loop")
                    break
        health.horizons_used = sorted(horizons_seen)



    health.print_summary()

    # Quality gate: fail the pipeline if key metrics are below thresholds.
    # Only apply when predictions were expected (not --no-predict).
    if not args.no_predict:
        if not health.check_quality_gate():
            logger.error("Pipeline quality gate FAILED — exiting with error.")
            sys.exit(1)

    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
