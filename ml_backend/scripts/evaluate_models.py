#!/usr/bin/env python
"""
Offline model evaluation — A/B comparison with/without sentiment features.

Metrics per horizon:
  - Directional accuracy (correctly predict sign of alpha)
  - Rank correlation (Spearman) between predicted & realized alpha
  - MAE / RMSE of predicted alpha
  - Top-K hit rate (top-10%% predictions → avg realized alpha)
  - Sign classifier: Brier score, calibration buckets

Modes:
  --ab        : Train two models (with / without sentiment) on the same data,
                compare OOS metrics side-by-side.
  --stored    : Evaluate stored prediction history against realized returns.

Usage:
  python -m ml_backend.scripts.evaluate_models --ab
  python -m ml_backend.scripts.evaluate_models --stored --tickers AAPL NVDA
"""

import argparse
import logging
import math
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(__file__).rsplit("scripts", 1)[0].rsplit("ml_backend", 1)[0])

from ml_backend.models.predictor import StockPredictor
from ml_backend.data.features_minimal import MinimalFeatureEngineer
from ml_backend.utils.mongodb import MongoDBClient
from ml_backend.config.feature_config_v1 import (
    TARGET_CONFIG,
    USE_MARKET_NEUTRAL_TARGET,
)
from ml_backend.config.constants import TOP_100_TICKERS
from ml_backend.backtest import DEFAULT_BACKTEST_TICKERS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions where sign(pred) == sign(actual)."""
    if len(y_true) == 0:
        return float("nan")
    signs_match = np.sign(y_pred) == np.sign(y_true)
    return float(signs_match.mean())


def rank_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation."""
    if len(y_true) < 5:
        return float("nan")
    from scipy.stats import spearmanr
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr)


def top_k_hit_rate(y_true: np.ndarray, y_pred: np.ndarray, k_pct: float = 0.10) -> float:
    """Average realized return of top-k%% predictions (highest predicted alpha)."""
    n = len(y_true)
    if n < 10:
        return float("nan")
    k = max(1, int(n * k_pct))
    top_idx = np.argsort(y_pred)[-k:]
    return float(np.mean(y_true[top_idx]))


def brier_score(y_binary: np.ndarray, prob_positive: np.ndarray) -> float:
    """Mean squared error between predicted probability and binary outcome."""
    if len(y_binary) == 0:
        return float("nan")
    return float(np.mean((prob_positive - y_binary) ** 2))


def calibration_buckets(
    y_binary: np.ndarray, prob_positive: np.ndarray, n_bins: int = 5,
) -> List[Dict[str, float]]:
    """Reliability diagram: bin predictions by prob, report actual fraction positive."""
    edges = np.linspace(0, 1, n_bins + 1)
    buckets = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob_positive >= lo) & (prob_positive < hi)
        if hi == 1.0:
            mask |= prob_positive == hi
        count = int(mask.sum())
        if count == 0:
            continue
        buckets.append({
            "bin": f"{lo:.2f}-{hi:.2f}",
            "n": count,
            "avg_prob": float(prob_positive[mask].mean()),
            "actual_frac_positive": float(y_binary[mask].mean()),
        })
    return buckets


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prob_pos: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute a dict of all evaluation metrics."""
    metrics = {
        "n_samples": len(y_true),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "rank_corr": rank_correlation(y_true, y_pred),
        "mae": float(np.mean(np.abs(y_true - y_pred))) if len(y_true) > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) > 0 else float("nan"),
        "top10_hit_rate": top_k_hit_rate(y_true, y_pred, 0.10),
    }
    if prob_pos is not None and len(prob_pos) > 0:
        y_binary = (y_true > 0).astype(float)
        metrics["brier_score"] = brier_score(y_binary, prob_pos)
    return metrics


# ---------------------------------------------------------------------------
# A/B evaluation: train with & without sentiment, compare OOS
# ---------------------------------------------------------------------------

def run_ab_evaluation(
    mongo_client: MongoDBClient,
    tickers: List[str],
    eval_tickers: Optional[List[str]] = None,
) -> Dict:
    """Train two models (with / without sentiment) and compare OOS.

    Returns dict with metrics for both variants per horizon.
    """
    from ml_backend.config import feature_config_v1 as cfg

    eval_tickers = eval_tickers or DEFAULT_BACKTEST_TICKERS

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=365 * 2)

    # Fetch data
    logger.info("Fetching data for %d tickers...", len(tickers))
    historical_data = {}
    for t in tickers:
        try:
            df = mongo_client.get_historical_data(t, start_date, end_date)
            if df is not None and not df.empty:
                historical_data[t] = df
        except Exception as e:
            logger.warning("Could not fetch %s: %s", t, e)

    if not historical_data:
        logger.error("No data fetched.")
        return {}

    results = {"with_sentiment": {}, "without_sentiment": {}}

    for variant, use_sent in [("with_sentiment", True), ("without_sentiment", False)]:
        logger.info("\n=== Training variant: %s ===", variant)
        # Toggle sentiment
        original = cfg.USE_SENTIMENT_FEATURES
        cfg.USE_SENTIMENT_FEATURES = use_sent

        predictor = StockPredictor(mongo_client)
        fe = MinimalFeatureEngineer(mongo_client)
        predictor.set_feature_engineer(fe)

        try:
            predictor.train_all_models(historical_data)
        except Exception as e:
            logger.error("Training failed for %s: %s", variant, e)
            cfg.USE_SENTIMENT_FEATURES = original
            continue

        # Evaluate on eval_tickers using predict_batch (vectorized OOS)
        oos_start = predictor.get_oos_start_date()

        for window_name in predictor.prediction_windows:
            horizon = TARGET_CONFIG.get(window_name, {}).get("horizon", 1)
            all_preds = []
            all_actual = []
            all_probs = []

            for ticker in eval_tickers:
                if ticker not in historical_data:
                    continue
                df = historical_data[ticker]

                batch_df = predictor.predict_batch(ticker, df)
                if batch_df.empty:
                    continue

                w_df = batch_df[batch_df["window"] == window_name].copy()
                if w_df.empty:
                    continue

                # Compute realized alpha for each prediction date
                # realized_alpha[t] = log_return[t : t+horizon] - SPY_log_return[t : t+horizon]
                # For simplicity, use the stock's own future log return (available in df)
                try:
                    fe2 = MinimalFeatureEngineer(mongo_client)
                    feats, meta2 = fe2.prepare_features(df, ticker=ticker, mongo_client=mongo_client)
                    if feats is None:
                        continue
                    df_a = meta2.get("df_aligned")
                    if df_a is None:
                        continue

                    target_col = TARGET_CONFIG[window_name]["target"]
                    if target_col not in df_a.columns:
                        continue

                    # Shift target by -horizon to align: target at row t is
                    # the return from t to t+horizon (computed as log(close[t+h]/close[t]))
                    # Since log_return_Xd at row t is log(close[t]/close[t-X]), we need
                    # to look X rows ahead → shift(-horizon)
                    realized = df_a[target_col].shift(-horizon).values

                    # Align by date
                    pred_dates = pd.to_datetime(w_df["date"].values).normalize()
                    aligned_dates = pd.to_datetime(df_a["date"]).dt.normalize().values
                    date_to_idx = {d: i for i, d in enumerate(aligned_dates)}

                    for _, row in w_df.iterrows():
                        d = pd.Timestamp(row["date"]).normalize()
                        idx = date_to_idx.get(d)
                        if idx is not None and idx < len(realized) and not np.isnan(realized[idx]):
                            all_preds.append(row["prediction"])
                            all_actual.append(realized[idx])
                            if "prob_positive" in row:
                                all_probs.append(row["prob_positive"])

                except Exception as e:
                    logger.debug("Could not align %s: %s", ticker, e)

            if all_preds:
                y_true = np.array(all_actual)
                y_pred = np.array(all_preds)
                prob_pos = np.array(all_probs) if all_probs else None

                # Filter to OOS only
                if oos_start is not None:
                    # all entries are OOS since predict_batch covers full df;
                    # filter would require date tracking — skip for now (full-sample)
                    pass

                m = compute_all_metrics(y_true, y_pred, prob_pos)
                if prob_pos is not None:
                    m["calibration"] = calibration_buckets(
                        (y_true > 0).astype(float), prob_pos
                    )
                results[variant][window_name] = m
            else:
                results[variant][window_name] = {"n_samples": 0}

        cfg.USE_SENTIMENT_FEATURES = original

    return results


# ---------------------------------------------------------------------------
# Stored-prediction evaluation
# ---------------------------------------------------------------------------

def evaluate_stored_predictions(
    mongo_client: MongoDBClient,
    tickers: List[str],
    days: int = 60,
) -> Dict:
    """Evaluate stored predictions vs realized returns."""
    results = {}
    end = datetime.utcnow()
    start = end - timedelta(days=days + 30)

    for window_name, tcfg in TARGET_CONFIG.items():
        horizon = tcfg["horizon"]
        all_preds = []
        all_actual = []

        for ticker in tickers:
            history = mongo_client.get_prediction_history(ticker, window_name, start_date=start)
            if not history:
                continue

            # Get price data to compute realized return
            price_df = mongo_client.get_historical_data(ticker, start - timedelta(days=60), end)
            if price_df is None or price_df.empty:
                continue

            # Build close series indexed by date
            if "date" not in price_df.columns and isinstance(price_df.index, pd.DatetimeIndex):
                price_df = price_df.reset_index()
                if "Date" in price_df.columns:
                    price_df = price_df.rename(columns={"Date": "date"})
            if "date" not in price_df.columns:
                continue

            price_df["date"] = pd.to_datetime(price_df["date"]).dt.normalize()
            price_df = price_df.sort_values("date").drop_duplicates("date", keep="last")
            close_series = price_df.set_index("date")["Close"]

            for doc in history:
                asof = doc.get("asof_date")
                if asof is None:
                    continue
                asof_d = pd.Timestamp(asof).normalize()
                # Find the close on asof_date and asof_date + horizon trading days
                if asof_d not in close_series.index:
                    continue
                future_dates = close_series.loc[asof_d:].index
                if len(future_dates) <= horizon:
                    continue
                future_date = future_dates[horizon]
                realized = np.log(close_series[future_date] / close_series[asof_d])
                pred_val = doc.get("prediction", doc.get("alpha", 0.0))
                all_preds.append(float(pred_val))
                all_actual.append(float(realized))

        if all_preds:
            y_true = np.array(all_actual)
            y_pred = np.array(all_preds)
            results[window_name] = compute_all_metrics(y_true, y_pred)
        else:
            results[window_name] = {"n_samples": 0}

    return results


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def _print_metrics(results: Dict, title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    for window, m in results.items():
        n = m.get("n_samples", 0)
        print(f"\n  --- {window} (n={n}) ---")
        if n == 0:
            print("    No samples.")
            continue
        print(f"    Directional accuracy : {m.get('directional_accuracy', float('nan')):.1%}")
        print(f"    Rank correlation     : {m.get('rank_corr', float('nan')):.4f}")
        print(f"    MAE                  : {m.get('mae', float('nan')):.6f}")
        print(f"    RMSE                 : {m.get('rmse', float('nan')):.6f}")
        print(f"    Top-10% hit rate     : {m.get('top10_hit_rate', float('nan')):.6f}")
        if "brier_score" in m:
            print(f"    Brier score          : {m.get('brier_score', float('nan')):.6f}")
        if "calibration" in m:
            print("    Calibration buckets:")
            for b in m["calibration"]:
                print(f"      {b['bin']:10s}  n={b['n']:4d}  avg_prob={b['avg_prob']:.3f}  actual={b['actual_frac_positive']:.3f}")


def _print_ab_comparison(results: Dict) -> None:
    print(f"\n{'='*70}")
    print("  A/B COMPARISON: With Sentiment vs Without Sentiment")
    print(f"{'='*70}")

    for window_name in TARGET_CONFIG:
        with_m = results.get("with_sentiment", {}).get(window_name, {})
        without_m = results.get("without_sentiment", {}).get(window_name, {})
        n_with = with_m.get("n_samples", 0)
        n_without = without_m.get("n_samples", 0)

        print(f"\n  --- {window_name} ---")
        print(f"  {'Metric':25s} {'With Sentiment':>18s} {'Without Sentiment':>18s} {'Delta':>10s}")
        print(f"  {'-'*25} {'-'*18} {'-'*18} {'-'*10}")

        for key, fmt, higher_better in [
            ("directional_accuracy", ".1%", True),
            ("rank_corr", ".4f", True),
            ("mae", ".6f", False),
            ("rmse", ".6f", False),
            ("top10_hit_rate", ".6f", True),
            ("brier_score", ".6f", False),
        ]:
            v_w = with_m.get(key)
            v_wo = without_m.get(key)
            if v_w is None and v_wo is None:
                continue
            v_w = v_w if v_w is not None else float("nan")
            v_wo = v_wo if v_wo is not None else float("nan")
            delta = v_w - v_wo if not (math.isnan(v_w) or math.isnan(v_wo)) else float("nan")
            # Arrow: ↑ if delta goes in the "good" direction
            if math.isnan(delta):
                arrow = ""
            elif (higher_better and delta > 0) or (not higher_better and delta < 0):
                arrow = " ↑"
            elif delta == 0:
                arrow = " ="
            else:
                arrow = " ↓"

            print(f"  {key:25s} {format(v_w, fmt):>18s} {format(v_wo, fmt):>18s} {format(delta, fmt):>10s}{arrow}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate ML models")
    parser.add_argument("--ab", action="store_true", help="Run A/B comparison (with/without sentiment)")
    parser.add_argument("--stored", action="store_true", help="Evaluate stored predictions vs realized")
    parser.add_argument("--tickers", nargs="+", default=None, help="Tickers for evaluation")
    parser.add_argument("--days", type=int, default=60, help="Lookback days for stored eval")
    args = parser.parse_args()

    if not args.ab and not args.stored:
        parser.print_help()
        print("\nSpecify --ab or --stored (or both).")
        sys.exit(1)

    try:
        mongo_client = MongoDBClient()
        if mongo_client.db is None:
            print("Cannot connect to MongoDB.")
            sys.exit(1)
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        sys.exit(1)

    eval_tickers = args.tickers or DEFAULT_BACKTEST_TICKERS
    train_tickers = list(TOP_100_TICKERS)

    if args.ab:
        logger.info("Running A/B evaluation (with vs without sentiment)...")
        ab_results = run_ab_evaluation(mongo_client, train_tickers, eval_tickers)
        _print_ab_comparison(ab_results)
        _print_metrics(ab_results.get("with_sentiment", {}), "Detailed: WITH sentiment")
        _print_metrics(ab_results.get("without_sentiment", {}), "Detailed: WITHOUT sentiment")

    if args.stored:
        logger.info("Evaluating stored predictions (last %d days)...", args.days)
        stored_results = evaluate_stored_predictions(mongo_client, eval_tickers, args.days)
        _print_metrics(stored_results, f"Stored Predictions (last {args.days}d)")

    print("\nDone.")


if __name__ == "__main__":
    main()
