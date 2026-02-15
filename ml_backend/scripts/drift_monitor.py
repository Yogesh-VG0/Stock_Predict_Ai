#!/usr/bin/env python
"""
Drift monitoring — detect model degradation using stored prediction history.

Reports:
  1. Prediction distribution shift (PSI between recent vs baseline periods)
  2. Directional accuracy trend (rolling 14-day windows)
  3. Calibration degradation (Brier score trend)
  4. Alpha magnitude decay (are predictions getting weaker / noisier?)
  5. Feature coverage (% sentiment features non-zero — canary for data pipeline)

Usage:
  python -m ml_backend.scripts.drift_monitor
  python -m ml_backend.scripts.drift_monitor --tickers AAPL NVDA --days 90
"""

import argparse
import logging
import math
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(__file__).rsplit("scripts", 1)[0].rsplit("ml_backend", 1)[0])

from ml_backend.utils.mongodb import MongoDBClient
from ml_backend.config.feature_config_v1 import TARGET_CONFIG
from ml_backend.backtest import DEFAULT_BACKTEST_TICKERS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PSI (Population Stability Index)
# ---------------------------------------------------------------------------

def compute_psi(baseline: np.ndarray, recent: np.ndarray, n_bins: int = 10) -> float:
    """Compute PSI between baseline and recent prediction distributions.

    PSI < 0.10 → stable
    PSI 0.10–0.25 → moderate shift
    PSI > 0.25 → significant shift (model may need retraining)
    """
    if len(baseline) < 10 or len(recent) < 10:
        return float("nan")

    # Use baseline percentiles as bin edges
    edges = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf

    base_counts = np.histogram(baseline, bins=edges)[0].astype(float) + 1e-6
    recent_counts = np.histogram(recent, bins=edges)[0].astype(float) + 1e-6

    base_pct = base_counts / base_counts.sum()
    recent_pct = recent_counts / recent_counts.sum()

    psi = float(np.sum((recent_pct - base_pct) * np.log(recent_pct / base_pct)))
    return psi


# ---------------------------------------------------------------------------
# Rolling accuracy
# ---------------------------------------------------------------------------

def rolling_directional_accuracy(
    dates: np.ndarray,
    predictions: np.ndarray,
    realized: np.ndarray,
    window_days: int = 14,
) -> pd.DataFrame:
    """Compute rolling directional accuracy over calendar windows."""
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "pred": predictions,
        "actual": realized,
        "correct": np.sign(predictions) == np.sign(realized),
    }).sort_values("date")

    if len(df) < window_days:
        return df

    df["rolling_acc"] = df["correct"].rolling(window_days, min_periods=5).mean()
    return df


# ---------------------------------------------------------------------------
# Main monitor
# ---------------------------------------------------------------------------

def run_drift_monitor(
    mongo_client: MongoDBClient,
    tickers: List[str],
    days: int = 60,
) -> Dict:
    """Run drift monitoring across all horizons."""
    results = {}
    end = datetime.utcnow()
    start = end - timedelta(days=days + 30)  # extra buffer for horizon alignment
    baseline_cutoff = end - timedelta(days=days)
    midpoint = baseline_cutoff + timedelta(days=days // 2)

    for window_name, tcfg in TARGET_CONFIG.items():
        horizon = tcfg["horizon"]
        all_preds = []
        all_realized = []
        all_dates = []
        all_confidence = []

        for ticker in tickers:
            history = mongo_client.get_prediction_history(ticker, window_name, start_date=start)
            if not history:
                continue

            # Get price data
            price_df = mongo_client.get_historical_data(
                ticker, start - timedelta(days=60), end
            )
            if price_df is None or price_df.empty:
                continue

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
                if asof_d not in close_series.index:
                    continue
                future_dates = close_series.loc[asof_d:].index
                if len(future_dates) <= horizon:
                    continue
                future_date = future_dates[horizon]
                realized = float(np.log(close_series[future_date] / close_series[asof_d]))
                pred_val = float(doc.get("prediction", doc.get("alpha", 0.0)))
                conf = float(doc.get("confidence", doc.get("prob_positive", 0.5)))

                all_preds.append(pred_val)
                all_realized.append(realized)
                all_dates.append(asof_d)
                all_confidence.append(conf)

        if len(all_preds) < 10:
            results[window_name] = {"n_samples": len(all_preds), "status": "insufficient_data"}
            continue

        preds = np.array(all_preds)
        realized = np.array(all_realized)
        dates = np.array(all_dates)
        confs = np.array(all_confidence)

        # 1. PSI: split into first half (baseline) vs second half (recent)
        mask_base = dates < midpoint
        mask_recent = dates >= midpoint
        psi = compute_psi(preds[mask_base], preds[mask_recent])

        # 2. Overall directional accuracy
        dir_acc = float((np.sign(preds) == np.sign(realized)).mean())

        # 3. Split-half directional accuracy
        base_acc = float("nan")
        recent_acc = float("nan")
        if mask_base.sum() > 5:
            base_acc = float((np.sign(preds[mask_base]) == np.sign(realized[mask_base])).mean())
        if mask_recent.sum() > 5:
            recent_acc = float((np.sign(preds[mask_recent]) == np.sign(realized[mask_recent])).mean())

        # 4. Brier score (overall + trend)
        y_binary = (realized > 0).astype(float)
        brier = float(np.mean((confs - y_binary) ** 2))
        brier_base = float("nan")
        brier_recent = float("nan")
        if mask_base.sum() > 5:
            brier_base = float(np.mean((confs[mask_base] - y_binary[mask_base]) ** 2))
        if mask_recent.sum() > 5:
            brier_recent = float(np.mean((confs[mask_recent] - y_binary[mask_recent]) ** 2))

        # 5. Alpha magnitude: is the model's prediction strength decaying?
        alpha_mag_base = float(np.mean(np.abs(preds[mask_base]))) if mask_base.sum() > 0 else float("nan")
        alpha_mag_recent = float(np.mean(np.abs(preds[mask_recent]))) if mask_recent.sum() > 0 else float("nan")

        # 6. MAE / RMSE
        mae = float(np.mean(np.abs(preds - realized)))
        rmse = float(np.sqrt(np.mean((preds - realized) ** 2)))

        # Status classification
        status = "OK"
        alerts = []
        if not math.isnan(psi) and psi > 0.25:
            alerts.append(f"HIGH PSI={psi:.3f} (prediction distribution shift)")
            status = "WARNING"
        elif not math.isnan(psi) and psi > 0.10:
            alerts.append(f"Moderate PSI={psi:.3f}")
        if not math.isnan(recent_acc) and recent_acc < 0.48:
            alerts.append(f"Directional accuracy degraded: {recent_acc:.1%}")
            status = "WARNING"
        if not math.isnan(brier_recent) and not math.isnan(brier_base) and brier_recent > brier_base * 1.3:
            alerts.append(f"Brier score degraded: {brier_base:.4f} → {brier_recent:.4f}")
            status = "WARNING"

        results[window_name] = {
            "n_samples": len(preds),
            "status": status,
            "alerts": alerts,
            "psi": psi,
            "directional_accuracy": dir_acc,
            "dir_acc_baseline": base_acc,
            "dir_acc_recent": recent_acc,
            "brier_score": brier,
            "brier_baseline": brier_base,
            "brier_recent": brier_recent,
            "alpha_magnitude_baseline": alpha_mag_base,
            "alpha_magnitude_recent": alpha_mag_recent,
            "mae": mae,
            "rmse": rmse,
        }

    return results


# ---------------------------------------------------------------------------
# Sentiment data coverage check
# ---------------------------------------------------------------------------

def check_sentiment_coverage(mongo_client: MongoDBClient, tickers: List[str], days: int = 30) -> Dict:
    """Check if sentiment data is flowing for each ticker."""
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    coverage = {}
    for ticker in tickers:
        df = mongo_client.get_sentiment_timeseries(ticker, start, end)
        n_rows = len(df) if df is not None and not df.empty else 0
        n_nonzero = 0
        if n_rows > 0:
            n_nonzero = int((df["composite_sentiment"] != 0).sum())
        coverage[ticker] = {
            "days_checked": days,
            "rows_returned": n_rows,
            "nonzero_sentiment": n_nonzero,
            "coverage_pct": n_rows / max(1, days) * 100,
            "status": "OK" if n_rows > days * 0.3 else "LOW" if n_rows > 0 else "EMPTY",
        }
    return coverage


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def _print_drift_report(results: Dict, sent_coverage: Dict) -> None:
    print(f"\n{'='*70}")
    print("  DRIFT MONITORING REPORT")
    print(f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*70}")

    for window, m in results.items():
        status = m.get("status", "UNKNOWN")
        status_icon = "✓" if status == "OK" else "⚠" if status == "WARNING" else "?"
        n = m.get("n_samples", 0)
        print(f"\n  {status_icon} {window} (n={n}) — {status}")

        if n == 0:
            print("    Insufficient data.")
            continue

        alerts = m.get("alerts", [])
        for a in alerts:
            print(f"    ⚠ ALERT: {a}")

        psi = m.get("psi", float("nan"))
        psi_label = "stable" if psi < 0.10 else "moderate" if psi < 0.25 else "HIGH"
        print(f"    PSI                     : {psi:.4f} ({psi_label})")
        print(f"    Directional accuracy    : {m.get('directional_accuracy', float('nan')):.1%}")
        print(f"      Baseline / Recent     : {m.get('dir_acc_baseline', float('nan')):.1%} / {m.get('dir_acc_recent', float('nan')):.1%}")
        print(f"    Brier score             : {m.get('brier_score', float('nan')):.4f}")
        print(f"      Baseline / Recent     : {m.get('brier_baseline', float('nan')):.4f} / {m.get('brier_recent', float('nan')):.4f}")
        print(f"    Alpha magnitude         : {m.get('alpha_magnitude_baseline', float('nan')):.6f} → {m.get('alpha_magnitude_recent', float('nan')):.6f}")
        print(f"    MAE / RMSE              : {m.get('mae', float('nan')):.6f} / {m.get('rmse', float('nan')):.6f}")

    # Sentiment coverage
    print(f"\n  {'─'*50}")
    print("  Sentiment Data Coverage:")
    for ticker, sc in sent_coverage.items():
        st = sc.get("status", "?")
        icon = "✓" if st == "OK" else "⚠" if st == "LOW" else "✗"
        print(f"    {icon} {ticker:6s}: {sc['rows_returned']:3d} rows / {sc['days_checked']}d "
              f"({sc['coverage_pct']:.0f}%%), nonzero={sc['nonzero_sentiment']}")


def main():
    parser = argparse.ArgumentParser(description="Drift monitoring report")
    parser.add_argument("--tickers", nargs="+", default=None, help="Tickers to monitor")
    parser.add_argument("--days", type=int, default=60, help="Lookback period (days)")
    args = parser.parse_args()

    try:
        mongo_client = MongoDBClient()
        if mongo_client.db is None:
            print("Cannot connect to MongoDB.")
            sys.exit(1)
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        sys.exit(1)

    tickers = args.tickers or DEFAULT_BACKTEST_TICKERS

    logger.info("Running drift monitor for %d tickers over %d days...", len(tickers), args.days)
    drift_results = run_drift_monitor(mongo_client, tickers, args.days)
    sent_coverage = check_sentiment_coverage(mongo_client, tickers, min(args.days, 30))
    _print_drift_report(drift_results, sent_coverage)

    print("\nDone.")


if __name__ == "__main__":
    main()
