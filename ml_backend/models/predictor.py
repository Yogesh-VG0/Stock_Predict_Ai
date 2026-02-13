"""
StockPredictor - LightGBM-based predictor with walk-forward validation.
Leakage-proof, time-based splits, proper evaluation.
"""

import json
import math
import os
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
import lightgbm as lgb

from ..config.constants import PREDICTION_WINDOWS
from ..config.feature_config_v1 import (
    FEATURE_CONFIG_V1,
    LIGHTGBM_PARAMS,
    POOL_CONFIG,
    TARGET_CONFIG,
    USE_MARKET_NEUTRAL_TARGET,
    WALK_FORWARD_FOLDS,
    TRADE_MIN_ALPHA,
    TRADE_MIN_PROB_POSITIVE,
    TRADE_SIGMA_MULT,
    ROUND_TRIP_COST_BPS,
)

logger = logging.getLogger(__name__)

# Default model directory
MODEL_DIR = os.getenv("MODEL_DIR", "models")


class StockPredictor:
    """
    LightGBM predictor with:
    - Time-based train/val/holdout splits (no random split)
    - Purged walk-forward validation
    - Log-return targets
    - Minimal leakage-proof features
    """

    def __init__(self, mongo_client=None):
        self.mongo_client = mongo_client
        self.feature_engineer = None
        self.models = {}  # {(ticker, window): lgb.Booster}
        self.scalers = {}  # {(ticker, window): StandardScaler}
        self.metadata = {}  # {(ticker, window): dict}
        self.pooled_models = {}  # {window: lgb.LGBMRegressor}
        self.pooled_metadata = {}  # {window: dict}
        self.prediction_windows = list(PREDICTION_WINDOWS.keys())

    def set_feature_engineer(self, feature_engineer):
        """Set the feature engineer (MinimalFeatureEngineer or compatible)."""
        self.feature_engineer = feature_engineer

    def train_all_models(
        self,
        historical_data: Optional[Dict[str, pd.DataFrame]] = None,
        ticker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> bool:
        """
        Train models. Accepts either:
        - historical_data: Dict[ticker, DataFrame]
        - Or ticker + start_date + end_date (fetches from MongoDB)
        """
        try:
            if historical_data is None and ticker:
                historical_data = self._fetch_historical_data(ticker, start_date, end_date)
                if not historical_data:
                    return False
                historical_data = {ticker: historical_data[ticker]} if ticker in historical_data else {}

            if not historical_data:
                logger.error("No historical data provided")
                return False

            fe = self.feature_engineer
            if fe is None:
                from ..data.features_minimal import MinimalFeatureEngineer
                fe = MinimalFeatureEngineer(self.mongo_client)
                self.feature_engineer = fe
            
            self.train_pooled_models(historical_data, fe)

            for t, df in historical_data.items():
                if df is None or df.empty or len(df) < FEATURE_CONFIG_V1["min_rows"]:
                    logger.warning(f"Skipping {t}: insufficient data")
                    continue
                self._train_ticker(t, df, fe)
            self.save_models()
            self.print_training_summary()
            return True
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def print_training_summary(self) -> None:
        """Print count(has_holdout) per horizon, median eval_rmse, baseline, win_rate (holdout only)."""
        if not self.metadata:
            logger.info("No metadata to summarize")
            return
        from collections import defaultdict
        by_window = defaultdict(list)
        for (ticker, window_name), meta in self.metadata.items():
            by_window[window_name].append(meta)
        lines = ["Training summary:"]
        for window_name in sorted(by_window.keys()):
            items = by_window[window_name]
            n_holdout = sum(1 for m in items if m.get("has_holdout"))
            total = len(items)
            # Holdout-only stats: don't mix val-only models into baseline comparison
            holdout_items = [m for m in items if m.get("has_holdout")]
            if holdout_items:
                eval_rmses = [m["eval_rmse"] for m in holdout_items]
                baseline_rmses = [m["baseline_rmse"] for m in holdout_items]
                hit_rates = [m.get("hit_rate", 0.5) for m in holdout_items]
                correlations = [m.get("correlation", 0.0) for m in holdout_items]
                median_eval = float(np.median(eval_rmses))
                median_baseline = float(np.median(baseline_rmses))
                median_hit = float(np.median(hit_rates))
                median_corr = float(np.median(correlations))
                wins = sum(1 for m in holdout_items if m.get("beats_baseline"))
                wins_last = sum(1 for m in holdout_items if m.get("beats_last_baseline", m.get("beats_baseline")))
                prod_ready = sum(1 for m in holdout_items if m.get("production_ready"))
                win_rate = wins / len(holdout_items)
                prod_rate = prod_ready / len(holdout_items)
                s = (
                    f"  {window_name}: {n_holdout}/{total} has_holdout | "
                    f"median eval_rmse={median_eval:.4f} | baseline={median_baseline:.4f} | "
                    f"hit_rate={median_hit:.1%} | corr={median_corr:.3f} | "
                    f"win_rate={win_rate:.1%} | wins_last={wins_last}/{len(holdout_items)} | prod_ready={prod_rate:.1%}"
                )
            else:
                val_rmses = [m["eval_rmse"] for m in items]
                median_val = float(np.median(val_rmses))
                s = f"  {window_name}: {n_holdout}/{total} has_holdout | val_rmse only (no holdout): median={median_val:.4f}"
            lines.append(s)
        logger.info("\n".join(lines))

    def _fetch_historical_data(
        self,
        ticker: str,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Fetch historical data from MongoDB."""
        if self.mongo_client is None:
            return None
        end = datetime.utcnow() if not end_date else pd.to_datetime(end_date)
        start = end - timedelta(days=365 * 5) if not start_date else pd.to_datetime(start_date)
        df = self.mongo_client.get_historical_data(ticker, start, end)
        if df is None or df.empty:
            return None
        return {ticker: df}

    def train_pooled_models(
        self, historical_data: Dict[str, pd.DataFrame], feature_engineer
    ) -> None:
        """Train one model per horizon across all tickers (pooled)."""
        if not POOL_CONFIG.get("enabled", True):
            return

        rows_by_window = {w: [] for w in TARGET_CONFIG}
        y_by_window = {w: [] for w in TARGET_CONFIG}
        dates_by_window = {w: [] for w in TARGET_CONFIG}
        feature_cols_ref = None

        logger.info(f"Processing {len(historical_data)} tickers for pooled training")
        for ticker, df in historical_data.items():
            if df is None or df.empty or len(df) < FEATURE_CONFIG_V1["min_rows"]:
                continue
            feats, meta = feature_engineer.prepare_features(
                df, ticker=ticker, mongo_client=self.mongo_client
            )
            df_aligned = meta.get("df_aligned")
            if feats is None or df_aligned is None or "Close" not in df_aligned.columns:
                continue

            close = df_aligned["Close"].values
            dates_full = pd.to_datetime(df_aligned["date"]).values
            if feature_cols_ref is None:
                feature_cols_ref = meta.get("feature_columns", [])

            # SPY for market-neutral target (alpha = stock return - SPY return)
            spy_close = None
            if USE_MARKET_NEUTRAL_TARGET and "SPY_close" in df_aligned.columns:
                spy_close = df_aligned["SPY_close"].values
                if np.any(np.isnan(spy_close)) or np.any(spy_close <= 0):
                    spy_close = None

            for window_name, cfg in TARGET_CONFIG.items():
                horizon = cfg["horizon"]
                y = np.log(close[horizon:] / close[:-horizon])
                if spy_close is not None and len(spy_close) >= horizon:
                    spy_ret = np.log(spy_close[horizon:] / spy_close[:-horizon])
                    y = y - spy_ret  # alpha vs SPY
                X = feats[: len(y)]
                n_y_pre = len(df_aligned) - horizon
                dates_x = dates_full[:n_y_pre]
                dates_y = dates_full[horizon:]
                valid = np.isfinite(y)
                X = X[valid]
                y = y[valid]
                dates_x = dates_x[valid]
                dates_y = dates_y[valid]
                if len(X) < 50:
                    continue
                rows_by_window[window_name].append(X)
                y_by_window[window_name].append(y)
                dates_by_window[window_name].append(dates_x)

        for window_name in TARGET_CONFIG:
            if not rows_by_window[window_name]:
                continue
            X_all = np.vstack(rows_by_window[window_name])
            y_all = np.concatenate(y_by_window[window_name])
            dates_all = np.concatenate(dates_by_window[window_name])

            if len(y_all) < POOL_CONFIG.get("min_total_samples", 2000):
                logger.warning(
                    "Pooled %s: too few samples %d", window_name, len(y_all)
                )
                continue

            # Time-based split: sort by date so val is always future of train
            order = np.argsort(dates_all)
            X_all = X_all[order]
            y_all = y_all[order]
            dates_all = dates_all[order]

            n = len(y_all)
            n_folds = max(1, WALK_FORWARD_FOLDS)
            fold_rmses, fold_maes, fold_hits, fold_corrs = [], [], [], []
            fold_q90, fold_q95 = [], []
            model = None

            purge = FEATURE_CONFIG_V1["purge_days"]
            embargo = FEATURE_CONFIG_V1["embargo_days"]

            for fold in range(n_folds):
                if n_folds == 1:
                    train_end = int(n * 0.85)
                else:
                    train_end = int(n * (0.5 + fold * 0.12))

                # Purge/embargo gap to prevent horizon overlap leakage
                horizon = TARGET_CONFIG[window_name]["horizon"]
                gap = max(purge, horizon) + embargo
                val_start = min(train_end + gap, n - 100)

                if val_start >= n - 50:
                    break
                val_end = min(val_start + int(n * 0.1), n)

                X_train, X_val = X_all[:train_end], X_all[val_start:val_end]
                y_train, y_val = y_all[:train_end], y_all[val_start:val_end]
                if len(y_train) < 200 or len(y_val) < 20:
                    break
                # Skip fold if val too small → unstable conformal quantiles
                if len(y_val) < 20:
                    continue

                w = np.exp(np.linspace(-2.0, 0.0, len(y_train))).astype(np.float32)
                fold_model = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
                fold_model.fit(
                    X_train,
                    y_train,
                    sample_weight=w,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(15, verbose=False),
                        lgb.log_evaluation(0),
                    ],
                )
                pred = fold_model.predict(X_val)
                fold_rmses.append(float(np.sqrt(np.mean((y_val - pred) ** 2))))
                fold_maes.append(float(np.mean(np.abs(y_val - pred))))
                fold_hits.append(float(np.mean((y_val > 0) == (pred > 0))))
                c = np.corrcoef(y_val, pred)[0, 1] if len(y_val) > 1 else 0.0
                fold_corrs.append(float(c) if not np.isnan(c) else 0.0)
                # Conformal: absolute residual quantiles for calibrated intervals
                abs_resid = np.abs(y_val - pred)
                fold_q90.append(float(np.quantile(abs_resid, 0.90)))
                fold_q95.append(float(np.quantile(abs_resid, 0.95)))
                # Keep the last model
                model = fold_model

            if not fold_rmses:
                continue
            rmse = float(np.median(fold_rmses))
            mae = float(np.median(fold_maes))
            hit_rate = float(np.median(fold_hits))
            correlation = float(np.median(fold_corrs))
            q90_med = float(np.median(fold_q90)) if fold_q90 else rmse
            q95_med = float(np.median(fold_q95)) if fold_q95 else rmse * 1.3
            # Regime-adaptive threshold: mean + 2*sigma (Losing Loonies v4 - tradable moves)
            mean_ret = float(np.mean(y_all))
            std_ret = float(np.std(y_all)) if len(y_all) > 1 else 0.0
            threshold = mean_ret + TRADE_SIGMA_MULT * std_ret if std_ret > 0 else mean_ret
            # Persist feature importance for pooled model
            top_features_gain_pooled = []
            try:
                booster = model.booster_
                gains = booster.feature_importance(importance_type="gain")
                cols = feature_cols_ref
                pairs = sorted(zip(cols, gains), key=lambda x: x[1], reverse=True)[:30]
                top_features_gain_pooled = [{"name": n, "gain": float(g)} for n, g in pairs]
            except Exception:
                pass
            self.pooled_models[window_name] = model
            self.pooled_metadata[window_name] = {
                "val_rmse": rmse,
                "val_mae": mae,
                "conformal_q90": q90_med,
                "conformal_q95": q95_med,
                "top_features_gain": top_features_gain_pooled,
                "n": int(n),
                "hit_rate": hit_rate,
                "correlation": correlation,
                "market_neutral": USE_MARKET_NEUTRAL_TARGET,
                "feature_columns": feature_cols_ref,
                "mean_return": mean_ret,
                "std_return": std_ret,
                "trade_threshold": threshold,
            }
            logger.info(
                "Trained POOLED-%s: rmse=%.4f mae=%.4f n=%d hit_rate=%.1f%% corr=%.3f (folds=%d)",
                window_name, rmse, mae, n, hit_rate * 100, correlation, n_folds,
            )

    def _train_ticker(self, ticker: str, df: pd.DataFrame, feature_engineer) -> None:
        """Train all window models for one ticker."""
        features, meta = feature_engineer.prepare_features(df, ticker=ticker, mongo_client=self.mongo_client)
        if features is None or len(features) < 50:
            logger.warning(f" insufficient features for {ticker}")
            return

        # Use aligned df (exact same rows as features)
        df = meta.get("df_aligned")
        if df is None or "Close" not in df.columns:
            logger.error(f"No aligned df for {ticker}")
            return
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        if len(df) != len(features):
            logger.error(f"Alignment mismatch {ticker}: df_aligned={len(df)} features={len(features)}")
            return

        purge = FEATURE_CONFIG_V1["purge_days"]
        embargo = FEATURE_CONFIG_V1["embargo_days"]
        train_r = FEATURE_CONFIG_V1["train_ratio"]
        val_r = FEATURE_CONFIG_V1["val_ratio"]
        holdout_r = FEATURE_CONFIG_V1["holdout_ratio"]
        feature_cols = meta.get("feature_columns", [])

        for window_name, cfg in TARGET_CONFIG.items():
            horizon = cfg["horizon"]
            # gap = max(purge, horizon) + embargo (labels overlap for multi-step)
            gap = max(purge, horizon) + embargo

            # Build target: row i predicts return from close[i] to close[i+horizon]
            # Optionally use alpha (stock return - SPY return) for market-neutral ranking
            close = df["Close"].values
            y = np.log(close[horizon:] / close[:-horizon])  # length n - horizon
            if USE_MARKET_NEUTRAL_TARGET and "SPY_close" in df.columns:
                spy_close = df["SPY_close"].values
                if len(spy_close) >= horizon and not (np.any(np.isnan(spy_close)) or np.any(spy_close <= 0)):
                    spy_ret = np.log(spy_close[horizon:] / spy_close[:-horizon])
                    y = y - spy_ret  # alpha vs SPY
            X = features[: len(y)]  # align: first len(y) rows

            # dates_x / dates_y BEFORE filtering (unbreakable pattern)
            n_y_pre = len(df) - horizon
            dates_full = pd.to_datetime(df["date"]).values
            dates_x = dates_full[:n_y_pre]   # feature row i -> date of close[i]
            dates_y = dates_full[horizon:]  # label row i -> date of close[i+horizon]
            assert len(dates_x) == len(y) and len(dates_y) == len(y), (
                f"{ticker}-{window_name}: date/y length mismatch"
            )

            valid = np.isfinite(y)
            X = X[valid]
            y = y[valid]
            dates_x = dates_x[valid]
            dates_y = dates_y[valid]
            assert len(dates_x) == len(X) and len(dates_y) == len(y), (
                f"{ticker}-{window_name}: date/X/y length mismatch after valid"
            )
            if len(dates_x) > 0:
                assert np.all(dates_x < dates_y), (
                    f"Date order violated for {ticker}-{window_name} "
                    f"(n={len(dates_x)}, horizon={horizon}, gap={gap}) — feature date must be < label date"
                )

            if len(X) < 50:
                logger.warning(f"Skipping {ticker}-{window_name}: only {len(X)} valid samples")
                continue
            if horizon >= 21 and len(X) < 300:
                logger.warning(
                    f"Skipping {ticker}-{window_name}: n={len(X)} < 300 (gap={gap})"
                )
                continue

            n = len(X)
            train_end = int(n * train_r)
            val_end = int(n * (train_r + val_r))

            train_idx_end = max(30, train_end)
            val_idx_start = train_idx_end + gap
            val_idx_end = min(max(val_idx_start + 10, val_end), n)  # cap at n
            test_idx_start = val_idx_end + gap
            test_idx_end = n  # remainder of series for holdout

            # Only create holdout if gap leaves samples
            if test_idx_start >= n:
                test_idx_start = n
                test_idx_end = n

            if not (val_idx_start < val_idx_end):
                logger.warning(
                    f"{ticker}-{window_name}: skipping empty val "
                    f"(n={n}, horizon={horizon}, gap={gap}) "
                    f"train_idx_end={train_idx_end} val_idx_start={val_idx_start} val_idx_end={val_idx_end} "
                    f"test_idx_start={test_idx_start}"
                )
                continue
            if test_idx_start < n and not (test_idx_start < test_idx_end):
                logger.warning(
                    f"{ticker}-{window_name}: skipping empty test "
                    f"(n={n}, horizon={horizon}, gap={gap}) "
                    f"train_idx_end={train_idx_end} val_idx_start={val_idx_start} val_idx_end={val_idx_end} "
                    f"test_idx_start={test_idx_start}"
                )
                continue

            train_mask = np.zeros(n, dtype=bool)
            val_mask = np.zeros(n, dtype=bool)
            test_mask = np.zeros(n, dtype=bool)
            train_mask[:train_idx_end] = True
            val_mask[val_idx_start:val_idx_end] = True
            test_mask[test_idx_start:test_idx_end] = True

            assert not np.any(train_mask & val_mask), f"{ticker}-{window_name}: train/val overlap"
            assert not np.any(train_mask & test_mask), f"{ticker}-{window_name}: train/test overlap"
            assert not np.any(val_mask & test_mask), f"{ticker}-{window_name}: val/test overlap"

            logger.debug(
                f"{ticker}-{window_name} n={n} train={train_mask.sum()} val={val_mask.sum()} "
                f"test={test_mask.sum()} gap={gap}"
            )
            if train_mask.any():
                logger.debug(f"{ticker}-{window_name}: train max_date={pd.Timestamp(dates_x[train_mask].max())}")
            if val_mask.any():
                logger.debug(
                    f"{ticker}-{window_name}: val dates={pd.Timestamp(dates_x[val_mask].min())}.."
                    f"{pd.Timestamp(dates_x[val_mask].max())}"
                )
            if test_mask.any():
                logger.debug(
                    f"{ticker}-{window_name}: test min_date={pd.Timestamp(dates_x[test_mask].min())} "
                    f"n_test={test_mask.sum()}"
                )

            X_train = X[train_mask]
            y_train = y[train_mask]
            X_val = X[val_mask]
            y_val = y[val_mask]

            if len(X_train) < 30 or len(X_val) < 5:
                logger.warning(f"Skipping {ticker}-{window_name}: not enough samples")
                continue

            # Recency weighting + no scaling for tree models
            w = np.exp(np.linspace(-2.0, 0.0, len(y_train))).astype(np.float32)
            model = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
            model.fit(
                X_train,
                y_train,
                sample_weight=w,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=15, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )

            key = (ticker, window_name)
            self.models[key] = model
            self.scalers[key] = None  # No scaling
            # Persist feature importance for drift/debugging
            top_features_gain = []
            try:
                booster = model.booster_
                gains = booster.feature_importance(importance_type="gain")
                cols = feature_cols
                pairs = sorted(zip(cols, gains), key=lambda x: x[1], reverse=True)[:30]
                top_features_gain = [{"name": n, "gain": float(g)} for n, g in pairs]
            except Exception:
                pass
            val_pred = model.predict(X_val)
            val_rmse = float(np.sqrt(np.mean((y_val - val_pred) ** 2)))

            mean_ret = float(np.mean(y_train))
            std_ret = float(np.std(y_train)) if len(y_train) > 1 else 0.0
            threshold = mean_ret + TRADE_SIGMA_MULT * std_ret if std_ret > 0 else mean_ret
            meta_out = {
                "feature_columns": feature_cols,
                "top_features_gain": top_features_gain,
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
                "val_rmse": val_rmse,
                "market_neutral": USE_MARKET_NEUTRAL_TARGET,
                "mean_return": mean_ret,
                "std_return": std_ret,
                "trade_threshold": threshold,
                "splits": {
                    "train_end": int(train_idx_end),
                    "val_start": int(val_idx_start),
                    "val_end": int(val_idx_end),
                    "test_start": int(test_idx_start),
                    "test_end": int(test_idx_end),
                    "gap": int(gap),
                },
            }
            if test_mask.any():
                y_test = y[test_mask]
                test_pred = model.predict(X[test_mask])
                test_rmse = float(np.sqrt(np.mean((y_test - test_pred) ** 2)))
                test_mae = float(np.mean(np.abs(y_test - test_pred)))
                # Conformal: absolute residual quantiles for calibrated intervals
                abs_resid_test = np.abs(y_test - test_pred)
                conformal_q90 = float(np.quantile(abs_resid_test, 0.90))
                conformal_q95 = float(np.quantile(abs_resid_test, 0.95))
                # Baselines: naive 0, last-return (momentum), momentum (log_return_1d)
                baseline_rmse = float(np.sqrt(np.mean(y_test ** 2)))
                last_returns = y[test_idx_start - 1 : test_idx_end - 1] if test_idx_start > 0 else np.zeros_like(y_test)
                # Momentum baseline: predict next return = last 1d return (if feature present)
                if "log_return_1d" in feature_cols:
                    idx = feature_cols.index("log_return_1d")
                    mom_pred = X[test_mask][:, idx]
                    baseline_momentum_rmse = float(np.sqrt(np.mean((y_test - mom_pred) ** 2)))
                else:
                    baseline_momentum_rmse = baseline_rmse
                if len(last_returns) == len(y_test):
                    baseline_last_rmse = float(np.sqrt(np.mean((y_test - last_returns) ** 2)))
                else:
                    baseline_last_rmse = baseline_rmse
                beats_naive = test_rmse < baseline_rmse
                beats_last = test_rmse < baseline_last_rmse
                # Direction metrics (important for trading usefulness)
                hit_rate = float(np.mean((y_test > 0) == (test_pred > 0)))
                correlation = float(np.corrcoef(y_test, test_pred)[0, 1]) if len(y_test) > 1 else 0.0
                if np.isnan(correlation):
                    correlation = 0.0
                production_ready = (
                    beats_naive and beats_last and hit_rate > 0.50
                )
                meta_out.update({
                    "n_test": int(test_mask.sum()),
                    "test_rmse": test_rmse,
                    "test_mae": test_mae,
                    "conformal_q90": conformal_q90,
                    "conformal_q95": conformal_q95,
                    "baseline_rmse": baseline_rmse,
                    "baseline_last_rmse": baseline_last_rmse,
                    "baseline_momentum_rmse": baseline_momentum_rmse,
                    "beats_baseline": beats_naive,
                    "beats_last_baseline": beats_last,
                    "hit_rate": hit_rate,
                    "correlation": correlation,
                    "production_ready": production_ready,
                    "has_holdout": True,
                    "eval_rmse": test_rmse,
                })
                logger.info(
                    f"Trained {ticker}-{window_name}: val_rmse={val_rmse:.4f} test_rmse={test_rmse:.4f} mae={test_mae:.4f} "
                    f"baseline={baseline_rmse:.4f} last={baseline_last_rmse:.4f} hit_rate={hit_rate:.1%} corr={correlation:.3f}"
                )
            else:
                # Conformal from val residuals when no holdout
                val_pred = model.predict(X_val)
                abs_resid_val = np.abs(y_val - val_pred)
                meta_out.update({
                    "n_test": 0,
                    "has_holdout": False,
                    "production_ready": False,
                    "eval_rmse": val_rmse,
                    "conformal_q90": float(np.quantile(abs_resid_val, 0.90)),
                    "conformal_q95": float(np.quantile(abs_resid_val, 0.95)),
                })
                logger.info(f"Trained {ticker}-{window_name}: val_rmse={val_rmse:.4f} (no holdout)")
            self.metadata[key] = meta_out

    def predict_all_windows(
        self,
        ticker: str,
        df: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """
        Predict for all windows. Returns dict of window -> {prediction, confidence, price_change, current_price, ...}
        """
        if self.feature_engineer is None:
            from ..data.features_minimal import MinimalFeatureEngineer
            self.feature_engineer = MinimalFeatureEngineer(self.mongo_client)

        features, meta = self.feature_engineer.prepare_features(
            df, ticker=ticker, mongo_client=self.mongo_client
        )
        if features is None or len(features) == 0:
            return {}

        results = {}
        # Use last row for prediction; current_price from df_aligned (matches feature row)
        X = features[-1:].astype(np.float32)
        df_aligned = meta.get("df_aligned")
        if df_aligned is not None and "Close" in df_aligned.columns:
            current_price = float(df_aligned["Close"].iloc[-1])
        else:
            current_price = float(df["Close"].iloc[-1]) if "Close" in df.columns else 0

        results = {}
        for window_name in self.prediction_windows:
            # Pooled model as default (better generalization); per-ticker only when proven better
            model = self.pooled_models.get(window_name)
            meta_w = self.pooled_metadata.get(window_name, {})
            key = (ticker, window_name)
            ticker_meta = self.metadata.get(key)
            if ticker_meta is not None and ticker_meta.get("n_train", 0) >= 300 and ticker_meta.get("production_ready", False):
                model = self.models.get(key, model)
                meta_w = ticker_meta
            if model is None:
                # logger.warning(f"No model for {ticker}-{window_name}") # Too noisy
                results[window_name] = {
                    "prediction": 0.0,
                    "price_change": 0.0,
                    "predicted_price": current_price,
                    "confidence": 0.0,
                    "current_price": current_price,
                    "prob_positive": 0.5,
                    "prob_above_threshold": 0.5,
                    "trade_recommended": False,
                    "trade_threshold": TRADE_MIN_ALPHA,
                    "normalized_return": 0.0,
                    "horizon_days": TARGET_CONFIG.get(window_name, {}).get("horizon", 1),
                    "min_return_for_profit": ROUND_TRIP_COST_BPS / 10000,
                    "covers_transaction_cost": False,
                    "reason": "no_model" 
                }
                continue

            meta_w = self.metadata.get(key) or self.pooled_metadata.get(window_name, {})
            # No scaling (tree models)
            pred_return = float(model.predict(X)[0])
            pred_price = current_price * math.exp(pred_return)
            price_change = pred_price - current_price
            
            sigma = float(meta_w.get("val_rmse", 0.0))
            prob_positive = (
                0.5 * (1 + math.erf(pred_return / (sigma * math.sqrt(2)))) if sigma > 0 else 0.5
            )
            confidence = prob_positive
            
            # Restore price range
            price_low = current_price * math.exp(pred_return - 2 * sigma)
            price_high = current_price * math.exp(pred_return + 2 * sigma)

            # Skip-trade rule: use regime-adaptive threshold (mean+2*sigma) when available
            raw_threshold = meta_w.get("trade_threshold")
            if raw_threshold is None:
                raw_threshold = float(TRADE_MIN_ALPHA)
            else:
                raw_threshold = float(raw_threshold)
            min_alpha = max(raw_threshold, TRADE_MIN_ALPHA)
            trade_recommended = (
                pred_return >= min_alpha and prob_positive >= TRADE_MIN_PROB_POSITIVE
            )

            # P(return > threshold) - classification-style (Losing Loonies v4: tradable moves)
            prob_above_threshold = (
                0.5 * (1 - math.erf((raw_threshold - pred_return) / (sigma * math.sqrt(2))))
                if sigma > 0 else 0.5
            )
            prob_above_threshold = max(0.0, min(1.0, prob_above_threshold))

            # Transaction cost: round-trip in log space
            min_return_for_profit = (ROUND_TRIP_COST_BPS / 10000)
            covers_transaction_cost = abs(pred_return) >= min_return_for_profit

            # Normalized return for horizon selection (compound-equivalent daily log return)
            # For log returns, pred_return/horizon_days is correct (Losing Loonies v4: don't divide raw R by days)
            horizon = TARGET_CONFIG.get(window_name, {}).get("horizon", 1)
            horizon_days = max(1, horizon)
            normalized_return = pred_return / horizon_days

            results[window_name] = {
                "prediction": pred_return,
                "price_change": float(price_change),
                "predicted_price": float(pred_price),
                "confidence": float(confidence),
                "current_price": current_price,
                "price_range": {
                    "low": price_low,
                    "high": price_high,
                },
                "alpha": pred_return,
                "prob_positive": float(prob_positive),
                "prob_above_threshold": float(prob_above_threshold),
                "trade_recommended": trade_recommended,
                "trade_threshold": float(min_alpha),
                "normalized_return": float(normalized_return),
                "horizon_days": int(horizon_days),
                "min_return_for_profit": min_return_for_profit,
                "covers_transaction_cost": covers_transaction_cost,
            }

        # Add best_window: highest normalized_return among trade_recommended, else highest normalized_return
        tradeable = [(w, r) for w, r in results.items() if r.get("trade_recommended")]
        if tradeable:
            best_window = max(tradeable, key=lambda x: x[1].get("normalized_return", 0))[0]
        else:
            best_window = max(
                results.items(),
                key=lambda x: x[1].get("normalized_return", 0),
            )[0]
        results["_meta"] = {"best_window": best_window}
        return results

    def load_models(self) -> None:
        """Load models from disk (per-ticker + pooled)."""
        base = os.path.join(MODEL_DIR, "v1")
        if not os.path.exists(base):
            logger.info("No v1 models found to load")
            return
        pooled_path = os.path.join(base, "_pooled")
        if os.path.isdir(pooled_path):
            for f in os.listdir(pooled_path):
                if f.endswith(".joblib"):
                    for w in self.prediction_windows:
                        if w in f:
                            try:
                                self.pooled_models[w] = joblib.load(
                                    os.path.join(pooled_path, f)
                                )
                            except Exception as e:
                                logger.warning("Could not load pooled %s: %s", w, e)
            if self.pooled_models:
                logger.info("Loaded %d pooled models", len(self.pooled_models))
            meta_path = os.path.join(pooled_path, "metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        self.pooled_metadata = json.load(f)
                except Exception as e:
                    logger.warning("Could not load pooled metadata: %s", e)
        for ticker in os.listdir(base):
            if ticker == "_pooled":
                continue
            ticker_path = os.path.join(base, ticker)
            if not os.path.isdir(ticker_path):
                continue
            meta_path = os.path.join(ticker_path, "metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as fp:
                        ticker_meta = json.load(fp)
                    for w, meta in ticker_meta.items():
                        self.metadata[(ticker, w)] = meta
                except Exception as e:
                    logger.warning("Could not load metadata for %s: %s", ticker, e)
            for f in os.listdir(ticker_path):
                if f.endswith(".joblib") and "scaler" not in f:
                    for w in self.prediction_windows:
                        if w in f:
                            key = (ticker, w)
                            path = os.path.join(ticker_path, f)
                            try:
                                self.models[key] = joblib.load(path)
                                scaler_path = path.replace(".joblib", "_scaler.joblib")
                                if os.path.exists(scaler_path):
                                    self.scalers[key] = joblib.load(scaler_path)
                            except Exception as e:
                                logger.warning(f"Could not load {path}: {e}")
        logger.info(f"Loaded {len(self.models)} models")

    def save_models(self) -> None:
        """Save models to disk (per-ticker + pooled)."""
        base = os.path.join(MODEL_DIR, "v1")
        pooled_path = os.path.join(base, "_pooled")
        if self.pooled_models:
            os.makedirs(pooled_path, exist_ok=True)
            for window, model in self.pooled_models.items():
                try:
                    joblib.dump(model, os.path.join(pooled_path, f"lgb_{window}.joblib"))
                except Exception as e:
                    logger.warning("Could not save pooled %s: %s", window, e)
            if self.pooled_metadata:
                try:
                    with open(os.path.join(pooled_path, "metadata.json"), "w") as f:
                        json.dump(self.pooled_metadata, f, indent=2)
                except Exception as e:
                    logger.warning("Could not save pooled metadata: %s", e)
        for (ticker, window), model in self.models.items():
            path = os.path.join(base, ticker)
            os.makedirs(path, exist_ok=True)
            try:
                joblib.dump(model, os.path.join(path, f"lgb_{window}.joblib"))
                scaler = self.scalers.get((ticker, window))
                if scaler is not None:
                    joblib.dump(scaler, os.path.join(path, f"lgb_{window}_scaler.joblib"))
            except Exception as e:
                logger.warning(f"Could not save {ticker}-{window}: {e}")
        # Save per-ticker metadata
        ticker_meta_by_ticker = {}
        for (ticker, window), meta in self.metadata.items():
            if ticker not in ticker_meta_by_ticker:
                ticker_meta_by_ticker[ticker] = {}
            ticker_meta_by_ticker[ticker][window] = meta
        for ticker, meta_dict in ticker_meta_by_ticker.items():
            path = os.path.join(base, ticker)
            if os.path.isdir(path):
                try:
                    with open(os.path.join(path, "metadata.json"), "w") as f:
                        json.dump(meta_dict, f, indent=2)
                except Exception as e:
                    logger.warning("Could not save metadata for %s: %s", ticker, e)
        logger.info(f"Saved {len(self.models)} models")
