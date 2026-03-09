"""
StockPredictor - LightGBM-based predictor with walk-forward validation.
Leakage-proof, time-based splits, proper evaluation.
"""

import json
import hashlib
import math
import os
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV

from ..config.constants import PREDICTION_WINDOWS
from ..config.feature_config_v1 import (
    FEATURE_CONFIG_V1,
    FEATURE_PRUNING,
    FEATURE_PRUNING_TOP_K_BY_HORIZON,
    LIGHTGBM_PARAMS,
    LIGHTGBM_PARAMS_NEXT_DAY,
    POOL_CONFIG,
    TARGET_CONFIG,
    USE_MARKET_NEUTRAL_TARGET,
    WALK_FORWARD_FOLDS,
    TRADE_MIN_ALPHA,
    TRADE_MIN_PROB_POSITIVE,
    TRADE_MIN_PROB_BY_HORIZON,
    TRADE_SIGMA_MULT,
    TRADE_THRESHOLD_CAP,
    ROUND_TRIP_COST_BPS,
)

logger = logging.getLogger(__name__)

# Default model directory
MODEL_DIR = os.getenv("MODEL_DIR", "models")

# Model version — bump when feature set, hyperparams, or architecture changes.
# Stored in every prediction document for reproducibility.
# v5.0.0: Major accuracy overhaul:
#   - Ensemble blending (pooled + per-ticker weighted average) replaces per-ticker-only
#   - Confidence from regression prediction magnitude + model quality (removed useless sign classifier dependency)
#   - Walk-forward starts at 50% (was 40%), 5 folds for robustness
#   - Recency weighting moderated to 2x (was 3x) to preserve effective sample size
#   - LightGBM: deeper trees, more leaves, lower regularization to find weak signals
#   - Sign classifier removed from confidence calc (was producing 50% = coin flip)
#   - Trade thresholds substantially lowered to generate meaningful trade counts
#   - Early stopping patience increased to 50 rounds
MODEL_VERSION = "v5.0.0"


def _lgb_params_for_horizon(window_name: str) -> dict:
    """Return horizon-appropriate LightGBM hyperparameters.

    next_day uses heavier regularization (LIGHTGBM_PARAMS_NEXT_DAY) because
    1-day alpha is much noisier than 7d/30d.
    """
    if window_name == "next_day":
        return dict(LIGHTGBM_PARAMS_NEXT_DAY)
    return dict(LIGHTGBM_PARAMS)


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
        self.sign_models = {}  # {(ticker, window): lgb.LGBMClassifier}
        self.pooled_sign_models = {}  # {window: lgb.LGBMClassifier}
        self.prediction_windows = list(PREDICTION_WINDOWS.keys())
        # Track the latest training cutoff date across all tickers/windows
        # so the backtest can restrict itself to OOS dates only.
        self._train_cutoff_dates = {"pooled": [], "ticker": []}  # per model type
        self._feature_shortlist = {}  # {window_name: [col_names]} — pruned feature set per horizon

    def set_feature_engineer(self, feature_engineer):
        """Set the feature engineer (MinimalFeatureEngineer or compatible)."""
        self.feature_engineer = feature_engineer

    # ------------------------------------------------------------------
    # Feature pruning helpers
    # ------------------------------------------------------------------
    def _build_feature_shortlists(self) -> None:
        """Build per-horizon feature shortlists from pooled model importance.

        Uses Phase-1 fold-0 importance only (not averaged across all folds)
        to avoid information leakage from later validation folds informing
        feature selection.  Keeps protected features + top-k by gain.
        Called after Phase 1 pooled training so Phase 2 retrains with a
        cleaner feature set.
        """
        if not FEATURE_PRUNING.get("enabled", False):
            return
        top_k = FEATURE_PRUNING.get("top_k", 30)
        protected = set(FEATURE_PRUNING.get("protected_features", []))
        min_feats = FEATURE_PRUNING.get("min_features", 15)

        for window_name, meta in self.pooled_metadata.items():
            # Per-horizon top_k: next_day gets fewer features (simpler model for noisy target)
            horizon_top_k = FEATURE_PRUNING_TOP_K_BY_HORIZON.get(window_name, top_k)
            all_cols = meta.get("feature_columns", [])
            # Prefer fold-0 importance to prevent future-fold leakage
            importance = meta.get("fold0_features_gain") or meta.get("top_features_gain", [])
            if not all_cols or not importance:
                continue
            # Top-k feature names by gain (already sorted desc in metadata)
            top_names = [entry["name"] for entry in importance[:horizon_top_k]]
            # Merge protected (that actually exist) + top-k
            shortlist = sorted(
                set(top_names) | (protected & set(all_cols))
            )
            if len(shortlist) < min_feats:
                logger.info(
                    "Pruning %s: shortlist too small (%d < %d) — skipping",
                    window_name, len(shortlist), min_feats,
                )
                continue
            self._feature_shortlist[window_name] = shortlist
            logger.info(
                "Pruning %s: %d → %d features (top_k=%d, protected=%d)",
                window_name, len(all_cols), len(shortlist), horizon_top_k,
                len(protected & set(all_cols)),
            )

    def _select_features(
        self, X: np.ndarray, current_cols: list, model_cols: list,
        ticker: str = "", window: str = "",
    ) -> pd.DataFrame:
        """Select & reorder columns of *X* so they match *model_cols*.

        If model_cols is None or identical to current_cols, returns X as a
        DataFrame.  Missing columns are NaN-filled (LightGBM native missing)
        and a warning is logged.  Always returns a DataFrame with model_cols
        as column names to suppress sklearn feature-name warnings.
        """
        if not model_cols:
            return pd.DataFrame(X, columns=current_cols) if current_cols else pd.DataFrame(X)
        if model_cols == current_cols:
            return pd.DataFrame(X, columns=model_cols)

        col_to_idx = {c: i for i, c in enumerate(current_cols)}
        missing = [c for c in model_cols if c not in col_to_idx]
        extra = [c for c in current_cols if c not in set(model_cols)]

        if missing:
            logger.warning(
                "Feature mismatch [%s-%s]: %d missing cols (NaN-filled): %s",
                ticker, window, len(missing), missing[:10],
            )
        if extra and len(extra) <= 20:
            logger.debug(
                "Feature mismatch [%s-%s]: %d extra cols (ignored): %s",
                ticker, window, len(extra), extra[:10],
            )

        # Build output array with model_cols ordering, NaN-fill missing
        # (LightGBM treats NaN as missing and learns optimal split direction)
        n_rows = X.shape[0]
        out = np.full((n_rows, len(model_cols)), np.nan, dtype=np.float32)
        for j, col in enumerate(model_cols):
            if col in col_to_idx:
                out[:, j] = X[:, col_to_idx[col]]
            # else: stays NaN (LightGBM native missing-value)
        return pd.DataFrame(out, columns=model_cols)

    def get_oos_start_date(self) -> Optional[pd.Timestamp]:
        """Return the earliest date the backtest should start (after all training data).

        Uses true holdout/test start dates from per-ticker metadata splits
        and pooled holdout dates. For strict OOS, we need the date *after* the
        latest test_start across the model type that will be used at inference.
        Since inference prefers pooled unless ticker is prod_ready, we use the
        latest cutoff across all model types to be safe.
        """
        all_cutoffs = []
        # Pooled cutoffs
        for ts in self._train_cutoff_dates.get("pooled", []):
            all_cutoffs.append(ts)
        # Per-ticker test start dates from metadata splits
        for (ticker, window), meta in self.metadata.items():
            splits = meta.get("splits", {})
            tsd = splits.get("test_start_date_x")
            if tsd is not None:
                all_cutoffs.append(pd.Timestamp(tsd))
        # Fallback to ticker cutoffs if no test_start dates stored
        if not all_cutoffs:
            for ts in self._train_cutoff_dates.get("ticker", []):
                all_cutoffs.append(ts)
        if not all_cutoffs:
            return None
        # Use the *latest* cutoff so that no model's training data leaks
        cutoff = max(all_cutoffs)
        return cutoff + pd.Timedelta(days=1)

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
            
            # Phase 1: train pooled with ALL features → extract importance
            self.train_pooled_models(historical_data, fe)

            # Phase 2: build per-horizon feature shortlists, then retrain pooled
            if FEATURE_PRUNING.get("enabled", False):
                self._build_feature_shortlists()
                if self._feature_shortlist:
                    logger.info("Retraining pooled models with pruned features …")
                    self._train_cutoff_dates["pooled"] = []  # reset for clean Phase 2
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
        end = datetime.now(timezone.utc) if not end_date else pd.to_datetime(end_date)
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

        # --- Feature coverage guardrails ---
        # Log per-feature-group coverage so silent batch-wide missingness is visible
        if feature_cols_ref:
            _feature_groups = {
                "short_interest": [c for c in feature_cols_ref if c.startswith("si_")],
                "sentiment": [c for c in feature_cols_ref if c.startswith("sent_") or "sentiment" in c],
                "insider": [c for c in feature_cols_ref if c.startswith("insider_")],
                "earnings": [c for c in feature_cols_ref if c.startswith("earn_")],
                "fundamental": [c for c in feature_cols_ref if c.startswith("fund_")],
                "sector_etf": [c for c in feature_cols_ref if c.startswith("sector_etf_") or c.startswith("excess_vs_")],
            }
            first_window = next(iter(TARGET_CONFIG))
            if rows_by_window[first_window]:
                X_check = np.vstack(rows_by_window[first_window])
                col_idx = {c: i for i, c in enumerate(feature_cols_ref)}
                coverage_lines = ["[FEATURE-COVERAGE] Per-group non-zero coverage in pooled training data:"]
                for group_name, cols in _feature_groups.items():
                    if not cols:
                        continue
                    idxs = [col_idx[c] for c in cols if c in col_idx]
                    if not idxs:
                        continue
                    group_data = X_check[:, idxs]
                    nonzero_pct = float((group_data != 0).any(axis=1).mean()) * 100
                    flag = " ⚠️ LOW" if nonzero_pct < 30 else ""
                    coverage_lines.append(f"  {group_name}: {nonzero_pct:.1f}% rows have non-zero values ({len(cols)} features){flag}")
                logger.info("\n".join(coverage_lines))

        # --- SEC/FMP consistency metadata ---
        skip_sec_fmp = os.environ.get("SKIP_SEC_FMP", "false").lower() == "true"
        if skip_sec_fmp:
            logger.warning("[SEC/FMP] Training with SKIP_SEC_FMP=true — SEC/FMP features will be zero-filled. "
                           "Ensure SKIP_SEC_FMP is also set at inference time for consistency.")

        for window_name in TARGET_CONFIG:
            if not rows_by_window[window_name]:
                continue
            X_all = np.vstack(rows_by_window[window_name])
            y_all = np.concatenate(y_by_window[window_name])
            dates_all = np.concatenate(dates_by_window[window_name])

            # --- Feature pruning: select shortlisted columns (Phase 2) ---
            active_cols = list(feature_cols_ref) if feature_cols_ref else []
            if window_name in self._feature_shortlist:
                shortlist = self._feature_shortlist[window_name]
                col_to_idx = {c: i for i, c in enumerate(feature_cols_ref)}
                sel_idx = sorted([col_to_idx[c] for c in shortlist if c in col_to_idx])
                if len(sel_idx) >= FEATURE_PRUNING.get("min_features", 15):
                    X_all = X_all[:, sel_idx]
                    active_cols = [feature_cols_ref[i] for i in sel_idx]
                    logger.info(
                        "POOLED-%s: pruned %d → %d features",
                        window_name, len(feature_cols_ref), len(active_cols),
                    )

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

            # Verify dates are monotonically sorted after argsort
            assert np.all(dates_all[:-1] <= dates_all[1:]), (
                f"Pooled {window_name}: dates not sorted after argsort — data integrity error"
            )

            n = len(y_all)
            n_folds = max(1, WALK_FORWARD_FOLDS)
            fold_rmses, fold_maes, fold_hits, fold_corrs = [], [], [], []
            fold_q90, fold_q95 = [], []
            fold0_features_gain = []  # fold-0 importance for leakage-free pruning
            model = None

            purge = FEATURE_CONFIG_V1["purge_days"]
            embargo = FEATURE_CONFIG_V1["embargo_days"]

            for fold in range(n_folds):
                if n_folds == 1:
                    train_end = int(n * 0.85)
                else:
                    # v5.0: start at 50% (was 40%) — more training data per fold
                    # Each fold advances by 8% so 5 folds cover 50%→82% of data
                    train_end = int(n * (0.50 + fold * 0.08))

                # Purge/embargo gap to prevent horizon overlap leakage
                horizon = TARGET_CONFIG[window_name]["horizon"]
                gap = max(purge, horizon) + embargo
                val_start = min(train_end + gap, n - 100)

                if val_start >= n - 50:
                    break
                val_end = min(val_start + int(n * 0.1), n)

                # --- Cross-sectional purge ---
                # In pooled training, multiple tickers share the same date.
                # The temporal purge/embargo gap only removes samples ±gap
                # indices apart, but since data is sorted by date (with same-
                # date tickers adjacent), we must also exclude all samples
                # from dates that fall within the purge window of the val set.
                train_dates = dates_all[:train_end]
                val_dates = dates_all[val_start:val_end]
                if len(val_dates) > 0 and len(train_dates) > 0:
                    val_min_date = val_dates.min()
                    # Remove training samples whose date is within `gap` trading
                    # days of val_min_date (cross-sectional contamination).
                    purge_cutoff = val_min_date - np.timedelta64(gap, 'D')
                    cross_sect_mask = train_dates <= purge_cutoff
                    n_purged = int((~cross_sect_mask).sum())
                    if n_purged > 0:
                        logger.debug(
                            "POOLED-%s fold %d: cross-sectional purge removed %d/%d train samples",
                            window_name, fold, n_purged, train_end,
                        )
                    X_train = X_all[:train_end][cross_sect_mask]
                    y_train = y_all[:train_end][cross_sect_mask]
                else:
                    X_train = X_all[:train_end]
                    y_train = y_all[:train_end]

                X_val = X_all[val_start:val_end]
                y_val = y_all[val_start:val_end]
                if len(y_train) < 200 or len(y_val) < 20:
                    break

                # v5.0: Recency weighting ~2x ratio (exp(-0.7) to exp(0))
                # Previous 3x ratio (exp(-1.1)) was too aggressive, reducing effective
                # sample size and hurting generalization on noisy targets.
                w = np.exp(np.linspace(-0.7, 0.0, len(y_train))).astype(np.float32)
                _horizon_params = _lgb_params_for_horizon(window_name)
                fold_model = lgb.LGBMRegressor(**_horizon_params)
                X_train_df = pd.DataFrame(X_train, columns=active_cols)
                X_val_df = pd.DataFrame(X_val, columns=active_cols)
                fold_model.fit(
                    X_train_df,
                    y_train,
                    sample_weight=w,
                    eval_set=[(X_val_df, y_val)],
                    callbacks=[
                        lgb.early_stopping(50, verbose=False),
                        lgb.log_evaluation(0),
                    ],
                )
                pred = fold_model.predict(X_val_df)
                fold_rmses.append(float(np.sqrt(np.mean((y_val - pred) ** 2))))
                fold_maes.append(float(np.mean(np.abs(y_val - pred))))
                fold_hits.append(float(np.mean((y_val > 0) == (pred > 0))))
                # Guard: np.corrcoef raises RuntimeWarning on constant input
                if len(y_val) > 1 and np.std(pred) > 1e-12 and np.std(y_val) > 1e-12:
                    c = np.corrcoef(y_val, pred)[0, 1]
                else:
                    c = 0.0
                fold_corrs.append(float(c) if not np.isnan(c) else 0.0)
                # Conformal: absolute residual quantiles for calibrated intervals
                abs_resid = np.abs(y_val - pred)
                fold_q90.append(float(np.quantile(abs_resid, 0.90)))
                fold_q95.append(float(np.quantile(abs_resid, 0.95)))
                # Keep the last model
                model = fold_model

                # Capture fold-0 feature importance for leakage-free pruning
                if fold == 0:
                    try:
                        _booster = fold_model.booster_
                        _gains = _booster.feature_importance(importance_type="gain")
                        _pairs = sorted(zip(active_cols, _gains), key=lambda x: x[1], reverse=True)[:30]
                        fold0_features_gain = [{"name": n, "gain": float(g)} for n, g in _pairs]
                    except Exception:
                        pass

            if not fold_rmses:
                continue
            rmse = float(np.median(fold_rmses))
            mae = float(np.median(fold_maes))
            hit_rate = float(np.median(fold_hits))
            correlation = float(np.median(fold_corrs))
            q90_med = float(np.median(fold_q90)) if fold_q90 else rmse
            q95_med = float(np.median(fold_q95)) if fold_q95 else rmse * 1.3

            # === True tail holdout: data after the last fold's val_end ===
            # The walk-forward folds validate incrementally but the final model
            # has seen all fold training data. Reserve a held-out tail segment
            # with a proper purge/embargo gap for honest OOS evaluation.
            horizon = TARGET_CONFIG[window_name]["horizon"]
            holdout_gap = max(purge, horizon) + embargo
            holdout_start = min(val_end + holdout_gap, n)
            pooled_holdout_meta = {}
            if holdout_start < n - 20:
                X_holdout = X_all[holdout_start:]
                y_holdout = y_all[holdout_start:]
                X_holdout_df = pd.DataFrame(X_holdout, columns=active_cols)
                holdout_pred = model.predict(X_holdout_df)
                holdout_rmse = float(np.sqrt(np.mean((y_holdout - holdout_pred) ** 2)))
                holdout_mae = float(np.mean(np.abs(y_holdout - holdout_pred)))
                holdout_hit = float(np.mean((y_holdout > 0) == (holdout_pred > 0)))
                # Guard: np.corrcoef raises RuntimeWarning on constant input
                if len(y_holdout) > 1 and np.std(holdout_pred) > 1e-12 and np.std(y_holdout) > 1e-12:
                    h_c = np.corrcoef(y_holdout, holdout_pred)[0, 1]
                else:
                    h_c = 0.0
                holdout_corr = float(h_c) if not np.isnan(h_c) else 0.0
                # Conformal from holdout residuals (truest calibration)
                abs_resid_ho = np.abs(y_holdout - holdout_pred)
                holdout_q90 = float(np.quantile(abs_resid_ho, 0.90))
                holdout_q95 = float(np.quantile(abs_resid_ho, 0.95))
                # Override fold-based conformal with holdout-based (more honest)
                q90_med = holdout_q90
                q95_med = holdout_q95
                pooled_holdout_meta = {
                    "holdout_rmse": holdout_rmse,
                    "holdout_mae": holdout_mae,
                    "holdout_hit_rate": holdout_hit,
                    "holdout_correlation": holdout_corr,
                    "holdout_n": int(len(y_holdout)),
                    "holdout_start_date": str(pd.Timestamp(dates_all[holdout_start])),
                }
                logger.info(
                    "POOLED-%s holdout: rmse=%.4f hit=%.1f%% corr=%.3f n=%d start=%s",
                    window_name, holdout_rmse, holdout_hit * 100, holdout_corr,
                    len(y_holdout), pd.Timestamp(dates_all[holdout_start]).date(),
                )
                # OOS boundary = holdout start (the first truly unseen date)
                self._train_cutoff_dates["pooled"].append(pd.Timestamp(dates_all[holdout_start]))
            else:
                logger.warning(
                    "POOLED-%s: not enough data for tail holdout (val_end=%d, n=%d)",
                    window_name, val_end, n,
                )
                # Fallback: use val_end as OOS boundary
                if val_end < len(dates_all):
                    self._train_cutoff_dates["pooled"].append(pd.Timestamp(dates_all[val_end - 1]))

            # Trade threshold from PREDICTION distribution, not return distribution.
            # Model outputs are compressed (regression-to-mean), so std_pred << std_returns.
            # Using std_returns made the threshold unreachably high (0 trades).
            mean_ret = float(np.mean(y_all))
            std_ret = float(np.std(y_all)) if len(y_all) > 1 else 0.0
            # Compute threshold from last fold's validation predictions
            val_pred_all = model.predict(pd.DataFrame(X_all[val_start:val_end], columns=active_cols)) if val_start < val_end else model.predict(pd.DataFrame(X_all[-100:], columns=active_cols))
            pred_mean = float(np.mean(val_pred_all))
            pred_std = float(np.std(val_pred_all)) if len(val_pred_all) > 1 else 0.0
            threshold = pred_mean + TRADE_SIGMA_MULT * pred_std if pred_std > 0 else float(TRADE_MIN_ALPHA)
            # Cap threshold per horizon to prevent unreasonably high/low values
            _cap = TRADE_THRESHOLD_CAP.get(window_name, {})
            if _cap:
                threshold = max(_cap["min"], min(_cap["max"], threshold))
                logger.info("POOLED-%s threshold=%.5f (capped to [%.5f, %.5f])", window_name, threshold, _cap["min"], _cap["max"])
            # Persist feature importance for pooled model
            top_features_gain_pooled = []
            try:
                booster = model.booster_
                gains = booster.feature_importance(importance_type="gain")
                cols = active_cols
                pairs = sorted(zip(cols, gains), key=lambda x: x[1], reverse=True)[:30]
                top_features_gain_pooled = [{"name": n, "gain": float(g)} for n, g in pairs]
            except Exception:
                pass
            self.pooled_models[window_name] = model
            pooled_meta = {
                "val_rmse": rmse,
                "val_mae": mae,
                "conformal_q90": q90_med,
                "conformal_q95": q95_med,
                "top_features_gain": top_features_gain_pooled,
                "fold0_features_gain": fold0_features_gain,
                "n": int(n),
                "hit_rate": hit_rate,
                "correlation": correlation,
                "market_neutral": USE_MARKET_NEUTRAL_TARGET,
                "feature_columns": active_cols,
                "mean_return": mean_ret,
                "std_return": std_ret,
                "trade_threshold": threshold,
            }

            # --- Sign classifier (calibrated P(up)) ---
            # v5.0: Stronger sign classifier with deeper trees and lower regularization.
            # Previous version produced 45-50% accuracy (coin flip or worse) because
            # it was over-regularized (depth-3, 12 leaves, reg_alpha=0.3).
            try:
                sign_y = (y_all > 0).astype(int)
                n_pos = int(sign_y[:train_end].sum())
                n_neg = int(len(sign_y[:train_end]) - n_pos)
                spw = n_neg / max(n_pos, 1)
                sign_params = {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "boosting_type": "gbdt",
                    "n_estimators": 800,        # v5.0: more capacity
                    "max_depth": 5,             # v5.0: deeper (was 3) to find directional patterns
                    "learning_rate": 0.02,
                    "num_leaves": 31,           # v5.0: more leaves (was 12)
                    "min_child_samples": 25,    # v5.0: finer splits (was 30)
                    "scale_pos_weight": spw,
                    "reg_alpha": 0.05,          # v5.0: much lower L1 (was 0.3)
                    "reg_lambda": 0.3,          # v5.0: lower L2 (was 0.5)
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                    "verbosity": -1,
                    "n_jobs": -1,
                }
                if window_name == "next_day":
                    sign_params.update({
                        "max_depth": 4,
                        "num_leaves": 20,
                        "min_child_samples": 30,
                        "reg_alpha": 0.1,
                        "reg_lambda": 0.5,
                        "subsample": 0.75,
                        "colsample_bytree": 0.7,
                    })
                _SIGN_MIN_TREES = 80  # v5.0: need enough trees for meaningful signal
                if val_start < val_end and val_start < n:
                    sign_X_train = X_all[:train_end]
                    sign_y_train = sign_y[:train_end]
                    sign_X_val = X_all[val_start:val_end]
                    sign_y_val = sign_y[val_start:val_end]
                    if len(sign_y_train) >= 200 and len(sign_y_val) >= 20:
                        # v5.0: moderate recency weighting (2x ratio)
                        w_sign = np.exp(np.linspace(-0.7, 0.0, len(sign_y_train))).astype(np.float32)
                        sign_clf = lgb.LGBMClassifier(**sign_params)
                        sign_X_train_df = pd.DataFrame(sign_X_train, columns=active_cols)
                        sign_X_val_df = pd.DataFrame(sign_X_val, columns=active_cols)
                        sign_clf.fit(
                            sign_X_train_df,
                            sign_y_train,
                            sample_weight=w_sign,
                            eval_set=[(sign_X_val_df, sign_y_val)],
                            callbacks=[
                                lgb.early_stopping(50, verbose=False),
                                lgb.log_evaluation(0),
                            ],
                        )
                        n_trees = sign_clf.booster_.num_trees()
                        if n_trees < _SIGN_MIN_TREES:
                            logger.info(
                                "POOLED-%s sign clf stopped at %d trees (< %d); "
                                "retraining with fixed %d rounds",
                                window_name, n_trees, _SIGN_MIN_TREES, _SIGN_MIN_TREES,
                            )
                            sign_params_fixed = {**sign_params, "n_estimators": _SIGN_MIN_TREES}
                            sign_clf = lgb.LGBMClassifier(**sign_params_fixed)
                            sign_clf.fit(
                                sign_X_train_df,
                                sign_y_train,
                                sample_weight=w_sign,
                                eval_set=[(sign_X_val_df, sign_y_val)],
                                callbacks=[lgb.log_evaluation(0)],
                            )
                        self.pooled_sign_models[window_name] = sign_clf
                        # Platt-scale (sigmoid calibration) on the val set
                        try:
                            cal_clf = CalibratedClassifierCV(sign_clf, method="sigmoid", cv="prefit")
                            cal_clf.fit(sign_X_val_df, sign_y_val)
                            self.pooled_sign_models[window_name] = cal_clf
                        except Exception as cal_err:
                            logger.debug("Platt calibration skipped for pooled-%s: %s", window_name, cal_err)
                        # Evaluate on val set
                        sign_proba = self.pooled_sign_models[window_name].predict_proba(sign_X_val_df)[:, 1]
                        sign_acc = float(np.mean((sign_proba > 0.5) == sign_y_val))
                        pooled_meta["sign_classifier_accuracy"] = sign_acc
                        n_final_trees = sign_clf.booster_.num_trees()
                        logger.info(
                            "POOLED-%s sign classifier: accuracy=%.1f%% trees=%d spw=%.3f",
                            window_name, sign_acc * 100, n_final_trees, spw,
                        )
            except Exception as e:
                logger.warning("Could not train pooled sign classifier for %s: %s", window_name, e)
            pooled_meta.update(pooled_holdout_meta)
            pooled_meta["skip_sec_fmp_at_train"] = skip_sec_fmp
            self.pooled_metadata[window_name] = pooled_meta
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

            # --- Feature pruning: select shortlisted columns ---
            active_cols = feature_cols
            if window_name in self._feature_shortlist:
                shortlist = self._feature_shortlist[window_name]
                col_to_idx = {c: i for i, c in enumerate(feature_cols)}
                sel_idx = sorted([col_to_idx[c] for c in shortlist if c in col_to_idx])
                if len(sel_idx) >= FEATURE_PRUNING.get("min_features", 15):
                    X = X[:, sel_idx]
                    active_cols = [feature_cols[i] for i in sel_idx]

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

            # v5.0: Recency weighting ~2x ratio (exp(-0.7) to exp(0))
            w = np.exp(np.linspace(-0.7, 0.0, len(y_train))).astype(np.float32)
            _horizon_params = _lgb_params_for_horizon(window_name)
            model = lgb.LGBMRegressor(**_horizon_params)
            X_train_df = pd.DataFrame(X_train, columns=active_cols)
            X_val_df = pd.DataFrame(X_val, columns=active_cols)
            model.fit(
                X_train_df,
                y_train,
                sample_weight=w,
                eval_set=[(X_val_df, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )

            key = (ticker, window_name)
            self.models[key] = model
            self.scalers[key] = None  # No scaling

            # --- Per-ticker sign classifier ---
            # v5.0: Stronger per-ticker sign classifier with deeper trees
            try:
                sign_y_train = (y_train > 0).astype(int)
                sign_y_val = (y_val > 0).astype(int)
                if len(np.unique(sign_y_train)) == 2 and len(X_train) >= 60:
                    n_pos_tk = int(sign_y_train.sum())
                    n_neg_tk = int(len(sign_y_train) - n_pos_tk)
                    spw_tk = n_neg_tk / max(n_pos_tk, 1)
                    sign_params_tk = {
                        "objective": "binary", "metric": "binary_logloss",
                        "boosting_type": "gbdt", "n_estimators": 500,
                        "max_depth": 4, "learning_rate": 0.02, "num_leaves": 20,
                        "min_child_samples": 20, "scale_pos_weight": spw_tk,
                        "reg_alpha": 0.05, "reg_lambda": 0.3,
                        "subsample": 0.8, "colsample_bytree": 0.8,
                        "random_state": 42, "verbosity": -1, "n_jobs": -1,
                    }
                    if window_name == "next_day":
                        sign_params_tk.update({
                            "max_depth": 3, "num_leaves": 15,
                            "min_child_samples": 25,
                            "reg_alpha": 0.1, "reg_lambda": 0.5,
                            "subsample": 0.75, "colsample_bytree": 0.7,
                        })
                    _SIGN_MIN_TREES_TK = 50  # v5.0: increased from 30
                    sign_clf_tk = lgb.LGBMClassifier(**sign_params_tk)
                    sign_clf_tk.fit(
                        X_train_df, sign_y_train, sample_weight=w,
                        eval_set=[(X_val_df, sign_y_val)],
                        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
                    )
                    n_trees_tk = sign_clf_tk.booster_.num_trees()
                    if n_trees_tk < _SIGN_MIN_TREES_TK:
                        sign_params_tk_fixed = {**sign_params_tk, "n_estimators": _SIGN_MIN_TREES_TK}
                        sign_clf_tk = lgb.LGBMClassifier(**sign_params_tk_fixed)
                        sign_clf_tk.fit(
                            X_train_df, sign_y_train, sample_weight=w,
                            eval_set=[(X_val_df, sign_y_val)],
                            callbacks=[lgb.log_evaluation(0)],
                        )
                    self.sign_models[key] = sign_clf_tk
                    # Platt-scale on val set
                    try:
                        cal_tk = CalibratedClassifierCV(sign_clf_tk, method="sigmoid", cv="prefit")
                        cal_tk.fit(X_val_df, sign_y_val)
                        self.sign_models[key] = cal_tk
                    except Exception as cal_err:
                        logger.debug("Platt calibration skipped for %s-%s: %s", ticker, window_name, cal_err)
            except Exception as e:
                logger.debug("Could not train sign classifier for %s-%s: %s", ticker, window_name, e)
            # Record training cutoff for OOS backtest boundary (ticker bucket)
            if val_mask.any():
                self._train_cutoff_dates["ticker"].append(pd.Timestamp(dates_x[val_mask].max()))
            # Persist feature importance for drift/debugging
            top_features_gain = []
            try:
                booster = model.booster_
                gains = booster.feature_importance(importance_type="gain")
                cols = active_cols
                pairs = sorted(zip(cols, gains), key=lambda x: x[1], reverse=True)[:30]
                top_features_gain = [{"name": n, "gain": float(g)} for n, g in pairs]
            except Exception:
                pass
            val_pred = model.predict(X_val_df)
            val_rmse = float(np.sqrt(np.mean((y_val - val_pred) ** 2)))

            mean_ret = float(np.mean(y_train))
            std_ret = float(np.std(y_train)) if len(y_train) > 1 else 0.0
            # Threshold from PREDICTION distribution (not returns — model outputs are compressed)
            pred_mean = float(np.mean(val_pred))
            pred_std = float(np.std(val_pred)) if len(val_pred) > 1 else 0.0
            threshold = pred_mean + TRADE_SIGMA_MULT * pred_std if pred_std > 0 else float(TRADE_MIN_ALPHA)
            # Cap threshold per horizon to prevent unreasonably high/low values
            _cap = TRADE_THRESHOLD_CAP.get(window_name, {})
            if _cap:
                threshold = max(_cap["min"], min(_cap["max"], threshold))
            meta_out = {
                "feature_columns": active_cols,
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
                    "test_start_date_x": str(pd.Timestamp(dates_x[test_idx_start])) if test_idx_start < len(dates_x) else None,
                    "test_start_date_y": str(pd.Timestamp(dates_y[test_idx_start])) if test_idx_start < len(dates_y) else None,
                },
            }
            if test_mask.any():
                y_test = y[test_mask]
                test_pred = model.predict(pd.DataFrame(X[test_mask], columns=active_cols))
                test_rmse = float(np.sqrt(np.mean((y_test - test_pred) ** 2)))
                test_mae = float(np.mean(np.abs(y_test - test_pred)))
                # Conformal: absolute residual quantiles for calibrated intervals
                abs_resid_test = np.abs(y_test - test_pred)
                conformal_q90 = float(np.quantile(abs_resid_test, 0.90))
                conformal_q95 = float(np.quantile(abs_resid_test, 0.95))
                # Baselines: naive 0 (predict no alpha)
                baseline_rmse = float(np.sqrt(np.mean(y_test ** 2)))
                # Momentum baseline: predict next return = last 1d return (if feature present)
                if "log_return_1d" in active_cols:
                    idx = active_cols.index("log_return_1d")
                    mom_pred = X[test_mask][:, idx]
                    baseline_momentum_rmse = float(np.sqrt(np.mean((y_test - mom_pred) ** 2)))
                else:
                    baseline_momentum_rmse = baseline_rmse
                # Last-return baseline: use non-overlapping lookback (stride = horizon)
                # Adjacent samples share (horizon-1)/horizon days → artificially good.
                # Step back by `horizon` samples for an independent comparison.
                stride = max(1, horizon)
                if test_idx_start >= stride:
                    last_returns = y[test_idx_start - stride : test_idx_end - stride]
                else:
                    last_returns = np.zeros_like(y_test)
                if len(last_returns) == len(y_test):
                    baseline_last_rmse = float(np.sqrt(np.mean((y_test - last_returns) ** 2)))
                else:
                    baseline_last_rmse = baseline_rmse
                beats_naive = test_rmse < baseline_rmse
                beats_last = test_rmse < baseline_last_rmse
                # Direction metrics (important for trading usefulness)
                hit_rate = float(np.mean((y_test > 0) == (test_pred > 0)))
                # Guard: np.corrcoef raises RuntimeWarning on constant input
                if len(y_test) > 1 and np.std(test_pred) > 1e-12 and np.std(y_test) > 1e-12:
                    correlation = float(np.corrcoef(y_test, test_pred)[0, 1])
                else:
                    correlation = 0.0
                if np.isnan(correlation):
                    correlation = 0.0
                # Statistical significance gate: binomial test for directional accuracy
                n_test_samples = int(test_mask.sum())
                n_correct = int(round(hit_rate * n_test_samples))
                binom_pval = 1.0
                if n_test_samples >= 20:
                    try:
                        from scipy.stats import binomtest
                        binom_pval = float(binomtest(n_correct, n_test_samples, 0.5, alternative='greater').pvalue)
                    except ImportError:
                        binom_pval = 0.5 if hit_rate > 0.52 else 1.0
                production_ready = (
                    beats_naive
                    and hit_rate >= 0.50
                    and binom_pval < 0.30  # v5.0: 70% confidence (was 80%) — more tickers qualify
                    and n_test_samples >= 15  # v5.0: reduced from 20 — include smaller holdouts
                )
                # Detect anti-correlated models: if correlation is strongly negative
                # (< -0.15) with enough test samples, the model learned an inverse
                # signal.  Flag for sign-flip at prediction time.
                invert_signal = (
                    correlation < -0.15
                    and n_test_samples >= 30
                    and hit_rate < 0.45  # clearly worse than random
                )
                # If inverted, the model IS useful — just backwards. Mark as
                # production_ready so the ensemble uses it (with flipped sign).
                if invert_signal:
                    production_ready = True
                    logger.info(
                        "INVERT-SIGNAL %s-%s: corr=%.3f hit=%.1f%% → will flip sign at prediction",
                        ticker, window_name, correlation, hit_rate * 100,
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
                    "invert_signal": invert_signal,
                    "has_holdout": True,
                    "eval_rmse": test_rmse,
                })
                logger.info(
                    f"Trained {ticker}-{window_name}: val_rmse={val_rmse:.4f} test_rmse={test_rmse:.4f} mae={test_mae:.4f} "
                    f"baseline={baseline_rmse:.4f} last={baseline_last_rmse:.4f} hit_rate={hit_rate:.1%} corr={correlation:.3f}"
                )
            else:
                # Conformal from val residuals when no holdout
                val_pred = model.predict(pd.DataFrame(X_val, columns=active_cols))
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

        # --- SEC/FMP consistency check at inference ---
        infer_skip_sec_fmp = os.environ.get("SKIP_SEC_FMP", "false").lower() == "true"
        for _wn, _pm in self.pooled_metadata.items():
            trained_skip = _pm.get("skip_sec_fmp_at_train")
            if trained_skip is not None and trained_skip != infer_skip_sec_fmp:
                logger.warning(
                    "[SEC/FMP-MISMATCH] Model %s was trained with SKIP_SEC_FMP=%s but inference "
                    "is running with SKIP_SEC_FMP=%s — predictions may be unreliable.",
                    _wn, trained_skip, infer_skip_sec_fmp,
                )
            break  # Only need to check once

        results = {}
        # Use last row for prediction; current_price from df_aligned (matches feature row)
        X_full = features[-1:].astype(np.float32)
        current_cols = meta.get("feature_columns", [])
        df_aligned = meta.get("df_aligned")
        if df_aligned is not None and "Close" in df_aligned.columns:
            current_price = float(df_aligned["Close"].iloc[-1])
        else:
            current_price = float(df["Close"].iloc[-1]) if "Close" in df.columns else 0

        results = {}
        for window_name in self.prediction_windows:
            # --- v4.0: Per-ticker priority, pooled fallback (no ensemble) ---
            pooled_model = self.pooled_models.get(window_name)
            pooled_meta = self.pooled_metadata.get(window_name, {})
            key = (ticker, window_name)
            ticker_meta = self.metadata.get(key)
            ticker_model = self.models.get(key)

            # Kill switch: suppress pooled model if anti-correlated on holdout
            pooled_holdout_corr = pooled_meta.get("holdout_correlation", 0.0)
            if pooled_holdout_corr < -0.08 and pooled_model is not None:
                logger.warning(
                    "KILL-SWITCH: Pooled %s has negative holdout correlation (%.3f). "
                    "Pooled model suppressed for %s.",
                    window_name, pooled_holdout_corr, ticker,
                )
                pooled_model = None

            # v5.0: Ensemble blending — weighted average of pooled + per-ticker.
            # Per-ticker captures stock-specific patterns; pooled provides stability.
            # Weight per-ticker more when it's production-ready (proven OOS edge).
            has_pooled = pooled_model is not None
            has_ticker = (ticker_meta is not None and ticker_model is not None)
            ticker_prod_ready = has_ticker and ticker_meta.get("production_ready", False)

            if not has_pooled and not has_ticker:
                results[window_name] = {
                    "prediction": 0.0,
                    "alpha": 0.0,
                    "alpha_pct": 0.0,
                    "price_change": 0.0,
                    "predicted_price": current_price,
                    "alpha_implied_price": current_price,
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
                    "is_market_neutral": USE_MARKET_NEUTRAL_TARGET,
                    "reason": "no_model"
                }
                continue

            # Use pooled metadata as primary for feature columns and thresholds
            meta_w = pooled_meta if has_pooled else ticker_meta
            if not meta_w:
                meta_w = {}
            model_cols = meta_w.get("feature_columns")

            # Hard check: if >50% of model columns are missing, skip this window
            feature_coverage_penalty = 1.0
            missing_ratio = 0.0
            if model_cols:
                col_set = set(current_cols)
                n_missing = sum(1 for c in model_cols if c not in col_set)
                missing_ratio = n_missing / len(model_cols) if model_cols else 0
                if n_missing > len(model_cols) * 0.5:
                    logger.error(
                        "Skipping %s-%s: %d/%d model columns missing (>50%%)",
                        ticker, window_name, n_missing, len(model_cols),
                    )
                    results[window_name] = {
                        "prediction": 0.0, "alpha": 0.0, "alpha_pct": 0.0,
                        "price_change": 0.0, "predicted_price": current_price,
                        "alpha_implied_price": current_price,
                        "confidence": 0.0, "current_price": current_price,
                        "prob_positive": 0.5, "prob_above_threshold": 0.5,
                        "trade_recommended": False,
                        "trade_threshold": TRADE_MIN_ALPHA,
                        "normalized_return": 0.0,
                        "horizon_days": TARGET_CONFIG.get(window_name, {}).get("horizon", 1),
                        "min_return_for_profit": ROUND_TRIP_COST_BPS / 10000,
                        "covers_transaction_cost": False,
                        "is_market_neutral": USE_MARKET_NEUTRAL_TARGET,
                        "reason": "feature_mismatch",
                    }
                    continue
                feature_coverage_penalty = max(0.5, 1.0 - missing_ratio)

            # v5.0: Ensemble prediction — blend pooled + per-ticker predictions
            pred_return = 0.0
            model_source = "none"

            if has_pooled and has_ticker:
                # Both available: weighted blend
                pooled_cols = pooled_meta.get("feature_columns", model_cols)
                ticker_cols = ticker_meta.get("feature_columns", model_cols)

                X_pooled = self._select_features(X_full, current_cols, pooled_cols, ticker=ticker, window=window_name)
                X_pooled_df = pd.DataFrame(X_pooled, columns=pooled_cols)
                pooled_pred = float(pooled_model.predict(X_pooled_df)[0])

                X_ticker = self._select_features(X_full, current_cols, ticker_cols, ticker=ticker, window=window_name)
                X_ticker_df = pd.DataFrame(X_ticker, columns=ticker_cols)
                ticker_pred = float(ticker_model.predict(X_ticker_df)[0])

                # Sign-flip for anti-correlated per-ticker models
                if ticker_meta.get("invert_signal", False):
                    ticker_pred = -ticker_pred

                # Blend weights: prod_ready per-ticker gets 60%, otherwise 30%
                if ticker_prod_ready:
                    tk_weight = 0.60
                else:
                    tk_weight = 0.30
                pred_return = tk_weight * ticker_pred + (1 - tk_weight) * pooled_pred
                model_source = "ensemble"
            elif has_pooled:
                X_pred = self._select_features(X_full, current_cols, model_cols, ticker=ticker, window=window_name)
                X_pred_df = pd.DataFrame(X_pred, columns=model_cols)
                pred_return = float(pooled_model.predict(X_pred_df)[0])
                model_source = "pooled"
            elif has_ticker:
                ticker_cols = ticker_meta.get("feature_columns", model_cols)
                X_pred = self._select_features(X_full, current_cols, ticker_cols, ticker=ticker, window=window_name)
                X_pred_df = pd.DataFrame(X_pred, columns=ticker_cols)
                pred_return = float(ticker_model.predict(X_pred_df)[0])
                if ticker_meta.get("invert_signal", False):
                    pred_return = -pred_return
                model_source = "per_ticker"

            alpha_implied_price = current_price * math.exp(pred_return)
            price_change = alpha_implied_price - current_price
            alpha_pct = pred_return * 100
            
            sigma = float(meta_w.get("val_rmse", 0.0))
            # --- P(up) from sign classifier + Gaussian CDF blend ---
            sign_model = self.pooled_sign_models.get(window_name)
            tk_sign = self.sign_models.get(key)
            if tk_sign is not None:
                sign_model = tk_sign
            gauss_prob = (
                0.5 * (1 + math.erf(pred_return / (sigma * math.sqrt(2)))) if sigma > 0 else 0.5
            )
            if sign_model is not None:
                try:
                    # For sign classifier, use pooled features (most reliable)
                    _sign_cols = pooled_meta.get("feature_columns", model_cols) if has_pooled else model_cols
                    _X_sign = self._select_features(X_full, current_cols, _sign_cols, ticker=ticker, window=window_name)
                    _X_sign_df = pd.DataFrame(_X_sign, columns=_sign_cols)
                    clf_prob = float(sign_model.predict_proba(_X_sign_df)[0, 1])
                    # v5.0: blend sign classifier with Gaussian CDF 50/50
                    # This prevents a bad sign classifier from dominating
                    prob_positive = 0.5 * clf_prob + 0.5 * gauss_prob
                except Exception:
                    prob_positive = gauss_prob
            else:
                prob_positive = gauss_prob

            # v5.0: Completely redesigned confidence calibration.
            # Previous formula: directional_edge / 0.30 × quality_mult
            # Problem: sign classifier at 50% → edge = 0 → confidence = 0 always.
            #
            # New formula: confidence = f(prediction_magnitude, model_quality, prob_edge)
            # 1. Base confidence from prediction magnitude relative to noise (sigma)
            # 2. Model quality multiplier from hit rate and correlation
            # 3. Directional agreement bonus from sign classifier
            prob_positive = max(0.15, min(0.85, prob_positive))

            # (1) Prediction magnitude confidence: |pred| / sigma, capped at 2.0
            if sigma > 1e-8:
                magnitude_ratio = min(2.0, abs(pred_return) / sigma)
            else:
                magnitude_ratio = min(2.0, abs(pred_return) / 0.01)
            magnitude_conf = magnitude_ratio / 2.0  # normalized to [0, 1]

            # (2) Model quality: based on hit rate and correlation
            # Use the BEST available model's metrics
            best_hit = 0.50
            best_corr = 0.0
            if has_pooled:
                best_hit = max(best_hit, float(pooled_meta.get("holdout_hit_rate", pooled_meta.get("hit_rate", 0.50))))
                best_corr = max(best_corr, abs(float(pooled_meta.get("holdout_correlation", pooled_meta.get("correlation", 0.0)))))
            if has_ticker:
                best_hit = max(best_hit, float(ticker_meta.get("hit_rate", 0.50)))
                best_corr = max(best_corr, abs(float(ticker_meta.get("correlation", 0.0))))

            # Hit rate quality: 50% → 0.3, 55% → 0.6, 60%+ → 1.0
            hit_quality = min(1.0, max(0.3, (best_hit - 0.47) / 0.13))
            # Correlation quality: 0 → 0.3, 0.05 → 0.6, 0.1+ → 1.0
            corr_quality = min(1.0, max(0.3, best_corr / 0.10))
            quality_mult = 0.6 * hit_quality + 0.4 * corr_quality

            # (3) Directional edge from prob_positive
            dir_edge = abs(prob_positive - 0.50)  # 0 to 0.35
            dir_bonus = min(0.3, dir_edge)  # up to 0.3 bonus

            # Final confidence: magnitude × quality + directional bonus
            confidence = min(0.95, magnitude_conf * quality_mult + dir_bonus)
            # Floor: never below 0.05 if model produced a non-zero prediction
            if abs(pred_return) > 1e-6:
                confidence = max(0.05, confidence)

            # Feature coverage penalty
            confidence *= feature_coverage_penalty

            # Stale sentiment penalty
            if os.environ.get("SENTIMENT_FRESH", "true").lower() not in ("true", "1", "yes"):
                confidence *= 0.85

            # ── Regime-adaptive confidence & threshold adjustment ──
            _regime_score = 0.0
            _vol_ratio = 1.0
            if current_cols:
                _col_idx = {c: i for i, c in enumerate(current_cols)}
                if "regime_score" in _col_idx:
                    _regime_score = float(X_full[0, _col_idx["regime_score"]])
                if "vol_ratio_5_20" in _col_idx:
                    _vol_ratio = float(X_full[0, _col_idx["vol_ratio_5_20"]])

            # Bear regime → moderate confidence reduction (10% max, was 20%)
            if _regime_score < 0:
                confidence *= max(0.90, 1.0 + _regime_score * 0.10)
            # Vol expansion → moderate confidence reduction
            if _vol_ratio > 1.5:
                _vol_penalty = min(0.10, (_vol_ratio - 1.5) * 0.05)
                confidence *= (1.0 - _vol_penalty)

            # Restore price range — use conformal q90 for calibrated interval
            conformal_q = float(meta_w.get("conformal_q90", 0.0))
            if conformal_q <= 0:
                conformal_q = 2 * sigma  # fallback to ±2σ if no conformal data
            price_low = current_price * math.exp(pred_return - conformal_q)
            price_high = current_price * math.exp(pred_return + conformal_q)

            # Skip-trade rule: use regime-adaptive threshold (mean+2*sigma) when available
            raw_threshold = meta_w.get("trade_threshold")
            if raw_threshold is None:
                raw_threshold = float(TRADE_MIN_ALPHA)
            else:
                raw_threshold = float(raw_threshold)
            min_alpha = max(raw_threshold, TRADE_MIN_ALPHA)

            # Regime-adaptive threshold: tighten in bear/high-vol regimes
            if _regime_score < 0:
                # In bear regime, require 50% more alpha to trade
                min_alpha *= (1.0 - _regime_score * 0.50)  # regime_score is negative → multiplier > 1
            if _vol_ratio > 1.5:
                # In vol-expansion regime, require 30% more alpha
                min_alpha *= min(1.5, 1.0 + (_vol_ratio - 1.5) * 0.30)

            # Transaction cost: round-trip in log space (must be computed BEFORE trade_recommended)
            min_return_for_profit = (ROUND_TRIP_COST_BPS / 10000)
            covers_transaction_cost = abs(pred_return) >= min_return_for_profit

            # Per-horizon probability threshold: next_day uses a lower bar (0.505)
            # because its sign classifier accuracy is ~51.7% — the standard 0.52
            # threshold was filtering out nearly all trades.
            _min_prob = TRADE_MIN_PROB_BY_HORIZON.get(window_name, TRADE_MIN_PROB_POSITIVE)
            trade_recommended = (
                pred_return >= min_alpha
                and prob_positive >= _min_prob
                and covers_transaction_cost  # Fix #21: must clear transaction costs
            )

            # P(return > threshold) - classification-style (Losing Loonies v4: tradable moves)
            prob_above_threshold = (
                0.5 * (1 - math.erf((raw_threshold - pred_return) / (sigma * math.sqrt(2))))
                if sigma > 0 else 0.5
            )
            prob_above_threshold = max(0.0, min(1.0, prob_above_threshold))

            # Normalized return for horizon selection (compound-equivalent daily log return)
            horizon = TARGET_CONFIG.get(window_name, {}).get("horizon", 1)
            horizon_days = max(1, horizon)
            normalized_return = pred_return / horizon_days

            results[window_name] = {
                "prediction": pred_return,
                "alpha": pred_return,
                "alpha_pct": float(alpha_pct),
                "price_change": float(price_change),
                # alpha_implied_price: current_price * exp(alpha).  NOT a true price
                # forecast when target is market-neutral.  Kept for backward compat.
                "predicted_price": float(alpha_implied_price),
                "alpha_implied_price": float(alpha_implied_price),
                "confidence": float(confidence),
                "current_price": current_price,
                "price_range": {
                    "low": price_low,
                    "high": price_high,
                },
                "prob_positive": float(prob_positive),
                "prob_above_threshold": float(prob_above_threshold),
                "trade_recommended": trade_recommended,
                "trade_threshold": float(min_alpha),
                "normalized_return": float(normalized_return),
                "horizon_days": int(horizon_days),
                "min_return_for_profit": min_return_for_profit,
                "covers_transaction_cost": covers_transaction_cost,
                "is_market_neutral": USE_MARKET_NEUTRAL_TARGET,
                "model_version": MODEL_VERSION,
                "training_timestamp": datetime.now(timezone.utc).isoformat(),
                # ── regime context ──
                "regime_score": float(_regime_score),
                "detected_regime": (
                    "bull" if _regime_score > 0.3
                    else "bear" if _regime_score < -0.3
                    else "neutral"
                ),
                "vol_expansion": bool(_vol_ratio > 1.5),
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

    def predict_batch(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Efficiently predict for all rows in df at once (vectorized).
        Returns DataFrame with date, window, prediction, trade_recommended.
        """
        if self.feature_engineer is None:
            from ..data.features_minimal import MinimalFeatureEngineer
            self.feature_engineer = MinimalFeatureEngineer(self.mongo_client)

        features, meta = self.feature_engineer.prepare_features(
            df, ticker=ticker, mongo_client=self.mongo_client
        )
        if features is None or len(features) == 0:
            return pd.DataFrame()

        df_aligned = meta.get("df_aligned")
        if df_aligned is None or "date" not in df_aligned.columns:
            return pd.DataFrame()

        dates = df_aligned["date"].values
        closes = df_aligned["Close"].values
        current_cols = meta.get("feature_columns", [])
        results_list = []

        # Pre-calculate common metrics
        # Note: sigma comes from model metadata, not calculated here
        
        for window_name in self.prediction_windows:
            # --- Ensemble logic (mirrors predict_all_windows) ---
            pooled_model = self.pooled_models.get(window_name)
            pooled_meta = self.pooled_metadata.get(window_name, {})
            key = (ticker, window_name)
            ticker_meta = self.metadata.get(key)
            ticker_model = self.models.get(key)

            # Kill switch: suppress pooled if anti-correlated
            pooled_holdout_corr = pooled_meta.get("holdout_correlation", 0.0)
            if pooled_holdout_corr < -0.08 and pooled_model is not None:
                pooled_model = None

            # v5.0: Ensemble blending in batch mode
            has_pooled = pooled_model is not None
            has_ticker = (ticker_meta is not None and ticker_model is not None)
            ticker_prod_ready = has_ticker and ticker_meta.get("production_ready", False)

            if not has_pooled and not has_ticker:
                continue

            meta_w = pooled_meta if has_pooled else ticker_meta

            # Compute ensemble prediction
            if has_pooled and has_ticker:
                pooled_cols = pooled_meta.get("feature_columns")
                ticker_cols = ticker_meta.get("feature_columns")
                X_pooled = self._select_features(features, current_cols, pooled_cols)
                X_pooled = pd.DataFrame(X_pooled, columns=pooled_cols)
                pooled_preds = pooled_model.predict(X_pooled)

                X_ticker = self._select_features(features, current_cols, ticker_cols)
                X_ticker = pd.DataFrame(X_ticker, columns=ticker_cols)
                ticker_preds = ticker_model.predict(X_ticker)
                if ticker_meta.get("invert_signal", False):
                    ticker_preds = -ticker_preds

                tk_w = 0.60 if ticker_prod_ready else 0.30
                preds_ret = tk_w * ticker_preds + (1 - tk_w) * pooled_preds
            elif has_pooled:
                model_cols = pooled_meta.get("feature_columns")
                X_pred = self._select_features(features, current_cols, model_cols)
                X_pred = pd.DataFrame(X_pred, columns=model_cols)
                preds_ret = pooled_model.predict(X_pred)
            else:
                model_cols = ticker_meta.get("feature_columns")
                X_pred = self._select_features(features, current_cols, model_cols)
                X_pred = pd.DataFrame(X_pred, columns=model_cols)
                preds_ret = ticker_model.predict(X_pred)
                if ticker_meta.get("invert_signal", False):
                    preds_ret = -preds_ret
            
            # Vectorized trade logic
            sigma = float(meta_w.get("val_rmse", 0.0))
            raw_threshold = meta_w.get("trade_threshold")
            if raw_threshold is None:
                raw_threshold = float(TRADE_MIN_ALPHA)
            else:
                raw_threshold = float(raw_threshold)
            min_alpha = max(raw_threshold, TRADE_MIN_ALPHA)

            # Prob positive (vectorized)
            # Use sign classifier when available, else Gaussian CDF fallback.
            # Prefer per-ticker sign model (mirrors predict_all_windows).
            sign_model = self.pooled_sign_models.get(window_name)
            tk_sign = self.sign_models.get(key)
            if tk_sign is not None:
                sign_model = tk_sign

            if sign_model is not None:
                try:
                    probs = sign_model.predict_proba(X_pred)[:, 1].astype(np.float64)
                except Exception:
                    sign_model = None  # fall through to Gaussian

            if sign_model is None:
                sqrt2 = math.sqrt(2)
                if sigma > 0:
                    try:
                        from scipy.special import erf as _erf
                        probs = 0.5 * (1 + _erf(preds_ret / (sigma * sqrt2)))
                    except ImportError:
                        probs = np.array([0.5 * (1 + math.erf(p / (sigma * sqrt2))) for p in preds_ret])
                else:
                    probs = np.full(len(preds_ret), 0.5)

            # Clip probabilities to [0.15, 0.85] — consistent with single-prediction path
            probs = np.clip(probs, 0.15, 0.85)

            # Transaction cost filter (matches predict_all_windows)
            min_return_for_profit = ROUND_TRIP_COST_BPS / 10000
            covers_cost = np.abs(preds_ret) >= min_return_for_profit

            # Trade recommended mask (per-horizon probability threshold)
            _min_prob = TRADE_MIN_PROB_BY_HORIZON.get(window_name, TRADE_MIN_PROB_POSITIVE)
            trade_mask = (preds_ret >= min_alpha) & (probs >= _min_prob) & covers_cost
            
            # Normalized return
            horizon = TARGET_CONFIG.get(window_name, {}).get("horizon", 1)
            norm_ret = preds_ret / max(1, horizon)

            # Build result result_df for this window
            # We want: date, window, trade_recommended, normalized_return, current_price
            w_df = pd.DataFrame({
                "date": dates,
                "window": window_name,
                "prediction": preds_ret,
                "prob_positive": probs,
                "trade_recommended": trade_mask,
                "normalized_return": norm_ret,
                "current_price": closes,
                "horizon": horizon
            })
            results_list.append(w_df)

        if not results_list:
            return pd.DataFrame()

        return pd.concat(results_list, ignore_index=True)

    def load_models(self) -> None:
        """Load models from disk (per-ticker + pooled) with SHA-256 integrity verification."""
        base = os.path.join(MODEL_DIR, "v1")
        if not os.path.exists(base):
            logger.info("No v1 models found to load")
            return

        # Load expected hashes for integrity verification
        _expected_hashes = {}
        _meta_json_path = os.path.join(base, "model_metadata.json")
        if os.path.exists(_meta_json_path):
            try:
                with open(_meta_json_path) as _mf:
                    _model_meta = json.load(_mf)
                _expected_hashes = _model_meta.get("model_hashes", {})
            except Exception:
                pass

        def _verify_hash(filepath: str) -> bool:
            """Verify a model file's SHA-256 against the expected hash."""
            if not _expected_hashes:
                return True  # No hashes stored — skip verification (backward compat)
            rel = os.path.relpath(filepath, base)
            expected = _expected_hashes.get(rel)
            if expected is None:
                return True  # No hash for this specific file — skip
            h = hashlib.sha256()
            with open(filepath, "rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    h.update(chunk)
            actual = h.hexdigest()
            if actual != expected:
                logger.error(
                    "INTEGRITY FAILURE: %s hash mismatch (expected=%s, got=%s). "
                    "Model file may be corrupted.",
                    rel, expected[:12], actual[:12],
                )
                return False
            return True

        pooled_path = os.path.join(base, "_pooled")
        if os.path.isdir(pooled_path):
            for f in os.listdir(pooled_path):
                if f.endswith(".joblib"):
                    for w in self.prediction_windows:
                        if w in f:
                            fpath = os.path.join(pooled_path, f)
                            try:
                                if not _verify_hash(fpath):
                                    logger.warning("Skipping corrupted pooled model: %s", f)
                                    continue
                                if f.startswith("sign_"):
                                    self.pooled_sign_models[w] = joblib.load(fpath)
                                elif f.startswith("lgb_"):
                                    self.pooled_models[w] = joblib.load(fpath)
                            except Exception as e:
                                logger.warning("Could not load pooled %s: %s", f, e)
            if self.pooled_models:
                logger.info("Loaded %d pooled models", len(self.pooled_models))
            meta_path = os.path.join(pooled_path, "metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        self.pooled_metadata = json.load(f)
                except Exception as e:
                    logger.warning("Could not load pooled metadata: %s", e)
            # Load feature shortlists (pruning config)
            shortlist_path = os.path.join(pooled_path, "feature_shortlist.json")
            if os.path.exists(shortlist_path):
                try:
                    with open(shortlist_path) as f:
                        self._feature_shortlist = json.load(f)
                    logger.info("Loaded feature shortlists for %d horizons", len(self._feature_shortlist))
                except Exception as e:
                    logger.warning("Could not load feature shortlist: %s", e)
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
                                if not _verify_hash(path):
                                    logger.warning("Skipping corrupted model: %s", path)
                                    continue
                                if f.startswith("sign_"):
                                    self.sign_models[key] = joblib.load(path)
                                elif f.startswith("lgb_"):
                                    self.models[key] = joblib.load(path)
                                    scaler_path = path.replace(".joblib", "_scaler.joblib")
                                    if os.path.exists(scaler_path):
                                        self.scalers[key] = joblib.load(scaler_path)
                            except Exception as e:
                                logger.warning(f"Could not load {path}: {e}")
        logger.info(f"Loaded {len(self.models)} models + {len(self.sign_models)} sign classifiers")

    def save_models(self) -> None:
        """Save models to disk (per-ticker + pooled) with SHA-256 integrity hashes."""
        base = os.path.join(MODEL_DIR, "v1")
        pooled_path = os.path.join(base, "_pooled")
        model_hashes = {}  # {filename: sha256_hex}

        def _save_and_hash(model_obj, filepath: str) -> None:
            """Save a joblib model and record its SHA-256 hash."""
            joblib.dump(model_obj, filepath)
            h = hashlib.sha256()
            with open(filepath, "rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    h.update(chunk)
            rel = os.path.relpath(filepath, base)
            model_hashes[rel] = h.hexdigest()

        if self.pooled_models:
            os.makedirs(pooled_path, exist_ok=True)
            for window, model in self.pooled_models.items():
                try:
                    _save_and_hash(model, os.path.join(pooled_path, f"lgb_{window}.joblib"))
                except Exception as e:
                    logger.warning("Could not save pooled %s: %s", window, e)
            # Save pooled sign classifiers
            for window, sign_clf in self.pooled_sign_models.items():
                try:
                    _save_and_hash(sign_clf, os.path.join(pooled_path, f"sign_{window}.joblib"))
                except Exception as e:
                    logger.warning("Could not save pooled sign %s: %s", window, e)
            if self.pooled_metadata:
                try:
                    with open(os.path.join(pooled_path, "metadata.json"), "w") as f:
                        json.dump(self.pooled_metadata, f, indent=2)
                except Exception as e:
                    logger.warning("Could not save pooled metadata: %s", e)
            # Save feature shortlists (pruning config)
            if self._feature_shortlist:
                try:
                    with open(os.path.join(pooled_path, "feature_shortlist.json"), "w") as f:
                        json.dump(self._feature_shortlist, f, indent=2)
                except Exception as e:
                    logger.warning("Could not save feature shortlist: %s", e)
        for (ticker, window), model in self.models.items():
            path = os.path.join(base, ticker)
            os.makedirs(path, exist_ok=True)
            try:
                _save_and_hash(model, os.path.join(path, f"lgb_{window}.joblib"))
                scaler = self.scalers.get((ticker, window))
                if scaler is not None:
                    _save_and_hash(scaler, os.path.join(path, f"lgb_{window}_scaler.joblib"))
                # Save per-ticker sign classifier
                sign_clf = self.sign_models.get((ticker, window))
                if sign_clf is not None:
                    _save_and_hash(sign_clf, os.path.join(path, f"sign_{window}.joblib"))
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

        # Write model_metadata.json — single source of truth for the current model version
        try:
            model_meta = {
                "model_version": MODEL_VERSION,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "n_pooled_models": len(self.pooled_models),
                "n_ticker_models": len(self.models),
                "windows": list(self.pooled_models.keys()),
                "feature_counts": {
                    w: len(m.get("feature_columns", []))
                    for w, m in self.pooled_metadata.items()
                },
                "model_hashes": model_hashes,
            }
            meta_path = os.path.join(base, "model_metadata.json")
            with open(meta_path, "w") as f:
                json.dump(model_meta, f, indent=2)
            logger.info(
                "Saved model_metadata.json (version=%s, %d hashes)",
                MODEL_VERSION, len(model_hashes),
            )
        except Exception as e:
            logger.warning("Could not save model_metadata.json: %s", e)

