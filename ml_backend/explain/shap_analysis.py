"""
SHAP Feature Importance Analysis using LightGBM's native TreeSHAP.

Uses `model.predict(X, pred_contrib=True)` -- O(n * features * depth),
no external shap package needed.  Works on Python 3.14.

Produces:
  1. Per-prediction SHAP values (which features pushed AAPL up/down today?)
  2. Global gain-based importance (normalized -- comparable across horizons)
  3. Sanity check: pred == bias + sum(shap_vals) (catches feature misalignment)
  4. Rich document stored to MongoDB ``feature_importance`` collection

Usage (standalone):
    python -m ml_backend.explain.shap_analysis --tickers AAPL MSFT --horizon 30_day
"""

import argparse
import hashlib
import json
import logging
import math
import sys
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feature_list_hash(feature_names: List[str]) -> str:
    """Deterministic SHA-256 of the ordered feature list (first 12 hex chars)."""
    payload = ",".join(feature_names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Core SHAP computation
# ---------------------------------------------------------------------------

def compute_shap_for_prediction(
    model,
    X_pred: np.ndarray,
    feature_names: List[str],
    feature_values: Optional[np.ndarray] = None,
    top_k: int = 15,
) -> Dict:
    """
    Compute TreeSHAP values for a single prediction row using LightGBM's
    built-in ``pred_contrib=True``.

    Parameters
    ----------
    model : LGBMRegressor / LGBMClassifier / Booster
    X_pred : ndarray of shape (1, n_features) -- MUST have exactly the columns
             the model was trained on, in the same order.
    feature_names : ordered list of column names matching X_pred columns.
    feature_values : optional 1-D array of the actual feature values (for the
                     stored document).  Falls back to X_pred[0] if omitted.
    top_k : number of top features to keep in the compact output.

    Returns
    -------
    dict with shap_values (top-k), shap_values_all, base_value, prediction,
    sanity_ok, bullish_drivers, bearish_drivers, top_positive_contrib,
    top_negative_contrib, n_features.
    """
    booster = model.booster_ if hasattr(model, "booster_") else model
    
    # Cast X_pred to pd.DataFrame to pass valid feature names to LightGBM
    import pandas as pd
    X_pred_df = pd.DataFrame(X_pred, columns=feature_names)
    contrib = booster.predict(X_pred_df, pred_contrib=True)

    if contrib.ndim == 1:
        contrib = contrib.reshape(1, -1)

    # contrib shape: (1, n_features + 1)  -- last col is bias
    shap_vals = contrib[0, :-1]
    base_value = float(contrib[0, -1])

    n_shap = len(shap_vals)
    n_names = len(feature_names)

    # --- Strict alignment check ---
    if n_shap != n_names:
        raise ValueError(
            f"Feature misalignment: pred_contrib returned {n_shap} SHAP values "
            f"but {n_names} feature names supplied.  This means X_pred columns "
            f"do not match the model's training columns."
        )

    # --- Sanity: bias + sum(shap) == model.predict(X_pred) ---
    shap_sum = float(base_value + shap_vals.sum())
    model_pred = float(model.predict(pd.DataFrame(X_pred, columns=feature_names))[0])
    sanity_ok = abs(shap_sum - model_pred) < 1e-4
    if not sanity_ok:
        logger.warning(
            "SHAP sanity FAILED: bias+sum=%.6f vs predict=%.6f (delta=%.6f). "
            "Likely feature misalignment.",
            shap_sum, model_pred, abs(shap_sum - model_pred),
        )

    # Feature values (for the per-contrib document)
    if feature_values is None:
        feature_values = X_pred[0]

    # Build (name, value, contrib) triples sorted by |contrib|
    triples = sorted(
        zip(feature_names, feature_values, shap_vals),
        key=lambda x: abs(x[2]),
        reverse=True,
    )

    top_features = {n: float(c) for n, _v, c in triples[:top_k]}
    all_features = {n: float(c) for n, _v, c in triples}

    # Top positive / negative contrib (with feature value)
    top_positive = [
        {"feature": n, "value": float(v), "contrib": float(c)}
        for n, v, c in triples if c > 0
    ][:top_k]
    top_negative = [
        {"feature": n, "value": float(v), "contrib": float(c)}
        for n, v, c in triples if c < 0
    ][:top_k]

    # Legacy-compat bullish/bearish (impact = |contrib|)
    bullish = [{"feature": d["feature"], "impact": d["contrib"]} for d in top_positive]
    bearish = [{"feature": d["feature"], "impact": abs(d["contrib"])} for d in top_negative]

    return {
        "shap_values": top_features,
        "shap_values_all": all_features,
        "base_value": base_value,
        "prediction": shap_sum,
        "model_predict": model_pred,
        "sanity_ok": sanity_ok,
        "top_positive_contrib": top_positive,
        "top_negative_contrib": top_negative,
        "bullish_drivers": bullish,
        "bearish_drivers": bearish,
        "n_features": n_shap,
    }


# ---------------------------------------------------------------------------
# Global gain importance (normalized)
# ---------------------------------------------------------------------------

def compute_global_importance(
    model,
    feature_names: List[str],
    top_k: int = 30,
) -> List[Dict]:
    """
    Extract LightGBM gain-based feature importance (global, not per-prediction).

    Always normalized (gain_pct sums to 100) so values are comparable across
    horizons even if absolute gain differs.
    """
    booster = model.booster_ if hasattr(model, "booster_") else model
    gains = booster.feature_importance(importance_type="gain")
    splits = booster.feature_importance(importance_type="split")

    n = min(len(gains), len(feature_names))
    total_gain = float(gains[:n].sum()) or 1.0  # avoid /0

    pairs = []
    for i in range(n):
        pairs.append({
            "feature": feature_names[i],
            "gain": float(gains[i]),
            "gain_normalized": round(float(gains[i]) / total_gain, 6),
            "gain_pct": round(float(gains[i]) / total_gain * 100, 2),
            "splits": int(splits[i]),
        })

    pairs.sort(key=lambda x: x["gain"], reverse=True)
    return pairs[:top_k]


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------

def format_shap_analysis(
    ticker: str,
    window: str,
    shap_result: Dict,
    global_importance: List[Dict],
    prediction_info: Dict,
) -> str:
    """Format a human-readable SHAP analysis report."""
    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"  FEATURE IMPORTANCE ANALYSIS: {ticker} ({window})")
    lines.append(f"{'='*70}")
    lines.append("")

    # Prediction summary
    pred = prediction_info.get("predicted_value", prediction_info.get("prediction", 0))
    conf = prediction_info.get("prob_up", prediction_info.get("confidence", 0))
    price = prediction_info.get("current_price", 0)
    is_mkt_neutral = prediction_info.get("is_market_neutral", True)
    label = "Alpha (vs SPY)" if is_mkt_neutral else "Expected Return"
    price_label = "Alpha-Implied Price" if is_mkt_neutral else "Predicted Price"
    pred_price = prediction_info.get("predicted_price", price * math.exp(pred))
    p_down = 1 - conf

    lines.append(f"  Current Price:    ${price:,.2f}")
    lines.append(f"  {price_label}:  ${pred_price:,.2f}")
    lines.append(f"  {label}:   {pred * 100:+.3f}%")
    lines.append(f"  P(up):            {conf:.1%}   P(down): {p_down:.1%}")

    # Sanity check
    sanity = "OK" if shap_result.get("sanity_ok", True) else "MISMATCH"
    base = shap_result["base_value"]
    shap_pred = shap_result["prediction"]
    model_pred = shap_result.get("model_predict", shap_pred)
    lines.append(f"  Base Value:       {base:+.6f}")
    lines.append(f"  SHAP Sum:         {shap_pred:+.6f}  (sanity: {sanity})")
    lines.append(f"  Model Predict:    {model_pred:+.6f}")
    lines.append("")

    # SHAP: What's driving THIS prediction
    lines.append(f"  {'-'*66}")
    lines.append(f"  SHAP VALUES (what drove this specific prediction)")
    lines.append(f"  {'-'*66}")
    lines.append("")
    lines.append(f"  {'Feature':<35} {'Value':>10} {'SHAP':>12} {'Dir':>8}")
    lines.append(f"  {'-'*35} {'-'*10} {'-'*12} {'-'*8}")

    # Use top_positive + top_negative to show value
    shown = set()
    for item in (shap_result.get("top_positive_contrib", []) +
                 shap_result.get("top_negative_contrib", [])):
        feat = item["feature"]
        if feat in shown:
            continue
        shown.add(feat)
        val = item.get("value", 0)
        contrib = item["contrib"]
        direction = "UP" if contrib > 0 else "DOWN"
        bar_len = min(int(abs(contrib) * 5000), 20)
        bar = "+" * bar_len if contrib > 0 else "-" * bar_len
        lines.append(f"  {feat:<35} {val:>10.4f} {contrib:>+12.6f} {direction:>8}  {bar}")
        if len(shown) >= 15:
            break

    lines.append("")

    # Global importance (what matters MOST across all training data)
    lines.append(f"  {'-'*66}")
    lines.append(f"  GLOBAL FEATURE IMPORTANCE (normalized gain -- comparable across horizons)")
    lines.append(f"  {'-'*66}")
    lines.append("")
    lines.append(f"  {'#':<4} {'Feature':<35} {'Gain %':>8} {'Splits':>8}")
    lines.append(f"  {'-'*4} {'-'*35} {'-'*8} {'-'*8}")

    for i, fi in enumerate(global_importance[:20], 1):
        bar_len = min(int(fi["gain_pct"]), 30)
        bar = "#" * bar_len
        lines.append(
            f"  {i:<4} {fi['feature']:<35} {fi['gain_pct']:>7.1f}% {fi['splits']:>7}  {bar}"
        )

    lines.append("")
    lines.append(f"{'='*70}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_shap_analysis(
    tickers: List[str],
    horizons: Optional[List[str]] = None,
    store_to_mongo: bool = True,
) -> Dict:
    """
    Run full SHAP + feature importance analysis.

    Returns: {ticker: {window: {shap, global_importance, prediction_info}}}

    MongoDB document schema per (ticker, window, date):
        predicted_value       - raw model output (log-return or alpha)
        predicted_price       - current_price * exp(predicted_value)
        prob_up               - P(return > 0) from sign classifier
        top_positive_contrib  - [{feature, value, contrib}, ...]
        top_negative_contrib  - [{feature, value, contrib}, ...]
        global_gain_importance - top-N normalized gain [{feature, gain_pct, ...}]
        feature_list_hash     - SHA-256[:12] of ordered feature list
        sanity_ok             - bias + sum(shap) == predict?
        is_market_neutral     - True when target = stock - SPY
    """
    from ml_backend.utils.mongodb import MongoDBClient
    from ml_backend.models.predictor import StockPredictor
    from ml_backend.config.feature_config_v1 import USE_MARKET_NEUTRAL_TARGET

    mongo_client = MongoDBClient()
    predictor = StockPredictor(mongo_client)

    # Load trained models from disk
    predictor.load_models()
    if not predictor.pooled_models:
        logger.error("No trained models found. Run training first.")
        return {}

    horizons = horizons or list(predictor.prediction_windows)
    all_results = {}

    for ticker in tickers:
        logger.info("Analyzing %s ...", ticker)
        ticker_results = {}

        # -- Fetch historical data via yfinance --
        try:
            import yfinance as yf
            df = yf.download(ticker, period="10y", progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.reset_index()
                if "Date" in df.columns:
                    df = df.rename(columns={"Date": "date"})
            else:
                logger.warning("No data for %s", ticker)
                continue
        except Exception as e:
            logger.warning("Could not fetch %s: %s", ticker, e)
            continue

        # -- Prepare features (same path as predict_all_windows) --
        if predictor.feature_engineer is None:
            from ml_backend.data.features_minimal import MinimalFeatureEngineer
            predictor.feature_engineer = MinimalFeatureEngineer(mongo_client)

        features, meta = predictor.feature_engineer.prepare_features(
            df, ticker=ticker, mongo_client=mongo_client
        )
        if features is None or len(features) == 0:
            logger.warning("No features for %s", ticker)
            continue

        X_full = features[-1:].astype(np.float32)
        current_cols = meta.get("feature_columns", [])
        df_aligned = meta.get("df_aligned")
        if df_aligned is not None and "Close" in df_aligned.columns:
            current_price = float(df_aligned["Close"].iloc[-1])
        else:
            current_price = float(df["Close"].iloc[-1])

        for window_name in horizons:
            # -- Select model (mirrors predict_all_windows logic exactly) --
            model = predictor.pooled_models.get(window_name)
            meta_w = predictor.pooled_metadata.get(window_name, {})
            key = (ticker, window_name)
            ticker_meta = predictor.metadata.get(key)
            model_type = "pooled"
            if (ticker_meta
                    and ticker_meta.get("n_train", 0) >= 300
                    and ticker_meta.get("production_ready", False)):
                model = predictor.models.get(key, model)
                meta_w = ticker_meta
                model_type = "per-ticker"

            if model is None:
                continue

            # -- Align features to the model's training columns --
            model_cols = meta_w.get("feature_columns", current_cols)
            X_pred = predictor._select_features(X_full, current_cols, model_cols)

            # -- Prediction (log-return / market-neutral alpha) --
            pred_return = float(model.predict(X_pred)[0])
            # pred_return is log-return.  Convert to price via exp().
            # When market-neutral, this is alpha_implied_price (not a true forecast).
            pred_price = current_price * math.exp(pred_return)

            # -- P(up) from sign classifier --
            sign_model = predictor.pooled_sign_models.get(window_name)
            if ticker_meta and ticker_meta.get("production_ready", False):
                sign_model = predictor.sign_models.get(key, sign_model)
            sigma = float(meta_w.get("val_rmse", 0.01))
            if sign_model is not None:
                try:
                    prob_up = float(sign_model.predict_proba(X_pred)[0, 1])
                except Exception:
                    prob_up = (
                        0.5 * (1 + math.erf(pred_return / (sigma * math.sqrt(2))))
                        if sigma > 0 else 0.5
                    )
            else:
                prob_up = (
                    0.5 * (1 + math.erf(pred_return / (sigma * math.sqrt(2))))
                    if sigma > 0 else 0.5
                )

            prediction_info = {
                "predicted_value": pred_return,       # raw model output
                "predicted_price": pred_price,        # current_price * exp(pred)
                "prob_up": prob_up,                   # P(return > 0)
                "current_price": current_price,
                "model_type": model_type,
                "is_market_neutral": USE_MARKET_NEUTRAL_TARGET,
                # Legacy compat aliases
                "prediction": pred_return,
                "confidence": prob_up,
            }

            # -- Compute SHAP values (native TreeSHAP) --
            shap_result = compute_shap_for_prediction(
                model,
                X_pred,
                model_cols,
                feature_values=X_pred.values[0],
                top_k=15,
            )

            # -- Compute global importance (normalized gain) --
            global_importance = compute_global_importance(model, model_cols, top_k=30)

            # -- Format and print human-readable report --
            report = format_shap_analysis(
                ticker, window_name, shap_result, global_importance, prediction_info
            )
            print(report)

            ticker_results[window_name] = {
                "shap": shap_result,
                "global_importance": global_importance,
                "prediction_info": prediction_info,
                "model_type": model_type,
            }

            # -- Store to MongoDB (rich document) --
            if store_to_mongo and mongo_client:
                _store_explainability_doc(
                    mongo_client, ticker, window_name, model_type,
                    prediction_info, shap_result, global_importance, model_cols,
                )

        all_results[ticker] = ticker_results

    return all_results


def _store_explainability_doc(
    mongo_client,
    ticker: str,
    window: str,
    model_type: str,
    prediction_info: Dict,
    shap_result: Dict,
    global_importance: List[Dict],
    feature_list: List[str],
) -> None:
    """Upsert a rich feature_importance document to MongoDB.

    Schema (per ticker/window/date):
        predicted_value, predicted_price, prob_up,
        top_positive_contrib, top_negative_contrib,
        global_gain_importance, feature_list_hash,
        sanity_ok, is_market_neutral, base_value, ...
    """
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        doc = {
            # --- Keys ---
            "ticker": ticker,
            "window": window,
            "date": today,
            "timestamp": datetime.now(timezone.utc),

            # --- Prediction ---
            "predicted_value": prediction_info["predicted_value"],
            "predicted_price": prediction_info["predicted_price"],
            "prob_up": prediction_info["prob_up"],
            "current_price": prediction_info["current_price"],
            "model_type": model_type,
            "is_market_neutral": prediction_info.get("is_market_neutral", True),

            # --- SHAP (per-prediction) ---
            "base_value": shap_result["base_value"],
            "shap_prediction": shap_result["prediction"],
            "sanity_ok": shap_result["sanity_ok"],
            "top_positive_contrib": shap_result["top_positive_contrib"],
            "top_negative_contrib": shap_result["top_negative_contrib"],
            "shap_top_features": shap_result["shap_values"],     # top-k {feat: contrib}

            # Legacy compat
            "bullish_drivers": shap_result["bullish_drivers"],
            "bearish_drivers": shap_result["bearish_drivers"],

            # --- Global importance (normalized) ---
            "global_gain_importance": global_importance[:20],

            # --- Feature provenance ---
            "feature_list_hash": _feature_list_hash(feature_list),
            "n_features": shap_result["n_features"],
            "feature_columns": feature_list,          # full ordered list
        }
        mongo_client.db["feature_importance"].update_one(
            {"ticker": ticker, "window": window, "date": today},
            {"$set": doc},
            upsert=True,
        )
        logger.info("Stored feature importance for %s-%s (hash=%s)",
                     ticker, window, doc["feature_list_hash"])
    except Exception as e:
        logger.error("Failed to store feature importance: %s", e)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="SHAP Feature Importance Analysis")
    parser.add_argument("--tickers", nargs="+", default=["AAPL"],
                        help="Tickers to analyze (default: AAPL)")
    parser.add_argument("--horizon", nargs="+", default=None,
                        help="Horizons to analyze (default: all)")
    parser.add_argument("--no-mongo", action="store_true",
                        help="Skip storing results to MongoDB")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    results = run_shap_analysis(
        tickers=args.tickers,
        horizons=args.horizon,
        store_to_mongo=not args.no_mongo,
    )

    if not results:
        print("\nNo results generated. Ensure models are trained first.")
        sys.exit(1)

    print(f"\nAnalysis complete for {len(results)} ticker(s).")


if __name__ == "__main__":
    main()
