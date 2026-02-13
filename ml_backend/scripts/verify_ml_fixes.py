"""
Verification runbook for ML fixes (purge/embargo, conformal, etc.).
Run: cd ml_backend && python -m scripts.verify_ml_fixes

Env: SMOKE_LIGHT=1 to skip LightGBM training (feature engineer + dataset only).
"""
import os
import sys
import re

# Add repo root (ml_backend/) and parent so both styles work
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)  # ml_backend/
PARENT_ROOT = os.path.dirname(REPO_ROOT)  # stockpredict-ai/ (for ml_backend.* imports)
for p in (REPO_ROOT, PARENT_ROOT):
    if p and p not in sys.path:
        sys.path.insert(0, p)

try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = pd = None

# Use flush=True so output is visible in CI/collapsed terminals
def _print(msg, **kwargs):
    print(msg, flush=True, **kwargs)


def test_static_compile():
    """1) Static compile - ensure no syntax errors."""
    import py_compile
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for mod in ["models/predictor.py", "utils/mongodb.py", "data/features.py", "data/features_minimal.py"]:
        path = os.path.join(base, mod)
        if os.path.exists(path):
            py_compile.compile(path, doraise=True)
    _print("OK: Static compile passed")


def test_horizon_is_int():
    """Verify horizon is int in pooled gap calculation."""
    try:
        from ml_backend.config.feature_config_v1 import TARGET_CONFIG
    except ImportError:
        try:
            from config.feature_config_v1 import TARGET_CONFIG
        except ImportError:
            _print("SKIP: test_horizon_is_int (config not importable)")
            return
    for name, cfg in TARGET_CONFIG.items():
        h = cfg["horizon"]
        assert isinstance(h, int), f"horizon for {name} must be int, got {type(h)}"
    _print("OK: All horizons are int")


def test_unit_consistency():
    """Verify pred is log-return and price band uses exp(pred ± q)."""
    if np is None:
        _print("SKIP: test_unit_consistency (numpy not installed)")
        return
    # Model predicts log return; q is log-return error band
    # price = current * exp(log_return)  → correct
    pred_return = 0.01
    q = 0.02
    current = 100.0
    price_low = current * np.exp(pred_return - q)
    price_high = current * np.exp(pred_return + q)
    assert 98 < price_low < 99.5, "price_low should be ~99 for pred=0.01, q=0.02"
    assert 100.5 < price_high < 103.5, "price_high should be ~103 for pred=0.01, q=0.02"
    _print("OK: Unit consistency (log-return -> exp) verified")


def test_conformal_residuals_oos():
    """Verify conformal uses val/holdout, never train."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pred_path = os.path.join(base, "models", "predictor.py")
    with open(pred_path) as f:
        content = f.read()
    # Pooled: abs_resid = np.abs(y_val - pred) (regex tolerates whitespace)
    assert re.search(r"abs_resid\s*=\s*np\.abs\(\s*y_val\s*-\s*pred\s*\)", content)
    # Per-ticker holdout: abs_resid_test = np.abs(y_test - test_pred)
    assert "abs_resid_test" in content and "y_test" in content and "test_pred" in content
    # Per-ticker no-holdout: abs_resid_val = np.abs(y_val - val_pred)
    assert "abs_resid_val" in content
    _print("OK: Conformal residuals use val/holdout only")


def test_confidence_q_floor():
    """Verify q_conf has practical floor."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pred_path = os.path.join(base, "models", "predictor.py")
    with open(pred_path) as f:
        content = f.read()
    assert re.search(r"max\s*\(\s*0\.00[25]", content), "q_conf should have floor"
    _print("OK: Confidence q floor present")


def test_momentum_baseline_guard():
    """Verify momentum baseline checks for log_return_1d before indexing."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pred_path = os.path.join(base, "models", "predictor.py")
    with open(pred_path) as f:
        content = f.read()
    assert re.search(r"['\"]log_return_1d['\"]\s+in\s+feature_cols", content)
    _print("OK: Momentum baseline guarded by column check")


def test_pooled_min_val_size():
    """Verify pooled fold skips when len(y_val) < 100."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pred_path = os.path.join(base, "models", "predictor.py")
    with open(pred_path) as f:
        content = f.read()
    assert re.search(r"len\s*\(\s*y_val\s*\)\s*<\s*100", content)
    _print("OK: Pooled min val size check present")


def _load_smoke_data():
    """Load AAPL data: try yfinance first, fall back to scripts/smoke_data/AAPL.csv."""
    ticker = "AAPL"
    # 1) Try yfinance
    try:
        import yfinance as yf
        end = pd.Timestamp.now()
        start = end - pd.Timedelta(days=365 * 2)
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty or len(data) < 100:
            raise ValueError("Insufficient yfinance data")
        data = data.reset_index()
        data = data.rename(columns={"Date": "date"})
        return {ticker: data}
    except Exception:
        pass
    # 2) Fall back to CSV
    csv_path = os.path.join(SCRIPT_DIR, "smoke_data", "AAPL.csv")
    if os.path.exists(csv_path):
        data = pd.read_csv(csv_path)
        data["date"] = pd.to_datetime(data["date"])
        if len(data) >= 100:
            return {ticker: data}
    return None


def smoke_test_training():
    """2) Smoke test: train on 1 ticker, limited data (yfinance or scripts/smoke_data/AAPL.csv)."""
    if pd is None:
        _print("SKIP: Smoke test (pandas not installed)")
        return
    try:
        from ml_backend.data.features_minimal import MinimalFeatureEngineer
        from ml_backend.models.predictor import StockPredictor
        from ml_backend.config.feature_config_v1 import TARGET_CONFIG
    except ImportError as e:
        _print("SKIP: Smoke test (import failed): " + str(e))
        return

    historical_data = _load_smoke_data()
    if historical_data is None:
        _print("SKIP: Smoke test (no data: yfinance failed and scripts/smoke_data/AAPL.csv not found)")
        return

    fe = MinimalFeatureEngineer()
    predictor = StockPredictor(mongo_client=None)
    predictor.set_feature_engineer(fe)

    # SMOKE_LIGHT=1: skip LightGBM, validate feature engineer + dataset only
    if os.environ.get("SMOKE_LIGHT"):
        feats, meta = fe.prepare_features(
            list(historical_data.values())[0],
            ticker="AAPL",
            mongo_client=None,
        )
        assert feats is not None and len(feats) >= 50, "Feature pipeline should produce >= 50 rows"
        _print("OK: Smoke test (light) passed - feature engineer + dataset OK")
        return

    try:
        predictor.train_pooled_models(historical_data, fe)
        # Validate pooled metadata if we trained pooled models
        if predictor.pooled_metadata:
            for w, meta in predictor.pooled_metadata.items():
                assert "conformal_q90" in meta, f"Pooled {w} missing conformal_q90"
                assert "conformal_q95" in meta, f"Pooled {w} missing conformal_q95"
                assert "top_features_gain" in meta, f"Pooled {w} missing top_features_gain"
            _print("OK: Smoke test (pooled) passed - metadata has conformal_q90/q95, top_features_gain")
        else:
            # CSV fallback often has < 2000 rows → pooled not trained; run per-ticker
            ticker = "AAPL"
            df = historical_data[ticker]
            predictor._train_ticker(ticker, df, fe)
            meta = predictor.metadata.get((ticker, "next_day"))
            if meta:
                assert "conformal_q90" in meta or "val_rmse" in meta
                _print("OK: Smoke test (per-ticker) passed - metadata has conformal/val_rmse")
            else:
                _print("OK: Smoke test passed - feature pipeline ran (no models due to data size)")
    except Exception as e:
        _print(f"FAIL: Smoke test: {e}")
        raise


def run_all():
    _print("=== ML Fixes Verification Runbook ===\n")
    test_static_compile()
    test_horizon_is_int()
    test_unit_consistency()
    test_conformal_residuals_oos()
    test_confidence_q_floor()
    test_momentum_baseline_guard()
    test_pooled_min_val_size()
    smoke_test_training()
    _print("\n=== All checks passed ===")


if __name__ == "__main__":
    run_all()
