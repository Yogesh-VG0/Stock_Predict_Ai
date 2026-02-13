"""
Leakage sanity test: ensure feature row date < label date for all rows.
Run: python -m ml_backend.scripts.test_leakage
"""
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml_backend.data.features_minimal import MinimalFeatureEngineer
from ml_backend.config.feature_config_v1 import TARGET_CONFIG


def test_target_feature_alignment():
    """Assert: for every row i, feature date_i < label date (label uses date_{i+horizon})."""
    np.random.seed(42)
    n = 150
    dates = pd.date_range(end=datetime.now(), periods=n, freq="B")  # business days
    df = pd.DataFrame({
        "date": dates,
        "Open": 100 + np.cumsum(np.random.randn(n) * 0.5),
        "High": 0, "Low": 0, "Close": 0, "Volume": 1e6 + np.random.randint(0, 5e5, n),
    })
    df["High"] = df["Open"] * 1.02
    df["Low"] = df["Open"] * 0.98
    df["Close"] = df["Open"] * (1 + np.random.randn(n) * 0.01)
    df["Close"] = df["Close"].clip(90, 110)

    fe = MinimalFeatureEngineer()
    features, meta = fe.prepare_features(df, ticker="AAPL")
    assert features is not None, "Features should not be None"
    df_aligned = meta.get("df_aligned")
    assert df_aligned is not None, "df_aligned required"
    assert len(df_aligned) == len(features), f"Alignment: df_aligned={len(df_aligned)} features={len(features)}"

    for window_name, cfg in TARGET_CONFIG.items():
        horizon = cfg["horizon"]
        close = df_aligned["Close"].values
        y = np.log(close[horizon:] / close[:-horizon])
        X = features[: len(y)]
        # Explicit dates_x / dates_y before filtering
        dates_full = pd.to_datetime(df_aligned["date"]).values
        dates_x = dates_full[:-horizon]  # feature dates (row i -> close[i])
        dates_y = dates_full[horizon:]   # label dates (row i -> close[i+horizon])
        assert len(dates_x) == len(y), f"dates_x len={len(dates_x)} y len={len(y)}"
        assert len(dates_y) == len(y), f"dates_y len={len(dates_y)} y len={len(y)}"

        valid = np.isfinite(y)
        X, y = X[valid], y[valid]
        dates_x, dates_y = dates_x[valid], dates_y[valid]
        assert len(dates_x) == len(y) and len(dates_y) == len(y), (
            f"dates_x/y len mismatch after valid: {len(dates_x)} {len(dates_y)} {len(y)}"
        )
        if len(X) < 5:
            continue
        for i in range(len(X)):
            assert pd.Timestamp(dates_x[i]) < pd.Timestamp(dates_y[i]), (
                f"LEAKAGE: row {i} feature_date={dates_x[i]} must be < label_date={dates_y[i]} "
                f"(horizon={horizon})"
            )
    print("PASS: feature date < label date for all rows (no leakage)")


if __name__ == "__main__":
    test_target_feature_alignment()
    print("All leakage checks passed.")
