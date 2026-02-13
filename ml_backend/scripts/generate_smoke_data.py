"""
Generate scripts/smoke_data/AAPL.csv for deterministic smoke test fallback.
Run once: python -m scripts.generate_smoke_data
Requires: pandas, numpy
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd

def main():
    np.random.seed(42)
    dates = pd.date_range("2014-01-02", "2024-06-01", freq="B")
    n = len(dates)
    ret = np.random.randn(n) * 0.01
    ret[0] = 0
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.roll(close, 1)
    open_[0] = 100
    high = np.maximum(open_, close) * (1 + np.abs(np.random.randn(n)) * 0.005)
    low = np.minimum(open_, close) * (1 - np.abs(np.random.randn(n)) * 0.005)
    vol = np.random.lognormal(14, 1, n).astype(int)
    df = pd.DataFrame({
        "date": dates,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol,
    })
    out_dir = os.path.join(SCRIPT_DIR, "smoke_data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "AAPL.csv")
    df.to_csv(out_path, index=False)
    print(f"Created {out_path} with {len(df)} rows")

if __name__ == "__main__":
    main()
