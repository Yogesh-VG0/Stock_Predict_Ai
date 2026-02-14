"""
Minimal V1 Feature Engine - Leakage-proof, point-in-time features.
Every feature for row t uses ONLY data available at/before market close of day t.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.tickers import get_ticker_id
from ..config.feature_config_v1 import FEATURE_CONFIG_V1

logger = logging.getLogger(__name__)

# Sector mapping for S&P 100 tickers → sector ETF
SECTOR_MAP = {
    # Technology → XLK
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "GOOGL": "XLK", "META": "XLK",
    "ORCL": "XLK", "CRM": "XLK", "AVGO": "XLK", "AMD": "XLK", "INTC": "XLK",
    "CSCO": "XLK", "ADBE": "XLK", "QCOM": "XLK", "TXN": "XLK", "NOW": "XLK",
    "INTU": "XLK", "IBM": "XLK", "ACN": "XLK", "AMAT": "XLK",
    # Consumer Discretionary → XLY
    "AMZN": "XLY", "HD": "XLY", "LOW": "XLY", "NFLX": "XLY", "SBUX": "XLY",
    "TSLA": "XLY", "NKE": "XLY", "MCD": "XLY", "DIS": "XLY", "BKNG": "XLY",
    "TGT": "XLY",
    # Financials → XLF
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF", "MS": "XLF",
    "V": "XLF", "MA": "XLF", "AXP": "XLF", "BLK": "XLF", "SCHW": "XLF",
    "C": "XLF", "COF": "XLF", "BK": "XLF", "MET": "XLF", "AIG": "XLF",
    "USB": "XLF", "PYPL": "XLF",
    # Energy → XLE
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE",
    # Healthcare → XLV
    "JNJ": "XLV", "UNH": "XLV", "PFE": "XLV", "ABBV": "XLV", "LLY": "XLV",
    "ABT": "XLV", "TMO": "XLV", "DHR": "XLV", "MRK": "XLV", "AMGN": "XLV",
    "GILD": "XLV", "ISRG": "XLV", "MDT": "XLV", "BMY": "XLV", "CVS": "XLV",
    # Consumer Staples → XLP
    "WMT": "XLP", "COST": "XLP", "PG": "XLP", "KO": "XLP", "PEP": "XLP",
    "MDLZ": "XLP", "CL": "XLP", "MO": "XLP", "PM": "XLP",
    # Industrials → XLI
    "CAT": "XLI", "HON": "XLI", "UNP": "XLI", "BA": "XLI", "RTX": "XLI",
    "LMT": "XLI", "DE": "XLI", "GE": "XLI", "GD": "XLI", "EMR": "XLI",
    "FDX": "XLI", "UPS": "XLI", "MMM": "XLI",
    # Communication → XLC
    "CMCSA": "XLC", "VZ": "XLC", "T": "XLC", "CHTR": "XLC", "TMUS": "XLC",
    # Utilities → XLU
    "NEE": "XLU", "SO": "XLU", "DUK": "XLU",
    # Real Estate → XLRE
    "AMT": "XLRE", "SPG": "XLRE",
    # Materials → XLB
    "LIN": "XLB",
    # Other / Conglomerates
    "BRK-B": "XLF", "PLTR": "XLK",
}
DEFAULT_SECTOR = "XLF"

# Sector ID mapping (module-level; stable across calls)
SECTOR_ID = {
    "XLK": 0, "XLF": 1, "XLE": 2, "XLV": 3, "XLP": 4, "XLY": 5,
    "XLI": 6, "XLC": 7, "XLU": 8, "XLRE": 9, "XLB": 10,
}


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure date column exists and is datetime."""
    if df is None or df.empty:
        return df
    if "date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        idx_name = df.index.name  # capture before reset_index() clears it
        df = df.reset_index()
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        elif idx_name == "Date" and "index" in df.columns:
            df = df.rename(columns={"index": "date"})
        elif "index" in df.columns:
            df = df.rename(columns={"index": "date"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _ensure_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Ensure standard OHLCV columns exist."""
    if df is None or df.empty:
        return df
    rename_map = {
        f"Close_{ticker}": "Close", f"Open_{ticker}": "Open",
        f"High_{ticker}": "High", f"Low_{ticker}": "Low",
        f"Volume_{ticker}": "Volume",
    }
    df = df.rename(columns=rename_map)
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df["Close"] = df["Adj Close"]
    # Pandas 3.0: collapse duplicate columns (e.g. from yfinance MultiIndex) to single Series
    for col in ["date", "Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns and isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]
    return df


def _to_series(x):
    """Ensure 1D Series for pandas 3.0 compatibility (avoids DataFrame assignment errors)."""
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x


class MinimalFeatureEngineer:
    """
    Leakage-proof feature engineering. All features for row t use only data
    available at market close of day t. No lookahead, no future information.
    """

    def __init__(self, mongo_client=None):
        self.mongo_client = mongo_client
        self.scaler = None
        self.feature_columns = []
        self.sector_cache = {}
        self.macro_cache = {}
        from .cache_fetch import FrameCache

        self.price_cache = FrameCache(max_items=64)
        self._macro_frame_cache = {}

    def prepare_features(
        self,
        df: pd.DataFrame,
        ticker: str = None,
        as_of_date: Optional[datetime] = None,
        mongo_client=None,
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Build minimal, leakage-proof features.
        Returns (features_array, metadata_dict).
        """
        try:
            df = df.copy()
            df = _ensure_date_column(df)
            df = _ensure_ohlcv(df, ticker or "UNKNOWN")

            required = ["Open", "High", "Low", "Close", "Volume"]
            if any(c not in df.columns for c in required):
                logger.error(f"Missing OHLCV columns: {[c for c in required if c not in df.columns]}")
                return None, {}

            # Flatten MultiIndex columns (pandas 3.0 / yfinance compatibility)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.sort_values("date").reset_index(drop=True)
            lookback = FEATURE_CONFIG_V1["lookback_days"]
            min_rows = FEATURE_CONFIG_V1["min_rows"]

            if len(df) < min_rows:
                logger.warning(f"Insufficient data: {len(df)} rows, need {min_rows}")
                return None, {}

            mc = mongo_client or self.mongo_client

            # Ensure OHLCV are Series (pandas 3.0 / yfinance MultiIndex compatibility)
            close = _to_series(df["Close"])
            vol = _to_series(df["Volume"])

            # 1. Returns (CRITICAL - use shift to avoid leakage)
            df["log_return_1d"] = np.log(close / close.shift(1))
            df["log_return_5d"] = np.log(close / close.shift(5))
            df["log_return_21d"] = np.log(close / close.shift(21))

            # 2. Volatility (past only)
            df["volatility_20d"] = df["log_return_1d"].rolling(20, min_periods=5).std()
            df["intraday_range"] = (_to_series(df["High"]) - _to_series(df["Low"])) / close.where(close != 0, np.nan)
            df["overnight_gap"] = (_to_series(df["Open"]) / close.shift(1) - 1).replace([np.inf, -np.inf], 0)

            # 3. Volume features
            df["volume_ma20"] = vol.rolling(20, min_periods=5).mean()
            vol_ma = df["volume_ma20"]
            df["volume_ratio"] = (vol / vol_ma.where(vol_ma != 0, np.nan)).replace([np.inf, -np.inf], np.nan)
            df["dollar_volume"] = close * vol
            # Volume z-score over 60-day window (replaces slow rolling percentile rank)
            vol_m60 = vol.rolling(60, min_periods=20).mean()
            vol_s60 = vol.rolling(60, min_periods=20).std()
            df["volume_z60"] = ((vol - vol_m60) / vol_s60.replace(0, np.nan)).fillna(0).clip(-5, 5)

            # 4. Trend indicators (shifted - use past close only)
            df["sma_20"] = close.rolling(20, min_periods=20).mean()
            df["sma_50"] = close.rolling(50, min_periods=20).mean()
            sma20 = _to_series(df["sma_20"])
            sma50 = _to_series(df["sma_50"])
            df["price_vs_sma20"] = (close / sma20 - 1).replace([np.inf, -np.inf], 0)
            df["price_vs_sma50"] = (close / sma50 - 1).replace([np.inf, -np.inf], 0)
            df["trend_20d"] = (close > sma20).astype(int)
            df["momentum_5d"] = close.pct_change(5).replace([np.inf, -np.inf], 0)

            # 4b. Additional high-value features (leakage-free)
            # Momentum acceleration: change in momentum (second derivative)
            df["momentum_accel"] = _to_series(df["momentum_5d"]).diff().fillna(0)
            # Volume momentum: 5-day change in volume (demand shifts)
            df["volume_momentum_5d"] = vol.pct_change(5).replace([np.inf, -np.inf], 0).fillna(0)
            # Volume/volatility ratio: high volume + low vol = conviction (informed flow)
            vol_20d = _to_series(df["volatility_20d"])
            df["volume_vol_ratio"] = (
                _to_series(df["volume_ratio"]) / (vol_20d + 1e-6)
            ).replace([np.inf, -np.inf], 0).fillna(0)
            # Bollinger Band position: where price sits within ±2σ band (-1 to +1)
            # Use price-space std so numerator/denominator are both in dollars
            # min_periods=20 for stable BB(20) — matches standard Bollinger Band definition
            std_20_price = close.rolling(20, min_periods=20).std()
            # Guard std==0 (illiquid/flat periods) → NaN → fillna(0)
            bb_raw = (close - sma20) / (2 * std_20_price.replace(0, np.nan))
            df["bb_position"] = bb_raw.replace([np.inf, -np.inf], 0).clip(-3, 3).fillna(0)
            # Price/volume divergence: price up + volume down (or vice-versa) = weak move
            ret_sign = np.sign(_to_series(df["log_return_1d"]))
            vol_sign = np.sign(vol.pct_change(5).fillna(0))
            df["price_vol_divergence"] = (ret_sign * vol_sign * -1).fillna(0)  # -1 when divergent

            # 5. RSI (14-period, standard)
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14, min_periods=5).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=5).mean()
            rs = gain / loss.where(loss != 0, np.nan)
            df["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

            # 5b. Sector and ticker IDs for pooled model (deterministic; stable across runs)
            sector = SECTOR_MAP.get((ticker or "").upper(), DEFAULT_SECTOR)
            df["sector_id"] = SECTOR_ID.get(sector, 1)
            df["ticker_id"] = get_ticker_id(ticker)

            # 6. Relative strength vs SPY (if available)
            df = self._add_relative_strength(df, ticker, mc)

            # 7. Minimal macro (lagged - never same-day release)
            df = self._add_macro_features(df, mc)

            # 7b. Cross-asset features: VIX proxy + sector ETF returns (lagged)
            df = self._add_cross_asset_features(df, ticker, mc)

            # 8. Regime flags (quantile uses prior window only - no self-referential threshold)
            q = df["volatility_20d"].shift(1).rolling(60, min_periods=20).quantile(0.7)
            df["vol_regime"] = (df["volatility_20d"] > q).astype(int).fillna(0)

            # 9. Earnings proximity (days to/since - safe, no outcomes)
            df = self._add_earnings_proximity(df, ticker, mc)

            # Define feature columns. KEEP returns (log_return_1d/5d/21d at t) - safe for predicting r_{t+1}
            # SPY_close used for market-neutral target only, not as feature
            # Exclude raw OHLCV, intermediate helpers, and anything that embeds absolute price level
            # Also block stray merge artifacts, Mongo metadata, and CV markers
            exclude = {
                "date", "date_norm", "macro_date", "date_merge", "timestamp", "_id",
                "Open", "High", "Low", "Close", "Volume", "Adj Close",
                "sma_20", "sma_50", "volume_ma20", "dollar_volume",
                "SPY_close", "VIX_close", "sector_etf_close",
                "split", "fold", "ticker",
            }
            self.feature_columns = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]

            # Safety: assert no column names that smell like targets/labels/future data
            # Use startswith/exact-token checks to avoid false positives on legitimate
            # feature names like volatility_20d, intraday_range, spy_vol_20d
            _leakage_substrings = ["target", "label", "future", "forward"]
            _leakage_prefixes = ["y_"]  # catches y_train, y_test, y_pred etc.
            _bad = [c for c in self.feature_columns
                    if any(k in c.lower() for k in _leakage_substrings)
                    or any(c.lower().startswith(p) for p in _leakage_prefixes)]
            if _bad:
                logger.error("Potential leakage columns detected in features: %s — removing them", _bad)
                self.feature_columns = [c for c in self.feature_columns if c not in _bad]

            # Stricter warmup: require core features to exist (no fake zeros in warmup)
            core = ["log_return_1d", "volatility_20d", "volume_ratio", "price_vs_sma20", "rsi"]
            core = [c for c in core if c in df.columns]
            if len(core) < 5:
                missing = set(["log_return_1d", "volatility_20d", "volume_ratio", "price_vs_sma20", "rsi"]) - set(df.columns)
                logger.error("Missing core columns: %s (df has: %s)", missing, list(df.columns[:15]))
                return None, {}
            df_clean = df.dropna(subset=core).copy()
            cols_to_fill = [c for c in self.feature_columns if c in df_clean.columns]
            df_clean[cols_to_fill] = df_clean[cols_to_fill].fillna(0)

            # Ensure we have feature columns
            available = [c for c in self.feature_columns if c in df_clean.columns]
            if len(available) < 5:
                logger.error(f"Too few features: {available}")
                return None, {}

            feature_df = df_clean[available].copy()
            for col in feature_df.columns:
                feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce").fillna(0)

            # Replace inf
            feature_df = feature_df.replace([np.inf, -np.inf], 0)

            self.feature_columns = available
            features = feature_df.values.astype(np.float32)
            # df_aligned must match feature rows exactly (same index)
            df_aligned = df_clean.loc[feature_df.index].copy()

            # If as_of_date specified, filter to that row only
            if as_of_date is not None and "date" in df_aligned.columns:
                mask = df_aligned["date"] <= pd.Timestamp(as_of_date)
                idx = np.where(mask)[0]
                if len(idx) > 0:
                    last_idx = int(idx[-1])
                    features = features[last_idx : last_idx + 1]
                    df_aligned = df_aligned.iloc[last_idx : last_idx + 1]
                else:
                    features = features[-1:]
                    df_aligned = df_aligned.iloc[-1:]

            metadata = {
                "feature_columns": self.feature_columns,
                "n_features": len(self.feature_columns),
                "n_rows": len(features),
                "date_range": (str(df["date"].min()), str(df["date"].max())),
                "df_aligned": df_aligned,  # Exact same rows as features
            }
            return features, metadata

        except Exception as e:
            logger.error(f"Error in prepare_features: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, {}

    def _add_relative_strength(self, df: pd.DataFrame, ticker: str, mongo_client) -> pd.DataFrame:
        """Add stock return minus SPY return using Mongo-first SPY data."""
        try:
            from .cache_fetch import fetch_price_df_mongo_first

            dates = _to_series(df["date"])
            start = pd.Timestamp(dates.min()) - timedelta(days=30)
            # Use 'now' or a fixed far future date to ensure stable cache key during loop
            # df merge handles point-in-time safety (left join on df dates)
            end = datetime.utcnow() + timedelta(days=2)

            spy_df = fetch_price_df_mongo_first(
                mongo_client,
                "SPY",
                start,
                end,
                cache=self.price_cache,
                allow_fallback_yfinance=True,
            )
            if spy_df is None or spy_df.empty:
                df["rel_strength_spy"] = 0.0
                df["spy_vol_20d"] = 0.0
                df["spy_vol_regime"] = 0
                return df

            spy_df = _ensure_date_column(spy_df)
            spy_df = _ensure_ohlcv(spy_df, "SPY")

            spy_df = spy_df.sort_values("date").drop_duplicates("date", keep="last")
            spy_df["date_norm"] = pd.to_datetime(_to_series(spy_df["date"])).dt.normalize()
            spy_close = _to_series(spy_df["Close"])
            spy_df["SPY_return"] = np.log(spy_close / spy_close.shift(1))
            spy_df["spy_vol_20d"] = spy_df["SPY_return"].rolling(20, min_periods=5).std()
            q = spy_df["spy_vol_20d"].shift(1).rolling(60, min_periods=20).quantile(0.7)
            spy_df["spy_vol_regime"] = (spy_df["spy_vol_20d"] > q).astype(int).fillna(0)
            spy_merge = spy_df[["date_norm", "SPY_return", "Close", "spy_vol_20d", "spy_vol_regime"]].copy()
            spy_merge = spy_merge.rename(columns={"Close": "SPY_close"})

            df["date_norm"] = pd.to_datetime(_to_series(df["date"])).dt.normalize()
            df = df.merge(spy_merge, on="date_norm", how="left")
            df["rel_strength_spy"] = (df["log_return_1d"] - df["SPY_return"]).fillna(0.0)
            df["spy_vol_20d"] = df["spy_vol_20d"].ffill().bfill().fillna(0)
            df["spy_vol_regime"] = df["spy_vol_regime"].fillna(0)
            df = df.drop(columns=["SPY_return", "date_norm"], errors="ignore")
        except Exception as e:
            logger.warning("Could not add SPY relative strength: %s", e)
            df["rel_strength_spy"] = 0.0
            if "spy_vol_20d" not in df.columns:
                df["spy_vol_20d"] = 0.0
            if "spy_vol_regime" not in df.columns:
                df["spy_vol_regime"] = 0
        return df

    def _add_macro_features(self, df: pd.DataFrame, mongo_client) -> pd.DataFrame:
        """Add minimal lagged macro (no same-day release). Cached by date range."""
        try:
            if mongo_client is None or not hasattr(mongo_client, "db"):
                df["macro_spread_2y10y"] = 0
                df["macro_fed_funds"] = 0
                return df
            dates = _to_series(df["date"])
            start = (pd.Timestamp(dates.min()) - timedelta(days=60)).strftime("%Y-%m-%d")
            end = (pd.Timestamp(dates.max()) + timedelta(days=1)).strftime("%Y-%m-%d")
            cache_key = ("MACRO", start, end)
            if cache_key in self._macro_frame_cache:
                macro_df = self._macro_frame_cache[cache_key].copy()
            else:
                from .fred_macro import fetch_and_store_all_fred_indicators

                data = fetch_and_store_all_fred_indicators(start, end, mongo_client)
                if not data:
                    df["macro_spread_2y10y"] = 0
                    df["macro_fed_funds"] = 0
                    return df
                series = {}
                if "TREASURY_10Y" in data and data["TREASURY_10Y"]:
                    series["10y"] = pd.Series(data["TREASURY_10Y"])
                if "TREASURY_2Y" in data and data["TREASURY_2Y"]:
                    series["2y"] = pd.Series(data["TREASURY_2Y"])
                if "FEDERAL_FUNDS_RATE" in data and data["FEDERAL_FUNDS_RATE"]:
                    series["ff"] = pd.Series(data["FEDERAL_FUNDS_RATE"])
                macro_df = pd.DataFrame(series)
                if macro_df.empty:
                    df["macro_spread_2y10y"] = 0
                    df["macro_fed_funds"] = 0
                    return df
                macro_df.index = pd.to_datetime(macro_df.index)
                macro_df = macro_df.resample("D").ffill()
                if "10y" in macro_df.columns and "2y" in macro_df.columns:
                    macro_df["spread"] = macro_df["10y"] - macro_df["2y"]
                else:
                    macro_df["spread"] = 0
                if "ff" not in macro_df.columns:
                    macro_df["ff"] = 0
                macro_df = macro_df.reset_index()
                macro_df = macro_df.rename(columns={"index": "macro_date"})
                macro_df["macro_date"] = pd.to_datetime(macro_df["macro_date"]).dt.normalize()
                macro_df = macro_df.rename(columns={"spread": "macro_spread_2y10y", "ff": "macro_fed_funds"})
                self._macro_frame_cache[cache_key] = macro_df.copy()

            df["date_norm"] = pd.to_datetime(df["date"]).dt.normalize()
            df = df.merge(
                macro_df[["macro_date", "macro_spread_2y10y", "macro_fed_funds"]],
                left_on="date_norm",
                right_on="macro_date",
                how="left",
            )
            df = df.drop(columns=["date_norm", "macro_date"], errors="ignore")
            # ffill then shift(1) then fillna(0) - macro usable at close(t+1)
            df["macro_spread_2y10y"] = df["macro_spread_2y10y"].ffill().shift(1).fillna(0)
            df["macro_fed_funds"] = df["macro_fed_funds"].ffill().shift(1).fillna(0)
        except Exception as e:
            logger.warning("Could not add macro: %s", e)
            df["macro_spread_2y10y"] = 0
            df["macro_fed_funds"] = 0
        return df

    def _add_cross_asset_features(self, df: pd.DataFrame, ticker: str, mongo_client) -> pd.DataFrame:
        """Add cross-asset regime context: VIX proxy + sector ETF returns.

        All features use shift(1) for strict point-in-time safety.
        VIX improves regime detection; sector ETFs capture rotation.
        """
        _ca_defaults = {
            "vix_return_1d": 0.0, "vix_vol_20d": 0.0, "vix_level": 20.0,
            "sector_etf_return_1d": 0.0, "sector_etf_return_5d": 0.0,
            "macro_spread_chg": 0.0, "macro_ff_chg": 0.0,
        }
        try:
            from .cache_fetch import fetch_price_df_mongo_first

            dates = _to_series(df["date"])
            start = pd.Timestamp(dates.min()) - timedelta(days=60)
            end = datetime.utcnow() + timedelta(days=2)
            df["date_norm"] = pd.to_datetime(dates).dt.normalize()

            # --- VIX proxy ---
            vix_df = fetch_price_df_mongo_first(
                mongo_client, "^VIX", start, end,
                cache=self.price_cache, allow_fallback_yfinance=True,
            )
            if vix_df is not None and not vix_df.empty:
                vix_df = _ensure_date_column(vix_df)
                vix_df = _ensure_ohlcv(vix_df, "^VIX")
                vix_df = vix_df.sort_values("date").drop_duplicates("date", keep="last")
                vix_df["date_norm"] = pd.to_datetime(_to_series(vix_df["date"])).dt.normalize()
                vix_close = _to_series(vix_df["Close"])
                vix_df["vix_return_1d"] = np.log(vix_close / vix_close.shift(1))
                vix_df["vix_vol_20d"] = vix_df["vix_return_1d"].rolling(20, min_periods=5).std()
                vix_df["vix_level"] = vix_close
                vix_merge = vix_df[["date_norm", "vix_return_1d", "vix_vol_20d", "vix_level"]].copy()
                df = df.merge(vix_merge, on="date_norm", how="left")
                # shift(1) for point-in-time safety: use yesterday's VIX data
                df["vix_return_1d"] = df["vix_return_1d"].shift(1).fillna(0)
                df["vix_vol_20d"] = df["vix_vol_20d"].shift(1).ffill().fillna(0)
                df["vix_level"] = df["vix_level"].shift(1).ffill().fillna(20)
            else:
                for c in ["vix_return_1d", "vix_vol_20d", "vix_level"]:
                    df[c] = _ca_defaults[c]

            # --- Sector ETF return (ticker's own sector, not SPY) ---
            sector_etf = SECTOR_MAP.get((ticker or "").upper(), DEFAULT_SECTOR)
            if sector_etf and sector_etf != "SPY":
                etf_df = fetch_price_df_mongo_first(
                    mongo_client, sector_etf, start, end,
                    cache=self.price_cache, allow_fallback_yfinance=True,
                )
                if etf_df is not None and not etf_df.empty:
                    etf_df = _ensure_date_column(etf_df)
                    etf_df = _ensure_ohlcv(etf_df, sector_etf)
                    etf_df = etf_df.sort_values("date").drop_duplicates("date", keep="last")
                    etf_df["date_norm"] = pd.to_datetime(_to_series(etf_df["date"])).dt.normalize()
                    etf_close = _to_series(etf_df["Close"])
                    etf_df["sector_etf_return_1d"] = np.log(etf_close / etf_close.shift(1))
                    etf_df["sector_etf_return_5d"] = np.log(etf_close / etf_close.shift(5))
                    etf_merge = etf_df[["date_norm", "sector_etf_return_1d", "sector_etf_return_5d"]].copy()
                    df = df.merge(etf_merge, on="date_norm", how="left", suffixes=("", "_etf_dup"))
                    # Drop any duplicate columns from merge
                    dup_cols = [c for c in df.columns if c.endswith("_etf_dup")]
                    if dup_cols:
                        df = df.drop(columns=dup_cols)
                    df["sector_etf_return_1d"] = df["sector_etf_return_1d"].shift(1).fillna(0)
                    df["sector_etf_return_5d"] = df["sector_etf_return_5d"].shift(1).fillna(0)
                else:
                    df["sector_etf_return_1d"] = 0.0
                    df["sector_etf_return_5d"] = 0.0
            else:
                df["sector_etf_return_1d"] = 0.0
                df["sector_etf_return_5d"] = 0.0

            df = df.drop(columns=["date_norm"], errors="ignore")

            # --- Macro rate CHANGES (complement levels from _add_macro_features) ---
            if "macro_spread_2y10y" in df.columns:
                df["macro_spread_chg"] = _to_series(df["macro_spread_2y10y"]).diff().fillna(0)
            else:
                df["macro_spread_chg"] = 0.0
            if "macro_fed_funds" in df.columns:
                df["macro_ff_chg"] = _to_series(df["macro_fed_funds"]).diff().fillna(0)
            else:
                df["macro_ff_chg"] = 0.0

        except Exception as e:
            logger.warning("Could not add cross-asset features: %s", e)
            for col, default in _ca_defaults.items():
                if col not in df.columns:
                    df[col] = default
        return df

    def _add_earnings_proximity(self, df: pd.DataFrame, ticker: str, mongo_client) -> pd.DataFrame:
        """Add days_since_earnings only. Vectorized for speed."""
        df["days_since_earnings"] = 90
        try:
            if mongo_client is None or not hasattr(mongo_client, "db"):
                return df

            # Try dedicated fundamentals_events collection first, fall back to sentiment
            earnings = []
            for coll_name in ["fundamentals_events", "sentiment"]:
                coll = mongo_client.db[coll_name]
                if coll is None:
                    continue
                doc = coll.find_one({"ticker": ticker}, sort=[("timestamp", -1)])
                if not doc:
                    continue
                # Staleness guard: ignore docs older than 180 days
                doc_ts = doc.get("timestamp")
                if doc_ts and hasattr(doc_ts, "timestamp"):
                    age_days = (datetime.utcnow() - doc_ts).days
                    if age_days > 180:
                        logger.debug("Skipping stale %s doc for %s (%d days old)", coll_name, ticker, age_days)
                        continue
                fmp = doc.get("fmp_raw_data", {}) or {}
                earnings = fmp.get("earnings", []) or doc.get("earnings", [])
                if earnings:
                    break
            if not earnings:
                return df
            earnings_dates = []
            for ev in earnings[:50]:
                ed = ev.get("date") or ev.get("reportedDate")
                if ed:
                    try:
                        earnings_dates.append(pd.to_datetime(ed).normalize())
                    except Exception:
                        pass
            if not earnings_dates:
                return df
            earnings_dates = np.array(sorted(set(earnings_dates)), dtype="datetime64[D]")
            df_dates = pd.to_datetime(df["date"]).dt.normalize().values.astype("datetime64[D]")
            idx = np.searchsorted(earnings_dates, df_dates, side="right") - 1
            days_since = np.full(len(df), 90, dtype=int)
            mask = idx >= 0
            if mask.any():
                delta = (df_dates[mask] - earnings_dates[idx[mask]]).astype("timedelta64[D]").astype(int)
                days_since[mask] = np.minimum(delta, 90)
            df["days_since_earnings"] = days_since
        except Exception as e:
            logger.warning(f"Could not add earnings proximity: {e}")
        return df
