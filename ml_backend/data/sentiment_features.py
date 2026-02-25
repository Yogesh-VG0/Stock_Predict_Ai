"""
Sentiment feature adapter — converts daily sentiment docs in MongoDB into
leakage-proof rolling features aligned to the price DataFrame.

Every feature for day t uses ONLY sentiment published strictly before market
close of day t.  The shift(1) in prepare_features guarantees point-in-time safety.

Features produced (12 columns):
 - sent_mean_1d   : yesterday's composite sentiment score (NaN if missing)
 - sent_mean_7d   : rolling 7-day mean sentiment (NaN if missing)
 - sent_mean_30d  : rolling 30-day mean sentiment (NaN if missing)
 - sent_momentum  : 7d mean − 30d mean (sentiment regime change) (NaN if missing)
 - sent_std_7d    : 7-day sentiment volatility (disagreement signal) (NaN if missing)
 - news_count_1d  : yesterday's news/article count (0 if missing)
 - news_count_7d  : rolling 7-day total article count (0 if missing)
 - news_spike_1d  : yesterday's count / 30d mean count — detects unusual coverage bursts
 - analyst_sentiment_7d : 7d rolling analyst consensus (NaN if missing)
 - analyst_rating_7d    : 7d rolling analyst rating score (NaN if missing)
 - sent_available  : 1.0 if sentiment data exists, 0.0 if not (missingness indicator)
 - sent_source_count : number of providers that contributed data (0 if missing)

All features are shifted by 1 before returning so that row t only sees data
available at close of day t-1 (strict point-in-time).

PIT SHIFT POLICY: Each adapter (macro, VIX, sector, sentiment) shifts
internally.  There is NO global shift in prepare_features().  Do not add
a second shift outside this module.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def make_sentiment_features(
    price_df: pd.DataFrame,
    ticker: str,
    mongo_client,
    *,
    lookback_extra_days: int = 60,
) -> pd.DataFrame:
    """Build sentiment features aligned to *price_df* dates.

    Parameters
    ----------
    price_df : DataFrame
        Must have a ``date`` column (datetime).  Features are aligned to these
        dates via a left join → every price row gets a feature row (NaN-filled
        where sentiment is missing).
    ticker : str
        Stock ticker symbol.
    mongo_client : MongoDBClient
        Must expose ``get_sentiment_timeseries(ticker, start, end)``.
    lookback_extra_days : int
        Extra days before the earliest price date to fetch sentiment.  Needed
        for the 30-day rolling window to be warm at the start of the series.

    Returns
    -------
    DataFrame with the same index as *price_df* and 12 sentiment columns.
    Score-based columns use NaN when missing (LightGBM treats as unknown).
    Count-based columns use 0 (genuinely no news). Missingness indicators
    (sent_available, sent_source_count) let the model learn when data is absent.
    """
    # Defaults when sentiment is unavailable.
    # Score-based features → NaN so LightGBM treats missing as unknown,
    # NOT as neutral (0).  Count features → 0 (genuinely no news).
    _NAN = float("nan")
    _defaults = {
        "sent_mean_1d": _NAN,
        "sent_mean_7d": _NAN,
        "sent_mean_30d": _NAN,
        "sent_momentum": _NAN,
        "sent_std_7d": _NAN,
        "news_count_1d": 0.0,
        "news_count_7d": 0.0,
        "news_spike_1d": 0.0,
        # FMP analyst features (v1.5 — sourced from sentiment collection)
        "analyst_sentiment_7d": _NAN,
        "analyst_rating_7d": _NAN,
        # price_target_gap_7d removed — detected as leakage (contains "target")
        # Missingness indicators — let the model learn "no data" vs "neutral"
        "sent_available": 0.0,
        "sent_source_count": 0.0,
    }
    empty = pd.DataFrame(
        {col: [val] * len(price_df) for col, val in _defaults.items()},
        index=price_df.index,
    )

    if mongo_client is None or not hasattr(mongo_client, "get_sentiment_timeseries"):
        return empty

    try:
        dates = pd.to_datetime(price_df["date"])
        start = pd.Timestamp(dates.min()) - timedelta(days=lookback_extra_days)
        end = pd.Timestamp(dates.max()) + timedelta(days=1)

        sent_df = mongo_client.get_sentiment_timeseries(ticker, start, end)
        if sent_df is None or sent_df.empty:
            return empty

        # Ensure sorted by date
        sent_df = sent_df.sort_values("date").reset_index(drop=True)
        sent_df["date"] = pd.to_datetime(sent_df["date"]).dt.normalize()

        # --- Build rolling features on sentiment timeseries ---
        score = sent_df["composite_sentiment"]
        count = sent_df["news_count"].astype(float)

        sent_df["sent_mean_1d"] = score  # will be shifted later
        sent_df["sent_mean_7d"] = score.rolling(7, min_periods=3).mean()
        sent_df["sent_mean_30d"] = score.rolling(30, min_periods=10).mean()
        sent_df["sent_momentum"] = sent_df["sent_mean_7d"] - sent_df["sent_mean_30d"]
        # std: leave NaN where insufficient data (don't fill 0 — unknown != no disagreement)
        sent_df["sent_std_7d"] = score.rolling(7, min_periods=3).std()
        sent_df["news_count_1d"] = count
        sent_df["news_count_7d"] = count.rolling(7, min_periods=1).sum()
        # news_spike_1d: daily count / 30d rolling mean count — detects unusual bursts
        news_count_30d_mean = count.rolling(30, min_periods=10).mean()
        sent_df["news_spike_1d"] = count / (news_count_30d_mean + 1e-6)

        # --- Missingness indicators ---
        # sent_available: 1 if we have sentiment data for this day
        sent_df["sent_available"] = 1.0
        # sent_source_count: number of providers that contributed
        if "sentiment_source_count" in sent_df.columns:
            sent_df["sent_source_count"] = sent_df["sentiment_source_count"].astype(float).fillna(0.0)
        else:
            # Fallback: estimate from non-zero confidence columns
            sent_df["sent_source_count"] = 1.0  # at least 1 source if row exists

        # --- FMP analyst features (v1.5) ---
        # These fields are already stored in the sentiment collection but were
        # previously unused by the ML pipeline.  Rolling 7-day mean smooths
        # the noisy daily values into stable signals.
        for col_src, col_dst in [
            ("fmp_analyst", "analyst_sentiment_7d"),
            ("fmp_rating", "analyst_rating_7d"),
            # price_target_gap_7d removed — detected as leakage
        ]:
            if col_src in sent_df.columns:
                raw = sent_df[col_src].astype(float)
                sent_df[col_dst] = raw.rolling(7, min_periods=3).mean()
            else:
                sent_df[col_dst] = _NAN

        feature_cols = list(_defaults.keys())
        sent_features = sent_df[["date"] + feature_cols].copy()

        # --- Align to price dates via left join ---
        price_dates = pd.DataFrame(
            {"date": pd.to_datetime(price_df["date"]).dt.normalize()},
            index=price_df.index,
        )
        merged = price_dates.merge(
            sent_features, on="date", how="left",
        )
        # Re-index to price_df.index (merge resets it)
        merged.index = price_df.index

        # --- Point-in-time shift: use yesterday's sentiment ---
        # shift(1) ensures row t sees sentiment from t-1 only.
        for col in feature_cols:
            merged[col] = merged[col].shift(1)

        # Fill ONLY count-based and indicator columns.
        # Score-based columns stay NaN so LightGBM knows data is missing.
        _fill_cols = {
            "news_count_1d": 0.0,
            "news_count_7d": 0.0,
            "news_spike_1d": 0.0,
            "sent_available": 0.0,
            "sent_source_count": 0.0,
        }
        for col, default in _fill_cols.items():
            if col in merged.columns:
                merged[col] = merged[col].fillna(default)

        return merged[feature_cols]

    except Exception as e:
        logger.warning("Could not build sentiment features for %s: %s", ticker, e)
        return empty

