"""
Sentiment feature adapter — converts daily sentiment docs in MongoDB into
leakage-proof rolling features aligned to the price DataFrame.

Every feature for day t uses ONLY sentiment published strictly before market
close of day t.  The shift(1) in prepare_features guarantees point-in-time safety.

Features produced (8 columns):
 - sent_mean_1d   : yesterday's composite sentiment score
 - sent_mean_7d   : rolling 7-day mean sentiment
 - sent_mean_30d  : rolling 30-day mean sentiment
 - sent_momentum  : 7d mean − 30d mean (sentiment regime change)
 - sent_std_7d    : 7-day sentiment volatility (disagreement signal)
 - news_count_1d  : yesterday's news/article count
 - news_count_7d  : rolling 7-day total article count
 - news_spike_1d  : yesterday's count / 30d mean count — detects unusual coverage bursts

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
    DataFrame with the same index as *price_df* and 7 sentiment columns.
    Missing values are filled with neutral defaults (0 for scores, 0 for
    counts).
    """
    # Neutral defaults if sentiment is unavailable
    _defaults = {
        "sent_mean_1d": 0.0,
        "sent_mean_7d": 0.0,
        "sent_mean_30d": 0.0,
        "sent_momentum": 0.0,
        "sent_std_7d": 0.0,
        "news_count_1d": 0.0,
        "news_count_7d": 0.0,
        "news_spike_1d": 0.0,
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
        sent_df["sent_std_7d"] = score.rolling(7, min_periods=3).std().fillna(0)
        sent_df["news_count_1d"] = count
        sent_df["news_count_7d"] = count.rolling(7, min_periods=1).sum()
        # news_spike_1d: daily count / 30d rolling mean count — detects unusual bursts
        news_count_30d_mean = count.rolling(30, min_periods=10).mean()
        sent_df["news_spike_1d"] = count / (news_count_30d_mean + 1e-6)

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

        # Fill missing values with neutral defaults
        for col, default in _defaults.items():
            merged[col] = merged[col].fillna(default)

        return merged[feature_cols]

    except Exception as e:
        logger.warning("Could not build sentiment features for %s: %s", ticker, e)
        return empty
