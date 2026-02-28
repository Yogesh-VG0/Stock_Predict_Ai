"""
Earnings feature adapter — converts FMP earnings data stored in MongoDB into
leakage-proof features aligned to the price DataFrame.

Every feature for day t uses ONLY earnings that were reported strictly before
market close of day t.  The shift(1) ensures point-in-time safety.

Features produced (4 columns):
 - earnings_surprise     : EPS actual - EPS estimated (NaN if missing)
 - earnings_beat         : +1 if beat, -1 if miss, 0 if met/unknown (NaN if missing)
 - earnings_recency      : 1/(days_since_last_earnings + 1) decay (NaN if missing)
 - earnings_surprise_pct : surprise / |estimated| as percentage (NaN if missing)

Data sources:
  MongoDB collection ``alpha_vantage_data`` (endpoint: ``fmp_earnings``)
  Document schema: {ticker, endpoint, data: [{date, eps, epsEstimated, revenue, ...}], timestamp}

  Fallback: ``sentiment`` collection (fmp_raw_data.earnings field)

PIT SHIFT POLICY: Each adapter shifts internally.  There is NO global shift
in prepare_features().  Do not add a second shift outside this module.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def make_earnings_features(
    price_df: pd.DataFrame,
    ticker: str,
    mongo_client,
    *,
    lookback_extra_days: int = 400,
) -> pd.DataFrame:
    """Build earnings features aligned to *price_df* dates.

    Parameters
    ----------
    price_df : DataFrame
        Must have a ``date`` column (datetime).  Features are aligned to these
        dates via forward-fill of the latest known earnings data.
    ticker : str
        Stock ticker symbol.
    mongo_client : MongoDBClient
        Must expose ``db`` attribute for direct collection access.
    lookback_extra_days : int
        Extra days before the earliest price date to look for historical earnings.

    Returns
    -------
    DataFrame with the same index as *price_df* and 4 earnings columns.
    All score-based columns use NaN when missing (LightGBM treats as unknown).
    """
    _NAN = float("nan")
    _defaults = {
        "earnings_surprise": _NAN,
        "earnings_beat": _NAN,
        "earnings_recency": _NAN,
        "earnings_surprise_pct": _NAN,
    }
    empty = pd.DataFrame(
        {col: [val] * len(price_df) for col, val in _defaults.items()},
        index=price_df.index,
    )

    if mongo_client is None or not hasattr(mongo_client, "db"):
        return empty

    try:
        # --- Fetch earnings data from MongoDB ---
        earnings_records = []

        # Primary: alpha_vantage_data collection with endpoint=fmp_earnings
        try:
            col = mongo_client.db.get_collection("alpha_vantage_data")
            if col is not None:
                doc = col.find_one(
                    {"ticker": ticker.upper(), "endpoint": "fmp_earnings"},
                    sort=[("timestamp", -1)],
                )
                if doc and isinstance(doc.get("data"), list):
                    for entry in doc["data"]:
                        if isinstance(entry, dict) and entry.get("date"):
                            earnings_records.append(entry)
        except Exception as e:
            logger.debug("earnings_features: alpha_vantage_data lookup failed for %s: %s", ticker, e)

        # Fallback: sentiment collection (fmp_raw_data.earnings)
        if not earnings_records:
            try:
                sent_col = mongo_client.db.get_collection("sentiment")
                if sent_col is not None:
                    doc = sent_col.find_one(
                        {"ticker": ticker.upper()},
                        sort=[("date", -1)],
                    )
                    if doc:
                        fmp_data = doc.get("fmp_raw_data", {}) or {}
                        raw_earnings = fmp_data.get("earnings", [])
                        if isinstance(raw_earnings, list):
                            earnings_records = [e for e in raw_earnings if isinstance(e, dict) and e.get("date")]
            except Exception as e:
                logger.debug("earnings_features: sentiment fallback failed for %s: %s", ticker, e)

        # --- Fallback: yfinance earnings data ---
        if not earnings_records:
            try:
                import yfinance as yf
                stock = yf.Ticker(ticker)
                # quarterly earnings have 'Reported EPS' and 'Surprise(%)'
                eq = getattr(stock, 'quarterly_earnings', None)
                if eq is not None and not eq.empty:
                    for idx_date, row in eq.iterrows():
                        try:
                            eps_actual = row.get('Reported EPS') or row.get('Actual')
                            eps_estimated = row.get('Estimated') or row.get('EPS Estimate')
                            if eps_actual is not None and eps_estimated is not None:
                                earnings_records.append({
                                    'date': str(idx_date),
                                    'eps': float(eps_actual),
                                    'epsEstimated': float(eps_estimated),
                                })
                        except Exception:
                            continue
                if not earnings_records:
                    # Try earnings_dates as minimal fallback (dates only, no surprise)
                    dates = getattr(stock, 'earnings_dates', None)
                    if dates is not None and not dates.empty:
                        for idx_date, row in dates.iterrows():
                            try:
                                eps_actual = row.get('Reported EPS')
                                eps_est = row.get('EPS Estimate')
                                if eps_actual is not None and eps_est is not None:
                                    earnings_records.append({
                                        'date': str(idx_date.date()) if hasattr(idx_date, 'date') else str(idx_date),
                                        'eps': float(eps_actual),
                                        'epsEstimated': float(eps_est),
                                    })
                            except Exception:
                                continue
            except Exception as e:
                logger.debug("earnings_features: yfinance fallback failed for %s: %s", ticker, e)

        if not earnings_records:
            return empty

        # --- Parse earnings into a sorted DataFrame ---
        parsed = []
        for rec in earnings_records:
            try:
                report_date = pd.to_datetime(rec["date"]).normalize()
                eps_actual = rec.get("eps")
                eps_estimated = rec.get("epsEstimated")

                if eps_actual is None or eps_estimated is None:
                    continue

                eps_actual = float(eps_actual)
                eps_estimated = float(eps_estimated)
                surprise = eps_actual - eps_estimated
                beat = 1 if surprise > 0.001 else (-1 if surprise < -0.001 else 0)
                surprise_pct = (surprise / abs(eps_estimated)) if abs(eps_estimated) > 0.01 else 0.0

                parsed.append({
                    "report_date": report_date,
                    "earnings_surprise": surprise,
                    "earnings_beat": float(beat),
                    "earnings_surprise_pct": np.clip(surprise_pct, -5.0, 5.0),
                })
            except Exception:
                continue

        if not parsed:
            return empty

        earn_df = pd.DataFrame(parsed).sort_values("report_date").drop_duplicates("report_date", keep="last")

        # --- Align to price dates ---
        price_dates = pd.to_datetime(price_df["date"]).dt.normalize()
        result = pd.DataFrame(index=price_df.index)

        # For each price date, find the most recent earnings report BEFORE that date
        report_dates_np = earn_df["report_date"].values.astype("datetime64[D]")
        price_dates_np = price_dates.values.astype("datetime64[D]")

        # searchsorted: find index of latest report_date <= price_date
        idx = np.searchsorted(report_dates_np, price_dates_np, side="right") - 1

        # Build feature arrays
        surprise_arr = np.full(len(price_df), _NAN)
        beat_arr = np.full(len(price_df), _NAN)
        recency_arr = np.full(len(price_df), _NAN)
        surprise_pct_arr = np.full(len(price_df), _NAN)

        mask = idx >= 0
        if mask.any():
            valid_idx = idx[mask]
            surprise_arr[mask] = earn_df["earnings_surprise"].values[valid_idx]
            beat_arr[mask] = earn_df["earnings_beat"].values[valid_idx]
            surprise_pct_arr[mask] = earn_df["earnings_surprise_pct"].values[valid_idx]

            # Recency: 1 / (days_since + 1), decays from 1.0 to ~0
            days_since = (price_dates_np[mask] - report_dates_np[valid_idx]).astype("timedelta64[D]").astype(int)
            days_since = np.maximum(days_since, 0)  # safety
            recency_arr[mask] = 1.0 / (days_since + 1.0)

        result["earnings_surprise"] = surprise_arr
        result["earnings_beat"] = beat_arr
        result["earnings_recency"] = recency_arr
        result["earnings_surprise_pct"] = surprise_pct_arr

        # --- Point-in-time shift: use yesterday's data ---
        for col in _defaults.keys():
            result[col] = result[col].shift(1)

        return result

    except Exception as e:
        logger.warning("Could not build earnings features for %s: %s", ticker, e)
        return empty
