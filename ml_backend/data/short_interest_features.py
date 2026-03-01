"""
Short interest feature adapter — converts short interest data stored in
MongoDB into leakage-proof features aligned to the price DataFrame.

Short interest is updated bi-weekly (settlement-cycle), so the latest
available value is forward-filled across rows.  shift(1) ensures PIT safety.

Features produced (3 columns):
 - si_short_float_pct       : Short interest as % of float (NaN if missing)
 - si_days_to_cover         : Short interest / avg daily volume (NaN if missing)
 - si_available             : 1.0 if data exists, 0.0 if not (missingness indicator)

Data sources:
  Primary: MongoDB collection ``short_interest_data``
  Document schema: {ticker, settlementDate, short_interest, avgDailyShareVolume,
                    daysToCover, short_float_pct/shortFloatPct, fetched_at}

  Fallback: ``sentiment`` collection (short_interest_data field)

PIT SHIFT POLICY: Each adapter shifts internally.  There is NO global shift
in prepare_features().  Do not add a second shift outside this module.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def make_short_interest_features(
    price_df: pd.DataFrame,
    ticker: str,
    mongo_client,
) -> pd.DataFrame:
    """Build short interest features aligned to *price_df* dates.

    Parameters
    ----------
    price_df : DataFrame
        Must have a ``date`` column (datetime).  Short interest values are
        forward-filled from the latest settlement date.
    ticker : str
        Stock ticker symbol.
    mongo_client : MongoDBClient
        Must expose ``db`` attribute for direct collection access.

    Returns
    -------
    DataFrame with the same index as *price_df* and 3 short interest columns.
    Score-based columns use NaN when missing; availability uses 0.0.
    """
    _NAN = float("nan")
    _defaults = {
        "si_short_float_pct": _NAN,
        "si_days_to_cover": _NAN,
        "si_available": 0.0,
    }
    empty = pd.DataFrame(
        {col: [val] * len(price_df) for col, val in _defaults.items()},
        index=price_df.index,
    )

    if mongo_client is None or not hasattr(mongo_client, "db"):
        return empty

    try:
        si_records = []

        # --- Primary: short_interest_data collection (full time series) ---
        try:
            si_col = mongo_client.db.get_collection("short_interest_data")
            if si_col is not None:
                docs = list(si_col.find(
                    {"ticker": ticker.upper()},
                    sort=[("settlementDate", 1)],
                ).limit(50))
                for doc in docs:
                    settle_date = doc.get("settlementDate")
                    if settle_date is None:
                        continue
                    sfp = (doc.get("short_float_pct")
                           or doc.get("shortFloatPct")
                           or doc.get("shortFloatPercentage"))
                    dtc = (doc.get("daysToCover")
                           or doc.get("days_to_cover")
                           or doc.get("daysToCoVerShortInterest"))
                    try:
                        sfp_val = float(sfp) if sfp is not None else None
                    except (ValueError, TypeError):
                        sfp_val = None
                    try:
                        dtc_val = float(dtc) if dtc is not None else None
                    except (ValueError, TypeError):
                        dtc_val = None
                    if sfp_val is not None:
                        si_records.append({
                            "date": pd.to_datetime(settle_date).normalize(),
                            "si_short_float_pct": np.clip(sfp_val, 0.0, 100.0),
                            "si_days_to_cover": np.clip(dtc_val, 0.0, 60.0) if dtc_val is not None else _NAN,
                        })
        except Exception as e:
            logger.debug("short_interest_features: direct collection failed for %s: %s", ticker, e)

        # --- Fallback: sentiment collection (latest only) ---
        if not si_records:
            try:
                sent_col = mongo_client.db.get_collection("sentiment")
                if sent_col is not None:
                    doc = sent_col.find_one(
                        {"ticker": ticker.upper()},
                        sort=[("date", -1)],
                    )
                    if doc:
                        si_data = doc.get("short_interest_data", [])
                        if isinstance(si_data, list) and si_data:
                            latest = si_data[0] if isinstance(si_data[0], dict) else {}
                            sfp = latest.get("short_float_pct") or latest.get("short_float_percentage")
                            dtc = latest.get("daysToCover") or latest.get("days_to_cover")
                            sfp_val = None
                            dtc_val = None
                            if sfp is not None:
                                try:
                                    sfp_val = float(sfp)
                                except (ValueError, TypeError):
                                    pass
                            if dtc is not None:
                                try:
                                    dtc_val = float(dtc)
                                except (ValueError, TypeError):
                                    pass
                            if sfp_val is not None:
                                doc_date = pd.to_datetime(doc.get("date", datetime.now(timezone.utc))).normalize()
                                si_records.append({
                                    "date": doc_date,
                                    "si_short_float_pct": np.clip(sfp_val, 0.0, 100.0),
                                    "si_days_to_cover": np.clip(dtc_val, 0.0, 60.0) if dtc_val is not None else _NAN,
                                })
            except Exception as e:
                logger.debug("short_interest_features: sentiment fallback failed for %s: %s", ticker, e)

        if not si_records:
            return empty

        # --- Build time-series aligned features ---
        si_df = pd.DataFrame(si_records).sort_values("date").drop_duplicates("date", keep="last")

        # Align to price dates via asof merge (forward-fill from last settlement)
        price_dates = pd.DataFrame(
            {"date": pd.to_datetime(price_df["date"]).dt.normalize()},
            index=price_df.index,
        )

        # Normalise datetime resolution so merge_asof never sees
        # incompatible dtypes (e.g. datetime64[s] vs datetime64[us]).
        si_df["date"] = pd.to_datetime(si_df["date"]).astype("datetime64[ns]")
        price_dates["date"] = price_dates["date"].astype("datetime64[ns]")

        merged = pd.merge_asof(
            price_dates.sort_values("date"),
            si_df,
            on="date",
            direction="backward",
        )
        # Re-index to price_df order
        merged.index = price_dates.sort_values("date").index
        merged = merged.reindex(price_df.index)

        result = pd.DataFrame(index=price_df.index)
        result["si_short_float_pct"] = merged["si_short_float_pct"]
        result["si_days_to_cover"] = merged["si_days_to_cover"]
        result["si_available"] = result["si_short_float_pct"].notna().astype(float)

        # --- Point-in-time shift: use yesterday's data ---
        for col_name in ("si_short_float_pct", "si_days_to_cover", "si_available"):
            result[col_name] = result[col_name].shift(1)

        # Fill availability indicator after shift
        result["si_available"] = result["si_available"].fillna(0.0)

        return result

    except Exception as e:
        logger.warning("Could not build short interest features for %s: %s", ticker, e)
        return empty
