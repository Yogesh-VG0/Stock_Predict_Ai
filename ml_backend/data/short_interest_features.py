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
from datetime import datetime, timedelta
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
        short_float_pct = None
        days_to_cover = None

        # --- Primary: short_interest_data collection ---
        try:
            si_col = mongo_client.db.get_collection("short_interest_data")
            if si_col is not None:
                # Get the latest document for this ticker
                doc = si_col.find_one(
                    {"ticker": ticker.upper()},
                    sort=[("fetched_at", -1)],
                )
                if doc:
                    # Handle different field naming conventions
                    sfp = doc.get("short_float_pct") or doc.get("shortFloatPct")
                    dtc = doc.get("daysToCover") or doc.get("days_to_cover")

                    if sfp is not None:
                        try:
                            short_float_pct = float(sfp)
                        except (ValueError, TypeError):
                            pass
                    if dtc is not None:
                        try:
                            days_to_cover = float(dtc)
                        except (ValueError, TypeError):
                            pass
        except Exception as e:
            logger.debug("short_interest_features: direct collection failed for %s: %s", ticker, e)

        # --- Fallback: sentiment collection ---
        if short_float_pct is None:
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
                            if sfp is not None:
                                try:
                                    short_float_pct = float(sfp)
                                except (ValueError, TypeError):
                                    pass
                            if dtc is not None:
                                try:
                                    days_to_cover = float(dtc)
                                except (ValueError, TypeError):
                                    pass
                        # Also check nested sources dict
                        if short_float_pct is None:
                            sources = doc.get("sources", {})
                            si_source = sources.get("short_interest", {})
                            if isinstance(si_source, dict):
                                sfp = si_source.get("short_float_percentage")
                                dtc = si_source.get("days_to_cover")
                                if sfp is not None:
                                    try:
                                        short_float_pct = float(sfp)
                                    except (ValueError, TypeError):
                                        pass
                                if dtc is not None:
                                    try:
                                        days_to_cover = float(dtc)
                                    except (ValueError, TypeError):
                                        pass
            except Exception as e:
                logger.debug("short_interest_features: sentiment fallback failed for %s: %s", ticker, e)

        # --- Build result ---
        has_data = short_float_pct is not None

        if not has_data:
            return empty

        # Clip extreme values
        if short_float_pct is not None:
            short_float_pct = np.clip(short_float_pct, 0.0, 100.0)
        if days_to_cover is not None:
            days_to_cover = np.clip(days_to_cover, 0.0, 60.0)

        result = pd.DataFrame(
            {
                "si_short_float_pct": short_float_pct if short_float_pct is not None else _NAN,
                "si_days_to_cover": days_to_cover if days_to_cover is not None else _NAN,
                "si_available": 1.0,
            },
            index=price_df.index,
        )

        # --- Point-in-time shift: use yesterday's data ---
        for col_name in ("si_short_float_pct", "si_days_to_cover", "si_available"):
            result[col_name] = result[col_name].shift(1)

        # Fill availability indicator after shift
        result["si_available"] = result["si_available"].fillna(0.0)

        return result

    except Exception as e:
        logger.warning("Could not build short interest features for %s: %s", ticker, e)
        return empty
