"""
Fundamental feature adapter — converts Finnhub basic financials stored in
MongoDB into leakage-proof features aligned to the price DataFrame.

Fundamentals are slow-moving (updated quarterly/annually), so the latest
available value is broadcast across all rows.  shift(1) ensures PIT safety.

Features produced (5 columns):
 - fund_pe_ratio       : P/E ratio TTM, clipped to [-50, 200] (NaN if missing)
 - fund_pb_ratio       : Price-to-Book ratio, clipped to [0, 50] (NaN if missing)
 - fund_dividend_yield : Indicated annual dividend yield (NaN if missing)
 - fund_roe            : Return on equity TTM, clipped to [-100, 200] (NaN if missing)
 - fund_beta           : Stock beta vs market (NaN if missing)

Data source:
  MongoDB collection ``finnhub_basic_financials``
  Document schema: {ticker, data: {metric: {peBasicExclExtraTTM, pbAnnual,
                    dividendYieldIndicatedAnnual, roeTTM, beta, ...}}, fetched_at}

PIT SHIFT POLICY: Each adapter shifts internally.  There is NO global shift
in prepare_features().  Do not add a second shift outside this module.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def make_fundamental_features(
    price_df: pd.DataFrame,
    ticker: str,
    mongo_client,
) -> pd.DataFrame:
    """Build fundamental features aligned to *price_df* dates.

    Parameters
    ----------
    price_df : DataFrame
        Must have a ``date`` column (datetime).  Fundamental values are
        broadcast to every row (slow-moving data).
    ticker : str
        Stock ticker symbol.
    mongo_client : MongoDBClient
        Must expose ``db`` attribute for direct collection access.

    Returns
    -------
    DataFrame with the same index as *price_df* and 5 fundamental columns.
    All columns use NaN when missing (LightGBM treats as unknown).
    """
    _NAN = float("nan")
    _defaults = {
        "fund_pe_ratio": _NAN,
        "fund_pb_ratio": _NAN,
        "fund_dividend_yield": _NAN,
        "fund_roe": _NAN,
        "fund_beta": _NAN,
    }
    empty = pd.DataFrame(
        {col: [val] * len(price_df) for col, val in _defaults.items()},
        index=price_df.index,
    )

    if mongo_client is None or not hasattr(mongo_client, "db"):
        return empty

    try:
        col = mongo_client.db.get_collection("finnhub_basic_financials")
        if col is None:
            return empty

        # Fetch the latest financials document for this ticker
        doc = col.find_one(
            {"ticker": ticker.upper()},
            sort=[("fetched_at", -1)],
        )
        if not doc:
            # Try without fetched_at sort
            doc = col.find_one({"ticker": ticker.upper()})
        if not doc or not isinstance(doc.get("data"), dict):
            return empty

        # Extract metrics — handle both nested and flat schemas
        data = doc["data"]
        metric = data.get("metric", data)  # Finnhub wraps in {metric: {...}}
        if not isinstance(metric, dict):
            return empty

        # --- Extract and clip values ---
        def _safe_float(val, clip_min=None, clip_max=None):
            """Convert to float, clip, return NaN on failure."""
            if val is None:
                return _NAN
            try:
                v = float(val)
                if np.isnan(v) or np.isinf(v):
                    return _NAN
                if clip_min is not None:
                    v = max(v, clip_min)
                if clip_max is not None:
                    v = min(v, clip_max)
                return v
            except (ValueError, TypeError):
                return _NAN

        pe = _safe_float(
            metric.get("peBasicExclExtraTTM") or metric.get("peTTM"),
            clip_min=-50, clip_max=200,
        )
        pb = _safe_float(
            metric.get("pbAnnual") or metric.get("pbQuarterly"),
            clip_min=0, clip_max=50,
        )
        div_yield = _safe_float(
            metric.get("dividendYieldIndicatedAnnual"),
            clip_min=0, clip_max=30,
        )
        roe = _safe_float(
            metric.get("roeTTM"),
            clip_min=-100, clip_max=200,
        )
        beta = _safe_float(
            metric.get("beta"),
            clip_min=-3, clip_max=5,
        )

        # Build result — broadcast single values across all rows
        result = pd.DataFrame(
            {
                "fund_pe_ratio": pe,
                "fund_pb_ratio": pb,
                "fund_dividend_yield": div_yield,
                "fund_roe": roe,
                "fund_beta": beta,
            },
            index=price_df.index,
        )

        # --- Point-in-time shift: use yesterday's data ---
        # For fundamentals this is overly conservative (they barely change day-to-day)
        # but consistency with the rest of the pipeline is more important.
        for col_name in _defaults.keys():
            result[col_name] = result[col_name].shift(1)

        return result

    except Exception as e:
        logger.warning("Could not build fundamental features for %s: %s", ticker, e)
        return empty
