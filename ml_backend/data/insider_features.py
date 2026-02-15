"""
Insider-transaction feature adapter — converts raw insider-transaction
documents in MongoDB into leakage-proof rolling features aligned to the
price DataFrame.

Every feature for day t uses ONLY insider filings with filingDate strictly
before day t.  The shift(1) at the end guarantees point-in-time safety.

Features produced (11 columns):
 - insider_net_shares_30d   : buys - sells (share count) over 30 calendar days
 - insider_net_shares_90d   : buys - sells over 90 calendar days
 - insider_buy_count_30d    : number of purchase transactions in 30d
 - insider_sell_count_30d   : number of sale transactions in 30d
 - insider_buy_ratio_30d    : buys / (buys + sells + 1e-6) over 30d
 - insider_buy_value_30d    : sum(change * price) for buys over 30d
 - insider_sell_value_30d   : sum(|change * price|) for sells over 30d
 - insider_net_value_30d    : buy_value - sell_value over 30d
 - insider_activity_z_90d   : z-score of transaction count in 90d vs trailing 360d
 - insider_net_value_z_90d  : z-score of net_value in 90d vs trailing 360d
 - insider_cluster_buying   : 1 if >=3 buys and 0 sells in 30d, else 0

Data source:
  MongoDB collection ``insider_transactions``
  Document schema: {symbol, name, share, change, filingDate, transactionDate,
                    transactionCode, transactionPrice}

PIT SHIFT POLICY: Each adapter (macro, VIX, sector, sentiment, insider)
shifts internally.  There is NO global shift in prepare_features().
Do not add a second shift outside this module.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Transaction codes classified as BUY vs SELL
_BUY_CODES = frozenset({"P", "M", "A"})   # Purchase, Multiple, Award/Acquisition
_SELL_CODES = frozenset({"S", "D"})        # Sale, Disposition


def make_insider_features(
    price_df: pd.DataFrame,
    ticker: str,
    mongo_client,
    *,
    lookback_extra_days: int = 400,
) -> pd.DataFrame:
    """Build insider-transaction features aligned to *price_df* dates.

    Parameters
    ----------
    price_df : DataFrame
        Must have a ``date`` column (datetime).  Features are aligned to these
        dates via vectorized rolling windows.
    ticker : str
        Stock ticker symbol.
    mongo_client : MongoDBClient
        Must expose ``db['insider_transactions']``.
    lookback_extra_days : int
        Extra days before the earliest price date to fetch insider records.
        Needed for the 360-day z-score window to be warm.

    Returns
    -------
    DataFrame with the same index as *price_df* and 11 insider feature
    columns.  Missing values are filled with neutral defaults.
    """
    _defaults = {
        "insider_net_shares_30d":  0.0,
        "insider_net_shares_90d":  0.0,
        "insider_buy_count_30d":   0.0,
        "insider_sell_count_30d":  0.0,
        "insider_buy_ratio_30d":   0.5,   # neutral: equal buys/sells
        "insider_buy_value_30d":   0.0,
        "insider_sell_value_30d":  0.0,
        "insider_net_value_30d":   0.0,
        "insider_activity_z_90d":  0.0,
        "insider_net_value_z_90d": 0.0,
        "insider_cluster_buying":  0.0,
    }
    empty = pd.DataFrame(
        {col: [val] * len(price_df) for col, val in _defaults.items()},
        index=price_df.index,
    )

    if mongo_client is None or not hasattr(mongo_client, "db"):
        return empty

    try:
        coll = mongo_client.db["insider_transactions"]
        if coll is None:
            return empty

        dates = pd.to_datetime(price_df["date"])
        earliest = pd.Timestamp(dates.min()) - timedelta(days=lookback_extra_days)

        # Fetch all insider records for this ticker since earliest
        cursor = coll.find(
            {"symbol": ticker, "filingDate": {"$gte": earliest.strftime("%Y-%m-%d")}},
            {"filingDate": 1, "transactionDate": 1, "transactionCode": 1,
             "change": 1, "share": 1, "transactionPrice": 1, "_id": 0},
        )
        records = list(cursor)
        if not records:
            return empty

        # Build a DataFrame of insider transactions
        txn = pd.DataFrame(records)

        # Parse filing date (this is the date information becomes public)
        txn["filing_dt"] = pd.to_datetime(txn["filingDate"], errors="coerce")
        txn = txn.dropna(subset=["filing_dt"])
        if txn.empty:
            return empty

        # Classify buy vs sell
        txn["code"] = txn["transactionCode"].str.upper().fillna("")
        txn["is_buy"] = txn["code"].isin(_BUY_CODES) | (txn.get("change", pd.Series(0)).fillna(0) > 0)
        txn["is_sell"] = txn["code"].isin(_SELL_CODES) | (txn.get("change", pd.Series(0)).fillna(0) < 0)

        # Compute value per transaction
        txn["change_val"] = pd.to_numeric(txn.get("change", 0), errors="coerce").fillna(0)
        txn["price_val"] = pd.to_numeric(txn.get("transactionPrice", 0), errors="coerce").fillna(0).clip(lower=0)
        txn["value"] = txn["change_val"].abs() * txn["price_val"]

        # Signed shares: positive for buys, negative for sells
        txn["signed_shares"] = np.where(txn["is_buy"], txn["change_val"].abs(), -txn["change_val"].abs())
        # Signed value
        txn["signed_value"] = np.where(txn["is_buy"], txn["value"], -txn["value"])

        txn = txn.sort_values("filing_dt").reset_index(drop=True)

        # --- Build daily aggregates keyed by filing date ---
        txn["filing_date_norm"] = txn["filing_dt"].dt.normalize()
        daily = txn.groupby("filing_date_norm").agg(
            net_shares=("signed_shares", "sum"),
            buy_count=("is_buy", "sum"),
            sell_count=("is_sell", "sum"),
            buy_value=("value", lambda x: x[txn.loc[x.index, "is_buy"]].sum() if len(x) > 0 else 0),
            sell_value=("value", lambda x: x[txn.loc[x.index, "is_sell"]].sum() if len(x) > 0 else 0),
            net_value=("signed_value", "sum"),
            txn_count=("signed_shares", "count"),
        ).reset_index()

        # Re-aggregate buy_value / sell_value directly (the lambda above can be fragile)
        buy_daily = txn[txn["is_buy"]].groupby("filing_date_norm")["value"].sum().rename("buy_value_agg")
        sell_daily = txn[txn["is_sell"]].groupby("filing_date_norm")["value"].sum().rename("sell_value_agg")
        daily = daily.merge(buy_daily, on="filing_date_norm", how="left")
        daily = daily.merge(sell_daily, on="filing_date_norm", how="left")
        daily["buy_value"] = daily["buy_value_agg"].fillna(0)
        daily["sell_value"] = daily["sell_value_agg"].fillna(0)
        daily = daily.drop(columns=["buy_value_agg", "sell_value_agg"], errors="ignore")

        daily = daily.rename(columns={"filing_date_norm": "date"})
        daily = daily.sort_values("date").reset_index(drop=True)

        # --- Reindex to full calendar date range so rolling works cleanly ---
        full_range = pd.date_range(
            start=daily["date"].min(),
            end=pd.Timestamp(dates.max()) + timedelta(days=1),
            freq="D",
        )
        daily = daily.set_index("date").reindex(full_range).fillna(0)
        daily.index.name = "date"

        # --- Rolling window features ---
        feat = pd.DataFrame(index=daily.index)

        feat["insider_net_shares_30d"] = daily["net_shares"].rolling(30, min_periods=1).sum()
        feat["insider_net_shares_90d"] = daily["net_shares"].rolling(90, min_periods=1).sum()

        feat["insider_buy_count_30d"] = daily["buy_count"].rolling(30, min_periods=1).sum()
        feat["insider_sell_count_30d"] = daily["sell_count"].rolling(30, min_periods=1).sum()

        bc30 = feat["insider_buy_count_30d"]
        sc30 = feat["insider_sell_count_30d"]
        feat["insider_buy_ratio_30d"] = bc30 / (bc30 + sc30 + 1e-6)

        feat["insider_buy_value_30d"] = daily["buy_value"].rolling(30, min_periods=1).sum()
        feat["insider_sell_value_30d"] = daily["sell_value"].rolling(30, min_periods=1).sum()
        feat["insider_net_value_30d"] = daily["net_value"].rolling(30, min_periods=1).sum()

        # Z-score: 90d activity vs trailing 360d
        txn_count_90d = daily["txn_count"].rolling(90, min_periods=1).sum()
        txn_count_360d_mean = daily["txn_count"].rolling(360, min_periods=30).sum().rolling(1).mean()
        txn_count_360d_std = daily["txn_count"].rolling(360, min_periods=30).sum().rolling(1).std()
        # Proper rolling z-score: compare 90d sum to rolling 90d-sum mean/std over 360d
        txn_sum_90d = daily["txn_count"].rolling(90, min_periods=1).sum()
        txn_sum_90d_mean = txn_sum_90d.rolling(360, min_periods=30).mean()
        txn_sum_90d_std = txn_sum_90d.rolling(360, min_periods=30).std().replace(0, np.nan)
        feat["insider_activity_z_90d"] = ((txn_sum_90d - txn_sum_90d_mean) / txn_sum_90d_std).fillna(0).clip(-5, 5)

        nv_90d = daily["net_value"].rolling(90, min_periods=1).sum()
        nv_90d_mean = nv_90d.rolling(360, min_periods=30).mean()
        nv_90d_std = nv_90d.rolling(360, min_periods=30).std().replace(0, np.nan)
        feat["insider_net_value_z_90d"] = ((nv_90d - nv_90d_mean) / nv_90d_std).fillna(0).clip(-5, 5)

        # Cluster buying: >=3 buys AND 0 sells in 30d
        feat["insider_cluster_buying"] = (
            (feat["insider_buy_count_30d"] >= 3) & (feat["insider_sell_count_30d"] == 0)
        ).astype(float)

        # --- Align to price dates ---
        feat = feat.reset_index().rename(columns={"index": "date"})
        feat["date"] = pd.to_datetime(feat["date"]).dt.normalize()

        price_dates = pd.DataFrame(
            {"date": pd.to_datetime(price_df["date"]).dt.normalize()},
            index=price_df.index,
        )
        merged = price_dates.merge(feat, on="date", how="left")
        merged.index = price_df.index

        feature_cols = list(_defaults.keys())

        # --- Point-in-time shift: use yesterday's insider data ---
        for col in feature_cols:
            if col in merged.columns:
                merged[col] = merged[col].shift(1)

        # Fill missing with neutral defaults
        for col, default in _defaults.items():
            if col not in merged.columns:
                merged[col] = default
            else:
                merged[col] = merged[col].fillna(default)

        return merged[feature_cols]

    except Exception as e:
        logger.warning("Could not build insider features for %s: %s", ticker, e)
        return empty
