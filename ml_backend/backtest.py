"""
Lightweight backtest module for StockPredict strategy validation.

Simulates: buy when trade_recommended, hold for horizon, sell.
Compares portfolio returns vs buy-and-hold SPY.
Reports: total return, Sharpe ratio, max drawdown.

Note: Uses current model (trained on recent data). For true walk-forward
backtest, retrain at each step (computationally expensive). This simulation
gives a sense of strategy behavior with current model.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config.feature_config_v1 import TARGET_CONFIG, ROUND_TRIP_COST_BPS

logger = logging.getLogger(__name__)

# Default tickers for backtest (subset for speed)
DEFAULT_BACKTEST_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "V", "XOM", "JNJ"]


def run_backtest(
    predictor,
    historical_data: Dict[str, pd.DataFrame],
    spy_data: Optional[pd.DataFrame] = None,
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_positions: int = 5,
    horizon: str = "next_day",
) -> Dict:
    """
    Run backtest: simulate buy when trade_recommended, hold for horizon, sell.

    Args:
        predictor: StockPredictor instance with loaded models
        historical_data: Dict[ticker, DataFrame] with OHLCV
        spy_data: SPY DataFrame for benchmark (optional; computed from historical if missing)
        tickers: Tickers to simulate (default: DEFAULT_BACKTEST_TICKERS)
        start_date: Start of backtest (default: min date + 100 days)
        end_date: End of backtest (default: max date)
        max_positions: Max positions to hold at once
        horizon: Horizon key ("next_day", "7_day", "30_day")

    Returns:
        Dict with keys: total_return, sharpe_ratio, max_drawdown, n_trades,
        spy_return, spy_sharpe, trades_df
    """
    tickers = tickers or [t for t in DEFAULT_BACKTEST_TICKERS if t in historical_data]
    if not tickers:
        tickers = list(historical_data.keys())[:10]
    tickers = [t for t in tickers if t in historical_data and historical_data[t] is not None and not historical_data[t].empty]

    if not tickers:
        return {"error": "No valid tickers or historical data"}

    cfg = TARGET_CONFIG.get(horizon, TARGET_CONFIG["next_day"])
    hold_days = cfg["horizon"]
    round_trip_cost = ROUND_TRIP_COST_BPS / 10000

    # Build date index from all tickers
    all_dates = []
    for t in tickers:
        df = historical_data[t]
        if "date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])
        all_dates.extend(df["date"].tolist())
    if not all_dates:
        return {"error": "No dates in historical data"}
    all_dates = sorted(set(pd.to_datetime(all_dates)))
    min_date = min(all_dates)
    max_date = max(all_dates)

    warmup = 100
    start = pd.Timestamp(start_date) if start_date else min_date + timedelta(days=warmup)
    end = pd.Timestamp(end_date) if end_date else max_date

    # Trading dates
    trade_dates = [d for d in all_dates if start <= d <= end]
    if len(trade_dates) < hold_days + 5:
        return {"error": f"Too few trading dates: {len(trade_dates)}"}

    # Build SPY returns if available
    spy_returns = {}
    if spy_data is not None and not spy_data.empty:
        s = spy_data.copy()
        s["date"] = pd.to_datetime(s["date"])
        s = s.sort_values("date").set_index("date")
        if "Close" in s.columns:
            s["ret"] = np.log(s["Close"] / s["Close"].shift(1))
            for d in trade_dates:
                dt = pd.Timestamp(d)
                if dt in s.index:
                    spy_returns[d] = s.loc[dt, "ret"] if pd.notna(s.loc[dt, "ret"]) else 0.0
                else:
                    spy_returns[d] = 0.0

    # Portfolio state: ticker -> (entry_date, entry_price, weight)
    positions = {}
    prev_prices = {}  # ticker -> last known price (for daily mark-to-market)
    daily_returns = []
    trades_log = []

    def _get_price(ticker: str, as_of: pd.Timestamp) -> Optional[float]:
        df = historical_data.get(ticker)
        if df is None or df.empty:
            return None
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        mask = df["date"] <= as_of
        if not mask.any():
            return None
        row = df.loc[mask].iloc[-1]
        return float(row["Close"] if "Close" in row else row.get("Close_" + ticker, np.nan))


    
    # Pre-compute all predictions for all tickers
    # ticker -> DataFrame with columns [date, window, trade_recommended, normalized_return, current_price]
    all_preds_cache = {} 
    
    logger.info("Pre-computing predictions for backtest...")
    for ticker in tickers:
        try:
            df = historical_data[ticker]
            if df is None or df.empty or len(df) < 100:
                continue
            # We only need predictions for the simulation period
            # But feature eng requires lookback. Pass full df, filter output.
            preds_df = predictor.predict_batch(ticker, df)
            if not preds_df.empty:
                # Filter for trade_recommended only to save space/time lookup
                preds_df["date"] = pd.to_datetime(preds_df["date"])
                # We only care about rows where trade is recommended for the requested horizon
                mask = (preds_df["window"] == horizon) & (preds_df["trade_recommended"])
                relevant = preds_df[mask].copy()
                if not relevant.empty:
                    # Index by date for fast lookup
                    all_preds_cache[ticker] = relevant.set_index("date")
        except Exception as e:
            logger.warning(f"Batch predict failed for {ticker}: {e}")
            
    logger.info("Pre-computation complete.")

    for i, trade_date in enumerate(trade_dates):
        trade_date = pd.Timestamp(trade_date)
        day_return = 0.0

        # 1. Mark-to-market: daily return from open positions (equal weight)
        n_pos = len(positions)
        weight = 1.0 / n_pos if n_pos > 0 else 0.0
        if i > 0 and n_pos > 0:
            for ticker, (entry_date, entry_price, _) in list(positions.items()):
                prev_p = prev_prices.get(ticker, entry_price)
                curr_p = _get_price(ticker, trade_date)
                if curr_p is not None and prev_p is not None and prev_p > 0 and curr_p > 0:
                    day_return += weight * np.log(curr_p / prev_p)
                if curr_p is not None:
                    prev_prices[ticker] = curr_p

        # 2. Sell positions that have reached hold_days
        to_close = []
        for ticker, (entry_date, entry_price, weight) in list(positions.items()):
            days_held = (trade_date - pd.Timestamp(entry_date)).days
            if days_held >= hold_days:
                to_close.append(ticker)
        n_before_close = len(positions)
        for ticker in to_close:
            entry_date, entry_price, _ = positions.pop(ticker)
            prev_prices.pop(ticker, None)
            exit_price = _get_price(ticker, trade_date)
            if exit_price is None or exit_price <= 0:
                continue
            ret = np.log(exit_price / entry_price) - round_trip_cost
            w = 1.0 / n_before_close if n_before_close > 0 else 0.0
            day_return += w * ret
            trades_log.append({
                "ticker": ticker,
                "entry_date": entry_date,
                "exit_date": trade_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return": ret,
                "weight": w,
            })

        # 3. Buy new positions if we have capacity
        n_open = len(positions)
        if n_open < max_positions:
            candidates = []
            for ticker in tickers:
                if ticker in positions:
                    continue
                # Fast lookup from pre-computed cache
                if ticker in all_preds_cache:
                    p_df = all_preds_cache[ticker]
                    if trade_date in p_df.index:
                        # Handle duplicate index (rare but possible if multiple windows collide or data issue)
                        row = p_df.loc[trade_date]
                        if isinstance(row, pd.DataFrame):
                            row = row.iloc[0] # Take first if multiple
                        
                        candidates.append((ticker, float(row["normalized_return"]), float(row["current_price"])))
            
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                to_buy = candidates[: max_positions - n_open]
                for ticker, _, price in to_buy:
                    positions[ticker] = (trade_date, price, 0.0)  # weight computed daily
                    prev_prices[ticker] = price

        daily_returns.append(day_return)

    portfolio_value = 1.0
    for r in daily_returns:
        portfolio_value *= np.exp(r)

    total_return = portfolio_value - 1.0
    daily_arr = np.array(daily_returns)
    sharpe = 0.0
    if len(daily_arr) > 1 and np.std(daily_arr) > 1e-10:
        sharpe = np.mean(daily_arr) / np.std(daily_arr) * np.sqrt(252)

    # Max drawdown
    cum = np.exp(np.cumsum(daily_arr))
    running_max = np.maximum.accumulate(cum)
    drawdowns = (cum - running_max) / running_max
    max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    # SPY benchmark
    spy_return = 0.0
    spy_sharpe = 0.0
    if spy_returns:
        spy_daily = [spy_returns.get(d, 0.0) for d in trade_dates[: len(daily_returns)]]
        if len(spy_daily) > len(daily_returns):
            spy_daily = spy_daily[: len(daily_returns)]
        elif len(spy_daily) < len(daily_returns):
            spy_daily = spy_daily + [0.0] * (len(daily_returns) - len(spy_daily))
        spy_arr = np.array(spy_daily)
        spy_return = float(np.exp(np.sum(spy_arr)) - 1.0)
        if len(spy_arr) > 1 and np.std(spy_arr) > 1e-10:
            spy_sharpe = float(np.mean(spy_arr) / np.std(spy_arr) * np.sqrt(252))

    return {
        "total_return": float(total_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "n_trades": len(trades_log),
        "spy_return": float(spy_return),
        "spy_sharpe": float(spy_sharpe),
        "trades_df": pd.DataFrame(trades_log) if trades_log else pd.DataFrame(),
        "start_date": str(trade_dates[0]) if trade_dates else None,
        "end_date": str(trade_dates[-1]) if trade_dates else None,
    }
