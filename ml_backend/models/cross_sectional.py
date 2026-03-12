"""
Cross-sectional ranking layer for stock predictions.

v10.0: Instead of evaluating each ticker in isolation, rank all tickers by
predicted alpha and select the top quintile.  Even a pooled model with low
absolute correlation (e.g. 0.019) becomes useful when it only needs to RANK
tickers correctly — ranking is strictly easier than level prediction.

Reference: Jegadeesh & Titman (1993), Fama-French cross-sectional factors.
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CrossSectionalRanker:
    """Rank tickers by predicted alpha and select top quintile for trading."""

    def __init__(self, top_pct: float = 0.20, min_tickers: int = 5,
                 confidence_boost: float = 0.10):
        self.top_pct = top_pct
        self.min_tickers = min_tickers
        self.confidence_boost = confidence_boost

    def rank_predictions(
        self,
        all_predictions: Dict[str, Dict],
        horizon: str,
    ) -> List[Tuple[str, int, float]]:
        """Rank tickers by predicted alpha for a given horizon.

        Args:
            all_predictions: {ticker: predict_all_windows() result}
            horizon: "next_day", "7_day", or "30_day"

        Returns:
            List of (ticker, rank, predicted_alpha) sorted by rank (1 = best).
        """
        scored = []
        for ticker, preds in all_predictions.items():
            if not isinstance(preds, dict):
                continue
            hz_pred = preds.get(horizon)
            if hz_pred is None or not isinstance(hz_pred, dict):
                continue
            # Use raw prediction (alpha) for ranking — before any trade filters
            alpha = hz_pred.get("prediction", hz_pred.get("alpha", 0.0))
            if alpha is None:
                alpha = 0.0
            scored.append((ticker, float(alpha)))

        if len(scored) < self.min_tickers:
            return []

        # Sort descending by predicted alpha
        scored.sort(key=lambda x: x[1], reverse=True)

        # Assign ranks (1-indexed)
        ranked = []
        for rank_idx, (ticker, alpha) in enumerate(scored):
            ranked.append((ticker, rank_idx + 1, alpha))

        return ranked

    def apply_ranking(
        self,
        all_predictions: Dict[str, Dict],
        horizon: str,
    ) -> Dict[str, Dict]:
        """Apply cross-sectional ranking to modify predictions.

        Adds ranking metadata and adjusts trade_recommended based on rank.
        Top quintile tickers get a confidence boost; bottom quintile get
        trade_recommended set to False regardless of per-ticker signal.

        Args:
            all_predictions: {ticker: predict_all_windows() result} — MODIFIED in place
            horizon: horizon to rank on

        Returns:
            The same dict with ranking metadata added.
        """
        ranked = self.rank_predictions(all_predictions, horizon)
        if not ranked:
            return all_predictions

        n_total = len(ranked)
        n_top = max(1, int(n_total * self.top_pct))

        top_tickers = set()
        bottom_tickers = set()
        rank_map = {}

        for ticker, rank, alpha in ranked:
            rank_map[ticker] = rank
            quintile = 1 + int((rank - 1) / max(1, n_total / 5))
            quintile = min(quintile, 5)

            if rank <= n_top:
                top_tickers.add(ticker)

            # Bottom quintile: suppress trades
            if quintile >= 5:
                bottom_tickers.add(ticker)

        # Apply ranking to predictions
        for ticker, preds in all_predictions.items():
            hz_pred = preds.get(horizon)
            if hz_pred is None or not isinstance(hz_pred, dict):
                continue

            rank = rank_map.get(ticker)
            if rank is None:
                continue

            quintile = 1 + int((rank - 1) / max(1, n_total / 5))
            quintile = min(quintile, 5)

            # Add ranking metadata
            hz_pred["cross_sectional_rank"] = rank
            hz_pred["rank_quintile"] = quintile
            hz_pred["rank_total_tickers"] = n_total

            # Top quintile: boost confidence
            if ticker in top_tickers:
                hz_pred["rank_signal"] = "top"
                old_conf = hz_pred.get("confidence", 0.0)
                hz_pred["confidence"] = min(
                    hz_pred.get("confidence", 0.0) + self.confidence_boost,
                    0.95,
                )

            # Bottom quintile: suppress trade recommendation
            elif ticker in bottom_tickers:
                hz_pred["rank_signal"] = "bottom"
                hz_pred["trade_recommended"] = False

            else:
                hz_pred["rank_signal"] = "middle"

        logger.info(
            "Cross-sectional ranking [%s]: %d tickers ranked, top %d, bottom %d suppressed",
            horizon, n_total, len(top_tickers), len(bottom_tickers),
        )

        return all_predictions

    def select_backtest_candidates(
        self,
        ticker_predictions: Dict[str, float],
        max_positions: int = 5,
    ) -> List[Tuple[str, float]]:
        """Select tickers for backtest based on cross-sectional ranking.

        Args:
            ticker_predictions: {ticker: predicted_alpha} for a single date
            max_positions: maximum number of positions to open

        Returns:
            List of (ticker, predicted_alpha) for the top-ranked candidates.
        """
        if len(ticker_predictions) < self.min_tickers:
            return []

        sorted_tickers = sorted(
            ticker_predictions.items(), key=lambda x: x[1], reverse=True
        )

        n_top = max(1, int(len(sorted_tickers) * self.top_pct))
        # Only select from top quintile, but cap at max_positions
        candidates = sorted_tickers[:min(n_top, max_positions)]

        # Filter: require positive predicted alpha (don't go long on negative alpha
        # even if it's the "best" in a down market)
        candidates = [(t, a) for t, a in candidates if a > 0]

        return candidates
