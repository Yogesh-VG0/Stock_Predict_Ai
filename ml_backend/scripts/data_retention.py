"""
Data Retention Policy — Automatic Cleanup of Old MongoDB Data

Removes documents older than the configured retention period from each
collection.  Designed to run as a GitHub Actions step after the daily
prediction pipeline.

Collections and their retention periods:
  - sentiment:                 12 months
  - stock_predictions:         12 months
  - prediction_explanations:   12 months
  - feature_importance:        12 months
  - insider_transactions:      18 months  (longer for trend analysis)
  - historical_data:           24 months  (needed for backtesting)
  - macro_data_raw / macro_data: never   (small, cumulative)
  - model_evaluations:         12 months
  - drift_reports:             12 months
  - short_interest_data:       12 months
  - economic_events:           12 months
  - sec_filings:               18 months
  - finnhub_basic_financials:  6 months   (refreshed daily, stale quickly)
  - finnhub_recommendation_trends: 12 months
  - finnhub_company_peers:     6 months
  - alpha_vantage_data:        12 months

Usage:
    python -m ml_backend.scripts.data_retention
    python -m ml_backend.scripts.data_retention --dry-run
    python -m ml_backend.scripts.data_retention --retention-months 6
"""

import argparse
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Collection → (retention_months, timestamp_field)
RETENTION_CONFIG: Dict[str, Tuple[int, str]] = {
    "sentiment":                     (12, "timestamp"),
    "stock_predictions":             (12, "timestamp"),
    "prediction_explanations":       (12, "timestamp"),
    "feature_importance":            (12, "timestamp"),
    "insider_transactions":          (18, "filingDate"),
    "model_evaluations":             (12, "timestamp"),
    "drift_reports":                 (12, "timestamp"),
    "short_interest_data":           (12, "fetched_at"),
    "economic_events":               (12, "timestamp"),
    "sec_filings":                   (18, "timestamp"),
    "finnhub_basic_financials":      (6,  "fetched_at"),
    "finnhub_recommendation_trends": (12, "fetched_at"),
    "finnhub_company_peers":         (6,  "fetched_at"),
    "alpha_vantage_data":            (12, "timestamp"),
}

# Collections that should NEVER be pruned (small, cumulative, or critical)
SKIP_COLLECTIONS = {"macro_data_raw", "macro_data", "historical_data"}


def run_retention(
    dry_run: bool = False,
    retention_override: int = 0,
) -> Dict[str, int]:
    """Delete old documents from MongoDB collections per retention policy.

    Args:
        dry_run: If True, only count documents to delete without actually deleting.
        retention_override: If > 0, override all retention periods with this value (months).

    Returns:
        Dict mapping collection name → number of deleted (or would-be-deleted) documents.
    """
    from pymongo import MongoClient

    uri = os.getenv("MONGODB_URI")
    if not uri:
        logger.error("MONGODB_URI not set — aborting retention cleanup")
        return {}

    client = MongoClient(uri, serverSelectionTimeoutMS=10000, connectTimeoutMS=10000)
    client.admin.command("ping")
    db = client["stock_predictor"]

    results: Dict[str, int] = {}
    now = datetime.utcnow()

    for coll_name, (default_months, ts_field) in RETENTION_CONFIG.items():
        months = retention_override if retention_override > 0 else default_months
        cutoff = now - timedelta(days=months * 30)

        try:
            coll = db[coll_name]

            # Build query: match documents with timestamp before cutoff
            # Handle both datetime objects and string dates
            query = {"$or": [
                {ts_field: {"$lt": cutoff}},
                {ts_field: {"$lt": cutoff.strftime("%Y-%m-%d")}},
            ]}

            count = coll.count_documents(query)

            if count == 0:
                logger.info("  %s: no documents older than %d months — skipping", coll_name, months)
                results[coll_name] = 0
                continue

            if dry_run:
                logger.info("  %s: [DRY RUN] would delete %d documents (older than %s)",
                            coll_name, count, cutoff.strftime("%Y-%m-%d"))
                results[coll_name] = count
            else:
                result = coll.delete_many(query)
                deleted = result.deleted_count
                logger.info("  %s: deleted %d documents (older than %s)",
                            coll_name, deleted, cutoff.strftime("%Y-%m-%d"))
                results[coll_name] = deleted

        except Exception as e:
            logger.warning("  %s: retention cleanup failed — %s", coll_name, e)
            results[coll_name] = -1

    client.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="MongoDB Data Retention Cleanup")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count documents to delete without actually deleting")
    parser.add_argument("--retention-months", type=int, default=0,
                        help="Override all retention periods with this value (months)")
    args = parser.parse_args()

    mode = "DRY RUN" if args.dry_run else "LIVE"
    logger.info("Starting data retention cleanup (%s)", mode)

    results = run_retention(
        dry_run=args.dry_run,
        retention_override=args.retention_months,
    )

    print(f"\n{'='*50}")
    print(f"Data Retention Cleanup Complete ({mode})")
    print(f"{'='*50}")
    total = 0
    for coll, count in sorted(results.items()):
        status = "would delete" if args.dry_run else "deleted"
        if count < 0:
            print(f"  {coll}: ERROR")
        elif count == 0:
            print(f"  {coll}: nothing to clean")
        else:
            print(f"  {coll}: {status} {count} documents")
            total += count
    print(f"\nTotal: {total} documents {'would be ' if args.dry_run else ''}cleaned up")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
