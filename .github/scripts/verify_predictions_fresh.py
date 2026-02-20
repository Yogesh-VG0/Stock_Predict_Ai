"""Verify stock_predictions has fresh docs (last 3h). Used by run-daily-pipeline-local.ps1."""
import os
from datetime import datetime, timedelta
from pymongo import MongoClient

uri = os.getenv("MONGODB_URI")
if not uri:
    print("Missing MONGODB_URI — skip verify")
    raise SystemExit(0)
client = MongoClient(uri, serverSelectionTimeoutMS=8000)
client.admin.command("ping")
db = client["stock_predictor"]
col = db["stock_predictions"]
now = datetime.utcnow()
cutoff = now - timedelta(hours=3)
canary = ["AAPL", "AMZN", "JPM", "XOM", "WMT", "CAT", "CMCSA", "PLTR"]
missing = []
stale = []
for t in canary:
    for w in ["next_day", "7_day", "30_day"]:
        doc = col.find_one({"ticker": t, "window": w}, sort=[("timestamp", -1)])
        if not doc:
            missing.append((t, w))
            continue
        ts = doc.get("timestamp")
        if not ts or ts < cutoff:
            stale.append((t, w, ts))
total = col.count_documents({"timestamp": {"$gte": cutoff}})
print("Fresh docs in last 3h:", total)
if missing:
    print("Missing:", missing)
if stale:
    print("Stale:", stale)
assert not missing, "Missing predictions: %s" % missing
assert not stale, "Stale predictions: %s" % stale
assert total >= 200, "Expected >= 200 fresh docs, got %s" % total
print("Freshness verified")
