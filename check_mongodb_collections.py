#!/usr/bin/env python3
"""
Check MongoDB Collections and Data Structure

This script checks all collections in MongoDB to understand 
the actual data structure and find historical stock data.
"""

import sys
import pandas as pd
from datetime import datetime

# Add ML backend to path
sys.path.append('ml_backend')

from ml_backend.utils.mongodb import MongoDBClient
from ml_backend.config.constants import MONGODB_URI

def check_all_collections():
    """Check all collections in MongoDB."""
    print("🔍 Checking all MongoDB collections...")
    
    mongo_client = MongoDBClient(MONGODB_URI)
    db = mongo_client.db
    
    # Get all collection names
    collections = db.list_collection_names()
    print(f"📊 Found {len(collections)} collections:")
    
    for collection_name in collections:
        collection = db[collection_name]
        count = collection.count_documents({})
        print(f"   {collection_name}: {count} documents")
        
        # Show sample document if collection has data
        if count > 0:
            sample = collection.find_one()
            print(f"      Sample keys: {list(sample.keys())}")
            
            # Check if it looks like stock data
            if any(key in sample for key in ['ticker', 'symbol', 'Close', 'price']):
                print(f"      ✅ Looks like stock data!")
                
                # Show more details for potential stock data
                if 'ticker' in sample:
                    tickers = collection.distinct('ticker')
                    print(f"      Tickers: {tickers[:5]}{'...' if len(tickers) > 5 else ''}")
                
                if 'date' in sample:
                    # Get date range
                    pipeline = [
                        {"$group": {
                            "_id": None,
                            "min_date": {"$min": "$date"},
                            "max_date": {"$max": "$date"}
                        }}
                    ]
                    result = list(collection.aggregate(pipeline))
                    if result:
                        min_date = result[0]['min_date']
                        max_date = result[0]['max_date']
                        print(f"      Date range: {min_date} to {max_date}")
        print()

def check_specific_ticker_data(ticker: str = "AAPL"):
    """Check for specific ticker data across all collections."""
    print(f"🎯 Searching for {ticker} data across all collections...")
    
    mongo_client = MongoDBClient(MONGODB_URI)
    db = mongo_client.db
    
    collections = db.list_collection_names()
    found_data = False
    
    for collection_name in collections:
        collection = db[collection_name]
        
        # Try different field names for ticker
        queries = [
            {'ticker': ticker},
            {'symbol': ticker},
            {'stock': ticker},
            {'ticker': ticker.upper()},
            {'symbol': ticker.upper()}
        ]
        
        for query in queries:
            try:
                count = collection.count_documents(query)
                if count > 0:
                    print(f"✅ Found {count} {ticker} records in '{collection_name}'")
                    
                    # Get sample data
                    sample = collection.find(query).limit(3)
                    for i, doc in enumerate(sample):
                        print(f"   Sample {i+1}: {dict(list(doc.items())[:5])}...")
                    
                    # Check date range if date field exists
                    sample_doc = collection.find_one(query)
                    if 'date' in sample_doc:
                        pipeline = [
                            {"$match": query},
                            {"$group": {
                                "_id": None,
                                "min_date": {"$min": "$date"},
                                "max_date": {"$max": "$date"},
                                "count": {"$sum": 1}
                            }}
                        ]
                        result = list(collection.aggregate(pipeline))
                        if result:
                            stats = result[0]
                            print(f"   Date range: {stats['min_date']} to {stats['max_date']}")
                            print(f"   Total records: {stats['count']}")
                    
                    found_data = True
                    print()
                    break
            except Exception as e:
                continue
    
    if not found_data:
        print(f"❌ No {ticker} data found in any collection")

def main():
    """Main function."""
    print("🔍 MONGODB COLLECTIONS AND DATA CHECK")
    print("="*50)
    
    # Check all collections
    check_all_collections()
    
    # Check specifically for AAPL data
    check_specific_ticker_data("AAPL")
    
    print("✅ Collection check complete!")
    print("\n💡 Look for collections with stock data above")
    print("💡 Update the backtesting script to use the correct collection")

if __name__ == "__main__":
    main() 