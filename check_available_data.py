#!/usr/bin/env python3
"""
Check Available Historical Data in MongoDB

This script checks what stock data is available in MongoDB 
so we can set appropriate holdout periods for backtesting.
"""

import sys
import pandas as pd
from datetime import datetime, timedelta

# Add ML backend to path
sys.path.append('ml_backend')

from ml_backend.utils.mongodb import MongoDBClient
from ml_backend.config.constants import MONGODB_URI

def check_available_data(ticker: str = "AAPL"):
    """Check what historical data is available for a ticker."""
    print(f"🔍 Checking available data for {ticker}...")
    
    mongo_client = MongoDBClient(MONGODB_URI)
    collection = mongo_client.db.stock_data
    
    # Get all data for the ticker
    query = {'ticker': ticker}
    cursor = collection.find(query).sort('date', 1)
    data = list(cursor)
    
    if not data:
        print(f"❌ No data found for {ticker}")
        return None
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Get date range
    start_date = df['date'].min()
    end_date = df['date'].max()
    total_days = len(df)
    
    print(f"📊 Data Summary for {ticker}:")
    print(f"   Start Date: {start_date.strftime('%Y-%m-%d')}")
    print(f"   End Date: {end_date.strftime('%Y-%m-%d')}")
    print(f"   Total Records: {total_days}")
    print(f"   Date Range: {(end_date - start_date).days} days")
    
    # Check recent data availability
    today = datetime.now()
    recent_periods = [30, 60, 90, 120, 180, 365]
    
    print(f"\n📅 Data Availability by Period:")
    for days in recent_periods:
        cutoff_date = today - timedelta(days=days)
        recent_data = df[df['date'] >= cutoff_date]
        print(f"   Last {days} days: {len(recent_data)} records")
    
    # Suggest good holdout periods
    print(f"\n💡 Suggested Holdout Periods:")
    
    # Find periods with sufficient data
    for months_back in [1, 2, 3, 4, 6]:
        holdout_end = today - timedelta(days=months_back * 30)
        holdout_start = holdout_end - timedelta(days=60)  # 60 days for testing
        
        holdout_data = df[(df['date'] >= holdout_start) & (df['date'] <= holdout_end)]
        
        if len(holdout_data) >= 30:
            print(f"   ✅ {months_back} months back: {len(holdout_data)} records available")
            print(f"      Period: {holdout_start.strftime('%Y-%m-%d')} to {holdout_end.strftime('%Y-%m-%d')}")
        else:
            print(f"   ❌ {months_back} months back: Only {len(holdout_data)} records available")
    
    # Show sample of recent data
    print(f"\n📈 Recent Data Sample:")
    recent_sample = df.tail(5)[['date', 'Close']].copy()
    for _, row in recent_sample.iterrows():
        print(f"   {row['date'].strftime('%Y-%m-%d')}: ${row['Close']:.2f}")
    
    return df

def main():
    """Main function."""
    print("🔍 CHECKING AVAILABLE HISTORICAL DATA")
    print("="*50)
    
    # Check AAPL data
    df = check_available_data("AAPL")
    
    if df is not None:
        print(f"\n✅ Data check complete!")
        print(f"💡 Use the suggested holdout periods for proper backtesting")
    else:
        print(f"\n❌ No data found. Please ensure historical data is loaded in MongoDB.")

if __name__ == "__main__":
    main() 