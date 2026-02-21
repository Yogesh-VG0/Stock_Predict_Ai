"""
Optimized Sentiment Pipeline Configuration for 24/7 Cloud Operation
"""

import os
from datetime import datetime, time

# Ticker prioritization for efficient processing
PRIORITY_TICKERS = {
    'high': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA'],  # Always process
    'medium': ['META', 'NFLX', 'DIS', 'AMD', 'CRM', 'ORCL'],    # Process during market hours
    'low': ['BABA', 'ADBE', 'INTC', 'PYPL', 'UBER', 'SNOW']     # Process during low activity
}

# Time-based processing optimization
MARKET_HOURS = {
    'pre_market': time(4, 0),    # 4:00 AM EST
    'market_open': time(9, 30),  # 9:30 AM EST  
    'market_close': time(16, 0), # 4:00 PM EST
    'after_hours': time(20, 0)   # 8:00 PM EST
}

# Rate limiting and timeout settings
RATE_LIMITS = {
    'requests_per_minute': 30,
    'timeout_seconds': 10,
    'retry_attempts': 3,
    'backoff_factor': 2
}

# Source priority during different times
def get_active_sources(current_hour):
    """Return sentiment sources to process based on time of day"""
    
    if 9 <= current_hour <= 16:  # Market hours - full analysis
        return ['news', 'reddit', 'sec']
    elif 6 <= current_hour <= 9 or 16 <= current_hour <= 20:  # Pre/post market
        return ['news', 'reddit']
    else:  # Overnight - minimal processing
        return ['news']

def get_tickers_for_time(current_hour):
    """Return ticker list based on time of day"""
    
    if 9 <= current_hour <= 16:  # Market hours
        return PRIORITY_TICKERS['high'] + PRIORITY_TICKERS['medium']
    elif 6 <= current_hour <= 9 or 16 <= current_hour <= 20:  # Pre/post market  
        return PRIORITY_TICKERS['high']
    else:  # Overnight
        return PRIORITY_TICKERS['high'][:3]  # Just top 3

# Cloud platform optimizations
CLOUD_CONFIG = {
    'github_actions': {
        'max_runtime_minutes': 25,  # Leave 5min buffer
        'concurrent_tickers': 2,
        'enable_caching': True
    },
    'vercel': {
        'max_runtime_seconds': 300,  # 5 minute limit
        'concurrent_tickers': 1,
        'enable_caching': False
    },
    'render': {
        'max_runtime_minutes': 60,   # More generous
        'concurrent_tickers': 3,
        'enable_caching': True
    }
}

# Error handling and notifications
MONITORING = {
    'send_alerts': True,
    'alert_threshold': 0.5,  # Alert if >50% failures
    'log_retention_days': 7,
    'health_check_url': None  # Set webhook URL for monitoring
} 