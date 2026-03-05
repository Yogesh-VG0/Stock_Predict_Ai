"""
Economic Calendar data fetcher for market-moving events.
Uses FMP data from sentiment pipeline stored in MongoDB.
"""

import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Any, Dict, List, Optional
import json
import os
import numpy as np

logger = logging.getLogger(__name__)

# Event impact mapping to tickers
EVENT_TICKER_MAPPING = {
    # Federal Reserve Events (High Impact)
    'fomc': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'COF', 'AXP', 'V', 'MA', 'BLK', 'SCHW'],  # Financial sector
    'fed interest rate': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'COF', 'AXP', 'V', 'MA', 'BLK', 'SCHW'],
    'fomc statement': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'COF', 'AXP', 'V', 'MA', 'BLK', 'SCHW'],
    'fomc press conference': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'COF', 'AXP', 'V', 'MA', 'BLK', 'SCHW'],
    'fomc meeting minutes': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'COF', 'AXP', 'V', 'MA', 'BLK', 'SCHW'],
    'fomc economic projections': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'COF', 'AXP', 'V', 'MA', 'BLK', 'SCHW'],
    'fed chair powell': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'COF', 'AXP', 'V', 'MA', 'BLK', 'SCHW'],
    
    # Employment Data (High Impact)
    'nonfarm payrolls': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'AMZN', 'HD', 'LOW', 'JPM', 'BAC', 'WFC', 'GS', 'MS'],
    'unemployment rate': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'AMZN', 'HD', 'LOW', 'JPM', 'BAC', 'WFC', 'GS', 'MS'],
    'jobless claims': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'AMZN', 'HD', 'LOW', 'JPM', 'BAC', 'WFC', 'GS', 'MS'],
    'adp nonfarm': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'AMZN', 'HD', 'LOW', 'JPM', 'BAC', 'WFC', 'GS', 'MS'],
    'jolts': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'AMZN', 'HD', 'LOW', 'JPM', 'BAC', 'WFC', 'GS', 'MS'],
    'average hourly earnings': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'AMZN', 'HD', 'LOW', 'JPM', 'BAC', 'WFC', 'GS', 'MS'],
    
    # Inflation Indicators (High Impact)
    'cpi': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'PG', 'KO', 'PEP', 'XOM', 'CVX', 'COP', 'JPM', 'BAC', 'WFC', 'GS', 'MS'],
    'core cpi': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'PG', 'KO', 'PEP', 'XOM', 'CVX', 'COP', 'JPM', 'BAC', 'WFC', 'GS', 'MS'],
    'ppi': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'PG', 'KO', 'PEP', 'XOM', 'CVX', 'COP', 'GE', 'BA', 'CAT', 'DE', 'HON'],
    'core pce': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'PG', 'KO', 'PEP', 'XOM', 'CVX', 'COP', 'JPM', 'BAC', 'WFC', 'GS', 'MS'],
    
    # Retail and Consumer Data (Medium Impact)
    'retail sales': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'AMZN', 'HD', 'LOW', 'PG', 'KO', 'PEP', 'PM', 'MO'],
    'core retail sales': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'AMZN', 'HD', 'LOW', 'PG', 'KO', 'PEP', 'PM', 'MO'],
    'consumer confidence': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'AMZN', 'HD', 'LOW', 'PG', 'KO', 'PEP', 'PM', 'MO'],
    
    # Manufacturing and Industrial Data (Medium Impact)
    'ism manufacturing': ['GE', 'BA', 'CAT', 'DE', 'HON', 'MMM', 'UPS', 'FDX', 'RTX', 'LMT', 'NVDA', 'AMD', 'INTC', 'TXN', 'QCOM'],
    'ism non-manufacturing': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'AMZN', 'HD', 'LOW', 'JPM', 'BAC', 'WFC', 'GS', 'MS'],
    'durable goods': ['GE', 'BA', 'CAT', 'DE', 'HON', 'MMM', 'UPS', 'FDX', 'RTX', 'LMT', 'NVDA', 'AMD', 'INTC', 'TXN', 'QCOM'],
    'philadelphia fed': ['GE', 'BA', 'CAT', 'DE', 'HON', 'MMM', 'UPS', 'FDX', 'RTX', 'LMT', 'NVDA', 'AMD', 'INTC', 'TXN', 'QCOM'],
    'chicago pmi': ['GE', 'BA', 'CAT', 'DE', 'HON', 'MMM', 'UPS', 'FDX', 'RTX', 'LMT', 'NVDA', 'AMD', 'INTC', 'TXN', 'QCOM'],
    
    # Housing Data (Medium Impact)
    'new home sales': ['HD', 'LOW', 'LMT', 'RTX', 'BA', 'SPG', 'AMT'],
    'existing home sales': ['HD', 'LOW', 'LMT', 'RTX', 'BA', 'SPG', 'AMT'],
    
    # Energy Data (Medium Impact)
    'crude oil': ['XOM', 'CVX', 'COP'],
    
    # GDP and Economic Growth (High Impact)
    'gdp': ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOG', 'GOOGL', 'META', 'TSLA', 'JPM', 'BAC', 'WFC', 'GS', 'MS'],
    
    # Tech and Innovation Events
    'semiconductor': ['NVDA', 'AMD', 'INTC', 'TXN', 'QCOM'],
    'ai': ['NVDA', 'AMD', 'INTC', 'MSFT', 'GOOG', 'GOOGL', 'META', 'AAPL'],
    'cloud': ['MSFT', 'AMZN', 'GOOG', 'GOOGL', 'ORCL', 'CRM', 'NOW'],
    
    # Healthcare Events
    'healthcare': ['JNJ', 'UNH', 'ABBV', 'ABT', 'MRK', 'PFE', 'BMY', 'MDT', 'ISRG'],
    'pharma': ['JNJ', 'ABBV', 'ABT', 'MRK', 'PFE', 'BMY', 'LLY'],
    
    # Financial Events
    'banking': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'COF'],
    'insurance': ['UNH', 'AIG', 'MET'],
    
    # Consumer Events
    'consumer': ['WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'PG', 'KO', 'PEP', 'PM', 'MO'],
    'discretionary': ['AMZN', 'HD', 'LOW', 'MCD', 'SBUX', 'NKE', 'DIS'],
    
    # Industrial Events
    'industrial': ['GE', 'BA', 'CAT', 'DE', 'HON', 'MMM', 'UPS', 'FDX', 'RTX', 'LMT'],
    
    # Real Estate Events
    'real estate': ['SPG', 'AMT'],
    
    # Telecom Events
    'telecom': ['T', 'TMUS', 'VZ', 'CMCSA', 'CHTR'],
    
    # Utilities Events
    'utilities': ['DUK', 'SO', 'NEE']
}

# Event importance scoring
EVENT_IMPORTANCE = {
    'high': 1.0,
    'medium': 0.6,
    'low': 0.3
}

class EconomicCalendar:
    """
    Fetches and processes economic calendar events.
    Maps events to affected tickers for event-driven feature engineering.
    Uses FMP data from sentiment pipeline stored in MongoDB.
    """
    
    def __init__(self, mongo_client=None):
        self.mongo_client = mongo_client
        
        # Instance-level cache to prevent multiple fetches per session
        self._events_cache = {
            'data': None,
            'timestamp': None,
            'cache_duration': timedelta(hours=6)
        }
        
    def fetch_economic_events(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Fetch economic calendar events for a date range from MongoDB cache.
        """
        if self.mongo_client:
            cached_events = self._get_cached_events(start_date, end_date)
            if cached_events:
                return cached_events
        return []
    
    def _parse_importance(self, bull_count: int) -> str:
        """Convert bull count to importance level."""
        if bull_count >= 3:
            return 'high'
        elif bull_count == 2:
            return 'medium'
        else:
            return 'low'
    
    def _parse_event_date(self, date_str: str) -> datetime:
        """Parse event date string to datetime."""
        try:
            # Handle various date formats
            if '/' in date_str:
                return datetime.strptime(date_str, '%m/%d/%Y')
            elif '-' in date_str:
                return datetime.strptime(date_str, '%Y-%m-%d')
            else:
                # Default to today if parsing fails
                return datetime.now()
        except:
            return datetime.now()
    
    def _extract_country_from_class(self, class_str: str) -> str:
        """Extract country code from CSS class."""
        # Class format: "ceFlags USA" or similar
        parts = class_str.split()
        if len(parts) > 1:
            country = parts[1]
            # Map common country names to codes
            country_map = {
                'USA': 'US',
                'United_States': 'US',
                'UnitedStates': 'US',
                'United_States_of_America': 'US'
            }
            return country_map.get(country, country)
        return 'US'  # Default to US if no country found
    
    def _map_event_to_tickers(self, event_data: Dict) -> List[str]:
        """
        Map economic event to affected tickers based on event type.
        """
        affected_tickers = set()
        event_name = event_data.get('event', '').lower()
        
        # Check each mapping keyword
        for keyword, tickers in EVENT_TICKER_MAPPING.items():
            if keyword in event_name:
                affected_tickers.update(tickers)
        
        # High importance events affect broad market
        if event_data.get('importance') == 'high' and not affected_tickers:
            affected_tickers.update(['SPY', 'QQQ', 'IWM', 'DIA'])
        
        # Filter to only S&P 100 tickers if needed
        from ..config.constants import TOP_100_TICKERS
        affected_tickers = [t for t in affected_tickers if t in TOP_100_TICKERS]
        
        return list(affected_tickers)
    
    def get_event_features(self, ticker: str, date: datetime, lookback_days: int = 7) -> Dict:
        """Get event-driven features for a specific ticker and date."""
        features = {
            'has_high_impact_event_today': 0,
            'days_to_next_high_impact': 30,  # Changed from 999 to 30 (more realistic default)
            'days_since_last_high_impact': 30,  # Changed from 999 to 30
            'event_density_7d': 0,
            'event_importance_sum_7d': 0.0,
            'has_earnings_today': 0,
            'days_to_next_earnings': 90,  # Changed from 999 to 90 (quarterly earnings)
            'days_since_last_earnings': 90,  # Changed from 999 to 90
            'has_dividend_today': 0,
            'days_to_next_dividend': 90,  # Changed from 999 to 90 (quarterly dividends)
            'days_since_last_dividend': 90,  # Changed from 999 to 90
            'dividend_amount': 0.0,
            'dividend_yield': 0.0,
            'next_earnings_eps_estimate': None,
            'next_earnings_revenue_estimate': None,
            'next_dividend_record_date': None,
            'next_dividend_payment_date': None,
            'dividend_frequency': None,
            'data_sources_checked': [],
            'fmp_earnings_count': 0,
            'fmp_dividends_count': 0,
            'economic_events_count': 0,
            'sentiment_data_available': False
        }
        
        try:
            # Get economic events
            start_date = date - timedelta(days=lookback_days)
            end_date = date + timedelta(days=lookback_days)
            events = self.fetch_economic_events(start_date, end_date)
            
            # Get latest sentiment data from MongoDB which includes FMP data
            features['data_sources_checked'].append('sentiment_data')
            if self.mongo_client and self.mongo_client.db is not None:
                sentiment_data = self.mongo_client.get_latest_sentiment(ticker)
                if sentiment_data:
                    features['sentiment_data_available'] = True
                    logger.info(f"✓ Found sentiment data for {ticker} with keys: {list(sentiment_data.keys())}")
                    
                    # Check multiple locations for FMP data
                    fmp_data_sources = [
                        sentiment_data.get('fmp_raw_data', {}),  # Primary location
                        sentiment_data,  # Direct in sentiment data
                    ]
                    
                    fmp_data = {}
                    for source in fmp_data_sources:
                        if isinstance(source, dict) and source:
                            fmp_data = source
                            break
                    
                    if fmp_data:
                        logger.info(f"Found FMP data for {ticker} with keys: {list(fmp_data.keys())}")
                        
                        # Check earnings data using multiple possible locations
                        earnings_data = []
                        
                        # Try different possible locations for earnings data
                        earnings_sources = [
                            fmp_data.get('earnings', []),           # Direct earnings array
                            fmp_data.get('earnings_calendar', []),  # Calendar-specific earnings
                            sentiment_data.get('fmp_earnings', []), # Legacy FMP location
                            sentiment_data.get('fmp_earnings_calendar', [])  # Legacy location
                        ]
                        
                        for source in earnings_sources:
                            if source and isinstance(source, list) and len(source) > 0:
                                earnings_data = source
                                logger.info(f"Found earnings data from source with {len(source)} entries")
                                break
                                
                        if earnings_data:
                            features['fmp_earnings_count'] = len(earnings_data)
                            features['data_sources_checked'].append('fmp_earnings')
                            logger.info(f"Processing {len(earnings_data)} FMP earnings entries for {ticker}")
                            
                            for event in earnings_data:
                                try:
                                    # Handle both FMP API structures
                                    event_date_str = event.get('date') or event.get('reportedDate')
                                    if not event_date_str:
                                        continue
                                        
                                    event_date = datetime.strptime(event_date_str, '%Y-%m-%d')
                                    days_diff = (event_date - date).days
                                    
                                    logger.info(f"Processing earnings event: {event_date_str}, days_diff: {days_diff}")
                                    
                                    if days_diff == 0:
                                        features['has_earnings_today'] = 1
                                        features['next_earnings_eps_estimate'] = event.get('epsEstimated') or event.get('estimatedEps')
                                        features['next_earnings_revenue_estimate'] = event.get('revenueEstimated') or event.get('estimatedRevenue')
                                        logger.info(f"✓ Earnings TODAY for {ticker}: EPS={features['next_earnings_eps_estimate']}, Revenue={features['next_earnings_revenue_estimate']}")
                                    elif days_diff > 0 and days_diff < features['days_to_next_earnings']:
                                        features['days_to_next_earnings'] = days_diff
                                        features['next_earnings_eps_estimate'] = event.get('epsEstimated') or event.get('estimatedEps')
                                        features['next_earnings_revenue_estimate'] = event.get('revenueEstimated') or event.get('estimatedRevenue')
                                        logger.info(f"✓ Next earnings for {ticker} in {days_diff} days: EPS={features['next_earnings_eps_estimate']}, Revenue={features['next_earnings_revenue_estimate']}")
                                    elif days_diff < 0 and abs(days_diff) < features['days_since_last_earnings']:
                                        features['days_since_last_earnings'] = abs(days_diff)
                                        logger.info(f"✓ Last earnings for {ticker} was {abs(days_diff)} days ago: EPS_Actual={event.get('epsActual')}")
                                except Exception as e:
                                    logger.warning(f"Error processing earnings event for {ticker}: {e}")
                        else:
                            logger.warning(f"No earnings data found in any location for {ticker}")
                        
                        # Check dividends data using multiple possible locations
                        dividends_data = []
                        
                        # Try different possible locations for dividends data
                        dividends_sources = [
                            fmp_data.get('dividends', []),           # Direct dividends array
                            fmp_data.get('dividends_calendar', []),  # Calendar-specific dividends
                            sentiment_data.get('fmp_dividends', []), # Legacy FMP location
                            sentiment_data.get('fmp_dividends_calendar', [])  # Legacy location
                        ]
                        
                        for source in dividends_sources:
                            if source and isinstance(source, list) and len(source) > 0:
                                dividends_data = source
                                logger.info(f"Found dividends data from source with {len(source)} entries")
                                break
                                
                        if dividends_data:
                            features['fmp_dividends_count'] = len(dividends_data)
                            features['data_sources_checked'].append('fmp_dividends')
                            logger.info(f"Processing {len(dividends_data)} FMP dividends entries for {ticker}")
                            
                            for div in dividends_data:
                                try:
                                    # Handle both FMP API structures
                                    div_date_str = div.get('date') or div.get('exDividendDate')  # Ex-dividend date
                                    if not div_date_str:
                                        continue
                                        
                                    div_date = datetime.strptime(div_date_str, '%Y-%m-%d')
                                    days_diff = (div_date - date).days
                                    
                                    logger.info(f"Processing dividend event: {div_date_str}, days_diff: {days_diff}")
                                    
                                    if days_diff == 0:
                                        features['has_dividend_today'] = 1
                                        features['dividend_amount'] = float(div.get('dividend', 0) or div.get('adjDividend', 0))
                                        features['dividend_yield'] = float(div.get('yield', 0))
                                        features['next_dividend_record_date'] = div.get('recordDate')
                                        features['next_dividend_payment_date'] = div.get('paymentDate')
                                        features['dividend_frequency'] = div.get('frequency')
                                        logger.info(f"✓ Dividend TODAY for {ticker}: ${features['dividend_amount']:.3f} ({features['dividend_yield']:.2f}% yield)")
                                    elif days_diff > 0 and days_diff < features['days_to_next_dividend']:
                                        features['days_to_next_dividend'] = days_diff
                                        features['dividend_amount'] = float(div.get('dividend', 0) or div.get('adjDividend', 0))
                                        features['dividend_yield'] = float(div.get('yield', 0))
                                        features['next_dividend_record_date'] = div.get('recordDate')
                                        features['next_dividend_payment_date'] = div.get('paymentDate')
                                        features['dividend_frequency'] = div.get('frequency')
                                        logger.info(f"✓ Next dividend for {ticker} in {days_diff} days: ${features['dividend_amount']:.3f} ({features['dividend_yield']:.2f}% yield)")
                                    elif days_diff < 0 and abs(days_diff) < features['days_since_last_dividend']:
                                        features['days_since_last_dividend'] = abs(days_diff)
                                        logger.info(f"✓ Last dividend for {ticker} was {abs(days_diff)} days ago: ${div.get('dividend', 0):.3f}")
                                except Exception as e:
                                    logger.warning(f"Error processing dividend event for {ticker}: {e}")
                        else:
                            logger.warning(f"No dividends data found in any location for {ticker}")
                                    
                    else:
                        logger.warning(f"No FMP data found in sentiment data for {ticker}")
                        
                    # Log final feature values for debugging
                    logger.info(f"Final economic features for {ticker}:")
                    logger.info(f"  - FMP earnings count: {features['fmp_earnings_count']}")
                    logger.info(f"  - FMP dividends count: {features['fmp_dividends_count']}")
                    logger.info(f"  - Days to next earnings: {features['days_to_next_earnings']}")
                    logger.info(f"  - Days to next dividend: {features['days_to_next_dividend']}")
                    logger.info(f"  - Data sources checked: {features['data_sources_checked']}")
                else:
                    features['sentiment_data_available'] = False
                    logger.warning(f"No FMP/earnings data in sentiment collection for {ticker}")

            # Process economic events - only high impact events
            features['economic_events_count'] = len(events)
            features['data_sources_checked'].append('economic_events')
            logger.info(f"Found {len(events)} economic events in date range for processing")
            
            ticker_events = [e for e in events if ticker in e.get('affected_tickers', [])]
            logger.info(f"Found {len(ticker_events)} events specifically affecting {ticker}")
            
            for event in ticker_events:
                event_date = datetime.strptime(event['date'], '%Y-%m-%d')
                days_diff = (event_date - date).days
                
                if days_diff == 0:
                        features['has_high_impact_event_today'] = 1
                
                if days_diff > 0 and days_diff <= 7:
                    features['event_density_7d'] += 1
                    features['event_importance_sum_7d'] += 1.0  # All events are high impact
                    
                    if days_diff < features['days_to_next_high_impact']:
                        features['days_to_next_high_impact'] = days_diff
                
                if days_diff < 0:
                    days_since = abs(days_diff)
                    if days_since < features['days_since_last_high_impact']:
                        features['days_since_last_high_impact'] = days_since
            
            # Add event volatility score
            features['event_volatility_score'] = self._calculate_event_volatility(features)
            
        except Exception as e:
            logger.error(f"Error getting event features for {ticker}: {e}")
        
        return features
    
    def _calculate_event_volatility(self, features: Dict) -> float:
        """Calculate expected volatility based on economic events and corporate events."""
        score = 0.0
        
        # Economic events impact
        score += features['has_high_impact_event_today'] * 1.0
        
        # Earnings impact
        if features['has_earnings_today']:
            score += 1.2  # Earnings have higher impact than economic events
        elif features['days_to_next_earnings'] <= 2:
            score += 0.6
        elif features['days_to_next_earnings'] <= 5:
            score += 0.3
        
        # Dividend impact
        if features['has_dividend_today']:
            score += 0.4  # Dividends have moderate impact
        elif features['days_to_next_dividend'] <= 2:
            score += 0.2
            
        # Event clustering increases volatility
        if features['event_density_7d'] > 5:
            score += 0.4
        elif features['event_density_7d'] > 3:
            score += 0.2
        
        return min(score, 2.0)  # Cap at 2.0
    
    def _calculate_real_earnings_days(self, fmp_data: Dict, current_date: datetime) -> Dict:
        """Calculate real days to/from earnings using FMP data."""
        earnings_features = {
            'days_to_next_earnings': 365,
            'days_since_last_earnings': 365,
            'fmp_earnings_count': 0,
            'next_earnings_eps_estimate': None,
            'next_earnings_revenue_estimate': None,
            'has_earnings_today': 0
        }
        
        try:
            # Combine earnings calendar and historical earnings
            earnings_calendar = fmp_data.get('earnings_calendar', [])
            historical_earnings = fmp_data.get('historical_earnings', [])
            
            all_earnings = earnings_calendar + historical_earnings
            if not all_earnings:
                return earnings_features
            
            earnings_features['fmp_earnings_count'] = len(all_earnings)
            
            for event in all_earnings:
                try:
                    event_date_str = event.get('date') or event.get('reportedDate')
                    if not event_date_str:
                        continue
                        
                    event_date = datetime.strptime(event_date_str, '%Y-%m-%d')
                    days_diff = (event_date - current_date).days
                    
                    if days_diff == 0:
                        earnings_features['has_earnings_today'] = 1
                        earnings_features['next_earnings_eps_estimate'] = event.get('epsEstimated') or event.get('estimatedEps')
                        earnings_features['next_earnings_revenue_estimate'] = event.get('revenueEstimated') or event.get('estimatedRevenue')
                    elif days_diff > 0 and days_diff < earnings_features['days_to_next_earnings']:
                        earnings_features['days_to_next_earnings'] = days_diff
                        earnings_features['next_earnings_eps_estimate'] = event.get('epsEstimated') or event.get('estimatedEps')
                        earnings_features['next_earnings_revenue_estimate'] = event.get('revenueEstimated') or event.get('estimatedRevenue')
                    elif days_diff < 0 and abs(days_diff) < earnings_features['days_since_last_earnings']:
                        earnings_features['days_since_last_earnings'] = abs(days_diff)
                        
                except Exception as e:
                    logger.warning(f"Error processing earnings event: {e}")
                    
            return earnings_features
            
        except Exception as e:
            logger.error(f"Error calculating real earnings days: {e}")
            return earnings_features
    
    def _calculate_real_dividend_days(self, fmp_data: Dict, current_date: datetime) -> Dict:
        """Calculate real days to/from dividends using FMP data."""
        dividend_features = {
            'days_to_next_dividend': 365,
            'days_since_last_dividend': 365,
            'fmp_dividends_count': 0,
            'dividend_amount': None,
            'dividend_yield': None,
            'next_dividend_record_date': None,
            'next_dividend_payment_date': None,
            'dividend_frequency': None,
            'has_dividend_today': 0
        }
        
        try:
            # Combine dividends calendar and historical dividends
            dividends_calendar = fmp_data.get('dividends_calendar', [])
            historical_dividends = fmp_data.get('historical_dividends', [])
            
            all_dividends = dividends_calendar + historical_dividends
            if not all_dividends:
                return dividend_features
            
            dividend_features['fmp_dividends_count'] = len(all_dividends)
            
            for div in all_dividends:
                try:
                    div_date_str = div.get('date') or div.get('exDividendDate') or div.get('paymentDate')
                    if not div_date_str:
                        continue
                        
                    div_date = datetime.strptime(div_date_str, '%Y-%m-%d')
                    days_diff = (div_date - current_date).days
                    
                    if days_diff == 0:
                        dividend_features['has_dividend_today'] = 1
                        dividend_features['dividend_amount'] = float(div.get('dividend', 0) or div.get('adjDividend', 0))
                        dividend_features['dividend_yield'] = float(div.get('yield', 0))
                        dividend_features['next_dividend_record_date'] = div.get('recordDate')
                        dividend_features['next_dividend_payment_date'] = div.get('paymentDate')
                        dividend_features['dividend_frequency'] = div.get('frequency')
                    elif days_diff > 0 and days_diff < dividend_features['days_to_next_dividend']:
                        dividend_features['days_to_next_dividend'] = days_diff
                        dividend_features['dividend_amount'] = float(div.get('dividend', 0) or div.get('adjDividend', 0))
                        dividend_features['dividend_yield'] = float(div.get('yield', 0))
                        dividend_features['next_dividend_record_date'] = div.get('recordDate')
                        dividend_features['next_dividend_payment_date'] = div.get('paymentDate')
                        dividend_features['dividend_frequency'] = div.get('frequency')
                    elif days_diff < 0 and abs(days_diff) < dividend_features['days_since_last_dividend']:
                        dividend_features['days_since_last_dividend'] = abs(days_diff)
                        
                except Exception as e:
                    logger.warning(f"Error processing dividend event: {e}")
                    
            return dividend_features
            
        except Exception as e:
            logger.error(f"Error calculating real dividend days: {e}")
            return dividend_features
    
    def _get_cached_events(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get cached events from MongoDB"""
        try:
            if not self.mongo_client or self.mongo_client.db is None:
                return []
            
            # Query MongoDB for events in date range
            query = {
                'date': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            }
            
            events = list(self.mongo_client.db.economic_events.find(query))
            if events:
                logger.info(f"Found {len(events)} cached events")
            return events
            
        except Exception as e:
            logger.error(f"Error getting cached events: {e}")
            return []
    
    def _store_events(self, events: List[Dict]):
        """Store events in MongoDB with duplicate prevention"""
        try:
            if not self.mongo_client or self.mongo_client.db is None or not events:
                return
            
            stored_count = 0
            for event in events:
                # Use upsert to prevent duplicates based on date and event name
                unique_key = {
                    'date': event.get('date'),
                    'event': event.get('event'),
                    'country': event.get('country', 'US')
                }
                
                result = self.mongo_client.db.economic_events.replace_one(
                    unique_key,
                    event,
                    upsert=True
                )
                
                if result.upserted_id or result.modified_count > 0:
                    stored_count += 1
            
            logger.info(f"Stored/Updated {stored_count} unique events in MongoDB (processed {len(events)} total)")
            
        except Exception as e:
            logger.error(f"Error storing events: {e}")

    async def fetch_fresh_fmp_data(self, ticker: str) -> Dict:
        """
        Fetch fresh FMP earnings and dividend data using centralized FMP manager.
        This prevents duplicate API calls by using the same manager as sentiment.py.
        """
        try:
            # Import the FMP manager from sentiment.py to avoid duplicate API calls
            from .sentiment import FMPAPIManager
            
            # Create FMP manager instance (this will use caching)
            fmp_manager = FMPAPIManager(self.mongo_client)
            
            logger.info(f"Fetching fresh FMP data for {ticker} using centralized manager")
            
            # Get all FMP data in one consolidated call
            fresh_data = await fmp_manager.get_all_fmp_data(ticker)
            
            if not fresh_data:
                logger.warning(f"No FMP data returned for {ticker}")
                return {}
            
            logger.info(f"Successfully fetched FMP data for {ticker}: {list(fresh_data.keys())}")
            return fresh_data
            
        except Exception as e:
            logger.error(f"Error fetching fresh FMP data for {ticker}: {e}")
            return {}

    async def get_event_features_with_fresh_data(self, ticker: str, date: datetime, lookback_days: int = 7) -> Dict:
        """
        Get event features using fresh FMP data instead of placeholders.
        """
        try:
            # Get fresh FMP data (this uses centralized manager to avoid duplicate API calls)
            fresh_fmp_data = await self.fetch_fresh_fmp_data(ticker)
            
            # Start with the standard event features
            features = self.get_event_features(ticker, date, lookback_days)
            
            if fresh_fmp_data:
                # Update features with real FMP data
                features.update(self._calculate_real_earnings_days(fresh_fmp_data, date))
                features.update(self._calculate_real_dividend_days(fresh_fmp_data, date))
                
                logger.info(f"Updated economic features for {ticker} with fresh FMP data")
            else:
                logger.warning(f"No fresh FMP data available for {ticker}, using default features")
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting event features with fresh data for {ticker}: {e}")
            # Fallback to standard features
            return self.get_event_features(ticker, date, lookback_days)

def integrate_economic_events_sentiment(sentiment_dict: Dict, ticker: str, mongo_client=None) -> Dict:
    """
    Integrate economic event data into sentiment analysis.
    High-impact events increase uncertainty and potential volatility.
    Uses shared EconomicCalendar instance when possible to prevent multiple web scraping.
    """
    try:
        # Try to use shared calendar instance from sentiment analyzer
        from .sentiment import SHARED_ECONOMIC_CALENDAR
        if SHARED_ECONOMIC_CALENDAR is not None:
            calendar = SHARED_ECONOMIC_CALENDAR
        else:
            # Fallback to creating new instance
            calendar = EconomicCalendar(mongo_client)
        
        # Get event features for today
        event_features = calendar.get_event_features(ticker, datetime.now())
        
        # Calculate event-based sentiment adjustment
        event_sentiment = 0.0
        
        # High impact events today create uncertainty (slightly bearish)
        if event_features['has_high_impact_event_today']:
            event_sentiment -= 0.2
        
        # Many upcoming events create anticipation (neutral to slightly bearish)
        if event_features['event_density_7d'] > 5:
            event_sentiment -= 0.1
        
        # Recent high impact events may still affect sentiment
        if event_features['days_since_last_high_impact'] <= 2:
            event_sentiment -= 0.05
        
        # High volatility score indicates trading opportunity (can be positive)
        volatility_score = event_features['event_volatility_score']
        if volatility_score > 1.5:
            event_sentiment += 0.1  # High vol can mean opportunity
        
        # Add to sentiment dict
        sentiment_dict['economic_event_sentiment'] = event_sentiment
        sentiment_dict['economic_event_volatility'] = volatility_score
        sentiment_dict['economic_event_volume'] = 1 if event_features['has_high_impact_event_today'] else 0
        sentiment_dict['economic_event_confidence'] = 0.9  # High confidence in event data
        
        # Store event features for explainability
        sentiment_dict['economic_event_features'] = event_features
        
        logger.info(f"Economic event sentiment for {ticker}: {event_sentiment:.2f} (volatility: {volatility_score:.2f})")
        
    except Exception as e:
        logger.error(f"Error integrating economic events: {e}")
        sentiment_dict['economic_event_sentiment'] = 0.0
        sentiment_dict['economic_event_volume'] = 0
        sentiment_dict['economic_event_confidence'] = 0.0
    
    return sentiment_dict 