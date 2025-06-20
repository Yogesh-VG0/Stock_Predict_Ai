import os
import sys
import json
from datetime import datetime, timezone

# Add ml_backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml_backend'))

from data.sentiment import SentimentAnalyzer
from utils.mongodb import MongoDBClient

def handler(request, context):
    """Vercel serverless function for sentiment pipeline"""
    
    try:
        # Initialize services
        mongodb_uri = os.getenv('MONGODB_URI')
        if not mongodb_uri:
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'MongoDB URI not configured'})
            }
        
        mongo_client = MongoDBClient(mongodb_uri)
        sentiment_analyzer = SentimentAnalyzer(mongo_client)
        
        # Process sentiment for top tickers
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META']
        results = {}
        
        for ticker in tickers:
            try:
                # Quick sentiment analysis (limited for serverless constraints)
                news_sentiment = sentiment_analyzer.fetch_and_store_news_sentiment(ticker)
                results[ticker] = {
                    'news_sentiment': news_sentiment,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'status': 'success'
                }
            except Exception as e:
                results[ticker] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Sentiment pipeline completed',
                'results': results,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f'Pipeline failed: {str(e)}',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        } 