"""
Simple Sentiment Pipeline Monitoring Dashboard
View at: http://localhost:8080/dashboard
"""

import os
import json
from datetime import datetime, timedelta
from flask import Flask, render_template_string
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml_backend'))
from utils.mongodb import MongoDBClient

app = Flask(__name__)

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Pipeline Monitor</title>
    <meta http-equiv="refresh" content="300">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .status-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .success { border-left: 5px solid #27ae60; }
        .error { border-left: 5px solid #e74c3c; }
        .warning { border-left: 5px solid #f39c12; }
        .metric { display: flex; justify-content: space-between; margin: 10px 0; }
        .ticker { padding: 5px 10px; margin: 5px; border-radius: 15px; display: inline-block; }
        .ticker.good { background: #d5f4e6; color: #27ae60; }
        .ticker.bad { background: #fadbd8; color: #e74c3c; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Sentiment Pipeline Monitor</h1>
            <p>Last updated: {{ last_updated }} | Next run: {{ next_run }}</p>
        </div>
        
        <div class="status-grid">
            <div class="status-card success">
                <h3>✅ Pipeline Status</h3>
                <div class="metric">
                    <span>Success Rate (24h):</span>
                    <strong>{{ success_rate }}%</strong>
                </div>
                <div class="metric">
                    <span>Total Runs Today:</span>
                    <strong>{{ total_runs }}</strong>
                </div>
                <div class="metric">
                    <span>Active Tickers:</span>
                    <strong>{{ active_tickers }}</strong>
                </div>
            </div>
            
            <div class="status-card">
                <h3>📈 Recent Sentiment Data</h3>
                {% for ticker, data in recent_sentiment.items() %}
                <div class="ticker {{ 'good' if data.sentiment > 0 else 'bad' }}">
                    {{ ticker }}: {{ data.sentiment | round(3) }}
                    <small>({{ data.confidence | round(3) }})</small>
                </div>
                {% endfor %}
            </div>
            
            <div class="status-card {{ 'error' if errors|length > 0 else 'success' }}">
                <h3>🚨 Recent Errors</h3>
                {% if errors|length == 0 %}
                    <p style="color: #27ae60;">No errors in the last 24 hours! 🎉</p>
                {% else %}
                    {% for error in errors[:5] %}
                    <div style="margin: 10px 0; padding: 10px; background: #fadbd8; border-radius: 5px;">
                        <strong>{{ error.ticker }}</strong>: {{ error.message }}
                        <br><small>{{ error.timestamp }}</small>
                    </div>
                    {% endfor %}
                {% endif %}
            </div>
            
            <div class="status-card">
                <h3>⏱️ Performance Metrics</h3>
                <div class="metric">
                    <span>Avg Runtime:</span>
                    <strong>{{ avg_runtime }} min</strong>
                </div>
                <div class="metric">
                    <span>Data Freshness:</span>
                    <strong>{{ data_freshness }} min ago</strong>
                </div>
                <div class="metric">
                    <span>API Calls Today:</span>
                    <strong>{{ api_calls }}</strong>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/dashboard')
def dashboard():
    try:
        # Connect to MongoDB
        mongo_client = MongoDBClient(os.getenv('MONGODB_URI'))
        
        # Get recent sentiment data
        recent_sentiment = {}
        sentiment_collection = mongo_client.db['sentiment_data']
        
        for ticker in ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']:
            latest = sentiment_collection.find_one(
                {'ticker': ticker},
                sort=[('timestamp', -1)]
            )
            if latest:
                recent_sentiment[ticker] = {
                    'sentiment': latest.get('sources', {}).get('news', {}).get('sentiment', 0),
                    'confidence': latest.get('sources', {}).get('news', {}).get('confidence', 0)
                }
        
        # Calculate metrics
        now = datetime.utcnow()
        last_24h = now - timedelta(days=1)
        
        # Mock data for demo (replace with real MongoDB queries)
        dashboard_data = {
            'last_updated': now.strftime('%Y-%m-%d %H:%M:%S UTC'),
            'next_run': (now + timedelta(hours=4)).strftime('%H:%M UTC'),
            'success_rate': 94.2,
            'total_runs': 6,
            'active_tickers': len(recent_sentiment),
            'recent_sentiment': recent_sentiment,
            'errors': [],  # No recent errors
            'avg_runtime': 8.5,
            'data_freshness': 45,
            'api_calls': 1240
        }
        
        return render_template_string(DASHBOARD_HTML, **dashboard_data)
        
    except Exception as e:
        error_data = {
            'last_updated': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
            'next_run': 'Unknown',
            'success_rate': 0,
            'total_runs': 0,
            'active_tickers': 0,
            'recent_sentiment': {},
            'errors': [{'ticker': 'SYSTEM', 'message': str(e), 'timestamp': datetime.utcnow().isoformat()}],
            'avg_runtime': 0,
            'data_freshness': 999,
            'api_calls': 0
        }
        return render_template_string(DASHBOARD_HTML, **error_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True) 