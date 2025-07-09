# StockPredict AI - Railway Deployment Guide

This guide explains how to deploy StockPredict AI on Railway with both frontend and backend services.

## Architecture

The application now runs both the Next.js frontend and Node.js backend in a single Railway service using PM2 process manager:

- **Frontend**: Next.js app running on port 3000
- **Backend**: Node.js API server running on port 5000
- **Process Manager**: PM2 manages both services

## Required Environment Variables

Set these environment variables in your Railway project settings:

### Required API Keys

```bash
# Finnhub API (Required for stock data)
FINNHUB_API_KEY=your_finnhub_api_key

# ML Backend URL (Required for predictions)
ML_BACKEND_URL=https://stockpredict-ai-ml-api.onrender.com

# News APIs (Optional but recommended)
MARKETAUX_API_KEY=your_marketaux_api_key
NEWSAPI_KEY=your_newsapi_key
EODHD_API_KEY=your_eodhd_api_key

# Market Data APIs (Optional)
CALENDARIFIC_API_KEY=your_calendarific_api_key
RAPIDAPI_KEY=your_rapidapi_key

# Redis (Optional - for caching)
REDIS_URL=redis://your_redis_url
```

## Deployment Steps

1. **Connect your GitHub repository to Railway**
2. **Set environment variables** in Railway dashboard
3. **Deploy** - Railway will automatically build and deploy using the Dockerfile

## How to Set Environment Variables in Railway

1. Go to your Railway project dashboard
2. Click on your service
3. Go to the "Variables" tab
4. Add each environment variable listed above
5. Redeploy the service

## API Key Setup

### Finnhub API Key (Required)
1. Go to [Finnhub.io](https://finnhub.io/)
2. Create a free account
3. Get your API key from the dashboard
4. Add as `FINNHUB_API_KEY` in Railway

### Other API Keys (Optional)
- **MarketAux**: [marketaux.com](https://marketaux.com/) for news data
- **NewsAPI**: [newsapi.org](https://newsapi.org/) for news aggregation
- **EODHD**: [eodhd.com](https://eodhd.com/) for financial data
- **Calendarific**: [calendarific.com](https://calendarific.com/) for market holidays
- **RapidAPI**: [rapidapi.com](https://rapidapi.com/) for various APIs

## Verification

After deployment, check:

1. **Frontend**: Access your Railway URL - should show the home page
2. **API Health**: Visit `your-railway-url/health` - should return `{"status": "healthy"}`
3. **News API**: Check if news data loads on the news page
4. **Predictions**: Check if stock predictions work

## Troubleshooting

### Common Issues

1. **API calls failing**: Check if environment variables are set correctly
2. **Backend not responding**: Check Railway logs for PM2 startup errors
3. **No predictions**: Verify `ML_BACKEND_URL` is set to the correct ML service
4. **No news data**: Add the news API keys (MARKETAUX_API_KEY, NEWSAPI_KEY)

### Checking Logs

View logs in Railway dashboard:
1. Go to your service
2. Click "Deployments" tab
3. Click on latest deployment
4. View build and runtime logs

## Service Architecture

```
Railway Service
├── PM2 Process Manager
│   ├── Frontend (Next.js) - Port 3000
│   └── Backend (Node.js) - Port 5000
├── MongoDB Atlas (External)
└── ML Backend API (External - Render)
```

The services communicate internally via localhost, and Railway exposes port 3000 publicly. 