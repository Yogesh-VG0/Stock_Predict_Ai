# Stock Prediction System

A production-grade backend and machine learning system for S&P 100 stock prediction, integrated with a Next.js frontend.

## Features

- Real-time stock price predictions for S&P 100 companies
- Multiple prediction windows (next day, 30-day, 90-day)
- Sentiment analysis from multiple sources (Reddit, News)
- Technical indicators and feature engineering
- LSTM-based deep learning models
- MongoDB data storage
- FastAPI backend
- Automated data updates
- Render deployment ready

## Prerequisites

- Python 3.9+
- MongoDB
- Node.js 16+ (for frontend)
- API keys for:
  - Reddit API
  - FRED API

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stockpredict-ai
```

2. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the backend directory with the following variables:
```env
MONGODB_URI=your_mongodb_uri
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_reddit_user_agent
FRED_API_KEY=your_fred_api_key
JWT_SECRET_KEY=your_jwt_secret_key
```