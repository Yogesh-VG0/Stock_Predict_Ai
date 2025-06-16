"""
Main FastAPI application for the stock prediction system.
"""

from fastapi import FastAPI, HTTPException, Depends, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import jwt
from pydantic import BaseModel
import time
from functools import wraps
import redis.asyncio as redis
import os
from dotenv import load_dotenv
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import logging
from starlette.concurrency import run_in_threadpool
import json
import httpx
import pandas as pd
import google.generativeai as genai
import numpy as np
from ml_backend.data.economic_calendar import EconomicCalendar
import asyncio

from ml_backend.config.constants import (
    API_PREFIX,
    API_VERSION,
    RATE_LIMIT,
    JWT_ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    TOP_100_TICKERS,
    API_CONFIG,
    PREDICTION_WINDOWS,
    TECHNICAL_INDICATORS,
    FEATURE_CONFIG,
    MODEL_CONFIG,
    MONGO_COLLECTIONS,
    REDDIT_SUBREDDITS,
    RSS_FEEDS,
    RETRY_CONFIG,
    HISTORICAL_DATA_YEARS
)
from ml_backend.utils.mongodb import MongoDBClient
from ml_backend.data.ingestion import DataIngestion
from ml_backend.data.sentiment import SentimentAnalyzer
from ml_backend.data.features import FeatureEngineer
from ml_backend.models.predictor import StockPredictor

# Load environment variables
load_dotenv()

# Only set up logging once
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stock Prediction API",
    description="API for S&P 100 stock predictions and analysis. Organized endpoints for predictions, sentiment, training, ingestion, and explainability.",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis client for rate limiting and caching
redis_client = None
try:
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        logger.warning("REDIS_URL environment variable not set, Redis features will be disabled")
    else:
        redis_client = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        # Test the connection will be done in startup event
except Exception as e:
    logger.warning(f"Could not connect to Redis: {str(e)}. Redis features will be disabled")
    redis_client = None

@app.on_event("startup")
async def startup():
    """Initialize rate limiter, DB, models, and other components on startup."""
    try:
        # Initialize MongoDB client and other components
        app.state.mongo_client = MongoDBClient(os.getenv("MONGODB_URI"))
        app.state.data_ingestion = DataIngestion(app.state.mongo_client)
        app.state.sentiment_analyzer = SentimentAnalyzer(app.state.mongo_client)
        app.state.feature_engineer = FeatureEngineer()
        app.state.stock_predictor = StockPredictor(feature_engineer=app.state.feature_engineer)
        app.state.calendar_fetcher = EconomicCalendar(app.state.mongo_client)
        if redis_client is not None:
            await redis_client.ping()
            await FastAPILimiter.init(redis_client)
        app.state.stock_predictor.load_models()
        logger.info("API startup completed")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

@app.on_event("shutdown")
async def shutdown():
    """Clean up resources on shutdown."""
    try:
        app.state.stock_predictor.save_models()
        app.state.mongo_client.close()
        if redis_client is not None:
            await redis_client.close()
        logger.info("API shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")

# Rate limiting middleware
def rate_limit(limit: int = RATE_LIMIT, window: int = 3600):
    """Rate limiting decorator using Redis."""
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = request.client.host
            key = f"rate_limit:{client_ip}"
            if redis_client is None:
                return await func(request, *args, **kwargs)
            # Get current count
            current = await redis_client.get(key)
            if current is None:
                # First request in the window
                await redis_client.setex(key, window, 1)
            else:
                current = int(current)
                if current >= limit:
                    raise HTTPException(
                        status_code=429,
                        detail="Rate limit exceeded. Please try again later."
                    )
                # Increment counter
                await redis_client.incr(key)
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

# Models
class PredictionResponse(BaseModel):
    ticker: str
    prediction: Dict[str, float]
    confidence: Dict[str, float]
    last_updated: datetime

class SentimentResponse(BaseModel):
    ticker: str
    sentiment_score: float
    sources: Dict[str, float]
    last_updated: datetime

class HistoricalDataResponse(BaseModel):
    ticker: str
    data: List[Dict[str, Any]]
    last_updated: datetime

# Routes
@app.get("/api/v1/metadata/{ticker}", tags=["Metadata"], summary="Get Metadata", description="Get metadata for a given ticker.")
async def get_metadata(ticker: str) -> Dict:
    """Get metadata for a specific ticker."""
    try:
        if ticker not in TOP_100_TICKERS:
            raise HTTPException(status_code=404, detail="Ticker not found in S&P 100")
        
        # Get latest predictions and sentiment
        predictions = app.state.mongo_client.get_latest_predictions(ticker)
        sentiment = app.state.mongo_client.get_latest_sentiment(ticker)
        
        # Get historical data for the last year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        historical_data = app.state.mongo_client.get_historical_data(ticker, start_date, end_date)
        
        # Calculate basic statistics
        if not historical_data.empty:
            stats = {
                "current_price": historical_data["Close"].iloc[-1],
                "price_change_1d": historical_data["Close"].pct_change().iloc[-1] * 100,
                "price_change_1w": historical_data["Close"].pct_change(5).iloc[-1] * 100,
                "price_change_1m": historical_data["Close"].pct_change(21).iloc[-1] * 100,
                "volume_avg_30d": historical_data["Volume"].rolling(30).mean().iloc[-1],
                "volatility_30d": historical_data["Close"].pct_change().rolling(30).std().iloc[-1] * 100
            }
        else:
            stats = {}
        
        return {
            "ticker": ticker,
            "predictions": predictions,
            "sentiment": sentiment,
            "statistics": stats,
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metadata for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/", tags=["Root"], summary="Root Endpoint", description="API root endpoint.")
async def root():
    return {"message": "Welcome to the Stock Prediction API"}

def load_all_historical_data_from_mongodb(mongo_client, tickers, start_date, end_date):
    results = {}
    for ticker in tickers:
        df = mongo_client.get_historical_data(ticker, start_date, end_date)
        if df is not None and not df.empty:
            results[ticker] = df
    return results

@app.post("/api/v1/train", tags=["Training"], summary="Train All Models", description="Train all models for all tickers using existing data in MongoDB.")
async def train_all_models():
    """
    Train all models for all tickers using existing data in MongoDB. Does NOT fetch new sentiment or ingest new data.
    """
    logger.info("Starting model training pipeline (no ingestion, no sentiment fetch)...")
    try:
        # Step 1: Load all historical data from MongoDB
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=HISTORICAL_DATA_YEARS * 365)
        historical_data = load_all_historical_data_from_mongodb(
            app.state.mongo_client, TOP_100_TICKERS, start_date, end_date
        )
        logger.info(f"Loaded historical data for tickers: {list(historical_data.keys())}")
        for t, df in historical_data.items():
            logger.info(f"{t}: {df.shape if df is not None else 'None'}")
        # Step 2: Train and predict using MongoDB data
        logger.info("Starting model training for all tickers...")
        app.state.stock_predictor.train_all_models(historical_data)
        logger.info("Model training completed.")
        for ticker, df in historical_data.items():
            if df is not None and not df.empty:
                # Rename OHLCV columns to standard names before feature engineering
                df = app.state.data_ingestion.rename_ohlcv_columns(df, ticker)
                # Ensure 'date' is a column and is datetime
                if 'date' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                if 'date' not in df.columns and 'index' in df.columns:
                    df = df.rename(columns={'index': 'date'})
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                features, _ = app.state.feature_engineer.prepare_features(df)
                if features is not None and features.size > 0:
                    try:
                        latest_features = features[-1]
                        # Try to get the latest price from Alpha Vantage quote endpoint
                        av_quote = app.state.mongo_client.get_alpha_vantage_data(ticker, 'quote')
                        raw_current_price = None
                        if av_quote and 'Global Quote' in av_quote and '05. price' in av_quote['Global Quote']:
                            try:
                                raw_current_price = float(av_quote['Global Quote']['05. price'])
                                logger.info(f"Using Alpha Vantage quote price for {ticker}: {raw_current_price}")
                            except Exception as e:
                                logger.warning(f"Could not parse Alpha Vantage price for {ticker}: {e}")
                        if raw_current_price is None:
                            if 'Close' in df.columns:
                                raw_current_price = float(df['Close'].iloc[-1])
                                logger.info(f"Using last Close price for {ticker}: {raw_current_price}")
                            else:
                                close_idx = app.state.feature_engineer.feature_columns.index('Close') if hasattr(app.state.feature_engineer, 'feature_columns') and 'Close' in app.state.feature_engineer.feature_columns else -1
                                raw_current_price = float(latest_features[close_idx]) if close_idx != -1 else None
                                logger.info(f"Using features array for Close price for {ticker}: {raw_current_price}")
                        predictions = app.state.stock_predictor.predict_all_windows(latest_features, ticker, raw_current_price=raw_current_price)
                        # Ensure predictions dict contains only floats
                        for window, vals in predictions.items():
                            try:
                                predictions[window]["prediction"] = float(predictions[window]["prediction"])
                            except Exception:
                                predictions[window]["prediction"] = 0.0
                            try:
                                predictions[window]["confidence"] = float(predictions[window]["confidence"])
                            except Exception:
                                predictions[window]["confidence"] = 0.0
                        logger.info(f"Storing predictions for {ticker}: {predictions}")
                        if predictions and all("prediction" in v and "confidence" in v for v in predictions.values()):
                            result = app.state.mongo_client.store_predictions(ticker, predictions)
                            if not result:
                                logger.error(f"Failed to store predictions for {ticker}")
                        else:
                            logger.warning(f"Predictions for {ticker} are empty or malformed: {predictions}")
                    except Exception as e:
                        logger.error(f"Error predicting/storing for {ticker}: {str(e)}")
                else:
                    logger.warning(f"No features for {ticker}, skipping prediction.")
            else:
                logger.warning(f"No historical data for {ticker}, skipping.")
        logger.info("Training and prediction pipeline completed successfully.")
        return {"status": "Training and prediction completed (no ingestion, no sentiment fetch)"}
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise HTTPException(status_code=500, detail="Training failed")

@app.get("/api/v1/predictions/{ticker}", tags=["Predictions"], summary="Get Predictions", description="Get model predictions for a given ticker.")
async def get_predictions(ticker: str) -> Dict:
    """Get predictions for a specific ticker."""
    try:
        if ticker not in TOP_100_TICKERS:
            raise HTTPException(status_code=404, detail="Ticker not found in S&P 100")
        # Get latest predictions
        predictions = app.state.mongo_client.get_latest_predictions(ticker)
        if not predictions:
            # Generate new predictions
            df = app.state.mongo_client.get_historical_data(ticker)
            if df is None or df.empty:
                raise HTTPException(status_code=404, detail="No historical data available")
            features, _ = app.state.feature_engineer.prepare_features(df)
            if features is None or features.size == 0:
                raise HTTPException(status_code=404, detail="No features available for prediction")
            latest_features = features[-1]
            predictions = app.state.stock_predictor.predict_all_windows(latest_features)
            app.state.mongo_client.store_predictions(ticker, predictions)
        return predictions
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting predictions for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/sentiment/{ticker}", tags=["Sentiment"], summary="Get Sentiment", description="Get sentiment analysis for a given ticker.")
async def get_sentiment(ticker: str) -> Dict:
    """Get sentiment analysis for a specific ticker."""
    try:
        if ticker not in TOP_100_TICKERS:
            raise HTTPException(status_code=404, detail="Ticker not found in S&P 100")
        sentiment = app.state.mongo_client.get_latest_sentiment(ticker)
        if not sentiment:
            sentiment = app.state.sentiment_analyzer.get_combined_sentiment(ticker)
        return sentiment
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sentiment for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/historical/{ticker}", tags=["Historical Data"], summary="Get Historical Data", description="Get historical OHLCV and related data for a given ticker.")
async def get_historical(ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict:
    """Get historical data for a specific ticker."""
    try:
        if ticker not in TOP_100_TICKERS:
            raise HTTPException(status_code=404, detail="Ticker not found in S&P 100")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        data = app.state.mongo_client.get_historical_data(ticker, start_dt, end_dt)
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail="No historical data available")
        return data.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting historical data for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

def build_explanation_prompt(ticker, date, sentiment, prediction, technicals=None, shap_top_factors=None, news=None):
    """Build a comprehensive explanation prompt using all available data"""
    try:
        # Get sentiment breakdown
        sentiment_sources = sentiment.get('sources', {})
        sentiment_breakdown = []
        for source, data in sentiment_sources.items():
            if data and isinstance(data, dict):
                score = data.get('sentiment_score', 0)
                volume = data.get('volume', 0)
                confidence = data.get('confidence', 0)
                if score is not None and volume > 0:
                    sentiment_breakdown.append(f"{source}: {score:.2f} (volume: {volume}, confidence: {confidence:.2f})")

        # Get technical analysis
        technical_analysis = []
        if technicals:
            for indicator, value in technicals.items():
                if value is not None:
                    technical_analysis.append(f"{indicator}: {value:.2f}")

        # Get news highlights
        news_highlights = []
        if news:
            for source, articles in news.items():
                if articles and isinstance(articles, list):
                    for article in articles[:3]:  # Top 3 articles per source
                        if article.get('title') and article.get('sentiment'):
                            news_highlights.append(f"{source}: {article['title']} (sentiment: {article['sentiment']:.2f})")

        # Get SHAP feature importance
        feature_importance = []
        if shap_top_factors:
            for factor, importance in shap_top_factors.items():
                feature_importance.append(f"{factor}: {importance:.2f}")

        # Build the prompt
        prompt = f"""
        Analyze the following stock prediction for {ticker} on {date}:

        Prediction: {prediction}

        Sentiment Analysis:
        {chr(10).join(sentiment_breakdown)}

        Technical Analysis:
        {chr(10).join(technical_analysis)}

        News Highlights:
        {chr(10).join(news_highlights)}

        Key Factors:
        {chr(10).join(feature_importance)}

        Please provide a comprehensive explanation of the prediction, considering:
        1. The overall market sentiment and its sources
        2. Technical indicators and their implications
        3. Recent news and events affecting the stock
        4. The most influential factors in the prediction
        5. Potential risks and opportunities
        """

        return prompt

    except Exception as e:
        logger.error(f"Error building explanation prompt: {e}")
        return f"Analyze the prediction for {ticker} on {date}: {prediction}"

def get_top_shap_features(shap_values, feature_names, top_n=3):
    """Get top SHAP features with proper validation"""
    try:
        if shap_values is None or feature_names is None:
            return {}

        # Ensure correct shape
        if len(shap_values.shape) > 1:
            shap_values = shap_values.reshape(-1)

        # Get absolute values and sort
        feature_importance = dict(zip(feature_names, np.abs(shap_values)))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        # Return top N features
        return dict(sorted_features[:top_n])

    except Exception as e:
        logger.error(f"Error getting top SHAP features: {e}")
        return {}

@app.get("/api/v1/explain/{ticker}/{date}", tags=["Explainability"], summary="Explain Prediction", description="Get SHAP-based explanation for a prediction on a given date.")
async def explain_prediction(ticker: str, date: str, model: str = Query("gemini-pro", description="LLM provider: gemini-pro (Google Gemini Pro only)")):
    """
    Generate a natural-language explanation for a prediction using stored sentiment/news data, technicals, SHAP, and Google Gemini Pro (Gemini LLM).
    """
    try:
        # Fetch sentiment and prediction data for the given date/ticker
        sentiment = app.state.mongo_client.get_sentiment_data(ticker, date)
        prediction = app.state.mongo_client.get_prediction(ticker, date)
        if not sentiment or not prediction:
            raise HTTPException(status_code=404, detail="No sentiment or prediction data available for this date/ticker.")

        # Fetch technical indicators (last available)
        df = app.state.mongo_client.get_historical_data(ticker)
        technicals = None
        if df is not None and not df.empty:
            tech_cols = [col for col in df.columns if col not in ["date", "ticker"]]
            last_row = df.iloc[-1]
            technicals = {col: last_row[col] for col in tech_cols if col in last_row}

        # Fetch SHAP values (top 3 features)
        features, _ = app.state.feature_engineer.prepare_features(df)
        latest_features = features[-1:]
        window = list(prediction.keys())[0] if prediction else 'next_day'
        shap_result = app.state.stock_predictor.explain_prediction(latest_features, window)
        feature_names = getattr(app.state.feature_engineer, 'feature_columns', df.columns.tolist())
        shap_top_factors = get_top_shap_features(shap_result.get('shap_values', np.zeros(len(feature_names))), feature_names)

        # Fetch recent news (from sentiment/news data)
        news = {}
        for source in ['rss_news', 'seeking_alpha', 'yahoo_news', 'marketaux']:
            if source in sentiment and isinstance(sentiment[source], list):
                news[source] = sentiment[source][:3]

        # Compose prompt
        prompt = build_explanation_prompt(ticker, date, sentiment, prediction, technicals, shap_top_factors, news)
        # Call Gemini Pro LLM
        explanation = await call_gemini_api(prompt)
        return {
            "ticker": ticker,
            "date": date,
            "explanation": explanation,
            "sentiment": sentiment,
            "prediction": prediction,
            "technicals": technicals,
            "shap_top_factors": shap_top_factors,
            "news": news
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating explanation for {ticker} on {date}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate explanation.")

# --- Gemini Pro LLM call ---
async def call_gemini_api(prompt: str) -> str:
    """
    Call Google Gemini Pro (via google-generativeai SDK) to generate a natural-language explanation.
    Requires GOOGLE_API_KEY in .env.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "[Google Gemini API key not set. Cannot generate explanation.]"
    try:
        genai.configure(api_key=api_key)
        # Gemini Pro model name is "gemini-pro"
        model = genai.GenerativeModel("gemini-pro")
        # Use async generation
        response = await run_in_threadpool(model.generate_content, prompt)
        if hasattr(response, 'text'):
            return response.text.strip()
        elif hasattr(response, 'candidates') and response.candidates:
            return response.candidates[0].text.strip()
        else:
            return "[No explanation generated by Gemini Pro.]"
    except Exception as e:
        logger.error(f"Error calling Gemini Pro: {str(e)}")
        return f"[Error generating explanation: {str(e)}]"

@app.post("/api/v1/ingest", tags=["Data Ingestion"], summary="Ingest Data", description="Ingest historical data for all tickers or a specific ticker.")
async def ingest_data(ticker: Optional[str] = None):
    """Ingest historical data for all tickers or a specific ticker."""
    try:
        if ticker:
            logger.info(f"Ingesting data for {ticker} via API endpoint...")
            result = await run_in_threadpool(app.state.data_ingestion.fetch_historical_data, ticker)
            if result is not None:
                return {"status": "success", "message": f"Ingested data for {ticker}", "rows": len(result)}
            else:
                raise HTTPException(status_code=500, detail=f"Failed to ingest data for {ticker}")
        else:
            logger.info("Ingesting data for all tickers via API endpoint...")
            result = await run_in_threadpool(app.state.data_ingestion.fetch_all_tickers)
            return {"status": "success", "message": f"Ingested data for {len(result)} tickers"}
    except Exception as e:
        logger.error(f"Error in /api/v1/ingest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/sentiment", tags=["Sentiment"], summary="Fetch Sentiment", description="Fetch and store sentiment data for all tickers or a specific ticker.")
async def fetch_sentiment(ticker: Optional[str] = None):
    """Fetch and store sentiment for all tickers or a specific ticker."""
    try:
        if ticker:
            logger.info(f"Fetching sentiment for {ticker} via API endpoint...")
            await app.state.sentiment_analyzer.get_combined_sentiment(ticker, force_refresh=True)
            return {"status": "success", "message": f"Fetched and stored sentiment for {ticker}"}
        else:
            logger.info("Fetching sentiment for all tickers via API endpoint...")
            tasks = [app.state.sentiment_analyzer.get_combined_sentiment(t, force_refresh=True) for t in TOP_100_TICKERS]
            await asyncio.gather(*tasks)
            return {"status": "success", "message": "Fetched and stored sentiment for all tickers"}
    except Exception as e:
        logger.error(f"Error in /api/v1/sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 