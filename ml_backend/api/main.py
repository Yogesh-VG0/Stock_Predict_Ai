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
import traceback
from starlette.concurrency import run_in_threadpool
import json
import httpx
import pandas as pd
import google.generativeai as genai
import numpy as np
from ml_backend.data.economic_calendar import EconomicCalendar
import asyncio
import sys
import random

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
    HISTORICAL_DATA_YEARS,
    MONGODB_URI
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
    version="2.0.0"
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

# Global instances
mongo_client = None
predictor = None

@app.on_event("startup")
async def startup():
    """Initialize rate limiter, DB, models, and other components on startup."""
    try:
        logger.info("Starting API initialization...")
        
        # Initialize MongoDB client and other components
        logger.info("Initializing MongoDB client...")
        mongo_client = MongoDBClient(MONGODB_URI)
        if mongo_client.db is None:
            raise Exception("Failed to connect to MongoDB")
        app.state.mongo_client = mongo_client
        logger.info("MongoDB connected successfully")
        
        logger.info("Initializing data ingestion...")
        app.state.data_ingestion = DataIngestion(mongo_client)
        
        logger.info("Initializing economic calendar...")
        app.state.calendar_fetcher = EconomicCalendar(app.state.mongo_client)
        
        logger.info("Initializing sentiment analyzer...")
        app.state.sentiment_analyzer = SentimentAnalyzer(mongo_client, calendar_fetcher=app.state.calendar_fetcher)
        
        logger.info("Initializing feature engineer...")
        app.state.feature_engineer = FeatureEngineer(
            sentiment_analyzer=app.state.sentiment_analyzer,
            mongo_client=app.state.mongo_client,
            calendar_fetcher=app.state.calendar_fetcher
        )
        
        logger.info("Initializing stock predictor...")
        predictor = StockPredictor(app.state.mongo_client)
        predictor.set_feature_engineer(app.state.feature_engineer)
        app.state.stock_predictor = predictor
        
        if redis_client is not None:
            logger.info("Initializing Redis...")
            await redis_client.ping()
            await FastAPILimiter.init(redis_client)
        else:
            logger.info("Redis not configured, skipping rate limiting")
            
        logger.info("Loading models...")
        predictor.load_models()
        
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"CRITICAL: Error during startup: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Set default values to prevent attribute errors
        if not hasattr(app.state, 'sentiment_analyzer'):
            app.state.sentiment_analyzer = None
        if not hasattr(app.state, 'mongo_client'):
            app.state.mongo_client = None
        raise e  # Re-raise to prevent app from starting with broken state
        
@app.on_event("shutdown")
async def shutdown():
    """Clean up resources on shutdown."""
    try:
        if hasattr(app.state, 'stock_predictor') and app.state.stock_predictor:
            app.state.stock_predictor.save_models()
        if hasattr(app.state, 'mongo_client') and app.state.mongo_client:
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

class PredictionRequest(BaseModel):
    ticker: str
    days_back: Optional[int] = 252  # 1 year of trading days

class TrainingRequest(BaseModel):
    ticker: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    retrain: Optional[bool] = False

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
    return {"message": "Stock Prediction API v2.0 - Enhanced with consistent feature engineering"}

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
                        predictions = app.state.stock_predictor.predict_all_windows(ticker, df)
                        # Ensure predictions dict contains valid values before storing
                        if predictions:
                            for window, vals in predictions.items():
                                # Validate prediction structure
                                if not isinstance(vals, dict):
                                    logger.error(f"Invalid prediction structure for {ticker}-{window}: {vals}")
                                    continue
                                    
                                # Ensure prediction value exists and is valid
                                if "price_change" in vals:
                                    try:
                                        predictions[window]["prediction"] = float(vals["price_change"])
                                    except (ValueError, TypeError) as e:
                                        logger.error(f"Invalid price_change for {ticker}-{window}: {vals['price_change']}, error: {e}")
                                        predictions[window]["prediction"] = 0.0
                                elif "prediction" in vals:
                                    try:
                                        predictions[window]["prediction"] = float(vals["prediction"])
                                    except (ValueError, TypeError) as e:
                                        logger.error(f"Invalid prediction for {ticker}-{window}: {vals['prediction']}, error: {e}")
                                        predictions[window]["prediction"] = 0.0
                                else:
                                    logger.error(f"No prediction or price_change found for {ticker}-{window}: {vals}")
                                    predictions[window]["prediction"] = 0.0
                                
                                # Ensure confidence value exists and is valid
                                try:
                                    predictions[window]["confidence"] = float(vals.get("confidence", 0.0))
                                except (ValueError, TypeError) as e:
                                    logger.error(f"Invalid confidence for {ticker}-{window}: {vals.get('confidence')}, error: {e}")
                                    predictions[window]["confidence"] = 0.0
                        logger.info(f"Storing complete predictions for {ticker}: {predictions}")
                        if predictions:
                            # Store complete prediction data (the new store_predictions method handles all fields)
                            result = app.state.mongo_client.store_predictions(ticker, predictions)
                            if result:
                                logger.info(f"Successfully stored complete predictions for {ticker}")
                            else:
                                logger.error(f"Failed to store predictions for {ticker}")
                        else:
                            logger.warning(f"No predictions generated for {ticker}")
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
        # Check if sentiment analyzer is available
        if not hasattr(app.state, 'sentiment_analyzer') or app.state.sentiment_analyzer is None:
            logger.error("Sentiment analyzer not initialized during startup")
            raise HTTPException(status_code=503, detail="Sentiment analyzer service unavailable")
        
        if ticker:
            logger.info(f"Fetching sentiment for {ticker} via API endpoint...")
            await app.state.sentiment_analyzer.get_combined_sentiment(ticker, force_refresh=True)
            return {"status": "success", "message": f"Fetched and stored sentiment for {ticker}"}
        else:
            logger.info("Fetching sentiment for all tickers via API endpoint with sequential processing...")
            results = []
            for i, ticker in enumerate(TOP_100_TICKERS):
                try:
                    logger.info(f"Processing ticker {i+1}/{len(TOP_100_TICKERS)}: {ticker}")
                    result = await app.state.sentiment_analyzer.get_combined_sentiment(ticker, force_refresh=True)
                    results.append({"ticker": ticker, "status": "success"})
                    
                    # Add delay between tickers to prevent rate limiting
                    if i < len(TOP_100_TICKERS) - 1:
                        await asyncio.sleep(random.uniform(2, 3))  # Increased from 1 second
                        
                except Exception as e:
                    logger.error(f"Error processing sentiment for {ticker}: {e}")
                    results.append({"ticker": ticker, "status": "error", "error": str(e)})
            
            success_count = len([r for r in results if r["status"] == "success"])
            return {
                "status": "completed", 
                "message": f"Processed {success_count}/{len(TOP_100_TICKERS)} tickers successfully",
                "results": results
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /api/v1/sentiment: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/optimization/insights/{ticker}")
async def get_optimization_insights(ticker: str):
    """Get optimization insights and feature importance for a ticker."""
    try:
        # Use shared MongoDB client
        if not hasattr(app.state, 'mongo_client') or app.state.mongo_client is None:
            raise HTTPException(status_code=500, detail="Database connection not available")
        mongo_client = app.state.mongo_client
        
        # Get feature importance
        feature_importance_doc = mongo_client.db['feature_importance'].find_one(
            {'ticker': ticker.upper()},
            sort=[('timestamp', -1)]
        )
        
        # Get data utilization stats
        collections_stats = {}
        for collection_name in ['sentiment_data', 'sec_filings', 'short_interest_data', 'seeking_alpha_sentiment']:
            try:
                count = mongo_client.db[collection_name].count_documents({'ticker': ticker.upper()})
                collections_stats[collection_name] = count
            except:
                collections_stats[collection_name] = 0
        
        # Get API cache efficiency
        cache_stats = mongo_client.db['api_cache'].aggregate([
            {
                '$match': {
                    'cache_key': {'$regex': ticker.upper()}
                }
            },
            {
                '$group': {
                    '_id': None,
                    'total_cached_requests': {'$sum': 1},
                    'avg_cache_age_hours': {
                        '$avg': {
                            '$divide': [
                                {'$subtract': [datetime.utcnow(), '$timestamp']},
                                3600000  # Convert to hours
                            ]
                        }
                    }
                }
            }
        ])
        
        cache_efficiency = list(cache_stats)
        
        optimization_insights = {
            'ticker': ticker.upper(),
            'feature_importance': feature_importance_doc.get('feature_scores', {}) if feature_importance_doc else {},
            'total_features_tracked': feature_importance_doc.get('total_features', 0) if feature_importance_doc else 0,
            'last_optimization': feature_importance_doc.get('timestamp') if feature_importance_doc else None,
            'data_utilization': {
                'sentiment_records': collections_stats.get('sentiment_data', 0),
                'sec_filings': collections_stats.get('sec_filings', 0),
                'short_interest_records': collections_stats.get('short_interest_data', 0),
                'seeking_alpha_records': collections_stats.get('seeking_alpha_sentiment', 0),
            },
            'cache_efficiency': {
                'total_cached_requests': cache_efficiency[0].get('total_cached_requests', 0) if cache_efficiency else 0,
                'avg_cache_age_hours': cache_efficiency[0].get('avg_cache_age_hours', 0) if cache_efficiency else 0,
            },
            'optimization_recommendations': []
        }
        
        # Add optimization recommendations
        if collections_stats.get('sentiment_data', 0) < 10:
            optimization_insights['optimization_recommendations'].append(
                "Low sentiment data volume - consider increasing sentiment analysis frequency"
            )
        
        if collections_stats.get('sec_filings', 0) == 0:
            optimization_insights['optimization_recommendations'].append(
                "No SEC filings data found - this is a valuable data source for prediction accuracy"
            )
        
        if feature_importance_doc and len(feature_importance_doc.get('feature_scores', {})) < 20:
            optimization_insights['optimization_recommendations'].append(
                "Low feature count - consider adding more external data sources"
            )
        
        # Don't close shared connection
        
        return optimization_insights
        
    except Exception as e:
        logger.error(f"Error getting optimization insights for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting insights: {str(e)}")

@app.post("/optimization/trigger/{ticker}")
async def trigger_optimization(ticker: str):
    """Trigger a comprehensive data optimization for a ticker."""
    try:
        # Use shared components
        if not hasattr(app.state, 'mongo_client') or app.state.mongo_client is None:
            raise HTTPException(status_code=500, detail="Database connection not available")
        mongo_client = app.state.mongo_client
        
        # Create advanced indexes
        mongo_client.create_advanced_indexes()
        
        # Use shared sentiment analyzer
        if not hasattr(app.state, 'sentiment_analyzer') or app.state.sentiment_analyzer is None:
            raise HTTPException(status_code=500, detail="Sentiment analyzer not available")
        sentiment_data = await app.state.sentiment_analyzer.get_combined_sentiment(ticker.upper())
        
        # Use shared feature engineer
        if not hasattr(app.state, 'feature_engineer') or app.state.feature_engineer is None:
            raise HTTPException(status_code=500, detail="Feature engineer not available")
        optimized_features = app.state.feature_engineer.get_optimized_feature_set(ticker.upper(), mongo_client)
        
        results = {
            'ticker': ticker.upper(),
            'optimization_completed': True,
            'sentiment_updated': bool(sentiment_data),
            'optimized_feature_count': len(optimized_features) if optimized_features else 0,
            'indexes_created': True,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Don't close shared connection
        
        return results
        
    except Exception as e:
        logger.error(f"Error triggering optimization for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Error triggering optimization: {str(e)}")

@app.post("/predict/{ticker}")
async def predict_stock(ticker: str, request: PredictionRequest):
    """
    Get predictions for next day, 7 days, and 30 days with consistent feature engineering.
    """
    try:
        logger.info(f"Prediction request for {ticker}")
        
        # Fetch historical data
        collection = app.state.mongo_client.db.stock_data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.days_back)
        
        query = {
            'ticker': ticker,
            'date': {'$gte': start_date, '$lte': end_date}
        }
        
        cursor = collection.find(query).sort('date', 1)
        data = list(cursor)
        
        if len(data) < 50:
            raise HTTPException(
                status_code=404, 
                detail=f"Insufficient historical data for {ticker}. Found {len(data)} records, need at least 50."
            )
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df = df.set_index('date') if 'date' in df.columns else df
        
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Make predictions for all windows
        predictions = app.state.stock_predictor.predict_all_windows(ticker, df)
        
        if not predictions:
            raise HTTPException(
                status_code=500,
                detail=f"No predictions could be generated for {ticker}. Ensure models are trained."
            )
        
        return {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "current_price": float(df['Close'].iloc[-1]),
            "predictions": predictions,
            "data_points_used": len(df),
            "api_version": "2.0.0"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/{ticker}")
async def train_models(ticker: str, request: TrainingRequest):
    """
    Train models for a specific ticker with improved feature engineering.
    """
    try:
        logger.info(f"Training request for {ticker}")
        
        # Check if models exist and retrain is not forced
        model_dir = f"models/{ticker}"
        if os.path.exists(model_dir) and not request.retrain:
            return {
                "ticker": ticker,
                "status": "models_exist",
                "message": f"Models for {ticker} already exist. Use retrain=true to force retraining.",
                "model_directory": model_dir
            }
        
        # Start training
        success = app.state.stock_predictor.train_all_models(
            ticker=ticker,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        if success:
            return {
                "ticker": ticker,
                "status": "success",
                "message": f"Models trained successfully for {ticker}",
                "windows": app.state.stock_predictor.prediction_windows,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Training failed for {ticker}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_available_models():
    """List all available trained models."""
    try:
        models_info = {}
        models_dir = "models"
        
        if os.path.exists(models_dir):
            for ticker_dir in os.listdir(models_dir):
                ticker_path = os.path.join(models_dir, ticker_dir)
                if os.path.isdir(ticker_path):
                    models_info[ticker_dir] = {
                        "windows": [],
                        "model_files": []
                    }
                    
                    for file in os.listdir(ticker_path):
                        if file.endswith(('.h5', '.joblib')):
                            models_info[ticker_dir]["model_files"].append(file)
                            
                            # Extract window from filename
                            for window in ['next_day', '7_day', '30_day']:
                                if window in file and window not in models_info[ticker_dir]["windows"]:
                                    models_info[ticker_dir]["windows"].append(window)
        
        return {
            "available_models": models_info,
            "total_tickers": len(models_info),
            "supported_windows": ["next_day", "7_day", "30_day"]
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test MongoDB connection
        if hasattr(app.state, 'mongo_client') and app.state.mongo_client and app.state.mongo_client.db is not None:
            stats = app.state.mongo_client.db.command("ping")
            mongodb_status = "connected" if stats else "disconnected"
        else:
            mongodb_status = "not_initialized"
        
        return {
            "status": "healthy",
            "mongodb": mongodb_status,
            "timestamp": datetime.now().isoformat(),
            "api_version": "2.0.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/debug/sec-filing-extraction/{ticker}")
async def debug_sec_filing_extraction(ticker: str):
    """Debug endpoint to test SEC filing text extraction for a specific ticker."""
    try:
        if not hasattr(app.state, 'sentiment_analyzer') or app.state.sentiment_analyzer is None:
            return {"error": "Sentiment analyzer not available"}
        
        # Get SEC analyzer
        sec_analyzer = app.state.sentiment_analyzer.sec_analyzer
        if not sec_analyzer:
            return {"error": "SEC analyzer not available"}
            
        logger.info(f"Debugging SEC filing extraction for {ticker}")
        result = await sec_analyzer.analyze_filings_sentiment(ticker, lookback_days=30)
        
        # Get raw filings data from MongoDB to show the difference
        raw_filings = []
        if app.state.mongo_client:
            try:
                collection = app.state.mongo_client.db['sec_filings_raw']
                raw_docs = collection.find(
                    {"ticker": ticker}, 
                    {"form_type": 1, "filing_date": 1, "sentiment_score": 1, "text_content": 1, "processed_at": 1}
                ).sort("processed_at", -1).limit(5)
                raw_filings = [
                    {
                        "form_type": doc.get("form_type", ""),
                        "filing_date": doc.get("filing_date", ""),
                        "sentiment_score": doc.get("sentiment_score", 0),
                        "text_preview": doc.get("text_content", "")[:500] + "..." if doc.get("text_content") else "",
                        "text_length": len(doc.get("text_content", "")),
                        "processed_at": doc.get("processed_at", "")
                    }
                    for doc in raw_docs
                ]
            except Exception as e:
                logger.warning(f"Error fetching raw SEC data: {e}")
        
        return {
            "ticker": ticker,
            "sec_sentiment_result": result,
            "raw_filings_sample": raw_filings,
            "total_raw_filings": len(raw_filings),
            "debug_info": {
                "narrative_extraction_approach": "Targets MD&A, Risk Factors, Business sections for sentiment",
                "extraction_methods": [
                    "1. Section-by-heading extraction (MD&A, Risk Factors, Business Overview)",
                    "2. Content following section headings (h1-h4, div, p, span)",
                    "3. General business narrative as fallback",
                    "4. Minimal XBRL metadata for context"
                ],
                "target_sections": [
                    "Management's Discussion and Analysis",
                    "Risk Factors", 
                    "Business Overview",
                    "Legal Proceedings",
                    "Forward-Looking Statements",
                    "Results of Operations",
                    "Financial Condition"
                ],
                "expected_improvements": {
                    "sentiment_scores": "Should be non-zero with narrative content",
                    "text_length": "Should see >2000 chars of business narrative",
                    "content_quality": "Actual business discussions, not just HTML headers",
                    "section_detection": "Identifies specific SEC filing sections"
                },
                "boilerplate_filtering": "Removes SEC headers, commission info, form references"
            },
            "extraction_validation": {
                "narrative_indicators": ["management", "risk", "business", "operations", "results", "financial condition"],
                "section_numbers": ["Item 1", "Item 1A", "Item 7"],
                "quality_threshold": "500+ chars per section with business keywords"
            },
            "debug_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in SEC filing debug for {ticker}: {e}")
        return {"error": str(e), "ticker": ticker}

@app.get("/api/v1/predictions/{ticker}/complete")
async def get_complete_predictions(ticker: str):
    """Get complete prediction data for a ticker including all fields."""
    try:
        ticker = ticker.upper()
        predictions = app.state.mongo_client.get_latest_predictions(ticker)
        
        if not predictions:
            raise HTTPException(status_code=404, detail=f"No predictions found for {ticker}")
        
        return {
            "ticker": ticker,
            "predictions": predictions,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting complete predictions for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/predictions/{ticker}/explanation")
async def get_prediction_explanation(ticker: str, window: str = "next_day"):
    """Get prediction explanation for a ticker and window."""
    try:
        ticker = ticker.upper()
        explanation = app.state.mongo_client.get_prediction_explanation(ticker, window)
        
        if not explanation:
            raise HTTPException(status_code=404, detail=f"No explanation found for {ticker}-{window}")
        
        return {
            "ticker": ticker,
            "window": window,
            "explanation": explanation,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting prediction explanation for {ticker}-{window}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/predictions/{ticker}/history")
async def get_prediction_history(ticker: str, days: int = 30):
    """Get prediction history for a ticker over specified days."""
    try:
        ticker = ticker.upper()
        history = app.state.mongo_client.get_prediction_history(ticker, days)
        
        return {
            "ticker": ticker,
            "days": days,
            "history": history,
            "count": len(history),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting prediction history for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/predictions/bulk/complete")
async def get_bulk_complete_predictions(tickers: str):
    """Get complete predictions for multiple tickers."""
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(',')]
        
        all_predictions = {}
        for ticker in ticker_list:
            predictions = app.state.mongo_client.get_latest_predictions(ticker)
            if predictions:
                all_predictions[ticker] = predictions
        
        return {
            "tickers": ticker_list,
            "predictions": all_predictions,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting bulk complete predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 