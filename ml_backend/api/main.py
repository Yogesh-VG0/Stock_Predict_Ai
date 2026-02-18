"""
Main FastAPI application for the stock prediction system.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import redis.asyncio as redis
import os
from dotenv import load_dotenv
import logging

from ml_backend.api.errors import setup_error_handling
from ml_backend.api.rate_limiter import (
    InMemoryRateLimiter,
    RateLimitMiddleware,
    cleanup_rate_limiter_task,
)
from ml_backend.api.routes.batch_predictions import router as batch_router
import traceback
from starlette.concurrency import run_in_threadpool
import json
import pandas as pd
import numpy as np
import asyncio
import random
import uuid

from ml_backend.data.economic_calendar import EconomicCalendar

from ml_backend.config.constants import (
    TOP_100_TICKERS,
    HISTORICAL_DATA_YEARS,
    MONGODB_URI,
)
from ml_backend.utils.mongodb import MongoDBClient
from ml_backend.data.ingestion import DataIngestion
from ml_backend.data.sentiment import SentimentAnalyzer
from ml_backend.data.features_minimal import MinimalFeatureEngineer
from ml_backend.models.predictor import StockPredictor
from ml_backend.api.utils import normalize_prediction_dict

# Load environment variables
load_dotenv()


def _normalize_ticker(ticker: str) -> str:
    """Normalize ticker for DB lookups: BRK.B → BRK-B (ML pipeline standard)."""
    return ticker.upper().replace(".", "-")

# Only set up logging once
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stock Prediction API",
    description="API for S&P 100 stock predictions and analysis. Organized endpoints for predictions, sentiment, training, ingestion, and explainability.",
    version="2.0.0"
)

setup_error_handling(app)

# CORS: use CORS_ORIGINS env for prod (e.g. ["https://yourdomain.com"]), else ["*"] for dev
_cors_origins = os.getenv("CORS_ORIGINS", "*")
if _cors_origins != "*":
    _cors_origins = [o.strip() for o in _cors_origins.split(",") if o.strip()]
else:
    _cors_origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize fallback rate limiter (used when Redis unavailable)
fallback_limiter = InMemoryRateLimiter()

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

# Rate limiting with Redis/in-memory fallback
app.add_middleware(
    RateLimitMiddleware,
    redis_client=redis_client,
    fallback_limiter=fallback_limiter,
    limit=100,
    window=3600,
    exclude_paths=["/health", "/docs", "/openapi.json", "/redoc"],
)

# Include batch predictions router
app.include_router(batch_router, prefix="/api/v1/predictions")

# Cache TTL for predictions (seconds)
PREDICTIONS_CACHE_TTL = 60  # 1 minute
PREDICTIONS_CACHE_VERSION = "v1"

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
        
        logger.info("Initializing minimal feature engineer (v1 - leakage-proof)...")
        app.state.feature_engineer = MinimalFeatureEngineer(mongo_client=app.state.mongo_client)
        
        logger.info("Initializing stock predictor...")
        stock_predictor = StockPredictor(app.state.mongo_client)
        stock_predictor.set_feature_engineer(app.state.feature_engineer)
        app.state.stock_predictor = stock_predictor
        
        if redis_client is not None:
            logger.info("Initializing Redis...")
            await redis_client.ping()
            app.state.redis_client = redis_client
        else:
            logger.info("Redis not configured, using in-memory rate limiter only")
            app.state.redis_client = None

        # Start rate limiter cleanup task (prevents memory buildup)
        asyncio.create_task(cleanup_rate_limiter_task(fallback_limiter, interval=3600))
            
        logger.info("Loading models...")
        app.state.stock_predictor.load_models()
        app.state.training_jobs = {}  # Job status: use Redis in production for persistence
        
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
        ticker = _normalize_ticker(ticker)
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


def _run_training_job(job_id: str) -> None:
    """Background task: run full training pipeline. Updates app.state.training_jobs[job_id]."""
    if not hasattr(app.state, "training_jobs") or app.state.training_jobs is None:
        app.state.training_jobs = {}
    jobs = app.state.training_jobs
    try:
        jobs[job_id] = {"status": "running", "progress": 0, "message": "Loading historical data..."}
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=HISTORICAL_DATA_YEARS * 365)
        historical_data = load_all_historical_data_from_mongodb(
            app.state.mongo_client, TOP_100_TICKERS, start_date, end_date
        )
        jobs[job_id] = {"status": "running", "progress": 20, "message": f"Training {len(historical_data)} tickers..."}
        app.state.stock_predictor.train_all_models(historical_data)
        jobs[job_id] = {"status": "running", "progress": 70, "message": "Storing predictions..."}
        total = len(historical_data)
        for i, (ticker, df) in enumerate(historical_data.items()):
            if df is not None and not df.empty:
                df = app.state.data_ingestion.rename_ohlcv_columns(df, ticker)
                if 'date' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                if 'date' not in df.columns and 'index' in df.columns:
                    df = df.rename(columns={'index': 'date'})
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                features, _ = app.state.feature_engineer.prepare_features(df, ticker=ticker, mongo_client=app.state.mongo_client)
                if features is not None and features.size > 0:
                    try:
                        predictions = app.state.stock_predictor.predict_all_windows(ticker, df)
                        predictions = normalize_prediction_dict(predictions)
                        if predictions:
                            app.state.mongo_client.store_predictions(ticker, predictions)
                    except Exception as e:
                        logger.error(f"Error predicting/storing for {ticker}: {str(e)}")
            jobs[job_id] = {
                "status": "running",
                "progress": 70 + int(30 * (i + 1) / max(total, 1)),
                "message": f"Processed {i+1}/{total} tickers",
            }
        jobs[job_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Training and prediction completed",
            "completed_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}")
        jobs[job_id] = {
            "status": "failed",
            "progress": 0,
            "message": str(e),
            "error": str(e),
            "completed_at": datetime.utcnow().isoformat(),
        }


@app.post("/api/v1/train", tags=["Training"], summary="Train All Models", description="Start training job in background. Returns job_id to poll status.")
async def train_all_models(background_tasks: BackgroundTasks) -> Dict:
    """Start training job. Returns job_id immediately. Poll GET /api/v1/train/status/{job_id} for progress."""
    job_id = str(uuid.uuid4())
    if not hasattr(app.state, "training_jobs"):
        app.state.training_jobs = {}
    app.state.training_jobs[job_id] = {"status": "pending", "progress": 0, "message": "Job queued"}
    background_tasks.add_task(_run_training_job, job_id)
    return {"job_id": job_id, "status": "pending", "message": "Training job started. Poll /api/v1/train/status/{job_id} for progress."}


@app.get("/api/v1/train/status/{job_id}", tags=["Training"], summary="Get Training Job Status", description="Get status of a training job.")
async def get_training_status(job_id: str) -> Dict:
    """Get status of a training job."""
    jobs = getattr(app.state, "training_jobs", None) or {}
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

async def _get_predictions_cached(ticker: str) -> Optional[Dict]:
    """Get predictions from Redis cache if available."""
    if redis_client is None:
        return None
    try:
        key = f"predictions:{PREDICTIONS_CACHE_VERSION}:{ticker}"
        cached = await redis_client.get(key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        logger.debug(f"Cache read failed for {ticker}: {e}")
    return None


async def _set_predictions_cache(ticker: str, predictions: Dict) -> None:
    """Store predictions in Redis cache."""
    if redis_client is None or not predictions:
        return
    try:
        key = f"predictions:{PREDICTIONS_CACHE_VERSION}:{ticker}"
        await redis_client.setex(key, PREDICTIONS_CACHE_TTL, json.dumps(predictions, default=str))
    except Exception as e:
        logger.debug(f"Cache write failed for {ticker}: {e}")


@app.get("/api/v1/predictions/{ticker}", tags=["Predictions"], summary="Get Predictions", description="Get model predictions for a given ticker.")
async def get_predictions(ticker: str) -> Dict:
    """Get predictions for a specific ticker."""
    try:
        ticker = _normalize_ticker(ticker)
        if ticker not in TOP_100_TICKERS:
            raise HTTPException(status_code=404, detail="Ticker not found in S&P 100")
        cached = await _get_predictions_cached(ticker)
        if isinstance(cached, dict) and "windows" in cached:
            return cached
        predictions = None
        if isinstance(cached, dict) and "windows" not in cached:
            predictions = cached
        else:
            predictions = app.state.mongo_client.get_latest_predictions(ticker)
        if not predictions:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=365 * 2)
            df = app.state.mongo_client.get_historical_data(ticker, start_date, end_date)
            if df is None or df.empty:
                raise HTTPException(status_code=404, detail="No historical data available")
            features, _ = app.state.feature_engineer.prepare_features(df, ticker=ticker, mongo_client=app.state.mongo_client)
            if features is None or features.size == 0:
                raise HTTPException(status_code=404, detail="No features available for prediction")
            predictions = app.state.stock_predictor.predict_all_windows(ticker, df)
            predictions = normalize_prediction_dict(predictions)
            app.state.mongo_client.store_predictions(ticker, predictions)
        windows = normalize_prediction_dict(predictions) if predictions else {}
        result = {
            "ticker": ticker,
            "as_of": datetime.utcnow().isoformat(),
            "windows": windows,
        }
        await _set_predictions_cache(ticker, result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting predictions for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/sentiment/{ticker}", tags=["Sentiment"], summary="Get Sentiment", description="Get sentiment analysis for a given ticker.")
async def get_sentiment(ticker: str) -> Dict:
    """Get sentiment analysis for a specific ticker."""
    try:
        ticker = _normalize_ticker(ticker)
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
        ticker = _normalize_ticker(ticker)
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
        return {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "data": data.to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting historical data for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

def build_comprehensive_explanation_prompt(ticker, date, sentiment, prediction, technicals=None, shap_top_factors=None, news=None):
    """Build a comprehensive explanation prompt using ALL available raw data and intelligence"""
    try:
        prompt_sections = []
        
        # === CORE PREDICTION ANALYSIS ===
        prediction_analysis = f"""
STOCK PREDICTION ANALYSIS FOR {ticker} ON {date}

PREDICTION RESULTS:
{format_prediction_details(prediction)}
"""
        prompt_sections.append(prediction_analysis)
        
        # === COMPREHENSIVE SENTIMENT ANALYSIS ===
        sentiment_analysis = build_detailed_sentiment_analysis(sentiment, ticker)
        prompt_sections.append(sentiment_analysis)
        
        # === TECHNICAL ANALYSIS DEEP DIVE ===
        technical_analysis = build_advanced_technical_analysis(technicals, ticker)
        prompt_sections.append(technical_analysis)
        
        # === NEWS INTELLIGENCE ===
        news_intelligence = build_news_intelligence_analysis(sentiment, ticker)
        prompt_sections.append(news_intelligence)
        
        # === SOCIAL SENTIMENT BREAKDOWN ===
        social_analysis = build_social_sentiment_analysis(sentiment, ticker)
        prompt_sections.append(social_analysis)
        
        # === FUNDAMENTAL DATA ANALYSIS ===
        fundamental_analysis = build_fundamental_analysis(sentiment, ticker)
        prompt_sections.append(fundamental_analysis)
        
        # === INSIDER & INSTITUTIONAL ACTIVITY ===
        insider_analysis = build_insider_activity_analysis(sentiment, ticker)
        prompt_sections.append(insider_analysis)
        
        # === SHAP FEATURE IMPORTANCE ===
        feature_analysis = build_feature_importance_analysis(shap_top_factors, ticker)
        prompt_sections.append(feature_analysis)
        
        # === RISK ASSESSMENT DATA ===
        risk_analysis = build_risk_assessment_data(sentiment, technicals, ticker)
        prompt_sections.append(risk_analysis)
        
        # === SIMPLE FRONTEND-OPTIMIZED PROMPT ===
        instruction_prompt = f"""
ANALYSIS INSTRUCTIONS:
Generate a concise, easy-to-read AI analysis for {ticker} for a trading dashboard. Use ALL the real-time data provided.

Format your response EXACTLY as follows, using markdown for formatting:

💡 **AI Analysis for {ticker}**
**Summary:** [A 1-2 sentence summary of the current situation and the AI's core conclusion.]
**Catalyst:** [Identify the most significant recent event from the data, e.g., "Earnings beat," "Analyst upgrade," or "High volume trend."]

**🔼 Positive Factors**
* [List 2-3 key bullish points from the data. Be specific.]
* [Factor 2]

**🔽 Risk Factors**
* [List 2-3 key bearish points from the data. Be specific.]
* [Factor 2]

**Trend Context:** [Describe today's move in the context of the recent trend, e.g., "Today's upward move reverses a 5-day downtrend on high volume."]

**Suggested Action:** [Provide a clear, actionable suggestion, e.g., "Watch for a breakout above $X resistance," or "Consider entry near $Y support if bullish momentum continues."]

**Scenario to Watch:** [**If confidence is Medium/Low**, add this section. Describe what could change the outlook, e.g., "If RSI breaks 70, the stock may become overbought, signaling a potential pullback."]

**Outlook:** [Choose one: Bullish, Bearish, Neutral] (Confidence: [High, Medium, or Low])
**Rationale Tag:** [Choose relevant tags: Momentum, Value, Growth, Contrarian, High Risk, Event-Driven]

**Key Levels to Watch:**
* **Support:** $[A key support level based on technicals]
* **Resistance:** $[A key resistance level based on technicals]

REQUIREMENTS:
- Use all provided data to inform your analysis. Do NOT invent data.
- Keep the language direct and actionable for a trader.
- Ensure the total response is concise and under 2000 characters.
"""
        
        # Combine all sections
        complete_prompt = "\n".join(prompt_sections) + instruction_prompt
        
        return complete_prompt

    except Exception as e:
        logger.error(f"Error building comprehensive explanation prompt: {e}")
        return f"Analyze the prediction for {ticker} on {date}: {prediction}"

def format_prediction_details(prediction):
    """Format prediction details with confidence and range analysis"""
    try:
        details = []
        for window, pred_data in prediction.items():
            if isinstance(pred_data, dict):
                price = pred_data.get('predicted_price', pred_data.get('prediction', 'N/A'))
                confidence = pred_data.get('confidence', 0)
                change = pred_data.get('price_change', 0)
                current = pred_data.get('current_price', 0)
                
                change_pct = (change / current * 100) if current and change else 0
                confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
                
                details.append(f"  {window.upper()}: ${price:.2f} ({change_pct:+.2f}%) - Confidence: {confidence_level} ({confidence:.2f})")
        
        return "\n".join(details)
    except Exception as e:
        return str(prediction)

def build_detailed_sentiment_analysis(sentiment, ticker):
    """Build comprehensive sentiment analysis using ALL raw data sources"""
    try:
        analysis = ["\n=== COMPREHENSIVE SENTIMENT ANALYSIS ==="]
        
        # Aggregate sentiment overview
        sentiment_sources = sentiment.get('sources', {})
        total_volume = sum(data.get('volume', 0) for data in sentiment_sources.values() if isinstance(data, dict))
        
        analysis.append(f"\nOVERALL SENTIMENT OVERVIEW:")
        analysis.append(f"Total Data Points Analyzed: {total_volume:,}")
        analysis.append(f"Blended Sentiment Score: {sentiment.get('blended_sentiment', 0):.3f}")
        
        # Calculate sentiment distribution
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        # Source-by-source breakdown with details
        for source_name, source_data in sentiment_sources.items():
            if isinstance(source_data, dict):
                score = source_data.get('sentiment_score', 0)
                volume = source_data.get('volume', 0)
                confidence = source_data.get('confidence', 0)
                
                analysis.append(f"\n{source_name.upper()} ANALYSIS:")
                analysis.append(f"  Sentiment: {score:.3f} | Volume: {volume:,} | Confidence: {confidence:.2f}")
                
                if score > 0.1:
                    positive_count += volume
                elif score < -0.1:
                    negative_count += volume
                else:
                    neutral_count += volume
        
        # Overall sentiment distribution
        total_sentiment_items = positive_count + negative_count + neutral_count
        if total_sentiment_items > 0:
            pos_pct = (positive_count / total_sentiment_items) * 100
            neg_pct = (negative_count / total_sentiment_items) * 100
            neu_pct = (neutral_count / total_sentiment_items) * 100
            
            analysis.append(f"\nSENTIMENT DISTRIBUTION:")
            analysis.append(f"  Bullish: {positive_count:,} items ({pos_pct:.1f}%)")
            analysis.append(f"  Bearish: {negative_count:,} items ({neg_pct:.1f}%)")
            analysis.append(f"  Neutral: {neutral_count:,} items ({neu_pct:.1f}%)")
        
        # DETAILED FINVIZ NEWS ANALYSIS
        if 'finviz_raw_data' in sentiment and sentiment['finviz_raw_data']:
            finviz_headlines = sentiment['finviz_raw_data']
            finviz_nlp = sentiment.get('finviz_nlp_results', [])
            
            analysis.append(f"\nFINVIZ NEWS INTELLIGENCE ({len(finviz_headlines)} headlines):")
            
            # Categorize by sentiment
            bullish_headlines = []
            bearish_headlines = []
            neutral_headlines = []
            
            for i, headline in enumerate(finviz_headlines):
                if i < len(finviz_nlp):
                    sent_score = finviz_nlp[i].get('sentiment', 0)
                    headline_data = {'text': headline, 'sentiment': sent_score}
                    
                    if sent_score > 0.3:
                        bullish_headlines.append(headline_data)
                    elif sent_score < -0.3:
                        bearish_headlines.append(headline_data)
                    else:
                        neutral_headlines.append(headline_data)
            
            # Show most impactful headlines
            if bullish_headlines:
                analysis.append(f"  MOST BULLISH ({len(bullish_headlines)} headlines):")
                bullish_headlines.sort(key=lambda x: x['sentiment'], reverse=True)
                for headline in bullish_headlines[:3]:
                    analysis.append(f"    • {headline['text'][:85]}... (Score: {headline['sentiment']:+.2f})")
            
            if bearish_headlines:
                analysis.append(f"  MOST BEARISH ({len(bearish_headlines)} headlines):")
                bearish_headlines.sort(key=lambda x: x['sentiment'])
                for headline in bearish_headlines[:3]:
                    analysis.append(f"    • {headline['text'][:85]}... (Score: {headline['sentiment']:+.2f})")
        
        # DETAILED MARKETAUX ANALYSIS
        if 'marketaux_raw_data' in sentiment and sentiment['marketaux_raw_data']:
            marketaux_articles = sentiment['marketaux_raw_data']
            analysis.append(f"\nMARKETAUX PREMIUM NEWS ANALYSIS ({len(marketaux_articles)} articles):")
            
            high_relevance_articles = []
            for article in marketaux_articles:
                entities = article.get('entities', [])
                if entities:
                    entity = entities[0]
                    match_score = entity.get('match_score', 0)
                    sentiment_score = entity.get('sentiment_score', 0)
                    
                    if match_score > 70:  # High relevance
                        high_relevance_articles.append({
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'relevance': match_score,
                            'sentiment': sentiment_score,
                            'source': article.get('source', ''),
                            'url': article.get('url', '')
                        })
            
            if high_relevance_articles:
                analysis.append(f"  HIGH RELEVANCE ARTICLES ({len(high_relevance_articles)}):")
                # Sort by relevance
                high_relevance_articles.sort(key=lambda x: x['relevance'], reverse=True)
                for article in high_relevance_articles[:3]:
                    analysis.append(f"    • [{article['source']}] {article['title'][:70]}...")
                    analysis.append(f"      Relevance: {article['relevance']:.1f}% | Sentiment: {article['sentiment']:+.3f}")
                    if article['description']:
                        analysis.append(f"      Summary: {article['description'][:100]}...")
        
        # RSS NEWS DETAILED BREAKDOWN
        if 'rss_news_raw_data' in sentiment and sentiment['rss_news_raw_data']:
            rss_articles = sentiment['rss_news_raw_data']
            rss_nlp = sentiment.get('rss_news_nlp_results', [])
            
            analysis.append(f"\nRSS NEWS FEED ANALYSIS ({len(rss_articles)} articles):")
            
            # Source breakdown
            source_stats = {}
            sentiment_by_source = {}
            
            for i, article in enumerate(rss_articles):
                source = article.get('source', 'Unknown')
                if source not in source_stats:
                    source_stats[source] = 0
                    sentiment_by_source[source] = []
                
                source_stats[source] += 1
                
                if i < len(rss_nlp):
                    sent_score = rss_nlp[i].get('sentiment', 0)
                    sentiment_by_source[source].append(sent_score)
            
            analysis.append(f"  NEWS SOURCES BREAKDOWN:")
            for source, count in sorted(source_stats.items(), key=lambda x: x[1], reverse=True)[:5]:
                avg_sentiment = sum(sentiment_by_source[source]) / len(sentiment_by_source[source]) if sentiment_by_source[source] else 0
                analysis.append(f"    {source}: {count} articles (Avg Sentiment: {avg_sentiment:+.2f})")
            
            # Most impactful RSS articles
            impactful_articles = []
            for i, article in enumerate(rss_articles):
                if i < len(rss_nlp):
                    sent_score = rss_nlp[i].get('sentiment', 0)
                    confidence = rss_nlp[i].get('confidence', 0)
                    impact_score = abs(sent_score) * confidence
                    
                    if impact_score > 0.3:  # High impact threshold
                        impactful_articles.append({
                            'title': article.get('title', ''),
                            'sentiment': sent_score,
                            'confidence': confidence,
                            'impact': impact_score,
                            'source': article.get('source', '')
                        })
            
            if impactful_articles:
                analysis.append(f"  HIGH IMPACT ARTICLES ({len(impactful_articles)}):")
                impactful_articles.sort(key=lambda x: x['impact'], reverse=True)
                for article in impactful_articles[:3]:
                    analysis.append(f"    • [{article['source']}] {article['title'][:70]}...")
                    analysis.append(f"      Impact: {article['impact']:.2f} | Sentiment: {article['sentiment']:+.2f} | Confidence: {article['confidence']:.2f}")
        
        return "\n".join(analysis)
        
    except Exception as e:
        logger.error(f"Error building detailed sentiment analysis: {e}")
        return "\nSENTIMENT ANALYSIS: Error processing data"

def build_social_sentiment_analysis(sentiment, ticker):
    """Analyze social media sentiment with comprehensive post analysis"""
    try:
        analysis = ["\n=== SOCIAL MEDIA INTELLIGENCE ==="]
        
        # Reddit comprehensive analysis
        if 'reddit_raw_data' in sentiment and sentiment['reddit_raw_data']:
            posts = sentiment['reddit_raw_data']
            analysis.append(f"\nREDDIT COMPREHENSIVE ANALYSIS ({len(posts)} posts/comments):")
            
            # Categorize by subreddit with detailed metrics
            subreddit_stats = {}
            high_engagement = []
            sentiment_posts = []
            time_analysis = {}
            
            total_upvotes = 0
            total_downvotes = 0
            total_comments = 0
            
            for post in posts:
                subreddit = post.get('subreddit', 'unknown')
                score = post.get('score', 0)
                post_type = post.get('type', 'unknown')
                upvotes = post.get('upvotes', 0)
                downvotes = post.get('downvotes', 0)
                num_comments = post.get('num_comments', 0)
                created_utc = post.get('created_utc', 0)
                
                # Aggregate stats
                total_upvotes += upvotes
                total_downvotes += downvotes
                total_comments += num_comments
                
                # Subreddit breakdown
                if subreddit not in subreddit_stats:
                    subreddit_stats[subreddit] = {
                        'posts': 0, 'total_score': 0, 'total_upvotes': 0, 
                        'total_downvotes': 0, 'total_comments': 0
                    }
                subreddit_stats[subreddit]['posts'] += 1
                subreddit_stats[subreddit]['total_score'] += score
                subreddit_stats[subreddit]['total_upvotes'] += upvotes
                subreddit_stats[subreddit]['total_downvotes'] += downvotes
                subreddit_stats[subreddit]['total_comments'] += num_comments
                
                # High engagement posts (better threshold)
                if score > 5 or upvotes > 10 or num_comments > 5:
                    content = post.get('title', post.get('body', ''))
                    high_engagement.append({
                        'content': content[:150],
                        'score': score,
                        'upvotes': upvotes,
                        'downvotes': downvotes,
                        'comments': num_comments,
                        'subreddit': subreddit,
                        'type': post_type,
                        'engagement_score': score + upvotes + (num_comments * 2)  # Weighted engagement
                    })
                
                # Sentiment analysis from content
                content = post.get('title', '') + ' ' + post.get('body', '')
                if content.strip():
                    # Simple keyword-based sentiment (could be enhanced with NLP)
                    bullish_keywords = ['buy', 'moon', 'rocket', 'bull', 'calls', 'up', 'green', 'gains', 'hodl']
                    bearish_keywords = ['sell', 'crash', 'bear', 'puts', 'down', 'red', 'loss', 'dump', 'short']
                    
                    content_lower = content.lower()
                    bullish_count = sum(1 for word in bullish_keywords if word in content_lower)
                    bearish_count = sum(1 for word in bearish_keywords if word in content_lower)
                    
                    if bullish_count > bearish_count:
                        sentiment_label = 'bullish'
                    elif bearish_count > bullish_count:
                        sentiment_label = 'bearish'
                    else:
                        sentiment_label = 'neutral'
                    
                    sentiment_posts.append({
                        'sentiment': sentiment_label,
                        'content': content[:100],
                        'score': score,
                        'subreddit': subreddit,
                        'bullish_signals': bullish_count,
                        'bearish_signals': bearish_count
                    })
            
            # Overall Reddit metrics
            total_posts = len(posts)
            avg_score = sum(post.get('score', 0) for post in posts) / total_posts if total_posts > 0 else 0
            engagement_ratio = (total_upvotes - total_downvotes) / max(total_upvotes + total_downvotes, 1)
            
            analysis.append(f"\nREDDIT OVERVIEW METRICS:")
            analysis.append(f"  Total Posts/Comments: {total_posts:,}")
            analysis.append(f"  Average Score: {avg_score:.1f}")
            analysis.append(f"  Total Upvotes: {total_upvotes:,}")
            analysis.append(f"  Total Downvotes: {total_downvotes:,}")
            analysis.append(f"  Total Comments: {total_comments:,}")
            analysis.append(f"  Engagement Ratio: {engagement_ratio:.3f} ({'Positive' if engagement_ratio > 0 else 'Negative'})")
            
            # Subreddit breakdown with detailed metrics
            analysis.append(f"\nSUBREDDIT BREAKDOWN:")
            for subreddit, stats in sorted(subreddit_stats.items(), key=lambda x: x[1]['posts'], reverse=True):
                avg_score = stats['total_score'] / stats['posts'] if stats['posts'] > 0 else 0
                avg_upvotes = stats['total_upvotes'] / stats['posts'] if stats['posts'] > 0 else 0
                avg_comments = stats['total_comments'] / stats['posts'] if stats['posts'] > 0 else 0
                
                analysis.append(f"  r/{subreddit}:")
                analysis.append(f"    Posts: {stats['posts']} | Avg Score: {avg_score:.1f}")
                analysis.append(f"    Avg Upvotes: {avg_upvotes:.1f} | Avg Comments: {avg_comments:.1f}")
            
            # Sentiment distribution
            bullish_posts = [p for p in sentiment_posts if p['sentiment'] == 'bullish']
            bearish_posts = [p for p in sentiment_posts if p['sentiment'] == 'bearish']
            neutral_posts = [p for p in sentiment_posts if p['sentiment'] == 'neutral']
            
            if sentiment_posts:
                analysis.append(f"\nREDDIT SENTIMENT BREAKDOWN:")
                analysis.append(f"  Bullish Posts: {len(bullish_posts)} ({len(bullish_posts)/len(sentiment_posts)*100:.1f}%)")
                analysis.append(f"  Bearish Posts: {len(bearish_posts)} ({len(bearish_posts)/len(sentiment_posts)*100:.1f}%)")
                analysis.append(f"  Neutral Posts: {len(neutral_posts)} ({len(neutral_posts)/len(sentiment_posts)*100:.1f}%)")
            
            # High engagement content with better analysis
            if high_engagement:
                analysis.append(f"\nTOP ENGAGEMENT POSTS:")
                high_engagement.sort(key=lambda x: x['engagement_score'], reverse=True)
                for post in high_engagement[:5]:  # Show top 5
                    analysis.append(f"  • r/{post['subreddit']} [{post['score']} score, {post['upvotes']} ↑, {post['comments']} comments]")
                    analysis.append(f"    {post['content'][:120]}...")
            
            # Most bullish and bearish posts
            if bullish_posts:
                top_bullish = sorted(bullish_posts, key=lambda x: x['score'], reverse=True)[:2]
                analysis.append(f"\nTOP BULLISH POSTS:")
                for post in top_bullish:
                    analysis.append(f"  • [{post['score']} score] {post['content'][:100]}...")
                    analysis.append(f"    Bullish signals: {post['bullish_signals']} | r/{post['subreddit']}")
            
            if bearish_posts:
                top_bearish = sorted(bearish_posts, key=lambda x: x['score'], reverse=True)[:2]
                analysis.append(f"\nTOP BEARISH POSTS:")
                for post in top_bearish:
                    analysis.append(f"  • [{post['score']} score] {post['content'][:100]}...")
                    analysis.append(f"    Bearish signals: {post['bearish_signals']} | r/{post['subreddit']}")
        
        # Twitter analysis (if available)
        if 'twitter_raw_data' in sentiment and sentiment['twitter_raw_data']:
            twitter_posts = sentiment['twitter_raw_data']
            analysis.append(f"\nTWITTER ANALYSIS ({len(twitter_posts)} tweets):")
            # Add Twitter analysis similar to Reddit
        
        # Discord analysis (if available)
        if 'discord_raw_data' in sentiment and sentiment['discord_raw_data']:
            discord_posts = sentiment['discord_raw_data']
            analysis.append(f"\nDISCORD ANALYSIS ({len(discord_posts)} messages):")
            # Add Discord analysis similar to Reddit
        
        return "\n".join(analysis)
        
    except Exception as e:
        logger.error(f"Error building social sentiment analysis: {e}")
        return "\nSOCIAL ANALYSIS: Error processing data"

def build_news_intelligence_analysis(sentiment, ticker):
    """Build comprehensive news intelligence using RSS and other sources"""
    try:
        analysis = ["\n=== NEWS INTELLIGENCE ANALYSIS ==="]
        
        # RSS News detailed analysis
        if 'rss_news_raw_data' in sentiment and sentiment['rss_news_raw_data']:
            news_data = sentiment['rss_news_raw_data']
            analysis.append(f"\nRSS NEWS ANALYSIS ({len(news_data)} articles):")
            
            # Categorize by source and sentiment
            source_breakdown = {}
            sentiment_distribution = {'positive': 0, 'negative': 0, 'neutral': 0}
            
            for i, article in enumerate(news_data):
                source = article.get('source', 'Unknown')
                title = article.get('title', '')
                
                if source not in source_breakdown:
                    source_breakdown[source] = {'count': 0, 'titles': []}
                source_breakdown[source]['count'] += 1
                source_breakdown[source]['titles'].append(title)
                
                # Get sentiment from NLP results
                if 'rss_news_nlp_results' in sentiment and i < len(sentiment['rss_news_nlp_results']):
                    nlp_result = sentiment['rss_news_nlp_results'][i]
                    sent_score = nlp_result.get('sentiment', 0)
                    confidence = nlp_result.get('confidence', 0)
                    
                    if sent_score > 0.3:
                        sentiment_distribution['positive'] += 1
                    elif sent_score < -0.3:
                        sentiment_distribution['negative'] += 1
                    else:
                        sentiment_distribution['neutral'] += 1
            
            # Source breakdown
            analysis.append(f"\nNEWS SOURCE BREAKDOWN:")
            for source, data in source_breakdown.items():
                analysis.append(f"  {source}: {data['count']} articles")
            
            # Sentiment distribution
            total_articles = sum(sentiment_distribution.values())
            if total_articles > 0:
                pos_pct = (sentiment_distribution['positive'] / total_articles) * 100
                neg_pct = (sentiment_distribution['negative'] / total_articles) * 100
                neu_pct = (sentiment_distribution['neutral'] / total_articles) * 100
                
                analysis.append(f"\nNEWS SENTIMENT DISTRIBUTION:")
                analysis.append(f"  Positive: {sentiment_distribution['positive']} ({pos_pct:.1f}%)")
                analysis.append(f"  Negative: {sentiment_distribution['negative']} ({neg_pct:.1f}%)")
                analysis.append(f"  Neutral: {sentiment_distribution['neutral']} ({neu_pct:.1f}%)")
            
            # Key headlines by sentiment
            if 'rss_news_nlp_results' in sentiment:
                analysis.append(f"\nKEY HEADLINES BY SENTIMENT:")
                
                # Get most positive and negative headlines
                sentiment_headlines = []
                for i, nlp_result in enumerate(sentiment['rss_news_nlp_results']):
                    if i < len(news_data):
                        title = news_data[i].get('title', '')
                        sentiment_score = nlp_result.get('sentiment', 0)
                        confidence = nlp_result.get('confidence', 0)
                        sentiment_headlines.append({
                            'title': title,
                            'sentiment': sentiment_score,
                            'confidence': confidence
                        })
                
                # Sort by sentiment
                sentiment_headlines.sort(key=lambda x: x['sentiment'], reverse=True)
                
                # Most positive
                if sentiment_headlines and sentiment_headlines[0]['sentiment'] > 0:
                    analysis.append(f"  MOST POSITIVE: {sentiment_headlines[0]['title'][:80]}... (Sentiment: {sentiment_headlines[0]['sentiment']:+.2f})")
                
                # Most negative
                if sentiment_headlines and sentiment_headlines[-1]['sentiment'] < 0:
                    analysis.append(f"  MOST NEGATIVE: {sentiment_headlines[-1]['title'][:80]}... (Sentiment: {sentiment_headlines[-1]['sentiment']:+.2f})")
        
        return "\n".join(analysis)
        
    except Exception as e:
        logger.error(f"Error building news intelligence analysis: {e}")
        return "\nNEWS INTELLIGENCE: Error processing data"

def build_fundamental_analysis(sentiment, ticker):
    """Build comprehensive fundamental analysis using ALL available financial data"""
    try:
        analysis = ["\n=== COMPREHENSIVE FUNDAMENTAL ANALYSIS ==="]
        
        # FMP data comprehensive analysis
        if 'fmp_raw_data' in sentiment and sentiment['fmp_raw_data']:
            fmp_data = sentiment['fmp_raw_data']
            
            # Detailed earnings analysis with trends
            if 'company_earnings' in fmp_data and fmp_data['company_earnings']:
                earnings = fmp_data['company_earnings']
                analysis.append(f"\nEARNINGS PERFORMANCE ANALYSIS:")
                
                latest_earnings = earnings[0] if earnings else {}
                if latest_earnings.get('epsActual') is not None and latest_earnings.get('epsEstimated') is not None:
                    eps_actual = latest_earnings['epsActual']
                    eps_estimate = latest_earnings['epsEstimated']
                    eps_surprise = ((eps_actual - eps_estimate) / eps_estimate) * 100 if eps_estimate != 0 else 0
                    
                    analysis.append(f"  Latest Quarter (Q{latest_earnings.get('fiscalQuarter', 'N/A')} {latest_earnings.get('fiscalYear', 'N/A')}):")
                    analysis.append(f"    EPS: ${eps_actual:.3f} vs Est. ${eps_estimate:.3f} ({eps_surprise:+.1f}% surprise)")
                    
                    if latest_earnings.get('revenueActual') and latest_earnings.get('revenueEstimated'):
                        rev_actual = latest_earnings['revenueActual'] / 1e9  # Convert to billions
                        rev_estimate = latest_earnings['revenueEstimated'] / 1e9
                        rev_surprise = ((latest_earnings['revenueActual'] - latest_earnings['revenueEstimated']) / latest_earnings['revenueEstimated']) * 100
                        
                        analysis.append(f"    Revenue: ${rev_actual:.2f}B vs Est. ${rev_estimate:.2f}B ({rev_surprise:+.1f}% surprise)")
                    
                    # Earnings trend analysis (if multiple quarters available)
                    if len(earnings) >= 4:
                        eps_growth_rates = []
                        revenue_growth_rates = []
                        
                        for i in range(min(3, len(earnings)-1)):
                            current_eps = earnings[i].get('epsActual', 0)
                            previous_eps = earnings[i+1].get('epsActual', 0)
                            current_rev = earnings[i].get('revenueActual', 0)
                            previous_rev = earnings[i+1].get('revenueActual', 0)
                            
                            if previous_eps != 0:
                                eps_growth = ((current_eps - previous_eps) / previous_eps) * 100
                                eps_growth_rates.append(eps_growth)
                            
                            if previous_rev != 0:
                                rev_growth = ((current_rev - previous_rev) / previous_rev) * 100
                                revenue_growth_rates.append(rev_growth)
                        
                        if eps_growth_rates:
                            avg_eps_growth = sum(eps_growth_rates) / len(eps_growth_rates)
                            analysis.append(f"    Avg EPS Growth (QoQ): {avg_eps_growth:+.1f}%")
                        
                        if revenue_growth_rates:
                            avg_rev_growth = sum(revenue_growth_rates) / len(revenue_growth_rates)
                            analysis.append(f"    Avg Revenue Growth (QoQ): {avg_rev_growth:+.1f}%")
                
                # Beat rate analysis
                beats = 0
                misses = 0
                meets = 0
                
                for earning in earnings[:8]:  # Last 8 quarters
                    if earning.get('epsActual') is not None and earning.get('epsEstimated') is not None:
                        actual = earning['epsActual']
                        estimated = earning['epsEstimated']
                        
                        if actual > estimated:
                            beats += 1
                        elif actual < estimated:
                            misses += 1
                        else:
                            meets += 1
                
                total_reports = beats + misses + meets
                if total_reports > 0:
                    analysis.append(f"\n  EARNINGS TRACK RECORD (Last {total_reports} quarters):")
                    analysis.append(f"    Beats: {beats} ({beats/total_reports*100:.1f}%)")
                    analysis.append(f"    Misses: {misses} ({misses/total_reports*100:.1f}%)")
                    analysis.append(f"    Meets: {meets} ({meets/total_reports*100:.1f}%)")
            
            # Comprehensive dividend analysis
            if 'company_dividends' in fmp_data and fmp_data['company_dividends']:
                dividends = fmp_data['company_dividends']
                analysis.append(f"\nDIVIDEND ANALYSIS:")
                
                if dividends:
                    latest_div = dividends[0]
                    analysis.append(f"  Current Dividend Information:")
                    analysis.append(f"    Dividend per Share: ${latest_div.get('dividend', 0):.3f}")
                    analysis.append(f"    Dividend Yield: {latest_div.get('yield', 0):.2f}%")
                    analysis.append(f"    Ex-Dividend Date: {latest_div.get('exDividendDate', 'N/A')}")
                    analysis.append(f"    Payment Date: {latest_div.get('paymentDate', 'N/A')}")
                    analysis.append(f"    Record Date: {latest_div.get('recordDate', 'N/A')}")
                    
                    # Dividend growth analysis
                    if len(dividends) >= 4:
                        div_amounts = [div.get('dividend', 0) for div in dividends[:12]]  # Last 12 quarters
                        
                        # Calculate annual dividend
                        current_annual = sum(div_amounts[:4])
                        previous_annual = sum(div_amounts[4:8]) if len(div_amounts) >= 8 else 0
                        
                        if previous_annual > 0:
                            annual_growth = ((current_annual - previous_annual) / previous_annual) * 100
                            analysis.append(f"    Annual Dividend Growth: {annual_growth:+.1f}%")
                        
                        # Dividend consistency
                        increasing_quarters = 0
                        for i in range(min(3, len(div_amounts)-1)):
                            if div_amounts[i] >= div_amounts[i+1]:
                                increasing_quarters += 1
                        
                        consistency_rate = (increasing_quarters / min(3, len(div_amounts)-1)) * 100 if len(div_amounts) > 1 else 0
                        analysis.append(f"    Dividend Consistency: {consistency_rate:.1f}%")
            
            # Advanced analyst estimates analysis
            if 'analyst_estimates' in fmp_data and fmp_data['analyst_estimates']:
                estimates = fmp_data['analyst_estimates']
                analysis.append(f"\nANALYST ESTIMATES & FORECASTS:")
                
                for estimate in estimates[:3]:  # Next 3 periods
                    date = estimate.get('date', '')
                    eps_avg = estimate.get('epsAvg', 0)
                    eps_high = estimate.get('epsHigh', 0)
                    eps_low = estimate.get('epsLow', 0)
                    revenue_avg = estimate.get('revenueAvg', 0) / 1e9 if estimate.get('revenueAvg') else 0
                    revenue_high = estimate.get('revenueHigh', 0) / 1e9 if estimate.get('revenueHigh') else 0
                    revenue_low = estimate.get('revenueLow', 0) / 1e9 if estimate.get('revenueLow') else 0
                    analyst_count = estimate.get('numberAnalysts', 0)
                    
                    analysis.append(f"  {date} Forecasts ({analyst_count} analysts):")
                    analysis.append(f"    EPS: ${eps_avg:.3f} (Range: ${eps_low:.3f} - ${eps_high:.3f})")
                    if revenue_avg > 0:
                        analysis.append(f"    Revenue: ${revenue_avg:.2f}B (Range: ${revenue_low:.2f}B - ${revenue_high:.2f}B)")
                    
                    # Calculate estimate dispersion (uncertainty measure)
                    if eps_high > eps_low and eps_avg > 0:
                        eps_dispersion = ((eps_high - eps_low) / eps_avg) * 100
                        analysis.append(f"    EPS Uncertainty: {eps_dispersion:.1f}% range")
            
            # Enhanced price target analysis
            if 'price_target_consensus' in fmp_data and fmp_data['price_target_consensus']:
                pt_data = fmp_data['price_target_consensus'][0]
                analysis.append(f"\nPRICE TARGET CONSENSUS:")
                
                target_consensus = pt_data.get('targetConsensus', 0)
                target_high = pt_data.get('targetHigh', 0)
                target_low = pt_data.get('targetLow', 0)
                target_median = pt_data.get('targetMedian', 0)
                analyst_count = pt_data.get('analystCount', 0)
                
                analysis.append(f"  Consensus Target: ${target_consensus:.2f} ({analyst_count} analysts)")
                analysis.append(f"  Target Range: ${target_low:.2f} - ${target_high:.2f}")
                analysis.append(f"  Median Target: ${target_median:.2f}")
                
                # Calculate target dispersion
                if target_high > target_low and target_consensus > 0:
                    target_dispersion = ((target_high - target_low) / target_consensus) * 100
                    analysis.append(f"  Price Target Dispersion: {target_dispersion:.1f}%")
                    
                    if target_dispersion > 30:
                        analysis.append(f"    HIGH DISPERSION - Significant analyst disagreement")
                    elif target_dispersion < 15:
                        analysis.append(f"    LOW DISPERSION - Strong analyst consensus")
            
            # Financial ratios analysis (if available)
            if 'financial_ratios' in fmp_data and fmp_data['financial_ratios']:
                ratios = fmp_data['financial_ratios']
                analysis.append(f"\nKEY FINANCIAL RATIOS:")
                
                if isinstance(ratios, list) and len(ratios) > 0:
                    latest_ratios = ratios[0]
                    
                    # Valuation ratios
                    pe_ratio = latest_ratios.get('priceEarningsRatio', 0)
                    pb_ratio = latest_ratios.get('priceToBookRatio', 0)
                    ps_ratio = latest_ratios.get('priceToSalesRatio', 0)
                    
                    analysis.append(f"  Valuation Metrics:")
                    analysis.append(f"    P/E Ratio: {pe_ratio:.2f}")
                    analysis.append(f"    P/B Ratio: {pb_ratio:.2f}")
                    analysis.append(f"    P/S Ratio: {ps_ratio:.2f}")
                    
                    # Profitability ratios
                    roe = latest_ratios.get('returnOnEquity', 0)
                    roa = latest_ratios.get('returnOnAssets', 0)
                    gross_margin = latest_ratios.get('grossProfitMargin', 0)
                    operating_margin = latest_ratios.get('operatingProfitMargin', 0)
                    
                    analysis.append(f"  Profitability Metrics:")
                    analysis.append(f"    ROE: {roe:.2f}%")
                    analysis.append(f"    ROA: {roa:.2f}%")
                    analysis.append(f"    Gross Margin: {gross_margin:.2f}%")
                    analysis.append(f"    Operating Margin: {operating_margin:.2f}%")
        
        # SEC Filing analysis (if available)
        if 'sec_raw_data' in sentiment and sentiment['sec_raw_data']:
            sec_data = sentiment['sec_raw_data']
            analysis.append(f"\nSEC FILINGS ANALYSIS:")
            
            # Recent filings summary
            if isinstance(sec_data, list):
                recent_filings = sec_data[:5]  # Last 5 filings
                analysis.append(f"  Recent SEC Filings ({len(recent_filings)}):")
                
                for filing in recent_filings:
                    form_type = filing.get('form_type', 'N/A')
                    filing_date = filing.get('filing_date', 'N/A')
                    sentiment_score = filing.get('sentiment_score', 0)
                    
                    analysis.append(f"    {form_type} filed {filing_date}: Sentiment {sentiment_score:+.3f}")
                    
                    # Extract key insights from filing
                    if filing.get('text_content'):
                        content = filing['text_content'][:200]
                        analysis.append(f"      Summary: {content}...")
        
        return "\n".join(analysis)
        
    except Exception as e:
        logger.error(f"Error building fundamental analysis: {e}")
        return "\nFUNDAMENTAL ANALYSIS: Error processing data"

def build_insider_activity_analysis(sentiment, ticker):
    """Analyze insider trading and institutional activity"""
    try:
        analysis = ["\n=== INSIDER & INSTITUTIONAL ACTIVITY ==="]
        
        # Finnhub insider data
        if 'finnhub_raw_data' in sentiment and sentiment['finnhub_raw_data']:
            finnhub_data = sentiment['finnhub_raw_data']
            
            if 'insider' in finnhub_data:
                insider = finnhub_data['insider']
                total_transactions = insider.get('transactions', 0)
                buy_transactions = insider.get('buy_transactions', 0)
                sell_transactions = insider.get('sell_transactions', 0)
                insider_sentiment = insider.get('sentiment', 0)
                
                analysis.append(f"\nINSIDER TRADING ACTIVITY:")
                analysis.append(f"  Total Transactions: {total_transactions:,}")
                analysis.append(f"  Buy Transactions: {buy_transactions:,}")
                analysis.append(f"  Sell Transactions: {sell_transactions:,}")
                
                buy_sell_ratio = buy_transactions / sell_transactions if sell_transactions > 0 else float('inf')
                if buy_sell_ratio > 1.5:
                    analysis.append(f"  Buy/Sell Ratio: {buy_sell_ratio:.2f} (BULLISH - More buying than selling)")
                elif buy_sell_ratio < 0.67:
                    analysis.append(f"  Buy/Sell Ratio: {buy_sell_ratio:.2f} (BEARISH - More selling than buying)")
                else:
                    analysis.append(f"  Buy/Sell Ratio: {buy_sell_ratio:.2f} (NEUTRAL)")
                
                analysis.append(f"  Insider Sentiment Score: {insider_sentiment:+.2f}")
            
            # Analyst recommendations
            if 'recommendations' in finnhub_data:
                recs = finnhub_data['recommendations']
                if 'finnhub_recommendation_breakdown' in recs:
                    breakdown = recs['finnhub_recommendation_breakdown']
                    total_recs = breakdown.get('total', 0)
                    
                    analysis.append(f"\nANALYST RECOMMENDATIONS ({total_recs} analysts):")
                    analysis.append(f"  Strong Buy: {breakdown.get('strongBuy', 0)}")
                    analysis.append(f"  Buy: {breakdown.get('buy', 0)}")
                    analysis.append(f"  Hold: {breakdown.get('hold', 0)}")
                    analysis.append(f"  Sell: {breakdown.get('sell', 0)}")
                    analysis.append(f"  Strong Sell: {breakdown.get('strongSell', 0)}")
                    
                    # Calculate recommendation score
                    if total_recs > 0:
                        bullish_recs = breakdown.get('strongBuy', 0) + breakdown.get('buy', 0)
                        bearish_recs = breakdown.get('sell', 0) + breakdown.get('strongSell', 0)
                        bullish_pct = (bullish_recs / total_recs) * 100
                        bearish_pct = (bearish_recs / total_recs) * 100
                        
                        analysis.append(f"  Bullish: {bullish_pct:.1f}% | Bearish: {bearish_pct:.1f}%")
        
        # Short interest analysis
        if 'short_interest_data' in sentiment and sentiment['short_interest_data']:
            short_data = sentiment['short_interest_data']
            analysis.append(f"\nSHORT INTEREST ANALYSIS:")
            
            if len(short_data) >= 2:
                latest = short_data[0]
                previous = short_data[1]
                
                latest_interest = latest.get('shortInterest', 0)
                previous_interest = previous.get('shortInterest', 0)
                
                change = latest_interest - previous_interest
                change_pct = (change / previous_interest) * 100 if previous_interest > 0 else 0
                
                analysis.append(f"  Current Short Interest: {latest_interest:,} shares")
                analysis.append(f"  Change from Previous: {change:+,} shares ({change_pct:+.1f}%)")
                analysis.append(f"  Days to Cover: {latest.get('daysToCoVerShortInterest', 0):.1f}")
                
                if change_pct > 10:
                    analysis.append(f"  ALERT: Significant increase in short interest (BEARISH)")
                elif change_pct < -10:
                    analysis.append(f"  ALERT: Significant decrease in short interest (BULLISH)")
        
        return "\n".join(analysis)
        
    except Exception as e:
        logger.error(f"Error building insider activity analysis: {e}")
        return "\nINSIDER ACTIVITY: Error processing data"

def build_advanced_technical_analysis(technicals, ticker):
    """Build advanced technical analysis with specific levels and signals"""
    try:
        analysis = ["\n=== ADVANCED TECHNICAL ANALYSIS ==="]
        
        if not technicals:
            analysis.append("Technical data not available")
            return "\n".join(analysis)
        
        analysis.append(f"\nKEY TECHNICAL INDICATORS:")
        
        # RSI Analysis
        if 'RSI' in technicals:
            rsi = technicals['RSI']
            analysis.append(f"  RSI (14): {rsi:.1f}")
            if rsi > 70:
                analysis.append(f"    Signal: OVERBOUGHT - Potential selling pressure")
            elif rsi < 30:
                analysis.append(f"    Signal: OVERSOLD - Potential buying opportunity")
            elif rsi > 50:
                analysis.append(f"    Signal: BULLISH momentum")
            else:
                analysis.append(f"    Signal: BEARISH momentum")
        
        # MACD Analysis
        if 'MACD' in technicals and 'MACD_Signal' in technicals:
            macd = technicals['MACD']
            macd_signal = technicals['MACD_Signal']
            macd_histogram = macd - macd_signal
            
            analysis.append(f"  MACD: {macd:.3f} | Signal: {macd_signal:.3f} | Histogram: {macd_histogram:+.3f}")
            
            if macd > macd_signal and macd_histogram > 0:
                analysis.append(f"    Signal: BULLISH crossover - Upward momentum")
            elif macd < macd_signal and macd_histogram < 0:
                analysis.append(f"    Signal: BEARISH crossover - Downward momentum")
        
        # Bollinger Bands Analysis
        if all(key in technicals for key in ['Bollinger_Upper', 'Bollinger_Lower', 'Bollinger_Mid']):
            bb_upper = technicals['Bollinger_Upper']
            bb_lower = technicals['Bollinger_Lower']
            bb_mid = technicals['Bollinger_Mid']
            
            # Assume current price is close to bb_mid for analysis
            if 'Close' in technicals:
                current_price = technicals['Close']
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) * 100
                
                analysis.append(f"  Bollinger Bands: Upper ${bb_upper:.2f} | Mid ${bb_mid:.2f} | Lower ${bb_lower:.2f}")
                analysis.append(f"  Price Position: {bb_position:.1f}% of band width")
                
                if bb_position > 80:
                    analysis.append(f"    Signal: Near upper band - Potential resistance")
                elif bb_position < 20:
                    analysis.append(f"    Signal: Near lower band - Potential support")
                else:
                    analysis.append(f"    Signal: Mid-range - No extreme levels")
        
        # Moving Averages Analysis
        moving_averages = []
        for key in technicals:
            if 'SMA' in key or 'EMA' in key:
                moving_averages.append((key, technicals[key]))
        
        if moving_averages:
            analysis.append(f"\n  MOVING AVERAGES:")
            for ma_name, ma_value in sorted(moving_averages):
                analysis.append(f"    {ma_name}: ${ma_value:.2f}")
        
        # Volume Analysis
        if 'Volume' in technicals:
            volume = technicals['Volume']
            analysis.append(f"\n  Current Volume: {volume:,.0f}")
            
            # Add volume trend analysis if available
            if 'Volume_SMA' in technicals:
                volume_sma = technicals['Volume_SMA']
                volume_ratio = volume / volume_sma if volume_sma > 0 else 1
                analysis.append(f"  Volume vs Average: {volume_ratio:.2f}x")
                
                if volume_ratio > 1.5:
                    analysis.append(f"    Signal: HIGH VOLUME - Increased interest")
                elif volume_ratio < 0.5:
                    analysis.append(f"    Signal: LOW VOLUME - Decreased interest")
        
        return "\n".join(analysis)
        
    except Exception as e:
        logger.error(f"Error building advanced technical analysis: {e}")
        return "\nTECHNICAL ANALYSIS: Error processing data"

def build_feature_importance_analysis(shap_top_factors, ticker):
    """Build detailed feature importance analysis"""
    try:
        analysis = ["\n=== MACHINE LEARNING FEATURE IMPORTANCE ==="]
        
        if not shap_top_factors:
            analysis.append("Feature importance data not available")
            return "\n".join(analysis)
        
        analysis.append(f"\nTOP PREDICTIVE FACTORS (SHAP Analysis):")
        
        # Sort factors by importance
        sorted_factors = sorted(shap_top_factors.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for i, (feature, importance) in enumerate(sorted_factors, 1):
            impact = "POSITIVE" if importance > 0 else "NEGATIVE"
            analysis.append(f"  {i}. {feature}: {importance:+.4f} ({impact} impact)")
            
            # Add interpretation based on feature name
            interpretation = interpret_feature_importance(feature, importance)
            if interpretation:
                analysis.append(f"     → {interpretation}")
        
        return "\n".join(analysis)
        
    except Exception as e:
        logger.error(f"Error building feature importance analysis: {e}")
        return "\nFEATURE IMPORTANCE: Error processing data"

def interpret_feature_importance(feature_name, importance):
    """Interpret what a feature importance score means"""
    try:
        feature_lower = feature_name.lower()
        impact = "increases" if importance > 0 else "decreases"
        
        if 'sentiment' in feature_lower:
            if 'reddit' in feature_lower:
                return f"Reddit sentiment {impact} price prediction"
            elif 'news' in feature_lower:
                return f"News sentiment {impact} price prediction"
            else:
                return f"Overall sentiment {impact} price prediction"
        elif 'rsi' in feature_lower:
            return f"RSI momentum indicator {impact} price prediction"
        elif 'macd' in feature_lower:
            return f"MACD trend indicator {impact} price prediction"
        elif 'volume' in feature_lower:
            return f"Trading volume {impact} price prediction"
        elif 'bollinger' in feature_lower:
            return f"Bollinger Bands position {impact} price prediction"
        elif 'sma' in feature_lower or 'ema' in feature_lower:
            return f"Moving average trend {impact} price prediction"
        elif 'short' in feature_lower:
            return f"Short interest levels {impact} price prediction"
        elif 'insider' in feature_lower:
            return f"Insider trading activity {impact} price prediction"
        else:
            return f"This factor {impact} price prediction"
            
    except Exception:
        return ""

def build_risk_assessment_data(sentiment, technicals, ticker):
    """Build comprehensive risk assessment"""
    try:
        analysis = ["\n=== COMPREHENSIVE RISK ASSESSMENT ==="]
        
        risk_factors = []
        opportunity_factors = []
        
        # Sentiment-based risks
        if sentiment:
            blended_sentiment = sentiment.get('blended_sentiment', 0)
            
            if blended_sentiment < -0.2:
                risk_factors.append(f"Negative market sentiment ({blended_sentiment:.2f}) indicates bearish outlook")
            elif blended_sentiment > 0.2:
                opportunity_factors.append(f"Positive market sentiment ({blended_sentiment:.2f}) supports upward movement")
            
            # Check sentiment divergence
            sentiment_sources = sentiment.get('sources', {})
            sentiment_scores = [data.get('sentiment_score', 0) for data in sentiment_sources.values() if isinstance(data, dict)]
            
            if len(sentiment_scores) > 1:
                sentiment_std = np.std(sentiment_scores) if sentiment_scores else 0
                if sentiment_std > 0.3:
                    risk_factors.append(f"High sentiment divergence ({sentiment_std:.2f}) indicates uncertainty")
        
        # Technical-based risks
        if technicals:
            # RSI risks
            if 'RSI' in technicals:
                rsi = technicals['RSI']
                if rsi > 80:
                    risk_factors.append(f"Extremely overbought RSI ({rsi:.1f}) suggests potential pullback")
                elif rsi < 20:
                    opportunity_factors.append(f"Extremely oversold RSI ({rsi:.1f}) suggests potential bounce")
            
            # Volume risks
            if 'Volume' in technicals and 'Volume_SMA' in technicals:
                volume_ratio = technicals['Volume'] / technicals['Volume_SMA']
                if volume_ratio < 0.3:
                    risk_factors.append(f"Very low volume ({volume_ratio:.2f}x avg) indicates lack of conviction")
        
        # Market structure risks
        if 'short_interest_data' in sentiment and sentiment['short_interest_data']:
            short_data = sentiment['short_interest_data'][0]
            days_to_cover = short_data.get('daysToCoVerShortInterest', 0)
            
            if days_to_cover > 5:
                risk_factors.append(f"High short interest ({days_to_cover:.1f} days to cover) indicates bearish pressure")
            elif days_to_cover > 2:
                opportunity_factors.append(f"Moderate short interest ({days_to_cover:.1f} days) potential for squeeze")
        
        # Compile risk assessment
        analysis.append(f"\nRISK FACTORS:")
        if risk_factors:
            for i, risk in enumerate(risk_factors, 1):
                analysis.append(f"  {i}. {risk}")
        else:
            analysis.append("  No significant risk factors identified")
        
        analysis.append(f"\nOPPORTUNITY FACTORS:")
        if opportunity_factors:
            for i, opp in enumerate(opportunity_factors, 1):
                analysis.append(f"  {i}. {opp}")
        else:
            analysis.append("  No significant opportunity factors identified")
        
        # Overall risk score
        risk_score = len(risk_factors) - len(opportunity_factors)
        if risk_score > 2:
            overall_risk = "HIGH RISK"
        elif risk_score > 0:
            overall_risk = "MODERATE RISK"
        elif risk_score < -2:
            overall_risk = "LOW RISK / HIGH OPPORTUNITY"
        else:
            overall_risk = "BALANCED RISK/REWARD"
        
        analysis.append(f"\nOVERALL RISK ASSESSMENT: {overall_risk}")
        
        return "\n".join(analysis)
        
    except Exception as e:
        logger.error(f"Error building risk assessment: {e}")
        return "\nRISK ASSESSMENT: Error processing data"

# Replace the old function name in the main explanation endpoint
def build_explanation_prompt(ticker, date, sentiment, prediction, technicals=None, shap_top_factors=None, news=None):
    """Wrapper to maintain backwards compatibility while using the new comprehensive function"""
    return build_comprehensive_explanation_prompt(ticker, date, sentiment, prediction, technicals, shap_top_factors, news)

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
    """Get optimization insights and feature importance for a ticker.

    Reads from the ``feature_importance`` collection populated by
    ``ml_backend.explain.shap_analysis``.
    """
    try:
        # Use shared MongoDB client
        if not hasattr(app.state, 'mongo_client') or app.state.mongo_client is None:
            raise HTTPException(status_code=500, detail="Database connection not available")
        mongo_client = app.state.mongo_client
        ticker_upper = ticker.upper()

        # ── Fetch latest feature_importance docs (one per horizon) ──
        fi_docs = list(
            mongo_client.db['feature_importance']
            .find({'ticker': ticker_upper})
            .sort('timestamp', -1)
            .limit(3)
        )

        # Build per-horizon explainability payloads
        horizons_data = {}
        for doc in fi_docs:
            window = doc.get("window", "unknown")
            horizons_data[window] = {
                "predicted_value": doc.get("predicted_value"),
                "predicted_price": doc.get("predicted_price"),
                "prob_up": doc.get("prob_up"),
                "current_price": doc.get("current_price"),
                "model_type": doc.get("model_type"),
                "is_market_neutral": doc.get("is_market_neutral", True),
                "sanity_ok": doc.get("sanity_ok", True),
                "base_value": doc.get("base_value"),
                "top_positive_contrib": doc.get("top_positive_contrib", []),
                "top_negative_contrib": doc.get("top_negative_contrib", []),
                "shap_top_features": doc.get("shap_top_features", {}),
                "global_gain_importance": doc.get("global_gain_importance", []),
                "feature_list_hash": doc.get("feature_list_hash"),
                "n_features": doc.get("n_features", 0),
                "date": doc.get("date"),
            }

        # Legacy compat: pick the latest doc for flat fields
        latest_doc = fi_docs[0] if fi_docs else None
        
        # Get data utilization stats
        collections_stats = {}
        for collection_name in ['sentiment', 'insider_transactions']:
            try:
                count = mongo_client.db[collection_name].count_documents({'ticker': ticker_upper})
                collections_stats[collection_name] = count
            except Exception:
                collections_stats[collection_name] = 0
        
        # Get API cache efficiency
        cache_stats = mongo_client.db['api_cache'].aggregate([
            {
                '$match': {
                    'cache_key': {'$regex': ticker_upper}
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
            'ticker': ticker_upper,
            # ── Per-horizon explainability ──
            'horizons': horizons_data,
            # ── Legacy flat fields (from latest doc) ──
            'feature_importance': latest_doc.get('shap_top_features', {}) if latest_doc else {},
            'total_features_tracked': latest_doc.get('n_features', 0) if latest_doc else 0,
            'last_optimization': latest_doc.get('timestamp') if latest_doc else None,
            'feature_list_hash': latest_doc.get('feature_list_hash') if latest_doc else None,
            'data_utilization': {
                'sentiment_records': collections_stats.get('sentiment', 0),
                'insider_transactions': collections_stats.get('insider_transactions', 0),
            },
            'cache_efficiency': {
                'total_cached_requests': cache_efficiency[0].get('total_cached_requests', 0) if cache_efficiency else 0,
                'avg_cache_age_hours': cache_efficiency[0].get('avg_cache_age_hours', 0) if cache_efficiency else 0,
            },
            'optimization_recommendations': []
        }
        
        # Add optimization recommendations
        if collections_stats.get('sentiment', 0) < 10:
            optimization_insights['optimization_recommendations'].append(
                "Low sentiment data volume - consider increasing sentiment analysis frequency"
            )
        
        if collections_stats.get('insider_transactions', 0) == 0:
            optimization_insights['optimization_recommendations'].append(
                "No insider transaction data - run sentiment cron to populate"
            )
        
        if latest_doc and latest_doc.get('n_features', 0) < 20:
            optimization_insights['optimization_recommendations'].append(
                "Low feature count - consider adding more external data sources"
            )
        
        if latest_doc and not latest_doc.get('sanity_ok', True):
            optimization_insights['optimization_recommendations'].append(
                "SHAP sanity check failed - feature misalignment detected, retrain models"
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
        
        # Fetch historical data from MongoDB
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.days_back)
        df = app.state.mongo_client.get_historical_data(ticker, start_date, end_date)
        
        if df is None or len(df) < 50:
            raise HTTPException(
                status_code=404, 
                detail=f"Insufficient historical data for {ticker}. Found {len(df) if df is not None else 0} records, need at least 50."
            )
        
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
        models_dir = "models/v1" if os.path.exists("models/v1") else "models"
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
    """Health check endpoint. Returns 503 when unhealthy."""
    try:
        # Test MongoDB connection
        if hasattr(app.state, 'mongo_client') and app.state.mongo_client and app.state.mongo_client.db is not None:
            stats = app.state.mongo_client.db.command("ping")
            mongodb_status = "connected" if stats else "disconnected"
        else:
            mongodb_status = "not_initialized"

        healthy = mongodb_status == "connected"
        payload = {
            "status": "healthy" if healthy else "unhealthy",
            "mongodb": mongodb_status,
            "timestamp": datetime.now().isoformat(),
            "api_version": "2.0.0",
        }
        return JSONResponse(content=payload, status_code=200 if healthy else 503)
    except Exception as e:
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            },
            status_code=503,
        )

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
        history = app.state.mongo_client.get_prediction_history_simple(ticker, days)
        
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

@app.get("/api/v1/explain/{ticker}/{date}", tags=["AI Explanations"], summary="Get Comprehensive AI Explanation", 
         description="Get comprehensive AI-generated explanation using all available data sources")
async def get_comprehensive_ai_explanation(ticker: str, date: str):
    """
    Generate comprehensive AI explanation using Google Gemini Pro with ALL available data.
    
    This endpoint:
    1. Gathers ALL sentiment data including raw headlines, Reddit posts, analyst reports
    2. Collects technical indicators and price data
    3. Retrieves fundamental data (earnings, dividends, analyst estimates)
    4. Gets SHAP feature importance from ML models
    5. Builds a comprehensive prompt using ALL raw data
    6. Calls Google Gemini Pro API for intelligent analysis
    7. Stores the explanation in MongoDB for future reference
    
    Returns detailed, actionable investment analysis.
    """
    try:
        ticker = _normalize_ticker(ticker)
        
        # Validate date format
        try:
            prediction_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        logger.info(f"Generating comprehensive AI explanation for {ticker} on {date}")
        
        # 1. Get comprehensive sentiment data with ALL raw data
        sentiment_data = app.state.mongo_client.get_latest_sentiment(ticker)
        if not sentiment_data:
            raise HTTPException(status_code=404, detail=f"No sentiment data found for {ticker}")
        
        # 2. Get latest predictions
        predictions = app.state.mongo_client.get_latest_predictions(ticker)
        if not predictions:
            raise HTTPException(status_code=404, detail=f"No predictions found for {ticker}")
        
        # 3. Get historical data for technical analysis
        try:
            end_date = prediction_date
            start_date = end_date - timedelta(days=365)  # 1 year of data
            historical_data = app.state.mongo_client.get_historical_data(ticker, start_date, end_date)
            
            # Calculate technical indicators
            if not historical_data.empty:
                technicals = calculate_comprehensive_technicals(historical_data)
            else:
                technicals = {}
        except Exception as e:
            logger.warning(f"Error getting technical data for {ticker}: {e}")
            technicals = {}
        
        # 4. Get SHAP feature importance from MongoDB feature_importance collection
        shap_factors = {}
        shap_contrib_data = {}
        try:
            # Prefer pre-computed SHAP from feature_importance collection
            fi_doc = app.state.mongo_client.db['feature_importance'].find_one(
                {'ticker': ticker},
                sort=[('timestamp', -1)]
            )
            if fi_doc:
                shap_factors = fi_doc.get('shap_top_features', {})
                shap_contrib_data = {
                    'top_positive_contrib': fi_doc.get('top_positive_contrib', []),
                    'top_negative_contrib': fi_doc.get('top_negative_contrib', []),
                    'global_gain_importance': fi_doc.get('global_gain_importance', []),
                    'prob_up': fi_doc.get('prob_up'),
                    'predicted_value': fi_doc.get('predicted_value'),
                    'sanity_ok': fi_doc.get('sanity_ok', True),
                    'feature_list_hash': fi_doc.get('feature_list_hash'),
                    'is_market_neutral': fi_doc.get('is_market_neutral', True),
                }
                logger.info(f"Loaded SHAP data from feature_importance for {ticker} (hash={fi_doc.get('feature_list_hash')})")
            else:
                logger.info(f"No pre-computed SHAP data for {ticker}, falling back to live")
                # Fallback: try live computation if predictor is available
                if hasattr(app.state, 'stock_predictor') and app.state.stock_predictor:
                    if hasattr(app.state, 'feature_engineer') and app.state.feature_engineer:
                        features_df = historical_data.tail(60) if not historical_data.empty else None
                        if features_df is not None:
                            features, _ = app.state.feature_engineer.prepare_features(features_df, ticker=ticker, mongo_client=app.state.mongo_client)
                            if features is not None and len(features) > 0:
                                latest_features = features[-1:].reshape(1, -1)
                                shap_result = getattr(app.state.stock_predictor, 'explain_prediction', lambda *a, **k: None)(latest_features, 'next_day', ticker)
                                if shap_result:
                                    sorted_factors = sorted(shap_result.items(), key=lambda x: abs(x[1]), reverse=True)
                                    shap_factors = dict(sorted_factors[:10])
        except Exception as e:
            logger.warning(f"Error getting SHAP factors for {ticker}: {e}")
        
        # 4. Get comprehensive financial data from ALL MongoDB collections
        try:
            logger.info(f"Fetching comprehensive financial data for {ticker}")
            comprehensive_financial_data = get_comprehensive_financial_data(app.state.mongo_client, ticker)
            logger.info(f"Retrieved financial data sources: {list(comprehensive_financial_data.keys())}")
        except Exception as e:
            logger.warning(f"Error getting comprehensive financial data for {ticker}: {e}")
            comprehensive_financial_data = {}
        
        # 5. Build enhanced prompt using ALL available data sources
        if comprehensive_financial_data:
            logger.info(f"Building enhanced financial prompt with comprehensive data")
            comprehensive_prompt = build_enhanced_financial_prompt(
                ticker=ticker,
                financial_data=comprehensive_financial_data,
                sentiment=sentiment_data,
                predictions=predictions,
                technicals=technicals
            )
        else:
            # Fallback to original prompt if comprehensive data not available
            comprehensive_prompt = build_comprehensive_explanation_prompt(
                ticker=ticker,
                date=date,
                sentiment=sentiment_data,
                prediction=predictions,
                technicals=technicals,
                shap_top_factors=shap_factors,
                news=sentiment_data.get('finviz_raw_data', [])
            )
        
        # 6. Call Google Gemini Pro API
        gemini_explanation = await call_google_gemini_api(comprehensive_prompt, ticker)
        
        # 7. Prepare comprehensive explanation data
        explanation_data = {
            "ticker": ticker,
            "explanation_date": date,
            "prediction_data": predictions,
            "sentiment_summary": {
                "blended_sentiment": sentiment_data.get('blended_sentiment', 0),
                "total_data_points": sum(
                    source.get('volume', 0) if isinstance(source, dict) else 0 
                    for source in sentiment_data.get('sources', {}).values()
                ),
                "finviz_articles": len(sentiment_data.get('finviz_raw_data', [])),
                "reddit_posts": len(sentiment_data.get('reddit_raw_data', [])),
                "rss_articles": len(sentiment_data.get('rss_news_raw_data', [])),
                "marketaux_articles": len(sentiment_data.get('marketaux_raw_data', []))
            },
            "financial_data_summary": {
                "earnings_quarters": len(comprehensive_financial_data.get('fmp_data', {}).get('earnings', [])),
                "analyst_estimates": len(comprehensive_financial_data.get('fmp_data', {}).get('analyst_estimates', [])),
                "insider_transactions": len(comprehensive_financial_data.get('insider_transactions', [])),
                "economic_events": len(comprehensive_financial_data.get('economic_events', [])),
                "current_price_available": bool(comprehensive_financial_data.get('current_price_data')),
                "finnhub_metrics_available": bool(comprehensive_financial_data.get('finnhub_data', {}).get('financials'))
            },
            "technical_indicators": technicals,
            "feature_importance": shap_factors,
            "shap_contrib": shap_contrib_data if shap_contrib_data else None,
            "ai_explanation": gemini_explanation,
            "data_sources_used": [
                "Finviz News Headlines",
                "MarketAux Premium News", 
                "Reddit Social Sentiment",
                "RSS News Feeds",
                "Technical Indicators",
                "ML Feature Importance"
            ],
            "explanation_quality": {
                "data_completeness": calculate_data_completeness(sentiment_data, technicals, shap_factors),
                "confidence_factors": extract_confidence_factors(predictions, sentiment_data),
                "risk_assessment": extract_risk_signals(sentiment_data, technicals),
                "financial_data_richness": len([k for k in comprehensive_financial_data.keys() if comprehensive_financial_data[k]])
            },
            "timestamp": datetime.utcnow().isoformat(),
            "prompt_length": len(comprehensive_prompt),
            "explanation_length": len(gemini_explanation) if gemini_explanation else 0
        }
        
        # 8. Store explanation in MongoDB
        try:
            app.state.mongo_client.store_prediction_explanation(ticker, 'comprehensive', explanation_data)
        except Exception as e:
            logger.warning(f"Could not store comprehensive explanation for {ticker}: {e}")
        
        # 9. Return comprehensive response
        return {
            "ticker": ticker,
            "date": date,
            "explanation": gemini_explanation,
            "data_summary": explanation_data["sentiment_summary"],
            "prediction_summary": {
                window: {
                    "predicted_price": pred.get("predicted_price", 0),
                    "confidence": pred.get("confidence", 0),
                    "price_change": pred.get("price_change", 0)
                }
                for window, pred in predictions.items()
            },
            "technical_summary": {
                "rsi": technicals.get("RSI", 0),
                "macd_signal": "Bullish" if technicals.get("MACD", 0) > technicals.get("MACD_Signal", 0) else "Bearish",
                "bollinger_position": calculate_bollinger_position(technicals),
                "volume_trend": "High" if technicals.get("Volume", 0) > technicals.get("Volume_SMA", 0) else "Normal"
            },
            "metadata": {
                "data_sources": explanation_data["data_sources_used"],
                "quality_score": explanation_data["explanation_quality"]["data_completeness"],
                "processing_time": datetime.utcnow().isoformat(),
                "api_version": "2.0.0"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating comprehensive AI explanation for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate explanation: {str(e)}")

# Default model: gemini-2.5-pro (free tier: 1.5K RPD, better quality)
# gemini-2.5-flash has only 20 RPD on free tier (too restrictive)
# Override: set GEMINI_MODEL env var (e.g., GEMINI_MODEL=gemini-2.5-flash)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")


async def call_google_gemini_api(prompt: str, ticker: str) -> str:
    """Call Google Gemini API with the comprehensive prompt."""
    try:
        from google import genai
        from google.genai import types
        import asyncio
        
        # Configure Gemini with new API
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            logger.error("GOOGLE_API_KEY not found in environment variables")
            return "AI explanation unavailable: API key not configured"
        
        # Initialize new Gemini client
        client = genai.Client(api_key=api_key)
        
        logger.info(f"Calling {GEMINI_MODEL} for {ticker} with prompt length: {len(prompt)}")
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=1000)
                )
            )
        )
        
        if response and response.text:
            logger.info(f"Received {GEMINI_MODEL} response for {ticker}: {len(response.text)} characters")
            return response.text
        else:
            logger.warning(f"Empty response from {GEMINI_MODEL} for {ticker}")
            return "AI explanation unavailable: Empty response from API"
            
    except ImportError as e:
        logger.error(f"Google Generative AI library not installed or outdated: {e}")
        return "AI explanation unavailable: Required library not installed or needs update"
    except Exception as e:
        error_str = str(e).lower()
        # Check for quota/rate limit errors
        if "quota" in error_str or "429" in error_str or "rate limit" in error_str:
            logger.error(f"{GEMINI_MODEL} quota/rate limit exceeded for {ticker}: {e}")
            return f"AI explanation unavailable: Gemini API quota exceeded ({GEMINI_MODEL} free tier limit reached). Please try again later or upgrade to paid tier."
        logger.error(f"Error calling {GEMINI_MODEL} for {ticker}: {e}")
        return f"AI explanation unavailable: {str(e)}"

def calculate_comprehensive_technicals(df: pd.DataFrame) -> Dict:
    """Calculate comprehensive technical indicators"""
    try:
        technicals = {}
        
        if len(df) < 20:
            return technicals
        
        # Price data
        close = df['close'] if 'close' in df.columns else df['Close']
        high = df['high'] if 'high' in df.columns else df['High'] 
        low = df['low'] if 'low' in df.columns else df['Low']
        volume = df['volume'] if 'volume' in df.columns else df['Volume']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        technicals['RSI'] = (100 - (100 / (1 + rs))).iloc[-1]
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        technicals['MACD'] = macd.iloc[-1]
        technicals['MACD_Signal'] = signal.iloc[-1]
        
        # Bollinger Bands
        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        technicals['Bollinger_Upper'] = (sma20 + (std20 * 2)).iloc[-1]
        technicals['Bollinger_Lower'] = (sma20 - (std20 * 2)).iloc[-1]
        technicals['Bollinger_Mid'] = sma20.iloc[-1]
        
        # Moving Averages
        technicals['SMA_20'] = close.rolling(window=20).mean().iloc[-1]
        technicals['SMA_50'] = close.rolling(window=50).mean().iloc[-1]
        technicals['EMA_12'] = ema12.iloc[-1]
        technicals['EMA_26'] = ema26.iloc[-1]
        
        # Volume
        technicals['Volume'] = volume.iloc[-1]
        technicals['Volume_SMA'] = volume.rolling(window=20).mean().iloc[-1]
        
        # Current price
        technicals['Close'] = close.iloc[-1]
        
        return technicals
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        return {}

def calculate_bollinger_position(technicals: Dict) -> str:
    """Calculate position within Bollinger Bands"""
    try:
        if all(key in technicals for key in ['Close', 'Bollinger_Upper', 'Bollinger_Lower']):
            price = technicals['Close']
            upper = technicals['Bollinger_Upper']
            lower = technicals['Bollinger_Lower']
            
            position = (price - lower) / (upper - lower) * 100
            
            if position > 80:
                return "Upper Band"
            elif position < 20:
                return "Lower Band"
            else:
                return "Mid-range"
        return "Unknown"
    except:
        return "Unknown"

def calculate_data_completeness(sentiment_data: Dict, technicals: Dict, shap_factors: Dict) -> float:
    """Calculate data completeness score (0-1)"""
    try:
        completeness_factors = []
        
        # Sentiment data completeness
        if sentiment_data:
            sentiment_score = 0
            if sentiment_data.get('finviz_raw_data'):
                sentiment_score += 0.25
            if sentiment_data.get('reddit_raw_data'):
                sentiment_score += 0.25
            if sentiment_data.get('rss_news_raw_data'):
                sentiment_score += 0.25
            if sentiment_data.get('marketaux_raw_data'):
                sentiment_score += 0.25
            completeness_factors.append(sentiment_score)
        
        # Technical data completeness
        if technicals:
            tech_score = min(len(technicals) / 10, 1.0)  # Expect ~10 technical indicators
            completeness_factors.append(tech_score)
        
        # SHAP factors completeness
        if shap_factors:
            shap_score = min(len(shap_factors) / 10, 1.0)  # Expect ~10 SHAP factors
            completeness_factors.append(shap_score)
        
        return sum(completeness_factors) / len(completeness_factors) if completeness_factors else 0.0
        
    except:
        return 0.5

def extract_confidence_factors(predictions: Dict, sentiment_data: Dict) -> List[str]:
    """Extract factors that contribute to prediction confidence"""
    factors = []
    
    try:
        # Check prediction consistency
        if len(predictions) > 1:
            confidences = [pred.get('confidence', 0) for pred in predictions.values() if isinstance(pred, dict)]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                if avg_confidence > 0.8:
                    factors.append("High model confidence across timeframes")
                elif avg_confidence > 0.6:
                    factors.append("Medium model confidence")
                else:
                    factors.append("Lower model confidence indicates uncertainty")
        
        # Check sentiment data volume
        sentiment_sources = sentiment_data.get('sources', {})
        total_volume = sum(data.get('volume', 0) for data in sentiment_sources.values() if isinstance(data, dict))
        if total_volume > 100:
            factors.append(f"High data volume ({total_volume:,} sentiment data points)")
        elif total_volume > 50:
            factors.append(f"Moderate data volume ({total_volume:,} sentiment data points)")
        
        # Check sentiment agreement
        sentiment_scores = [data.get('sentiment_score', 0) for data in sentiment_sources.values() if isinstance(data, dict)]
        if len(sentiment_scores) > 1:
            sentiment_std = np.std(sentiment_scores) if sentiment_scores else 0
            if sentiment_std < 0.2:
                factors.append("Strong sentiment agreement across sources")
            elif sentiment_std > 0.5:
                factors.append("High sentiment divergence indicates uncertainty")
        
    except Exception as e:
        logger.warning(f"Error extracting confidence factors: {e}")
    
    return factors if factors else ["Standard confidence assessment"]

def extract_risk_signals(sentiment_data: Dict, technicals: Dict) -> List[str]:
    """Extract risk signals from the data"""
    risks = []
    
    try:
        # Technical risk signals
        if technicals.get('RSI', 50) > 80:
            risks.append("Overbought conditions (RSI > 80)")
        elif technicals.get('RSI', 50) < 20:
            risks.append("Oversold conditions (RSI < 20)")
        
        # Volume risk signals
        if technicals.get('Volume', 0) < technicals.get('Volume_SMA', 1) * 0.3:
            risks.append("Very low volume indicates lack of conviction")
        
        # Sentiment risk signals
        blended_sentiment = sentiment_data.get('blended_sentiment', 0)
        if abs(blended_sentiment) > 0.8:
            risks.append("Extreme sentiment levels may indicate market extremes")
        
        # Short interest risk (if available)
        if 'short_interest_data' in sentiment_data and sentiment_data['short_interest_data']:
            short_data = sentiment_data['short_interest_data']
            if isinstance(short_data, list) and len(short_data) > 0:
                latest_short = short_data[0]
                days_to_cover = latest_short.get('daysToCoVerShortInterest', 0)
                if days_to_cover > 5:
                    risks.append(f"High short interest ({days_to_cover:.1f} days to cover)")
        
    except Exception as e:
        logger.warning(f"Error extracting risk signals: {e}")
    
    return risks if risks else ["Standard risk assessment"]

def get_comprehensive_financial_data(mongo_client, ticker: str) -> Dict:
    """Fetch comprehensive financial data from all MongoDB collections for AI analysis"""
    try:
        financial_data = {}
        
        # 1. Get Alpha Vantage / FMP data
        av_collection = mongo_client.db['alpha_vantage_data']
        av_data = list(av_collection.find({'ticker': ticker}).sort('timestamp', -1).limit(20))
        
        # Parse FMP data by type
        financial_data['fmp_data'] = {}
        for record in av_data:
            endpoint = record.get('endpoint', '')
            if 'earnings' in endpoint:
                financial_data['fmp_data']['earnings'] = record.get('data', [])
            elif 'dividends' in endpoint:
                financial_data['fmp_data']['dividends'] = record.get('data', [])
            elif 'analyst-estimates' in endpoint:
                financial_data['fmp_data']['analyst_estimates'] = record.get('data', [])
            elif 'ratings' in endpoint:
                financial_data['fmp_data']['ratings'] = record.get('data', [])
        
        # 2. Get current price data
        price_collection = mongo_client.db['current_prices']
        current_price = price_collection.find_one({'ticker': ticker}, sort=[('timestamp', -1)])
        financial_data['current_price_data'] = current_price or {}
        
        # 3. Get economic events affecting this ticker
        events_collection = mongo_client.db['economic_events']
        today = datetime.utcnow().strftime('%Y-%m-%d')
        economic_events = list(events_collection.find({
            'date': {'$gte': today},
            'affected_tickers': ticker
        }).limit(10))
        financial_data['economic_events'] = economic_events
        
        # 4. Get Finnhub financial metrics
        finnhub_collection = mongo_client.db['finnhub_data']
        finnhub_data = list(finnhub_collection.find({'ticker': ticker}).sort('fetched_at', -1).limit(10))
        
        financial_data['finnhub_data'] = {}
        for record in finnhub_data:
            api_source = record.get('api_source', '')
            if 'basic_financials' in api_source:
                financial_data['finnhub_data']['financials'] = record.get('data', {})
            elif 'insider_sentiment' in api_source:
                financial_data['finnhub_data']['insider_sentiment'] = record.get('data', {})
            elif 'company_peers' in api_source:
                financial_data['finnhub_data']['peers'] = record.get('data', [])
        
        # 5. Get insider transactions
        insider_collection = mongo_client.db['insider_transactions']
        insider_transactions = list(insider_collection.find({'symbol': ticker}).sort('filingDate', -1).limit(10))
        financial_data['insider_transactions'] = insider_transactions
        
        return financial_data
        
    except Exception as e:
        logger.error(f"Error fetching comprehensive financial data for {ticker}: {e}")
        return {}

def build_enhanced_financial_prompt(ticker: str, financial_data: Dict, sentiment: Dict, predictions: Dict, technicals: Dict) -> str:
    """Build enhanced prompt using ALL available financial data"""
    try:
        prompt_sections = []
        
        # Core analysis header
        prompt_sections.append(f"""
COMPREHENSIVE FINANCIAL ANALYSIS FOR {ticker}
Using REAL-TIME data from multiple premium sources for enhanced AI analysis.
""")
        
        # 1. CURRENT MARKET DATA
        current_data = financial_data.get('current_price_data', {})
        if current_data:
            prompt_sections.append(f"""
REAL-TIME MARKET DATA:
Current Price: ${current_data.get('current_price', 0):.2f}
Day Range: ${current_data.get('day_low', 0):.2f} - ${current_data.get('day_high', 0):.2f}
Volume: {current_data.get('volume', 0):,}
Day Change: {current_data.get('day_change_percent', 0):.2f}%
Previous Close: ${current_data.get('previous_close', 0):.2f}
""")
        
        # 2. EARNINGS & FUNDAMENTAL DATA
        fmp_data = financial_data.get('fmp_data', {})
        if fmp_data.get('earnings'):
            earnings = fmp_data['earnings'][:3]  # Latest 3 quarters
            prompt_sections.append(f"""
EARNINGS PERFORMANCE (Latest 3 quarters):""")
            for i, earning in enumerate(earnings):
                quarter = f"Q{earning.get('fiscalQuarter', 'N/A')} {earning.get('fiscalYear', 'N/A')}"
                eps_actual = earning.get('epsActual') or 0
                eps_estimate = earning.get('epsEstimated') or 0
                surprise = ((eps_actual - eps_estimate) / eps_estimate * 100) if eps_estimate != 0 else 0
                revenue = (earning.get('revenueActual') or 0) / 1e9  # Convert to billions
                
                prompt_sections.append(f"""
{quarter}: EPS ${eps_actual:.3f} vs Est. ${eps_estimate:.3f} ({surprise:+.1f}% surprise)
Revenue: ${revenue:.2f}B""")
        
        # 3. ANALYST ESTIMATES & RATINGS
        if fmp_data.get('analyst_estimates'):
            estimates = fmp_data['analyst_estimates'][:2]  # Next 2 periods
            prompt_sections.append(f"""
ANALYST FORECASTS:""")
            for estimate in estimates:
                period = estimate.get('date', 'N/A')
                eps_avg = estimate.get('epsAvg', 0)
                revenue_avg = estimate.get('revenueAvg', 0) / 1e9 if estimate.get('revenueAvg') else 0
                analyst_count = estimate.get('numberAnalysts', 0)
                prompt_sections.append(f"""
{period}: EPS ${eps_avg:.3f}, Revenue ${revenue_avg:.2f}B ({analyst_count} analysts)""")
        
        # 4. FINANCIAL METRICS (FINNHUB)
        finnhub_data = financial_data.get('finnhub_data', {})
        if finnhub_data.get('financials'):
            metrics = finnhub_data['financials'].get('metric', {})
            
            # Safely get values, defaulting to 0 or 1 to avoid division errors
            current_price = current_data.get('current_price') or 0
            high_52w = metrics.get('52WeekHigh') or (current_price if current_price > 0 else 1)

            prompt_sections.append(f"""
KEY FINANCIAL METRICS:
P/E Ratio: {metrics.get('peInclExtraTTM', 0):.1f}
Beta: {metrics.get('beta', 0):.2f}
52-Week Range: ${metrics.get('52WeekLow', 0):.2f} - ${high_52w:.2f}
Current vs 52W High: {((current_price / high_52w) - 1) * 100 if high_52w else 0:.1f}%
Dividend Yield: {metrics.get('currentDividendYieldTTM', 0):.2f}%
ROE: {metrics.get('roeRfy', 0):.1f}%
""")
        
        # 5. INSIDER ACTIVITY
        insider_transactions = financial_data.get('insider_transactions', [])
        if insider_transactions:
            recent_insider = insider_transactions[:3]  # Last 3 transactions
            prompt_sections.append(f"""
RECENT INSIDER ACTIVITY:""")
            for transaction in recent_insider:
                name = transaction.get('name', 'Unknown')
                action = 'SOLD' if transaction.get('transactionCode') == 'S' else 'BOUGHT'
                shares = abs(transaction.get('change', 0))
                price = transaction.get('transactionPrice', 0)
                date = transaction.get('transactionDate', 'N/A')
                prompt_sections.append(f"""
{date}: {name} {action} {shares:,} shares at ${price:.2f}""")
        
        # 6. ECONOMIC EVENTS IMPACT
        economic_events = financial_data.get('economic_events', [])
        if economic_events:
            prompt_sections.append(f"""
UPCOMING ECONOMIC EVENTS AFFECTING {ticker}:""")
            for event in economic_events[:3]:
                event_name = event.get('event', 'Unknown')
                importance = event.get('importance', 'medium')
                forecast = event.get('forecast', 'N/A')
                date = event.get('date', 'N/A')
                prompt_sections.append(f"""
{date}: {event_name} - {importance.upper()} impact (Forecast: {forecast})""")
        
        # 7. SENTIMENT & PREDICTION DATA
        prompt_sections.append(f"""
AI MODEL PREDICTIONS:
LSTM: {predictions.get('next_day', {}).get('model_predictions', {}).get('lstm', 0):.2f}%
XGBoost: {predictions.get('next_day', {}).get('model_predictions', {}).get('xgboost', 0):.2f}%
LightGBM: {predictions.get('next_day', {}).get('model_predictions', {}).get('lightgbm', 0):.2f}%

SENTIMENT ANALYSIS:
Overall Sentiment: {sentiment.get('blended_sentiment', 0):.3f}
News Articles: {sentiment.get('finviz_volume', 0)} (Sentiment: {sentiment.get('finviz_sentiment', 0):.2f})
Reddit Posts: {sentiment.get('reddit_volume', 0)} (Sentiment: {sentiment.get('reddit_sentiment', 0):.2f})

TECHNICAL INDICATORS:
RSI(14): {technicals.get('RSI', 0):.1f}
MACD Signal: {'Bullish' if technicals.get('MACD', 0) > technicals.get('MACD_Signal', 0) else 'Bearish'}
""")
        
        # 7.5. ADD KEY NARRATIVES FROM RAW TEXT
        prompt_sections.append("\nKEY NEWS & SOCIAL MEDIA NARRATIVES:")
        
        # Top News Headlines
        top_headlines = []
        if 'finviz_raw_data' in sentiment and sentiment['finviz_raw_data']:
            top_headlines.extend(sentiment['finviz_raw_data'][:2])
        if 'rss_news_raw_data' in sentiment and sentiment['rss_news_raw_data']:
            top_headlines.extend([item.get('title', '') for item in sentiment['rss_news_raw_data'][:2]])
        
        if top_headlines:
            prompt_sections.append("Top News Headlines:")
            for headline in set(top_headlines): # Use set to avoid duplicates
                prompt_sections.append(f"- {headline}")

        # Top Reddit Posts
        if 'reddit_raw_data' in sentiment and sentiment['reddit_raw_data']:
            prompt_sections.append("Top Reddit Posts:")
            for post in sentiment['reddit_raw_data'][:3]:
                prompt_sections.append(f"- (r/{post.get('subreddit', '')}) {post.get('title', '')}")

        # 8. AI INSTRUCTION FOR COMPREHENSIVE ANALYSIS
        instruction = f"""
ANALYSIS INSTRUCTIONS:
Generate a concise, easy-to-read AI analysis for {ticker} for a trading dashboard. Use ALL the real-time data provided.

Format your response EXACTLY as follows, using markdown for formatting:

💡 **AI Analysis for {ticker}**
**Summary:** [A 1-2 sentence summary of the current situation and the AI's core conclusion.]
**Catalyst:** [Identify the most significant recent event from the data, e.g., "Earnings beat," "Analyst upgrade," or "High volume trend."]

**🔼 Positive Factors**
* [List 2-3 key bullish points from the data. Be specific.]
* [Factor 2]

**🔽 Risk Factors**
* [List 2-3 key bearish points from the data. Be specific.]
* [Factor 2]

**Trend Context:** [Describe today's move in the context of the recent trend, e.g., "Today's upward move reverses a 5-day downtrend on high volume."]

**Suggested Action:** [Provide a clear, actionable suggestion, e.g., "Watch for a breakout above $X resistance," or "Consider entry near $Y support if bullish momentum continues."]

**Scenario to Watch:** [**If confidence is Medium/Low**, add this section. Describe what could change the outlook, e.g., "If RSI breaks 70, the stock may become overbought, signaling a potential pullback."]

**Outlook:** [Choose one: Bullish, Bearish, Neutral] (Confidence: [High, Medium, or Low])
**Rationale Tag:** [Choose relevant tags: Momentum, Value, Growth, Contrarian, High Risk, Event-Driven]

**Key Levels to Watch:**
* **Support:** $[A key support level based on technicals]
* **Resistance:** $[A key resistance level based on technicals]

REQUIREMENTS:
- Use all provided data to inform your analysis. Do NOT invent data.
- Keep the language direct and actionable for a trader.
- Ensure the total response is concise and under 2000 characters.
"""
        
        complete_prompt = "\n".join(prompt_sections) + instruction
        return complete_prompt
        
    except Exception as e:
        logger.error(f"Error building enhanced financial prompt: {e}")
        return f"Analyze {ticker} with available data"

@app.get("/api/v1/explanation/stored/{ticker}", tags=["AI Explanations"], summary="Get Stored AI Explanation", 
        description="Retrieve stored AI explanation from MongoDB without regenerating")
async def get_stored_ai_explanation(ticker: str, window: str = "comprehensive"):
    """
    Retrieve stored AI explanation from MongoDB database.
    
    This endpoint returns previously generated explanations without triggering
    new analysis, making it fast and efficient for frontend consumption.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL)
        window: Explanation window/type (default: comprehensive)
    
    Returns:
        Stored explanation data or 404 if not found
    """
    try:
        ticker = _normalize_ticker(ticker)
        collection = app.state.mongo_client.db['prediction_explanations']
        
        stored_doc = collection.find_one(
            {"ticker": ticker, "window": window},
            sort=[("timestamp", -1)]
        )
        
        if not stored_doc:
            raise HTTPException(
                status_code=404, 
                detail=f"No stored explanation found for {ticker.upper()}-{window}"
            )
        
        # Return the stored explanation data
        return {
            "ticker": ticker.upper(),
            "window": window,
            "timestamp": stored_doc.get("timestamp"),
            "explanation_data": stored_doc.get("explanation_data", {}),
            "source": "mongodb_stored",
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving stored explanation for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/api/v1/explain/batch", tags=["AI Explanations"], summary="Batch Generate AI Explanations", 
         description="Generate AI explanations for all 25 S&P tickers in batch")
async def batch_generate_ai_explanations(date: Optional[str] = None):
    """
    Generate comprehensive AI explanations for all 25 S&P tickers in batch.
    
    This endpoint will:
    1. Process all 25 tickers sequentially
    2. Generate AI explanations using Google Gemini 2.5 Pro
    3. Store all explanations in MongoDB
    4. Return summary statistics and results
    
    Args:
        date: Target date for explanations (defaults to today)
    
    Returns:
        Batch processing results with success/failure counts
    """
    try:
        # Use today's date if not provided
        target_date = date or datetime.utcnow().strftime('%Y-%m-%d')
        
        # Validate date format
        try:
            prediction_date = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        logger.info(f"Starting batch AI explanation generation for all 25 tickers on {target_date}")
        
        # Initialize results tracking
        results = []
        success_count = 0
        error_count = 0
        start_time = datetime.utcnow()
        
        # Process each ticker
        for i, ticker in enumerate(TOP_100_TICKERS, 1):
            ticker_start_time = datetime.utcnow()
            
            try:
                logger.info(f"[{i}/25] Processing {ticker}...")
                
                # 1. Get comprehensive sentiment data
                sentiment_data = app.state.mongo_client.get_latest_sentiment(ticker)
                if not sentiment_data:
                    # Try to generate fresh sentiment data
                    try:
                        sentiment_data = await app.state.sentiment_analyzer.get_combined_sentiment(ticker, force_refresh=False)
                    except Exception as e:
                        logger.warning(f"Could not get sentiment for {ticker}: {e}")
                        sentiment_data = {"blended_sentiment": 0, "sources": {}}
                
                # 2. Get latest predictions
                predictions = app.state.mongo_client.get_latest_predictions(ticker)
                if not predictions:
                    # Generate basic predictions if none exist
                    try:
                        end_date = prediction_date
                        start_date = end_date - timedelta(days=365)
                        historical_data = app.state.mongo_client.get_historical_data(ticker, start_date, end_date)
                        
                        if not historical_data.empty:
                            predictions = app.state.stock_predictor.predict_all_windows(ticker, historical_data)
                        else:
                            predictions = {"next_day": {"prediction": 0, "confidence": 0.5}}
                    except Exception as e:
                        logger.warning(f"Could not generate predictions for {ticker}: {e}")
                        predictions = {"next_day": {"prediction": 0, "confidence": 0.5}}
                
                # 3. Get technical indicators
                try:
                    end_date = prediction_date
                    start_date = end_date - timedelta(days=365)
                    historical_data = app.state.mongo_client.get_historical_data(ticker, start_date, end_date)
                    
                    if not historical_data.empty:
                        technicals = calculate_comprehensive_technicals(historical_data)
                    else:
                        technicals = {}
                except Exception as e:
                    logger.warning(f"Error getting technical data for {ticker}: {e}")
                    technicals = {}
                
                # 4. Get comprehensive financial data
                try:
                    comprehensive_financial_data = get_comprehensive_financial_data(app.state.mongo_client, ticker)
                except Exception as e:
                    logger.warning(f"Error getting comprehensive financial data for {ticker}: {e}")
                    comprehensive_financial_data = {}
                
                # 5. Build comprehensive prompt
                if comprehensive_financial_data:
                    comprehensive_prompt = build_enhanced_financial_prompt(
                        ticker=ticker,
                        financial_data=comprehensive_financial_data,
                        sentiment=sentiment_data,
                        predictions=predictions,
                        technicals=technicals
                    )
                else:
                    # Fallback to the enhanced prompt, passing empty financial data
                    comprehensive_prompt = build_enhanced_financial_prompt(
                        ticker=ticker,
                        financial_data={},
                        sentiment=sentiment_data,
                        predictions=predictions,
                        technicals=technicals
                    )
                
                # 6. Call Google Gemini API
                gemini_explanation = await call_google_gemini_api(comprehensive_prompt, ticker)
                
                # 7. Prepare and store explanation data
                explanation_data = {
                    "ticker": ticker,
                    "explanation_date": target_date,
                    "prediction_data": predictions,
                    "sentiment_summary": {
                        "blended_sentiment": sentiment_data.get('blended_sentiment', 0),
                        "total_data_points": sum(
                            source.get('volume', 0) if isinstance(source, dict) else 0 
                            for source in sentiment_data.get('sources', {}).values()
                        ),
                        "finviz_articles": len(sentiment_data.get('finviz_raw_data', [])),
                        "reddit_posts": len(sentiment_data.get('reddit_raw_data', [])),
                        "rss_articles": len(sentiment_data.get('rss_news_raw_data', [])),
                        "marketaux_articles": len(sentiment_data.get('marketaux_raw_data', []))
                    },
                    "technical_indicators": technicals,
                    "ai_explanation": gemini_explanation,
                    "data_sources_used": [
                        "Finviz News Headlines",
                        "MarketAux Premium News",
                        "Reddit Social Sentiment", 
                        "RSS News Feeds",
                        "Technical Indicators",
                        "ML Feature Importance",
                        "FMP Financial Data",
                        "Economic Events",
                        "Insider Transactions"
                    ],
                    "timestamp": datetime.utcnow().isoformat(),
                    "explanation_length": len(gemini_explanation) if gemini_explanation else 0
                }
                
                # 8. Store in MongoDB
                try:
                    app.state.mongo_client.store_prediction_explanation(ticker, 'comprehensive', explanation_data)
                    storage_success = True
                except Exception as e:
                    logger.warning(f"Could not store explanation for {ticker}: {e}")
                    storage_success = False
                
                # Calculate processing time
                ticker_end_time = datetime.utcnow()
                processing_time = (ticker_end_time - ticker_start_time).total_seconds()
                
                # Record success
                success_count += 1
                results.append({
                    "ticker": ticker,
                    "status": "success",
                    "explanation_length": len(gemini_explanation) if gemini_explanation else 0,
                    "data_sources": len(explanation_data["data_sources_used"]),
                    "processing_time_seconds": round(processing_time, 2),
                    "stored_in_mongodb": storage_success,
                    "sentiment_score": sentiment_data.get('blended_sentiment', 0),
                    "prediction_confidence": predictions.get('next_day', {}).get('confidence', 0) if isinstance(predictions.get('next_day'), dict) else 0
                })
                
                logger.info(f"✅ [{i}/25] {ticker} SUCCESS: {len(gemini_explanation) if gemini_explanation else 0} chars in {processing_time:.1f}s")
                
            except Exception as e:
                # Record error
                error_count += 1
                ticker_end_time = datetime.utcnow()
                processing_time = (ticker_end_time - ticker_start_time).total_seconds()
                
                results.append({
                    "ticker": ticker,
                    "status": "error",
                    "error_message": str(e),
                    "processing_time_seconds": round(processing_time, 2),
                    "stored_in_mongodb": False
                })
                
                logger.error(f"❌ [{i}/25] {ticker} ERROR: {str(e)}")
            
            # Brief pause between tickers to avoid overwhelming APIs
            if i < len(TOP_100_TICKERS):
                await asyncio.sleep(2)
        
        # Calculate total processing time
        end_time = datetime.utcnow()
        total_processing_time = (end_time - start_time).total_seconds()
        
        # Calculate summary statistics
        successful_results = [r for r in results if r["status"] == "success"]
        if successful_results:
            avg_explanation_length = sum(r["explanation_length"] for r in successful_results) / len(successful_results)
            avg_processing_time = sum(r["processing_time_seconds"] for r in successful_results) / len(successful_results)
            total_data_sources = sum(r["data_sources"] for r in successful_results)
        else:
            avg_explanation_length = 0
            avg_processing_time = 0
            total_data_sources = 0
        
        # Prepare final response
        batch_summary = {
            "batch_id": f"batch_{start_time.strftime('%Y%m%d_%H%M%S')}",
            "target_date": target_date,
            "total_tickers": len(TOP_100_TICKERS),
            "successful": success_count,
            "failed": error_count,
            "success_rate": round((success_count / len(TOP_100_TICKERS)) * 100, 1),
            "total_processing_time_seconds": round(total_processing_time, 2),
            "total_processing_time_minutes": round(total_processing_time / 60, 1),
            "performance_metrics": {
                "avg_explanation_length": round(avg_explanation_length, 0),
                "avg_processing_time_per_ticker": round(avg_processing_time, 2),
                "total_data_sources_used": total_data_sources,
                "explanations_stored_in_mongodb": len([r for r in successful_results if r.get("stored_in_mongodb", False)])
            },
            "tickers_processed": TOP_100_TICKERS,
            "detailed_results": results,
            "api_info": {
                "google_gemini_model": GEMINI_MODEL,
                "mongodb_collections_used": [
                    "prediction_explanations",
                    "sentiment_data", 
                    "alpha_vantage_data",
                    "current_prices",
                    "economic_events",
                    "finnhub_data",
                    "insider_transactions"
                ],
                "data_freshness": "Real-time market data and sentiment",
                "explanation_format": "Frontend-optimized with actionable insights"
            },
            "next_steps": {
                "view_explanations": "Check MongoDB prediction_explanations collection",
                "frontend_access": "AI explanations now available in stock detail widgets",
                "api_access": f"Individual explanations: GET /api/v1/explain/{{ticker}}/{target_date}"
            }
        }
        
        logger.info(f"🎉 Batch AI explanation generation completed: {success_count}/{len(TOP_100_TICKERS)} successful")
        
        return batch_summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch AI explanation generation: {e}")
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")

@app.get("/api/v1/explain/batch/status", tags=["AI Explanations"], summary="Check Batch Status", 
         description="Check the status of AI explanations for all tickers")
async def check_batch_explanation_status():
    """
    Check the status of AI explanations for all 25 tickers.
    
    Returns:
        Status summary showing which tickers have explanations and when they were generated
    """
    try:
        status_results = []
        
        for ticker in TOP_100_TICKERS:
            try:
                # Check if explanation exists in MongoDB
                explanation = app.state.mongo_client.get_prediction_explanation(ticker, 'comprehensive')
                
                if explanation:
                    status_results.append({
                        "ticker": ticker,
                        "has_explanation": True,
                        "explanation_date": explanation.get('explanation_date', 'Unknown'),
                        "generated_at": explanation.get('timestamp', 'Unknown'),
                        "explanation_length": len(explanation.get('ai_explanation', '')),
                        "data_sources": len(explanation.get('data_sources_used', [])),
                        "sentiment_score": explanation.get('sentiment_summary', {}).get('blended_sentiment', 0)
                    })
                else:
                    status_results.append({
                        "ticker": ticker,
                        "has_explanation": False,
                        "explanation_date": None,
                        "generated_at": None,
                        "explanation_length": 0,
                        "data_sources": 0,
                        "sentiment_score": 0
                    })
                    
            except Exception as e:
                status_results.append({
                    "ticker": ticker,
                    "has_explanation": False,
                    "error": str(e)
                })
        
        # Calculate summary statistics
        with_explanations = [r for r in status_results if r.get("has_explanation", False)]
        without_explanations = [r for r in status_results if not r.get("has_explanation", False)]
        
        summary = {
            "total_tickers": len(TOP_100_TICKERS),
            "with_explanations": len(with_explanations),
            "without_explanations": len(without_explanations),
            "coverage_percentage": round((len(with_explanations) / len(TOP_100_TICKERS)) * 100, 1),
            "tickers_with_explanations": [r["ticker"] for r in with_explanations],
            "tickers_without_explanations": [r["ticker"] for r in without_explanations],
            "detailed_status": status_results,
            "recommendations": {
                "generate_missing": f"POST /api/v1/explain/batch to generate all explanations",
                "individual_generation": f"GET /api/v1/explain/{{ticker}}/{{date}} for specific tickers"
            }
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error checking batch explanation status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 