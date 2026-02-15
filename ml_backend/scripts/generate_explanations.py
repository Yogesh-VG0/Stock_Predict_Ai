"""
Batch AI Explanation Generator for CI/CD

Generates Gemini AI explanations for all tickers with stored predictions
and stores them in the ``prediction_explanations`` MongoDB collection.
Model defaults to gemini-2.5-flash (free tier, high rate limits);
override with GEMINI_MODEL env var.

Usage (CI):
    python -m ml_backend.scripts.generate_explanations
    python -m ml_backend.scripts.generate_explanations --tickers AAPL MSFT
    python -m ml_backend.scripts.generate_explanations --max-tickers 20
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── S&P 100 tickers (same list as run_pipeline) ──────────────────────────
TOP_100_TICKERS: List[str] = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CRM", "AMD", "INTC",
    "CSCO", "ADBE", "QCOM", "TXN", "NOW", "INTU", "AMZN", "TSLA", "HD", "NFLX",
    "LOW", "SBUX", "NKE", "MCD", "DIS", "BKNG", "TGT", "JPM", "V", "MA",
    "BAC", "WFC", "GS", "MS", "AXP", "BLK", "SCHW", "C", "COF", "BK",
    "MET", "AIG", "USB", "XOM", "CVX", "COP", "JNJ", "UNH", "LLY", "PFE",
    "ABBV", "ABT", "TMO", "DHR", "MRK", "AMGN", "GILD", "ISRG", "MDT", "BMY",
    "CVS", "WMT", "COST", "PG", "KO", "PEP", "MDLZ", "CL", "MO", "CAT",
    "HON", "UNP", "BA", "RTX", "LMT", "DE", "GE", "GD", "EMR", "FDX",
    "UPS", "MMM", "CMCSA", "VZ", "T", "CHTR", "BRK-B", "ACN", "IBM", "PYPL",
    "LIN", "NEE", "SO", "DUK", "AMT", "SPG", "PLTR", "TMUS", "PM", "AMAT",
]


# ── Technical indicator calculator (standalone, no FastAPI dependency) ──
def calculate_technicals(df: pd.DataFrame) -> Dict:
    """Calculate key technicals from OHLCV DataFrame."""
    if df is None or len(df) < 20:
        return {}
    try:
        close = df["close"] if "close" in df.columns else df["Close"]
        high = df["high"] if "high" in df.columns else df["High"]
        low = df["low"] if "low" in df.columns else df["Low"]
        volume = df["volume"] if "volume" in df.columns else df["Volume"]

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()

        # Bollinger
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()

        # SMAs
        sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
        vol_sma = volume.rolling(20).mean().iloc[-1]

        return {
            "RSI": float(rsi),
            "MACD": float(macd.iloc[-1]),
            "MACD_Signal": float(signal.iloc[-1]),
            "Bollinger_Upper": float((sma20 + 2 * std20).iloc[-1]),
            "Bollinger_Lower": float((sma20 - 2 * std20).iloc[-1]),
            "SMA_20": float(sma20.iloc[-1]),
            "SMA_50": float(sma50) if sma50 is not None else None,
            "EMA_12": float(ema12.iloc[-1]),
            "EMA_26": float(ema26.iloc[-1]),
            "Close": float(close.iloc[-1]),
            "Volume": float(volume.iloc[-1]),
            "Volume_SMA": float(vol_sma),
        }
    except Exception as e:
        logger.warning("Technicals failed: %s", e)
        return {}


# Default model: gemini-2.5-flash (free tier, 15 RPM, fast)
# Override: set GEMINI_MODEL=gemini-2.5-pro for higher quality
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _call_gemini(prompt: str, ticker: str) -> str:
    """Synchronous Gemini API call using GEMINI_MODEL."""
    try:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "AI explanation unavailable: GOOGLE_API_KEY not set"

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=1000)
            ),
        )
        if response and response.text:
            logger.info("%s response for %s: %d chars", GEMINI_MODEL, ticker, len(response.text))
            return response.text
        return "AI explanation unavailable: empty response"
    except Exception as e:
        logger.error("%s API error for %s: %s", GEMINI_MODEL, ticker, e)
        return f"AI explanation unavailable: {e}"


def _build_prompt(
    ticker: str,
    date: str,
    predictions: Dict,
    sentiment: Dict,
    technicals: Dict,
    shap_data: Optional[Dict],
) -> str:
    """Build a concise explanation prompt."""
    sections = []

    # ── Predictions ──
    pred_lines = []
    for window in ("next_day", "7_day", "30_day"):
        pd_ = predictions.get(window, {})
        if isinstance(pd_, dict) and pd_:
            pred_lines.append(
                f"  {window}: price=${pd_.get('predicted_price', 'N/A')}, "
                f"change={pd_.get('price_change', 0):.2f}%, "
                f"confidence={pd_.get('confidence', 0):.2f}"
            )
    sections.append(f"PREDICTION DATA FOR {ticker} ON {date}:\n" + "\n".join(pred_lines))

    # ── Sentiment ──
    blended = sentiment.get("blended_sentiment", 0)
    sources = sentiment.get("sources", {})
    source_lines = []
    for src, data in sources.items():
        if isinstance(data, dict):
            source_lines.append(f"  {src}: score={data.get('score', 0):.3f}, volume={data.get('volume', 0)}")
    sections.append(f"SENTIMENT (blended={blended:.3f}):\n" + "\n".join(source_lines))

    # ── Recent News Headlines ──
    news_lines = []
    # Extract from finviz raw data (list of headline strings)
    finviz_headlines = sentiment.get("finviz_raw_data", [])
    if isinstance(finviz_headlines, list):
        for h in finviz_headlines[:5]:
            if isinstance(h, str) and h.strip():
                news_lines.append(f"  - {h.strip()}")
    # Extract from RSS news raw data (list of dicts with "title")
    rss_news = sentiment.get("rss_news_raw_data", [])
    if isinstance(rss_news, list):
        for item in rss_news[:5]:
            title = item.get("title", "") if isinstance(item, dict) else str(item)
            if title.strip() and title.strip() not in [l.strip().lstrip("- ") for l in news_lines]:
                news_lines.append(f"  - {title.strip()}")
    # Extract from reddit raw data
    reddit_posts = sentiment.get("reddit_raw_data", [])
    if isinstance(reddit_posts, list):
        for item in reddit_posts[:3]:
            title = item.get("title", "") if isinstance(item, dict) else str(item)
            if title.strip():
                news_lines.append(f"  - [Reddit] {title.strip()}")

    if news_lines:
        sections.append("RECENT NEWS HEADLINES:\n" + "\n".join(news_lines[:10]))

    # ── Technicals ──
    if technicals:
        tech_lines = [f"  {k}: {v}" for k, v in technicals.items() if v is not None]
        sections.append("TECHNICAL INDICATORS:\n" + "\n".join(tech_lines))

    # ── SHAP ──
    if shap_data:
        pos = shap_data.get("top_positive_contrib", [])
        neg = shap_data.get("top_negative_contrib", [])
        shap_lines = ["FEATURE IMPORTANCE (SHAP):"]
        if pos:
            shap_lines.append("  Bullish drivers: " + ", ".join(
                f"{f.get('feature', '?')} (+{f.get('contrib', 0):.4f})" for f in pos[:5]
            ))
        if neg:
            shap_lines.append("  Bearish drivers: " + ", ".join(
                f"{f.get('feature', '?')} ({f.get('contrib', 0):.4f})" for f in neg[:5]
            ))
        sections.append("\n".join(shap_lines))

    # ── Instructions ──
    sections.append(f"""
You are a trading dashboard AI. Generate a SHORT analysis for {ticker}.
STRICT RULES:
- MAX 600 characters total
- Reference specific news events if provided (e.g. "Earnings beat" or "AI fears slam sector")
- Use this EXACT format (no markdown headers, no bold):

Summary: [1 sentence, max 20 words, mention key news catalyst if any]

+ [Bullish factor 1]
+ [Bullish factor 2]

- [Risk factor 1]
- [Risk factor 2]

Outlook: [Bullish/Bearish/Neutral] | Confidence: [value]
Levels: Support $[price] | Resistance $[price]

Do NOT add disclaimers, caveats, or "not financial advice" text.
Use ONLY the provided data. Do NOT invent numbers.
""")
    return "\n\n".join(sections)


def generate_explanations(
    tickers: Optional[List[str]] = None,
    max_tickers: int = 100,
    date: Optional[str] = None,
) -> Dict:
    """Generate and store Gemini explanations for tickers with predictions."""
    from ml_backend.utils.mongodb import MongoDBClient

    mongo = MongoDBClient()
    target_date = date or datetime.utcnow().strftime("%Y-%m-%d")
    ticker_list = tickers or TOP_100_TICKERS
    ticker_list = ticker_list[:max_tickers]

    results = {"success": 0, "skipped": 0, "failed": 0, "details": []}

    for i, ticker in enumerate(ticker_list, 1):
        logger.info("[%d/%d] Processing %s…", i, len(ticker_list), ticker)

        # 1. Get predictions (required)
        predictions = mongo.get_latest_predictions(ticker)
        if not predictions:
            logger.warning("  No predictions for %s — skipping", ticker)
            results["skipped"] += 1
            results["details"].append({"ticker": ticker, "status": "skipped", "reason": "no predictions"})
            continue

        # 2. Get sentiment
        sentiment = mongo.get_latest_sentiment(ticker) or {"blended_sentiment": 0, "sources": {}}

        # 3. Get technicals from historical data
        try:
            end_dt = datetime.strptime(target_date, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=365)
            hist = mongo.get_historical_data(ticker, start_dt, end_dt)
            technicals = calculate_technicals(hist) if hist is not None and not hist.empty else {}
            # Fallback: if MongoDB has no historical data, try yfinance directly
            if not technicals:
                try:
                    import yfinance as yf
                    yf_data = yf.download(ticker, start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), progress=False)
                    if yf_data is not None and not yf_data.empty:
                        # Flatten MultiIndex columns if present
                        if hasattr(yf_data.columns, 'levels'):
                            yf_data.columns = yf_data.columns.get_level_values(0)
                        technicals = calculate_technicals(yf_data)
                        if technicals:
                            logger.info("  ✅ Got technicals from yfinance fallback for %s", ticker)
                except Exception as yf_err:
                    logger.warning("  yfinance fallback failed for %s: %s", ticker, yf_err)
        except Exception as e:
            logger.warning("  Technicals failed for %s: %s", ticker, e)
            technicals = {}

        # 4. Get SHAP data
        shap_data = None
        try:
            fi_doc = mongo.db["feature_importance"].find_one(
                {"ticker": ticker}, sort=[("timestamp", -1)]
            )
            if fi_doc:
                shap_data = {
                    "top_positive_contrib": fi_doc.get("top_positive_contrib", []),
                    "top_negative_contrib": fi_doc.get("top_negative_contrib", []),
                    "global_gain_importance": fi_doc.get("global_gain_importance", []),
                    "prob_up": fi_doc.get("prob_up"),
                    "predicted_value": fi_doc.get("predicted_value"),
                }
        except Exception as e:
            logger.warning("  SHAP lookup failed for %s: %s", ticker, e)

        # 5. Build prompt & call Gemini
        prompt = _build_prompt(ticker, target_date, predictions, sentiment, technicals, shap_data)
        explanation_text = _call_gemini(prompt, ticker)

        if "unavailable" in explanation_text.lower():
            logger.error("  Gemini failed for %s: %s", ticker, explanation_text[:80])
            results["failed"] += 1
            results["details"].append({"ticker": ticker, "status": "failed", "reason": explanation_text[:200]})
            continue

        # 6. Build explanation doc & store
        explanation_data = {
            "ticker": ticker,
            "explanation_date": target_date,
            "prediction_data": predictions,
            "sentiment_summary": {
                "blended_sentiment": sentiment.get("blended_sentiment", 0),
                "total_data_points": sum(
                    s.get("volume", 0) if isinstance(s, dict) else 0
                    for s in sentiment.get("sources", {}).values()
                ),
                "finviz_articles": len(sentiment.get("finviz_raw_data", [])),
                "reddit_posts": len(sentiment.get("reddit_raw_data", [])),
                "rss_articles": len(sentiment.get("rss_news_raw_data", [])),
                "marketaux_articles": len(sentiment.get("marketaux_raw_data", [])),
            },
            "technical_indicators": technicals,
            "feature_importance": shap_data or {},
            "ai_explanation": explanation_text,
            "data_sources_used": [
                "ML Predictions",
                "Sentiment Analysis",
                "Technical Indicators",
                "SHAP Feature Importance",
                f"Google {GEMINI_MODEL}",
            ],
            "explanation_quality": {
                "data_completeness": min(1.0, (
                    (0.3 if predictions else 0) +
                    (0.2 if sentiment.get("blended_sentiment") else 0) +
                    (0.2 if technicals else 0) +
                    (0.15 if shap_data else 0) +
                    (0.15 if explanation_text else 0)
                )),
            },
            "timestamp": datetime.utcnow().isoformat(),
            "prompt_length": len(prompt),
            "explanation_length": len(explanation_text),
        }

        stored = mongo.store_prediction_explanation(ticker, "comprehensive", explanation_data)
        if stored:
            results["success"] += 1
            results["details"].append({"ticker": ticker, "status": "success", "chars": len(explanation_text)})
            logger.info("  ✅ Stored for %s (%d chars)", ticker, len(explanation_text))
        else:
            results["failed"] += 1
            results["details"].append({"ticker": ticker, "status": "failed", "reason": "mongo store failed"})
            logger.error("  ❌ MongoDB store failed for %s", ticker)

        # Rate limit: gemini-2.5-flash free tier ~15 RPM; pro ~5 RPM
        sleep_secs = 4 if "pro" in GEMINI_MODEL else 2
        time.sleep(sleep_secs)

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch Gemini AI Explanation Generator")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Specific tickers (default: all with predictions)")
    parser.add_argument("--max-tickers", type=int, default=100,
                        help="Max tickers to process (default: 100)")
    parser.add_argument("--date", type=str, default=None,
                        help="Target date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    logger.info("Starting batch AI explanation generation")
    results = generate_explanations(
        tickers=args.tickers,
        max_tickers=args.max_tickers,
        date=args.date,
    )

    print(f"\n{'='*50}")
    print(f"Batch Explanation Generation Complete")
    print(f"  Success: {results['success']}")
    print(f"  Skipped: {results['skipped']}")
    print(f"  Failed:  {results['failed']}")
    print(f"{'='*50}")

    # Fail CI if more than half failed
    total_attempted = results["success"] + results["failed"]
    if total_attempted > 0 and results["failed"] > total_attempted * 0.5:
        print("Too many failures — exiting with error")
        sys.exit(1)


if __name__ == "__main__":
    main()
