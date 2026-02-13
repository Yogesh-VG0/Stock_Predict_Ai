// Use relative URLs - they'll be handled by Next.js rewrites (dev) or Vercel rewrites (prod)
const isProduction = typeof window !== 'undefined' && window.location.hostname !== 'localhost';
export const API_BASE_URL = isProduction ? '' : (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000');
export const NODE_BACKEND_URL = isProduction ? '' : (process.env.NEXT_PUBLIC_NODE_BACKEND_URL || 'http://localhost:5000');

export interface Prediction {
  predicted_price: number;
  predicted_change: number;
  current_price: number;
  confidence: number;
  price_change?: number;
}

export interface PredictionTimeframes {
  '1_day'?: Prediction;
  next_day?: Prediction;
  '7_day': Prediction;
  '30_day': Prediction;
}

export interface Predictions {
  [key: string]: PredictionTimeframes;
}

export interface TechnicalSummary {
  rsi: number;
  macd?: number;
  macd_signal: 'Bullish' | 'Bearish';
  bollinger_position: 'Upper Band' | 'Lower Band' | 'Mid-range';
  volume_trend: 'High' | 'Normal' | 'Low';
  // Extended technical indicators
  bollinger_upper?: number;
  bollinger_lower?: number;
  sma_20?: number;
  sma_50?: number;
  ema_12?: number;
  ema_26?: number;
  volume?: number;
  volume_sma?: number;
  close_price?: number;
}

export interface DataSummary {
  blended_sentiment: number;
  total_data_points: number;
  finviz_articles: number;
  reddit_posts: number;
  rss_articles: number;
  marketaux_articles: number;
}

export interface AIExplanation {
  ticker: string;
  date: string;
  explanation: string;
  data_summary: DataSummary;
  prediction_summary: {
    next_day: {
      predicted_price: number;
      confidence: number;
      price_change: number;
      price_range?: { low: number; high: number };
      model_predictions?: { lstm: number; xgboost: number; lightgbm: number };
      ensemble_weights?: { lstm: number; xgboost: number; lightgbm: number };
    };
    '7_day': {
      predicted_price: number;
      confidence: number;
      price_change: number;
      price_range?: { low: number; high: number };
      model_predictions?: { lstm: number; xgboost: number; lightgbm: number };
      ensemble_weights?: { lstm: number; xgboost: number; lightgbm: number };
    };
    '30_day': {
      predicted_price: number;
      confidence: number;
      price_change: number;
      price_range?: { low: number; high: number };
      model_predictions?: { lstm: number; xgboost: number; lightgbm: number };
      ensemble_weights?: { lstm: number; xgboost: number; lightgbm: number };
    };
  };
  technical_summary: TechnicalSummary;
  metadata: {
    data_sources: string[];
    quality_score: number;
    processing_time: string;
    api_version: string;
    explanation_length?: number;
    timestamp?: string;
  };
}

export interface Sentiment {
  reddit: number;
  weighted: number;
}

export interface MarketStatus {
  open: boolean;
  premarket: boolean;
  afterhours: boolean;
  session: string | null;
  next_open_time: string | null;
  next_close_time: string | null;
  next_session: string | null;
  exchange: string;
  timezone: string;
  holiday: string | null;
  timestamp: number;
}

export async function getPredictions(ticker?: string): Promise<Predictions | null> {
  try {
    // First try to get real predictions from Node.js backend which calls ML backend
    if (ticker) {
      try {
        const response = await fetch(`${NODE_BACKEND_URL}/api/stock/${ticker}/predictions`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
          cache: 'no-store', // Ensure fresh data from backend
        });

        if (response.ok) {
          const data = await response.json();
          console.log(`✅ Real ML predictions loaded for ${ticker}:`, data);
          return data;
        } else {
          console.log(`❌ No real predictions available for ${ticker}, status: ${response.status}`);
        }
      } catch (error) {
        console.log(`❌ Backend predictions unavailable for ${ticker}:`, error);
      }
    }

    // If Node.js backend fails, try ML backend directly
    const url = ticker
      ? `${API_BASE_URL}/api/v1/predictions/${ticker}`
      : `${API_BASE_URL}/api/v1/predictions`;

    const response = await fetch(url, { cache: 'no-store' });
    if (response.ok) {
      const data = await response.json();
      console.log(`✅ Direct ML predictions loaded for ${ticker || 'all stocks'}:`, data);
      // Support { windows } format from ML API
      const windows = data.windows || data;
      return ticker ? { [ticker]: windows } : data.predictions;
    } else {
      console.log(`❌ ML backend predictions unavailable for ${ticker || 'all stocks'}, status: ${response.status}`);
    }

    // Return null to trigger frontend fallback to mock data
    return null;
  } catch (error) {
    console.log(`❌ All prediction sources failed for ${ticker || 'all stocks'}, will use enhanced mock data:`, error);
    return null;
  }
}

export async function getSentiment(ticker: string): Promise<Sentiment | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/sentiment/${ticker}`, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error('Failed to fetch sentiment');
    }

    const data = await response.json();
    return data.sentiment;
  } catch (error) {
    console.error('Error fetching sentiment:', error);
    return null;
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/health`);
    if (!response.ok) {
      throw new Error('API health check failed');
    }

    const data = await response.json();
    return data.status === 'healthy';
  } catch (error) {
    console.error('Error checking API health:', error);
    return false;
  }
}

export async function getMarketStatus(): Promise<MarketStatus | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/market/status`);
    if (!response.ok) {
      throw new Error('Failed to fetch market status');
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching market status:', error);
    return null;
  }
}

export async function getStockPrice(symbol: string): Promise<{ price: number; change: number; changePercent: number } | null> {
  try {
    // This would be replaced with a real API call to Alpha Vantage, Finnhub, etc.
    // For now, returning mock data for demonstration
    const mockPrices: Record<string, any> = {
      'AAPL': { price: 191.45, change: 2.34, changePercent: 1.24 },
      'MSFT': { price: 378.85, change: -1.22, changePercent: -0.32 },
      'GOOGL': { price: 140.93, change: 0.87, changePercent: 0.62 },
      'AMZN': { price: 155.74, change: 3.21, changePercent: 2.10 },
      'TSLA': { price: 238.45, change: -5.67, changePercent: -2.32 },
      'NVDA': { price: 875.28, change: 12.45, changePercent: 1.44 },
    }

    return mockPrices[symbol.toUpperCase()] || {
      price: Math.random() * 500 + 50,
      change: Math.random() * 20 - 10,
      changePercent: Math.random() * 10 - 5
    }
  } catch (error) {
    console.error('Error fetching stock price:', error);
    return null;
  }
}

export async function searchStocks(query: string): Promise<Array<{ symbol: string; name: string; price?: number }> | null> {
  try {
    // Extended mock search results - replace with real API
    const mockStocks = [
      { symbol: "AAPL", name: "Apple Inc." },
      { symbol: "MSFT", name: "Microsoft Corporation" },
      { symbol: "GOOGL", name: "Alphabet Inc." },
      { symbol: "AMZN", name: "Amazon.com Inc." },
      { symbol: "TSLA", name: "Tesla Inc." },
      { symbol: "NVDA", name: "NVIDIA Corporation" },
      { symbol: "META", name: "Meta Platforms Inc." },
      { symbol: "NFLX", name: "Netflix Inc." },
      { symbol: "JPM", name: "JPMorgan Chase & Co." },
      { symbol: "V", name: "Visa Inc." },
      { symbol: "JNJ", name: "Johnson & Johnson" },
      { symbol: "WMT", name: "Walmart Inc." },
      { symbol: "PG", name: "Procter & Gamble Co." },
      { symbol: "UNH", name: "UnitedHealth Group Inc." },
      { symbol: "HD", name: "Home Depot Inc." },
      { symbol: "MA", name: "Mastercard Inc." },
      { symbol: "BAC", name: "Bank of America Corp." },
      { symbol: "XOM", name: "Exxon Mobil Corp." },
      { symbol: "LLY", name: "Eli Lilly and Co." },
      { symbol: "ABBV", name: "AbbVie Inc." }
    ]

    return mockStocks.filter(stock =>
      stock.symbol.toLowerCase().includes(query.toLowerCase()) ||
      stock.name.toLowerCase().includes(query.toLowerCase())
    ).slice(0, 10)
  } catch (error) {
    console.error('Error searching stocks:', error);
    return null;
  }
}

export function getSymbolFromCompanyName(companyName: string): string | null {
  const companyMap: Record<string, string> = {
    'apple': 'AAPL',
    'apple inc': 'AAPL',
    'microsoft': 'MSFT',
    'microsoft corporation': 'MSFT',
    'google': 'GOOGL',
    'alphabet': 'GOOGL',
    'alphabet inc': 'GOOGL',
    'amazon': 'AMZN',
    'amazon.com': 'AMZN',
    'tesla': 'TSLA',
    'tesla inc': 'TSLA',
    'nvidia': 'NVDA',
    'nvidia corporation': 'NVDA',
    'meta': 'META',
    'facebook': 'META',
    'meta platforms': 'META',
    'netflix': 'NFLX',
    'netflix inc': 'NFLX'
  }

  return companyMap[companyName.toLowerCase()] || null
}

export interface StockDetails {
  symbol: string;
  name: string;
  sector: string;
  industry: string;
  description: string;
  headquarters: string;
  founded: number;
  employees: number;
  ceo: string;
  website: string;
  price?: number;
  change?: number;
  changePercent?: number;
  aiAnalysis: {
    positiveFactors: string[];
    negativeFactors: string[];
  };
}

export async function getStockDetails(symbol: string): Promise<StockDetails | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/stock/${symbol}`, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error('Failed to fetch stock details');
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching stock details:', error);
    return null;
  }
}

export async function getAIAnalysis(symbol: string): Promise<{
  positiveFactors: string[];
  negativeFactors: string[];
  sentimentScore?: number;
  confidence?: number;
  lastUpdated?: string;
} | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/stock/${symbol}/ai-analysis`, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error('Failed to fetch AI analysis');
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching AI analysis:', error);
    return null;
  }
}

export async function getComprehensiveAIExplanation(ticker: string, date?: string): Promise<AIExplanation | null> {
  try {
    const targetDate = date || new Date().toISOString().split('T')[0]; // YYYY-MM-DD format

    // PRIORITY 1: Check if we have stored explanation in MongoDB (fastest)
    // Only try ML backend in development
    const isLocalDev = typeof window !== 'undefined' && window.location.hostname === 'localhost';
    if (isLocalDev) try {
      const storedResponse = await fetch(`http://127.0.0.1:8000/api/v1/explanation/stored/${ticker}?window=comprehensive`, { cache: 'no-store' });
      if (storedResponse.ok) {
        const storedData = await storedResponse.json();

        // Transform stored MongoDB data to frontend format
        if (storedData.explanation_data?.ai_explanation) {
          console.log(`✅ Using stored explanation for ${ticker} from MongoDB`);

          const explanationData = storedData.explanation_data;

          return {
            ticker: ticker,
            date: explanationData.explanation_date || targetDate,
            explanation: explanationData.ai_explanation,
            data_summary: {
              blended_sentiment: explanationData.sentiment_summary?.blended_sentiment || 0,
              total_data_points: (
                (explanationData.sentiment_summary?.finviz_articles || 0) +
                (explanationData.sentiment_summary?.reddit_posts || 0) +
                (explanationData.sentiment_summary?.rss_articles || 0) +
                (explanationData.sentiment_summary?.marketaux_articles || 0)
              ),
              finviz_articles: explanationData.sentiment_summary?.finviz_articles || 0,
              reddit_posts: explanationData.sentiment_summary?.reddit_posts || 0,
              rss_articles: explanationData.sentiment_summary?.rss_articles || 0,
              marketaux_articles: explanationData.sentiment_summary?.marketaux_articles || 0
            },
            prediction_summary: {
              next_day: {
                predicted_price: explanationData.prediction_data?.next_day?.predicted_price || 0,
                confidence: explanationData.prediction_data?.next_day?.confidence || 0,
                price_change: explanationData.prediction_data?.next_day?.price_change || 0,
                price_range: explanationData.prediction_data?.next_day?.price_range || {},
                model_predictions: explanationData.prediction_data?.next_day?.model_predictions || {},
                ensemble_weights: explanationData.prediction_data?.next_day?.ensemble_weights || {}
              },
              '7_day': {
                predicted_price: explanationData.prediction_data?.['7_day']?.predicted_price || 0,
                confidence: explanationData.prediction_data?.['7_day']?.confidence || 0,
                price_change: explanationData.prediction_data?.['7_day']?.price_change || 0,
                price_range: explanationData.prediction_data?.['7_day']?.price_range || {},
                model_predictions: explanationData.prediction_data?.['7_day']?.model_predictions || {},
                ensemble_weights: explanationData.prediction_data?.['7_day']?.ensemble_weights || {}
              },
              '30_day': {
                predicted_price: explanationData.prediction_data?.['30_day']?.predicted_price || 0,
                confidence: explanationData.prediction_data?.['30_day']?.confidence || 0,
                price_change: explanationData.prediction_data?.['30_day']?.price_change || 0,
                price_range: explanationData.prediction_data?.['30_day']?.price_range || {},
                model_predictions: explanationData.prediction_data?.['30_day']?.model_predictions || {},
                ensemble_weights: explanationData.prediction_data?.['30_day']?.ensemble_weights || {}
              }
            },
            technical_summary: {
              rsi: explanationData.technical_indicators?.RSI || 0,
              macd: explanationData.technical_indicators?.MACD || 0,
              macd_signal: explanationData.technical_indicators?.MACD_Signal > 0 ? 'Bullish' : 'Bearish',
              bollinger_position: explanationData.technical_indicators?.Close > explanationData.technical_indicators?.Bollinger_Upper ? 'Upper Band' :
                explanationData.technical_indicators?.Close < explanationData.technical_indicators?.Bollinger_Lower ? 'Lower Band' : 'Mid-range',
              volume_trend: explanationData.technical_indicators?.Volume > explanationData.technical_indicators?.Volume_SMA ? 'High' : 'Normal',
              // Additional technical data
              bollinger_upper: explanationData.technical_indicators?.Bollinger_Upper || 0,
              bollinger_lower: explanationData.technical_indicators?.Bollinger_Lower || 0,
              sma_20: explanationData.technical_indicators?.SMA_20 || 0,
              sma_50: explanationData.technical_indicators?.SMA_50 || 0,
              ema_12: explanationData.technical_indicators?.EMA_12 || 0,
              ema_26: explanationData.technical_indicators?.EMA_26 || 0,
              volume: explanationData.technical_indicators?.Volume || 0,
              volume_sma: explanationData.technical_indicators?.Volume_SMA || 0,
              close_price: explanationData.technical_indicators?.Close || 0
            },
            metadata: {
              data_sources: explanationData.data_sources_used || [],
              quality_score: 0.95,
              processing_time: "Retrieved from MongoDB",
              api_version: "v2.5-stored",
              explanation_length: explanationData.explanation_length || 0,
              timestamp: explanationData.timestamp || storedData.timestamp
            }
          };
        }
      }
    } catch (storedError) {
      console.log(`No stored explanation found for ${ticker}, will try generation`);
    }

    // PRIORITY 2: Try Node.js backend stored explanation endpoint
    try {
      const response = await fetch(`${API_BASE_URL}/api/stock/${ticker}/explanation?window=comprehensive`, { cache: 'no-store' });
      if (response.ok) {
        const data = await response.json();
        if (data) {
          console.log(`✅ Using Node.js backend stored explanation for ${ticker}`);
          return data;
        }
      }
    } catch (nodeError) {
      console.log('Node.js backend stored explanation not available');
    }

    // PRIORITY 3: Only generate if no stored data exists (this is expensive!)
    console.log(`⚠️ No stored explanation found for ${ticker}, this will trigger generation`);
    return null; // Return null to indicate no stored data, let component decide whether to generate

  } catch (error) {
    console.error('Error fetching comprehensive AI explanation:', error);
    return null;
  }
}

export async function generateAIExplanation(ticker: string, date?: string): Promise<AIExplanation | null> {
  try {
    const targetDate = date || new Date().toISOString().split('T')[0];

    const response = await fetch(`${API_BASE_URL}/api/stock/${ticker}/explanation/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ date: targetDate }),
      cache: 'no-store'
    });

    if (!response.ok) {
      throw new Error(`Failed to generate AI explanation: ${response.statusText}`);
    }

    const data = await response.json();
    return data.explanation;
  } catch (error) {
    console.error('Error generating AI explanation:', error);
    return null;
  }
}

export async function getStoredAIExplanation(ticker: string, window: string = 'comprehensive'): Promise<any | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/stock/${ticker}/explanation?window=${window}`, { cache: 'no-store' });

    if (!response.ok) {
      if (response.status === 404) {
        console.log(`No stored explanation available for ${ticker}-${window}`);
        return null;
      }
      throw new Error(`Failed to fetch stored explanation: ${response.statusText}`);
    }

    const data = await response.json();
    return data.explanation;
  } catch (error) {
    console.error('Error fetching stored AI explanation:', error);
    return null;
  }
}

export async function generateMockAIExplanation(ticker: string): Promise<AIExplanation> {
  // Generate a realistic mock explanation for demo purposes
  const currentDate = new Date().toISOString().split('T')[0];
  const currentPrice = await getStockPrice(ticker);
  const basePrice = currentPrice?.price || 150;

  return {
    ticker,
    date: currentDate,
    explanation: `
## Executive Summary

Based on comprehensive analysis of ${ticker}, our AI models predict moderate volatility with ${Math.random() > 0.5 ? 'bullish' : 'bearish'} bias over the next 30 days. Key drivers include technical momentum, earnings sentiment, and broader market conditions.

## Sentiment Intelligence Analysis

**Overall Sentiment Score:** ${(Math.random() * 0.4 - 0.2).toFixed(3)}

Our analysis processed ${Math.floor(Math.random() * 200 + 50)} data points across multiple sources:
- **News Sentiment:** ${Math.floor(Math.random() * 15 + 5)} articles show mixed sentiment with focus on recent earnings
- **Social Media:** ${Math.floor(Math.random() * 50 + 20)} Reddit posts indicate ${Math.random() > 0.5 ? 'growing bullish' : 'cautious bearish'} sentiment
- **Institutional Flow:** Recent analyst coverage suggests ${Math.random() > 0.5 ? 'accumulation' : 'profit-taking'} patterns

## Technical Analysis

**Key Indicators:**
- **RSI(14):** ${(Math.random() * 40 + 30).toFixed(1)} - ${Math.random() > 0.5 ? 'Neutral territory with room for movement' : 'Approaching oversold levels'}
- **MACD:** ${Math.random() > 0.5 ? 'Bullish crossover' : 'Bearish divergence'} indicating ${Math.random() > 0.5 ? 'momentum building' : 'potential reversal'}
- **Volume:** ${Math.random() > 0.5 ? 'Above average' : 'Below average'} confirming ${Math.random() > 0.5 ? 'strong conviction' : 'lack of participation'}

## Risk Assessment

**Primary Risks:**
- Market volatility remains elevated
- ${Math.random() > 0.5 ? 'Earnings guidance uncertainty' : 'Sector rotation pressures'}
- ${Math.random() > 0.5 ? 'Fed policy implications' : 'Economic data sensitivity'}

**Catalysts:**
- Upcoming earnings announcement
- ${Math.random() > 0.5 ? 'Product launch cycle' : 'Regulatory clarity'}
- Sector performance trends

## Actionable Insights

**Entry Levels:** Consider positions near $${(basePrice * 0.98).toFixed(2)} support
**Target Levels:** Initial target $${(basePrice * 1.05).toFixed(2)}, extended $${(basePrice * 1.12).toFixed(2)}
**Stop Loss:** Below $${(basePrice * 0.94).toFixed(2)} for risk management

*This analysis incorporates ${Math.floor(Math.random() * 10 + 5)} data sources and ${Math.floor(Math.random() * 50 + 100)} individual data points for comprehensive assessment.*
    `,
    data_summary: {
      blended_sentiment: Math.random() * 0.4 - 0.2,
      total_data_points: Math.floor(Math.random() * 200 + 50),
      finviz_articles: Math.floor(Math.random() * 15 + 5),
      reddit_posts: Math.floor(Math.random() * 50 + 20),
      rss_articles: Math.floor(Math.random() * 25 + 10),
      marketaux_articles: Math.floor(Math.random() * 10 + 3)
    },
    prediction_summary: {
      next_day: {
        predicted_price: basePrice * (1 + (Math.random() * 0.04 - 0.02)),
        confidence: Math.random() * 0.3 + 0.7,
        price_change: (Math.random() * 4 - 2)
      },
      '7_day': {
        predicted_price: basePrice * (1 + (Math.random() * 0.08 - 0.04)),
        confidence: Math.random() * 0.3 + 0.6,
        price_change: (Math.random() * 8 - 4)
      },
      '30_day': {
        predicted_price: basePrice * (1 + (Math.random() * 0.15 - 0.075)),
        confidence: Math.random() * 0.3 + 0.5,
        price_change: (Math.random() * 15 - 7.5)
      }
    },
    technical_summary: {
      rsi: Math.random() * 40 + 30,
      macd_signal: Math.random() > 0.5 ? 'Bullish' : 'Bearish',
      bollinger_position: ['Upper Band', 'Mid-range', 'Lower Band'][Math.floor(Math.random() * 3)] as any,
      volume_trend: ['High', 'Normal', 'Low'][Math.floor(Math.random() * 3)] as any
    },
    metadata: {
      data_sources: [
        'Finviz News Headlines',
        'Reddit Social Sentiment',
        'RSS News Feeds',
        'MarketAux Premium News',
        'Technical Indicators',
        'ML Feature Importance'
      ],
      quality_score: Math.random() * 0.3 + 0.7,
      processing_time: new Date().toISOString(),
      api_version: '2.0.0'
    }
  };
}

// Batch AI Explanation Functions
export interface BatchExplanationStatus {
  total_tickers: number;
  with_explanations: number;
  without_explanations: number;
  coverage_percentage: number;
  tickers_with_explanations: string[];
  tickers_without_explanations: string[];
  detailed_status: Array<{
    ticker: string;
    has_explanation: boolean;
    explanation_date?: string;
    generated_at?: string;
    explanation_length?: number;
    data_sources?: number;
    sentiment_score?: number;
  }>;
}

export interface BatchGenerationResult {
  batch_id: string;
  target_date: string;
  total_tickers: number;
  successful: number;
  failed: number;
  success_rate: number;
  total_processing_time_seconds: number;
  total_processing_time_minutes: number;
  performance_metrics: {
    avg_explanation_length: number;
    avg_processing_time_per_ticker: number;
    total_data_sources_used: number;
    explanations_stored_in_mongodb: number;
  };
  detailed_results: Array<{
    ticker: string;
    status: 'success' | 'error';
    explanation_length?: number;
    data_sources?: number;
    processing_time_seconds?: number;
    stored_in_mongodb?: boolean;
    sentiment_score?: number;
    prediction_confidence?: number;
    error_message?: string;
  }>;
}

export async function generateBatchAIExplanations(date?: string): Promise<BatchGenerationResult | null> {
  try {
    const targetDate = date || new Date().toISOString().split('T')[0];

    // Only try ML backend in development (localhost)
    const isLocalDev = typeof window !== 'undefined' && window.location.hostname === 'localhost';
    if (!isLocalDev) {
      console.log('Batch generation only available in development mode');
      return null;
    }

    // Try ML backend directly first
    const mlResponse = await fetch(`http://127.0.0.1:8000/api/v1/explain/batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ date: targetDate })
    });

    if (mlResponse.ok) {
      const result = await mlResponse.json();
      return result;
    }

    throw new Error(`Batch generation failed: ${mlResponse.statusText}`);
  } catch (error) {
    console.error('Error generating batch AI explanations:', error);
    return null;
  }
}

export async function getBatchExplanationStatus(): Promise<BatchExplanationStatus | null> {
  try {
    // Use Node.js backend which connects directly to MongoDB
    const backendResponse = await fetch(`${API_BASE_URL}/api/stock/batch/status`);

    if (backendResponse.ok) {
      const result = await backendResponse.json();
      return {
        total_tickers: result.total_tickers,
        with_explanations: result.with_explanations,
        without_explanations: result.without_explanations,
        coverage_percentage: result.coverage_percentage,
        tickers_with_explanations: result.available_tickers || [],
        tickers_without_explanations: [],
        detailed_status: []
      };
    }

    // Fallback to ML backend if Node.js backend fails (only in development)
    const isLocalDev = typeof window !== 'undefined' && window.location.hostname === 'localhost';
    if (isLocalDev) {
      const mlResponse = await fetch(`http://127.0.0.1:8000/api/v1/explain/batch/status`);

      if (mlResponse.ok) {
        const result = await mlResponse.json();
        return result;
      }
    }

    console.log('Batch status unavailable');
    return null;
  } catch (error) {
    console.error('Error checking batch explanation status:', error);
    return null;
  }
}

export async function getAvailableStocksWithExplanations(): Promise<string[]> {
  try {
    const response = await fetch(`${NODE_BACKEND_URL}/api/stock/batch/available`);
    if (!response.ok) {
      throw new Error('Failed to fetch available stocks');
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching available stocks:', error);
    return [];
  }
}

// Watchlist API functions
export interface WatchlistItem {
  symbol: string;
  name: string;
  sector: string;
  price: number;
  change: number;
  changePercent: number;
  high: number;
  low: number;
  open: number;
  previousClose: number;
  volume: number;
  tradeCount: number;
  timestamp: number;
  sparklineData: number[];
}

export interface WatchlistData {
  success: boolean;
  watchlist: WatchlistItem[];
  totalValue: number;
  totalChange: number;
  totalChangePercent: number;
}

export async function getWatchlist(userId: string = 'default'): Promise<WatchlistData | null> {
  try {
    const response = await fetch(`${NODE_BACKEND_URL}/api/watchlist/${userId}`);
    if (!response.ok) {
      throw new Error('Failed to fetch watchlist');
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching watchlist:', error);
    return null;
  }
}

export async function addToWatchlist(userId: string, symbol: string): Promise<{ success: boolean; message?: string; error?: string }> {
  try {
    const response = await fetch(`${NODE_BACKEND_URL}/api/watchlist/${userId}/add`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ symbol: symbol.toUpperCase() })
    });

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error adding to watchlist:', error);
    return { success: false, error: 'Failed to add to watchlist' };
  }
}

export async function removeFromWatchlist(userId: string, symbol: string): Promise<{ success: boolean; message?: string; error?: string }> {
  try {
    const response = await fetch(`${NODE_BACKEND_URL}/api/watchlist/${userId}/${symbol}`, {
      method: 'DELETE'
    });

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error removing from watchlist:', error);
    return { success: false, error: 'Failed to remove from watchlist' };
  }
}

export async function getRealtimeUpdates(symbols: string[]): Promise<{ success: boolean; updates?: any; timestamp?: number }> {
  try {
    const symbolsParam = symbols.join(',');
    const response = await fetch(`${NODE_BACKEND_URL}/api/watchlist/updates/realtime?symbols=${symbolsParam}`);

    if (!response.ok) {
      throw new Error('Failed to fetch real-time updates');
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching real-time updates:', error);
    return { success: false };
  }
}

export async function getWebSocketStatus(): Promise<{ success: boolean; connected?: boolean; subscribedSymbols?: string[] }> {
  try {
    const response = await fetch(`${NODE_BACKEND_URL}/api/watchlist/status/websocket`);

    if (!response.ok) {
      throw new Error('Failed to get WebSocket status');
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error getting WebSocket status:', error);
    return { success: false };
  }
}

export async function subscribeToUpdates(symbols: string[]): Promise<{ success: boolean; message?: string }> {
  try {
    const response = await fetch(`${NODE_BACKEND_URL}/api/watchlist/subscribe`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ symbols })
    });

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error subscribing to updates:', error);
    return { success: false };
  }
}

