export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

export interface Prediction {
  predicted_price: number;
  predicted_change: number;
  current_price: number;
}

export interface Predictions {
  [key: string]: {
    [key: string]: Prediction;
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
    const url = ticker 
      ? `${API_BASE_URL}/api/v1/predictions/${ticker}`
      : `${API_BASE_URL}/api/v1/predictions`;
    
    const response = await fetch(url);
    if (!response.ok) {
      // Log the specific error but don't throw - let the app use enhanced mocks
      console.log(`No predictions available for ${ticker || 'all stocks'} in database, will use enhanced mock data`);
      return null;
    }
    
    const data = await response.json();
    return data.predictions;
  } catch (error) {
    // Don't log as error since this is expected behavior when predictions aren't available
    console.log(`Predictions API unavailable for ${ticker || 'all stocks'}, using enhanced mock data:`, error);
    return null;
  }
}

export async function getSentiment(ticker: string): Promise<Sentiment | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/sentiment/${ticker}`);
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

export async function searchStocks(query: string): Promise<Array<{symbol: string; name: string; price?: number}> | null> {
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

