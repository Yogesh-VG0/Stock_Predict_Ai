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
      throw new Error('Failed to fetch predictions');
    }
    
    const data = await response.json();
    return data.predictions;
  } catch (error) {
    console.error('Error fetching predictions:', error);
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