const axios = require('axios');

// Polygon API DISABLED — free tier (5 req/min) causes infinite rate-limit loops
// with 26 stocks × 6 indicators = 156 calls needed. Using financialdata.net instead.
const MASSIVE_API_BASE = 'https://api.polygon.io';
const MASSIVE_API_KEY = ''; // Force disabled — use financialdata.net calculations

// Cache for technical indicators (24 hour TTL since it's end-of-day data)
const indicatorCache = new Map();
const CACHE_TTL = 24 * 60 * 60 * 1000; // 24 hours

// Cache for historical price data (shared across indicator calculations)
const historicalPriceCache = new Map();
const HIST_CACHE_TTL = 12 * 60 * 60 * 1000; // 12 hours

// financialdata.net API key (free tier, 300 req/day)
const FINANCIALDATA_API_KEY = process.env.FINANCIALDATA_API_KEY || '';

// In-flight request deduplication — prevents duplicate concurrent fetches
const inFlightRequests = new Map();

// Rate limiting (5 calls per minute for free tier)
let lastRequestTime = 0;
const MIN_REQUEST_INTERVAL = 12000; // 12 seconds between requests (5 per minute)
const MAX_RETRIES = 2; // Max retries on 429 to prevent infinite loops

/**
 * Rate-limited request helper (with max retry limit)
 */
async function makeRateLimitedRequest(url, retryCount = 0) {
  const now = Date.now();
  const timeSinceLastRequest = now - lastRequestTime;
  
  if (timeSinceLastRequest < MIN_REQUEST_INTERVAL) {
    await new Promise(resolve => setTimeout(resolve, MIN_REQUEST_INTERVAL - timeSinceLastRequest));
  }
  
  lastRequestTime = Date.now();
  
  try {
    const response = await axios.get(url, { timeout: 15000 });
    return response.data;
  } catch (error) {
    if (error.response?.status === 429 && retryCount < MAX_RETRIES) {
      console.warn(`⚠️ Massive API rate limit hit (attempt ${retryCount + 1}/${MAX_RETRIES}), waiting 60 seconds...`);
      await new Promise(resolve => setTimeout(resolve, 60000));
      return makeRateLimitedRequest(url, retryCount + 1);
    }
    throw error;
  }
}

/**
 * Get cached data or fetch fresh
 */
function getCachedOrFetch(cacheKey, fetchFn) {
  const cached = indicatorCache.get(cacheKey);
  if (cached && Date.now() < cached.expiry) {
    console.log(`📊 Cache hit for ${cacheKey}`);
    return Promise.resolve(cached.data);
  }
  return fetchFn();
}

/**
 * Calculate date range for indicators (last 2 months for better accuracy)
 */
function getDateRange() {
  const to = new Date();
  const from = new Date();
  from.setMonth(from.getMonth() - 2);
  
  return {
    from: from.toISOString().split('T')[0],
    to: to.toISOString().split('T')[0]
  };
}

/**
 * Fetch RSI (Relative Strength Index) from Massive API
 * RSI > 70 = Overbought, RSI < 30 = Oversold
 */
async function getRSI(symbol, timespan = 'day', window = 14) {
  const cacheKey = `rsi-${symbol}-${timespan}-${window}`;
  
  return getCachedOrFetch(cacheKey, async () => {
    if (!MASSIVE_API_KEY) {
      console.log('⚠️ Massive API key not configured, using calculated RSI');
      return calculateRSIFromOHLC(symbol);
    }
    
    try {
      const { from, to } = getDateRange();
      const url = `${MASSIVE_API_BASE}/v1/indicators/rsi/${symbol}?timespan=${timespan}&window=${window}&series_type=close&order=desc&limit=1&timestamp.gte=${from}&timestamp.lte=${to}&apiKey=${MASSIVE_API_KEY}`;
      
      console.log(`📈 Fetching RSI for ${symbol}...`);
      const data = await makeRateLimitedRequest(url);
      
      if (data.results?.values?.[0]) {
        const result = {
          value: data.results.values[0].value,
          timestamp: data.results.values[0].timestamp,
          signal: data.results.values[0].value > 70 ? 'Overbought' : 
                  data.results.values[0].value < 30 ? 'Oversold' : 'Neutral',
          window: window,
          source: 'massive_api'
        };
        
        indicatorCache.set(cacheKey, { data: result, expiry: Date.now() + CACHE_TTL });
        return result;
      }
      
      throw new Error('No RSI data in response');
    } catch (error) {
      console.warn(`⚠️ Failed to fetch RSI from Massive API: ${error.message}`);
      return calculateRSIFromOHLC(symbol);
    }
  });
}

/**
 * Fetch MACD (Moving Average Convergence Divergence) from Massive API
 */
async function getMACD(symbol, timespan = 'day', shortWindow = 12, longWindow = 26, signalWindow = 9) {
  const cacheKey = `macd-${symbol}-${timespan}-${shortWindow}-${longWindow}-${signalWindow}`;
  
  return getCachedOrFetch(cacheKey, async () => {
    if (!MASSIVE_API_KEY) {
      console.log('⚠️ Massive API key not configured, using calculated MACD');
      return calculateMACDFromOHLC(symbol);
    }
    
    try {
      const { from, to } = getDateRange();
      const url = `${MASSIVE_API_BASE}/v1/indicators/macd/${symbol}?timespan=${timespan}&short_window=${shortWindow}&long_window=${longWindow}&signal_window=${signalWindow}&series_type=close&order=desc&limit=1&timestamp.gte=${from}&timestamp.lte=${to}&apiKey=${MASSIVE_API_KEY}`;
      
      console.log(`📈 Fetching MACD for ${symbol}...`);
      const data = await makeRateLimitedRequest(url);
      
      if (data.results?.values?.[0]) {
        const macdData = data.results.values[0];
        const result = {
          value: macdData.value,
          signal: macdData.signal,
          histogram: macdData.histogram,
          timestamp: macdData.timestamp,
          trend: macdData.histogram > 0 ? 'Bullish' : 'Bearish',
          crossover: Math.abs(macdData.value - macdData.signal) < 0.5 ? 'Near Crossover' : 'Trending',
          source: 'massive_api'
        };
        
        indicatorCache.set(cacheKey, { data: result, expiry: Date.now() + CACHE_TTL });
        return result;
      }
      
      throw new Error('No MACD data in response');
    } catch (error) {
      console.warn(`⚠️ Failed to fetch MACD from Massive API: ${error.message}`);
      return calculateMACDFromOHLC(symbol);
    }
  });
}

/**
 * Fetch SMA (Simple Moving Average) from Massive API
 */
async function getSMA(symbol, timespan = 'day', window = 20) {
  const cacheKey = `sma-${symbol}-${timespan}-${window}`;
  
  return getCachedOrFetch(cacheKey, async () => {
    if (!MASSIVE_API_KEY) {
      console.log('⚠️ Massive API key not configured, using calculated SMA');
      return calculateSMAFromOHLC(symbol, window);
    }
    
    try {
      const { from, to } = getDateRange();
      const url = `${MASSIVE_API_BASE}/v1/indicators/sma/${symbol}?timespan=${timespan}&window=${window}&series_type=close&order=desc&limit=1&timestamp.gte=${from}&timestamp.lte=${to}&apiKey=${MASSIVE_API_KEY}`;
      
      console.log(`📈 Fetching SMA(${window}) for ${symbol}...`);
      const data = await makeRateLimitedRequest(url);
      
      if (data.results?.values?.[0]) {
        const result = {
          value: data.results.values[0].value,
          timestamp: data.results.values[0].timestamp,
          window: window,
          source: 'massive_api'
        };
        
        indicatorCache.set(cacheKey, { data: result, expiry: Date.now() + CACHE_TTL });
        return result;
      }
      
      throw new Error('No SMA data in response');
    } catch (error) {
      console.warn(`⚠️ Failed to fetch SMA from Massive API: ${error.message}`);
      return calculateSMAFromOHLC(symbol, window);
    }
  });
}

/**
 * Fetch EMA (Exponential Moving Average) from Massive API
 */
async function getEMA(symbol, timespan = 'day', window = 12) {
  const cacheKey = `ema-${symbol}-${timespan}-${window}`;
  
  return getCachedOrFetch(cacheKey, async () => {
    if (!MASSIVE_API_KEY) {
      console.log('⚠️ Massive API key not configured, using calculated EMA');
      return calculateEMAFromOHLC(symbol, window);
    }
    
    try {
      const { from, to } = getDateRange();
      const url = `${MASSIVE_API_BASE}/v1/indicators/ema/${symbol}?timespan=${timespan}&window=${window}&series_type=close&order=desc&limit=1&timestamp.gte=${from}&timestamp.lte=${to}&apiKey=${MASSIVE_API_KEY}`;
      
      console.log(`📈 Fetching EMA(${window}) for ${symbol}...`);
      const data = await makeRateLimitedRequest(url);
      
      if (data.results?.values?.[0]) {
        const result = {
          value: data.results.values[0].value,
          timestamp: data.results.values[0].timestamp,
          window: window,
          source: 'massive_api'
        };
        
        indicatorCache.set(cacheKey, { data: result, expiry: Date.now() + CACHE_TTL });
        return result;
      }
      
      throw new Error('No EMA data in response');
    } catch (error) {
      console.warn(`⚠️ Failed to fetch EMA from Massive API: ${error.message}`);
      return calculateEMAFromOHLC(symbol, window);
    }
  });
}

/**
 * Get all technical indicators for a symbol
 */
async function getAllIndicators(symbol) {
  const cacheKey = `all-indicators-${symbol}`;
  
  // Check cache first
  const cached = indicatorCache.get(cacheKey);
  if (cached && Date.now() < cached.expiry) {
    console.log(`📊 Cache hit for all indicators: ${symbol}`);
    return cached.data;
  }
  
  // In-flight deduplication: if this symbol is already being fetched, reuse the promise
  if (inFlightRequests.has(cacheKey)) {
    console.log(`📊 Dedup: reusing in-flight request for ${symbol}`);
    return inFlightRequests.get(cacheKey);
  }
  
  console.log(`📊 Fetching all technical indicators for ${symbol}...`);
  
  const fetchPromise = (async () => {
    try {
      // Fetch all indicators (with rate limiting handled internally)
      const [rsi, macd, sma20, sma50, ema12, ema26] = await Promise.all([
        getRSI(symbol).catch(e => ({ value: null, error: e.message })),
        getMACD(symbol).catch(e => ({ value: null, error: e.message })),
        getSMA(symbol, 'day', 20).catch(e => ({ value: null, error: e.message })),
        getSMA(symbol, 'day', 50).catch(e => ({ value: null, error: e.message })),
        getEMA(symbol, 'day', 12).catch(e => ({ value: null, error: e.message })),
        getEMA(symbol, 'day', 26).catch(e => ({ value: null, error: e.message }))
      ]);
      
      const result = {
        symbol: symbol.toUpperCase(),
        timestamp: new Date().toISOString(),
        indicators: {
          rsi: {
            value: rsi.value,
            signal: rsi.signal || 'Neutral',
            window: rsi.window || 14,
            interpretation: getRSIInterpretation(rsi.value)
          },
          macd: {
            value: macd.value,
            signal: macd.signal,
            histogram: macd.histogram,
            trend: macd.trend || 'Neutral',
            interpretation: getMACDInterpretation(macd)
          },
          sma: {
            sma20: sma20.value,
            sma50: sma50.value,
            trend: sma20.value && sma50.value ? 
              (sma20.value > sma50.value ? 'Bullish (Golden Cross potential)' : 'Bearish (Death Cross potential)') : 
              'Unknown'
          },
          ema: {
            ema12: ema12.value,
            ema26: ema26.value,
            trend: ema12.value && ema26.value ?
              (ema12.value > ema26.value ? 'Bullish' : 'Bearish') :
              'Unknown'
          }
        },
        summary: generateTechnicalSummary(rsi, macd, sma20, sma50),
        source: 'calculated',
        cached: false
      };
      
      // Cache for 24 hours
      indicatorCache.set(cacheKey, { data: result, expiry: Date.now() + CACHE_TTL });
      
      return result;
    } catch (error) {
      console.error(`Error fetching indicators for ${symbol}:`, error.message);
      throw error;
    } finally {
      // Remove from in-flight map once done
      inFlightRequests.delete(cacheKey);
    }
  })();
  
  // Store the promise for deduplication
  inFlightRequests.set(cacheKey, fetchPromise);
  
  return fetchPromise;
}

// Helper functions for interpretation
function getRSIInterpretation(rsi) {
  if (rsi === null || rsi === undefined) return 'Data unavailable';
  if (rsi >= 80) return 'Extremely overbought - Strong sell signal';
  if (rsi >= 70) return 'Overbought - Consider taking profits';
  if (rsi >= 60) return 'Bullish momentum - Uptrend likely';
  if (rsi >= 40) return 'Neutral - No clear direction';
  if (rsi >= 30) return 'Bearish momentum - Downtrend likely';
  if (rsi >= 20) return 'Oversold - Consider buying opportunity';
  return 'Extremely oversold - Strong buy signal';
}

function getMACDInterpretation(macd) {
  if (!macd || macd.value === null) return 'Data unavailable';
  
  if (macd.histogram > 0 && macd.value > macd.signal) {
    return 'Bullish momentum increasing';
  } else if (macd.histogram > 0 && macd.value < macd.signal) {
    return 'Bullish momentum weakening';
  } else if (macd.histogram < 0 && macd.value < macd.signal) {
    return 'Bearish momentum increasing';
  } else if (macd.histogram < 0 && macd.value > macd.signal) {
    return 'Bearish momentum weakening';
  }
  return 'Neutral - Watch for crossover';
}

function generateTechnicalSummary(rsi, macd, sma20, sma50) {
  let bullishSignals = 0;
  let bearishSignals = 0;
  
  // RSI signals
  if (rsi.value < 30) bullishSignals += 2;
  else if (rsi.value < 40) bullishSignals += 1;
  else if (rsi.value > 70) bearishSignals += 2;
  else if (rsi.value > 60) bearishSignals += 1;
  
  // MACD signals
  if (macd.trend === 'Bullish') bullishSignals += 2;
  else if (macd.trend === 'Bearish') bearishSignals += 2;
  
  // SMA signals
  if (sma20.value && sma50.value) {
    if (sma20.value > sma50.value) bullishSignals += 1;
    else bearishSignals += 1;
  }
  
  const total = bullishSignals + bearishSignals;
  if (total === 0) return { signal: 'Neutral', strength: 0, description: 'Insufficient data for analysis' };
  
  const bullishPercent = (bullishSignals / total) * 100;
  
  if (bullishPercent >= 70) {
    return { signal: 'Strong Buy', strength: bullishPercent, description: 'Multiple indicators confirm bullish trend' };
  } else if (bullishPercent >= 55) {
    return { signal: 'Buy', strength: bullishPercent, description: 'Technical indicators lean bullish' };
  } else if (bullishPercent >= 45) {
    return { signal: 'Neutral', strength: 50, description: 'Mixed signals - wait for confirmation' };
  } else if (bullishPercent >= 30) {
    return { signal: 'Sell', strength: 100 - bullishPercent, description: 'Technical indicators lean bearish' };
  } else {
    return { signal: 'Strong Sell', strength: 100 - bullishPercent, description: 'Multiple indicators confirm bearish trend' };
  }
}

// ── Historical Price Fetcher (financialdata.net - free, 300 req/day, no key) ──
async function fetchHistoricalPrices(symbol) {
  const cacheKey = `hist-prices-${symbol.toUpperCase()}`;
  const cached = historicalPriceCache.get(cacheKey);
  if (cached && Date.now() < cached.expiry) {
    return cached.data;
  }
  
  // In-flight dedup for historical price fetches
  if (inFlightRequests.has(cacheKey)) {
    return inFlightRequests.get(cacheKey);
  }
  
  const fetchPromise = (async () => {
  try {
    const keyParam = FINANCIALDATA_API_KEY ? `&key=${FINANCIALDATA_API_KEY}` : '';
    const url = `https://financialdata.net/api/v1/stock-prices?identifier=${symbol.toUpperCase()}${keyParam}`;
    console.log(`📊 Fetching historical prices from financialdata.net for ${symbol}...`);
    const response = await axios.get(url, { timeout: 15000 });
    const data = response.data;
    if (Array.isArray(data) && data.length > 0) {
      // Sort ascending by date
      data.sort((a, b) => new Date(a.date) - new Date(b.date));
      historicalPriceCache.set(cacheKey, { data, expiry: Date.now() + HIST_CACHE_TTL });
      console.log(`✅ Got ${data.length} historical prices for ${symbol} from financialdata.net`);
      return data;
    }
    return null;
  } catch (error) {
    console.warn(`⚠️ financialdata.net failed for ${symbol}: ${error.message}`);
    return null;
  } finally {
    inFlightRequests.delete(cacheKey);
  }
  })();
  
  inFlightRequests.set(cacheKey, fetchPromise);
  return fetchPromise;
}

// ── Local Technical Indicator Calculations ──
function localRSI(closes, window = 14) {
  if (!closes || closes.length < window + 1) return null;
  let gains = 0, losses = 0;
  for (let i = closes.length - window; i < closes.length; i++) {
    const change = closes[i] - closes[i - 1];
    if (change > 0) gains += change;
    else losses += Math.abs(change);
  }
  const avgGain = gains / window;
  const avgLoss = losses / window;
  if (avgLoss === 0) return 100;
  return 100 - (100 / (1 + avgGain / avgLoss));
}

function localEMA(closes, window) {
  if (!closes || closes.length < window) return null;
  const k = 2 / (window + 1);
  let ema = closes.slice(0, window).reduce((a, b) => a + b, 0) / window;
  for (let i = window; i < closes.length; i++) {
    ema = closes[i] * k + ema * (1 - k);
  }
  return ema;
}

function localSMA(closes, window) {
  if (!closes || closes.length < window) return null;
  const slice = closes.slice(-window);
  return slice.reduce((a, b) => a + b, 0) / window;
}

function localMACD(closes) {
  const ema12 = localEMA(closes, 12);
  const ema26 = localEMA(closes, 26);
  if (ema12 === null || ema26 === null) return { value: null, signal: null, histogram: null };
  const macdVal = ema12 - ema26;
  // Approximate signal line from recent MACD values
  const k = 2 / (9 + 1);
  // For signal line we need MACD series - approximate it
  const emaValues = [];
  const shortK = 2 / 13;
  const longK = 2 / 27;
  let shortEma = closes.slice(0, 12).reduce((a, b) => a + b, 0) / 12;
  let longEma = closes.slice(0, 26).reduce((a, b) => a + b, 0) / 26;
  for (let i = 26; i < closes.length; i++) {
    shortEma = closes[i] * shortK + shortEma * (1 - shortK);
    longEma = closes[i] * longK + longEma * (1 - longK);
    emaValues.push(shortEma - longEma);
  }
  if (emaValues.length < 9) return { value: macdVal, signal: null, histogram: null };
  let signalLine = emaValues.slice(0, 9).reduce((a, b) => a + b, 0) / 9;
  for (let i = 9; i < emaValues.length; i++) {
    signalLine = emaValues[i] * k + signalLine * (1 - k);
  }
  return { value: macdVal, signal: signalLine, histogram: macdVal - signalLine };
}

// Fallback functions (when Polygon API is unavailable) - calculate from free historical data
async function calculateRSIFromOHLC(symbol) {
  const prices = await fetchHistoricalPrices(symbol);
  if (prices && prices.length > 20) {
    const closes = prices.map(p => p.close);
    const rsi = localRSI(closes, 14);
    if (rsi !== null) {
      return {
        value: parseFloat(rsi.toFixed(2)),
        signal: rsi > 70 ? 'Overbought' : rsi < 30 ? 'Oversold' : 'Neutral',
        window: 14,
        source: 'financialdata_calculated'
      };
    }
  }
  return {
    value: null,
    signal: 'Unavailable',
    window: 14,
    source: 'unavailable'
  };
}

async function calculateMACDFromOHLC(symbol) {
  const prices = await fetchHistoricalPrices(symbol);
  if (prices && prices.length > 35) {
    const closes = prices.map(p => p.close);
    const macd = localMACD(closes);
    if (macd.value !== null) {
      return {
        value: parseFloat(macd.value.toFixed(4)),
        signal: macd.signal !== null ? parseFloat(macd.signal.toFixed(4)) : null,
        histogram: macd.histogram !== null ? parseFloat(macd.histogram.toFixed(4)) : null,
        trend: macd.histogram !== null ? (macd.histogram > 0 ? 'Bullish' : 'Bearish') : 'Neutral',
        source: 'financialdata_calculated'
      };
    }
  }
  return {
    value: null,
    signal: null,
    histogram: null,
    trend: 'Unavailable',
    source: 'unavailable'
  };
}

async function calculateSMAFromOHLC(symbol, window) {
  const prices = await fetchHistoricalPrices(symbol);
  if (prices && prices.length >= window) {
    const closes = prices.map(p => p.close);
    const sma = localSMA(closes, window);
    if (sma !== null) {
      return {
        value: parseFloat(sma.toFixed(2)),
        window: window,
        source: 'financialdata_calculated'
      };
    }
  }
  return {
    value: null,
    window: window,
    source: 'unavailable'
  };
}

async function calculateEMAFromOHLC(symbol, window) {
  const prices = await fetchHistoricalPrices(symbol);
  if (prices && prices.length >= window) {
    const closes = prices.map(p => p.close);
    const ema = localEMA(closes, window);
    if (ema !== null) {
      return {
        value: parseFloat(ema.toFixed(2)),
        window: window,
        source: 'financialdata_calculated'
      };
    }
  }
  return {
    value: null,
    window: window,
    source: 'unavailable'
  };
}

// Clear cache for a symbol
function clearCache(symbol) {
  const prefix = symbol ? symbol.toLowerCase() : '';
  for (const key of indicatorCache.keys()) {
    if (!symbol || key.toLowerCase().includes(prefix)) {
      indicatorCache.delete(key);
    }
  }
}

module.exports = {
  getRSI,
  getMACD,
  getSMA,
  getEMA,
  getAllIndicators,
  clearCache,
  // Export cache for debugging
  getCache: () => indicatorCache
};
