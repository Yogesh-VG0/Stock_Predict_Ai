const axios = require('axios');
const { DateTime } = require('luxon');
const path = require('path');
const redisClient = require('./redisClient');

// Ensure dotenv is loaded
require('dotenv').config({ path: path.resolve(__dirname, '..', '..', '.env') });

// US market sessions (Eastern Time)
const SESSIONS = [
  { name: 'pre-market', start: '04:00', end: '09:30' },
  { name: 'regular', start: '09:30', end: '16:00' },
  { name: 'after-hours', start: '16:00', end: '20:00' }
];
const TIMEZONE = 'America/New_York';
const EXCHANGE = 'US';

// In-memory fallback cache
const holidaysMemoryCache = {};
const fearGreedMemoryCache = { data: null, expiresAt: 0 };
const FEAR_GREED_CACHE_TTL_MS = 15 * 60 * 1000;

// Static NYSE market holidays (fallback when API is unavailable)
const STATIC_NYSE_HOLIDAYS = {
  2025: [
    '2025-01-01', // New Year's Day (Wednesday)
    '2025-01-20', // Martin Luther King, Jr. Day (Monday)
    '2025-02-17', // Washington's Birthday / Presidents' Day (Monday)
    '2025-04-18', // Good Friday (Friday)
    '2025-05-26', // Memorial Day (Monday)
    '2025-06-19', // Juneteenth National Independence Day (Thursday)
    '2025-07-04', // Independence Day (Friday)
    '2025-09-01', // Labor Day (Monday)
    '2025-11-27', // Thanksgiving Day (Thursday)
    '2025-12-25', // Christmas Day (Thursday)
  ],
  2026: [
    '2026-01-01', // New Year's Day (Thursday)
    '2026-01-19', // Martin Luther King, Jr. Day (Monday)
    '2026-02-16', // Washington's Birthday / Presidents' Day (Monday)
    '2026-04-03', // Good Friday (Friday)
    '2026-05-25', // Memorial Day (Monday)
    '2026-06-19', // Juneteenth National Independence Day (Friday)
    '2026-07-03', // Independence Day observed (Friday)
    '2026-09-07', // Labor Day (Monday)
    '2026-11-26', // Thanksgiving Day (Thursday)
    '2026-12-25', // Christmas Day (Friday)
  ],
};

// Helper with timeout for async operations
async function withTimeout(promise, ms, fallback) {
  let timeoutId;
  const timeoutPromise = new Promise((resolve) => {
    timeoutId = setTimeout(() => resolve(fallback), ms);
  });
  try {
    const result = await Promise.race([promise, timeoutPromise]);
    clearTimeout(timeoutId);
    return result;
  } catch (err) {
    clearTimeout(timeoutId);
    return fallback;
  }
}

// Helper to fetch US market holidays from a public API (e.g., NYSE holidays from Calendarific)
async function fetchUSHolidays(year) {
  // Try in-memory cache first (fastest)
  if (holidaysMemoryCache[year]) {
    return holidaysMemoryCache[year];
  }
  
  // Try Redis cache with timeout (don't hang if Redis is slow)
  const redisKey = `us_holidays_${year}`;
  try {
    if (redisClient && redisClient.isOpen) {
      const cached = await withTimeout(redisClient.get(redisKey), 2000, null);
      if (cached) {
        const parsed = JSON.parse(cached);
        holidaysMemoryCache[year] = parsed; // Also cache in memory
        return parsed;
      }
    }
  } catch (err) {
    // Silently ignore Redis errors - it's optional
  }
  
  // Check if Calendarific API key is configured
  const apiKey = process.env.CALENDARIFIC_API_KEY;
  if (!apiKey || apiKey === 'undefined' || apiKey === 'your_calendarific_api_key_here' || apiKey === '') {
    // Use static fallback - no need to make API call
    const staticHolidays = STATIC_NYSE_HOLIDAYS[year] || STATIC_NYSE_HOLIDAYS[2026] || [];
    holidaysMemoryCache[year] = staticHolidays;
    return staticHolidays;
  }
  
  // Fetch from API
  try {
    const url = `https://calendarific.com/api/v2/holidays?&api_key=${apiKey}&country=US&year=${year}&type=national`;
    const { data } = await axios.get(url, { timeout: 5000 });
    const holidays = data.response.holidays
      .filter(h => h.locations === 'All' || h.locations.includes('New York Stock Exchange'))
      .map(h => h.date.iso);
    
    // Cache in Redis (1 year TTL)
    try {
      if (redisClient.isOpen) {
        await redisClient.set(redisKey, JSON.stringify(holidays), { EX: 60 * 60 * 24 * 365 });
      }
    } catch (err) {
      // Silently ignore Redis errors
    }
    
    // Cache in memory
    holidaysMemoryCache[year] = holidays;
    return holidays;
  } catch (error) {
    // Log specific error type
    if (error.response?.status === 401) {
      console.warn('⚠️ Calendarific API 401 Unauthorized - using static holiday list');
    } else {
      console.warn('⚠️ Calendarific API unavailable - using static holiday list:', error.message);
    }
    
    // Use static fallback
    const staticHolidays = STATIC_NYSE_HOLIDAYS[year] || STATIC_NYSE_HOLIDAYS[2026] || [];
    holidaysMemoryCache[year] = staticHolidays;
    return staticHolidays;
  }
}

function getSession(now) {
  for (const session of SESSIONS) {
    const [startHour, startMinute] = session.start.split(':').map(Number);
    const [endHour, endMinute] = session.end.split(':').map(Number);
    const sessionStart = now.set({ hour: startHour, minute: startMinute, second: 0, millisecond: 0 });
    const sessionEnd = now.set({ hour: endHour, minute: endMinute, second: 0, millisecond: 0 });
    if (now >= sessionStart && now < sessionEnd) {
      return session.name;
    }
  }
  // If after after-hours (after 8:00 PM), return 'closed-after-hours' for clarity
  const afterHoursEnd = now.set({ hour: 20, minute: 0, second: 0, millisecond: 0 });
  if (now >= afterHoursEnd) {
    return 'closed-after-hours';
  }
  // If before pre-market (before 4:00 AM), return 'closed-before-pre-market'
  const preMarketStart = now.set({ hour: 4, minute: 0, second: 0, millisecond: 0 });
  if (now < preMarketStart) {
    return 'closed-before-pre-market';
  }
  return null;
}

function getNextSession(now, holidays) {
  // Find the next session start time (pre-market, regular, after-hours) for today or next valid day
  let day = now;
  for (let i = 0; i < 10; i++) { // look ahead up to 10 days
    const iso = day.toISODate();
    const isWeekend = day.weekday > 5;
    const isHoliday = holidays.includes(iso);
    if (!isWeekend && !isHoliday) {
      // For today, check which session is next
      for (const session of SESSIONS) {
        const [startHour, startMinute] = session.start.split(':').map(Number);
        const sessionStart = day.set({ hour: startHour, minute: startMinute, second: 0, millisecond: 0 });
        if (sessionStart > now) {
          return { name: session.name, time: sessionStart };
        }
      }
      // If no session left today, go to next valid day
    }
    day = day.plus({ days: 1 }).set({ hour: 0, minute: 0, second: 0, millisecond: 0 });
  }
  return null;
}

async function fetchMarketStatus() {
  const now = DateTime.now().setZone(TIMEZONE);
  const holidays = await fetchUSHolidays(now.year);
  const todayISO = now.toISODate();
  const isWeekend = now.weekday > 5;
  const isHoliday = holidays.includes(todayISO);

  let session = null;
  let open = false;
  let premarket = false;
  let afterhours = false;
  let next_open_time = null;
  let next_close_time = null;
  let next_session = null;

  if (!isWeekend && !isHoliday) {
    session = getSession(now);
    open = session === 'regular';
    premarket = session === 'pre-market';
    afterhours = session === 'after-hours';
    if (session === 'pre-market' || session === 'regular' || session === 'after-hours') {
      // Find session end time
      const sessionMeta = SESSIONS.find(s => s.name === session);
      const [endHour, endMinute] = sessionMeta.end.split(':').map(Number);
      next_close_time = now.set({ hour: endHour, minute: endMinute, second: 0, millisecond: 0 }).toISO();
    }
  }

  // Find next session (for frontend display)
  const nextSessionObj = getNextSession(now, holidays);
  if (nextSessionObj) {
    next_open_time = nextSessionObj.time.toISO();
    next_session = nextSessionObj.name;
  }

  return {
    open,
    premarket,
    afterhours,
    session,
    next_open_time,
    next_close_time,
    next_session,
    exchange: EXCHANGE,
    timezone: TIMEZONE,
    holiday: isHoliday ? todayISO : null,
    timestamp: now.toSeconds(),
  };
}

function clamp(value, min = 0, max = 100) {
  return Math.min(max, Math.max(min, value));
}

function getFearGreedLabel(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'Unavailable';
  const score = Number(value);
  if (score <= 20) return 'Extreme Fear';
  if (score <= 40) return 'Fear';
  if (score <= 60) return 'Neutral';
  if (score <= 80) return 'Greed';
  return 'Extreme Greed';
}

function buildFearGreedPoint(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return { value: null, valueText: 'Unavailable' };
  }

  const rounded = Math.round(clamp(Number(value)));
  return { value: rounded, valueText: getFearGreedLabel(rounded) };
}

function average(values) {
  const nums = values.filter(v => Number.isFinite(v));
  if (!nums.length) return null;
  return nums.reduce((sum, value) => sum + value, 0) / nums.length;
}

function extractYahooCloses(data) {
  const quote = data?.chart?.result?.[0]?.indicators?.quote?.[0];
  if (!quote?.close) return [];
  return quote.close.map(Number).filter(Number.isFinite);
}

async function fetchYahooCloses(symbol, range = '2y') {
  const encoded = encodeURIComponent(symbol);
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encoded}?range=${range}&interval=1d`;
  const response = await axios.get(url, {
    timeout: 8000,
    headers: {
      'User-Agent': 'StockPredictAI/1.0 (+https://stockpredict.dev)',
      Accept: 'application/json,text/plain,*/*',
    },
  });

  return extractYahooCloses(response.data);
}

function momentumScore(closes, offset = 0) {
  const end = closes.length - 1 - offset;
  if (end < 20) return null;

  const latest = closes[end];
  const window = closes.slice(Math.max(0, end - 124), end + 1);
  const sma = average(window);
  if (!Number.isFinite(latest) || !sma) return null;

  // Above the 125-day moving average = greed; below it = fear.
  const pctFromAverage = ((latest - sma) / sma) * 100;
  return clamp(50 + pctFromAverage * 4);
}

function volatilityScore(closes, offset = 0) {
  const end = closes.length - 1 - offset;
  if (end < 10) return null;

  const latest = closes[end];
  const window = closes.slice(Math.max(0, end - 49), end + 1);
  const sma = average(window);
  if (!Number.isFinite(latest) || !sma) return null;

  // Lower VIX vs its 50-day average = greed; elevated VIX = fear.
  const pctBelowAverage = ((sma - latest) / sma) * 100;
  return clamp(50 + pctBelowAverage * 5);
}

function combineMarketScores(momentum, volatility) {
  const hasMomentum = Number.isFinite(momentum);
  const hasVolatility = Number.isFinite(volatility);

  if (hasMomentum && hasVolatility) return momentum * 0.65 + volatility * 0.35;
  if (hasMomentum) return momentum;
  if (hasVolatility) return volatility;
  return null;
}

function scoreAtOffset(spxCloses, vixCloses, offset) {
  return combineMarketScores(
    momentumScore(spxCloses, offset),
    volatilityScore(vixCloses, offset),
  );
}

async function fetchFearGreedMarketProxy() {
  const [spxCloses, vixCloses] = await Promise.all([
    fetchYahooCloses('^GSPC', '2y'),
    fetchYahooCloses('^VIX', '2y'),
  ]);

  if (!spxCloses.length && !vixCloses.length) {
    throw new Error('Yahoo Finance returned no market data');
  }

  const now = scoreAtOffset(spxCloses, vixCloses, 0);
  if (now === null) {
    throw new Error('Insufficient market data to build Fear & Greed proxy');
  }

  return {
    fgi: {
      now: buildFearGreedPoint(now),
      previousClose: buildFearGreedPoint(scoreAtOffset(spxCloses, vixCloses, 1)),
      oneWeekAgo: buildFearGreedPoint(scoreAtOffset(spxCloses, vixCloses, 5)),
      oneMonthAgo: buildFearGreedPoint(scoreAtOffset(spxCloses, vixCloses, 21)),
      oneYearAgo: buildFearGreedPoint(scoreAtOffset(spxCloses, vixCloses, 252)),
    },
    lastUpdated: {
      epochUnixSeconds: Math.floor(Date.now() / 1000),
      humanDate: new Date().toISOString(),
    },
    source: 'market_proxy_yahoo_finance',
    methodology: 'Composite market sentiment proxy using S&P 500 momentum versus its 125-day average and VIX versus its 50-day average.',
  };
}

function cacheFearGreedData(data) {
  fearGreedMemoryCache.data = data;
  fearGreedMemoryCache.expiresAt = Date.now() + FEAR_GREED_CACHE_TTL_MS;
  return data;
}

function getHistoryPoint(recent, daysAgo) {
  if (!Array.isArray(recent) || !recent.length) return null;

  const targetTime = Date.now() - daysAgo * 24 * 60 * 60 * 1000;
  let closest = recent[0];
  let closestDiff = Math.abs(new Date(closest.date).getTime() - targetTime);

  for (const point of recent) {
    const diff = Math.abs(new Date(point.date).getTime() - targetTime);
    if (diff < closestDiff) {
      closest = point;
      closestDiff = diff;
    }
  }

  return closest;
}

function normalizeFearGreedChartResponse(payload) {
  const currentScore = Number(payload?.score?.score);
  if (!Number.isFinite(currentScore)) {
    throw new Error('feargreedchart.com response missing score.score');
  }

  const recent = Array.isArray(payload?.recent)
    ? payload.recent
        .map(point => ({ date: point.date, score: Number(point.score) }))
        .filter(point => point.date && Number.isFinite(point.score))
        .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
    : [];

  const previous = recent.length > 1 ? recent[recent.length - 2] : getHistoryPoint(recent, 1);
  const weekAgo = getHistoryPoint(recent, 7);
  const monthAgo = getHistoryPoint(recent, 30);
  const yearAgo = getHistoryPoint(recent, 365);
  const ts = Number(payload?.ts) || Date.now();

  return {
    fgi: {
      now: buildFearGreedPoint(currentScore),
      previousClose: buildFearGreedPoint(previous?.score),
      oneWeekAgo: buildFearGreedPoint(weekAgo?.score),
      oneMonthAgo: buildFearGreedPoint(monthAgo?.score),
      oneYearAgo: buildFearGreedPoint(yearAgo?.score),
    },
    components: payload?.score?.components || [],
    market: payload?.market || {},
    sectors: payload?.sectors || {},
    backtest: payload?.backtest || {},
    recent: recent.slice(-370),
    lastUpdated: {
      epochUnixSeconds: Math.floor(ts / 1000),
      humanDate: new Date(ts).toISOString(),
    },
    source: 'feargreedchart.com',
  };
}

async function fetchFearGreedChartIndex() {
  const response = await axios.get('https://feargreedchart.com/api/?action=all', {
    timeout: 8000,
    headers: {
      'User-Agent': 'StockPredictAI/1.0 (+https://stockpredict.dev)',
      Accept: 'application/json,text/plain,*/*',
    },
  });

  return normalizeFearGreedChartResponse(response.data);
}

// Fetch Fear & Greed Index from a no-auth public JSON endpoint. If it is
// temporarily unavailable, fall back to the local market proxy so the dashboard
// still renders live sentiment instead of an empty state.
async function fetchFearGreedIndex() {
  if (fearGreedMemoryCache.data && Date.now() < fearGreedMemoryCache.expiresAt) {
    return fearGreedMemoryCache.data;
  }

  try {
    return cacheFearGreedData(await fetchFearGreedChartIndex());
  } catch (error) {
    console.warn('⚠️ feargreedchart.com unavailable, using market proxy:', error?.response?.data || error.message);

    try {
      return cacheFearGreedData(await fetchFearGreedMarketProxy());
    } catch (fallbackError) {
      console.error('Error fetching Fear & Greed Index:', fallbackError?.response?.data || fallbackError.message);
      return null;
    }
  }
}

module.exports = { fetchMarketStatus, fetchFearGreedIndex }; 