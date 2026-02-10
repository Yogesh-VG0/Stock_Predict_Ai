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

// Fetch Fear & Greed Index from RapidAPI
async function fetchFearGreedIndex() {
  try {
    const rapidApiKey = process.env.RAPIDAPI_KEY;
    if (!rapidApiKey) {
      console.warn('⚠️ RAPIDAPI_KEY not configured - using fallback Fear & Greed data');
      return null;
    }

    const response = await axios.get('https://fear-and-greed-index.p.rapidapi.com/v1/fgi', {
      headers: {
        'x-rapidapi-key': rapidApiKey,
        'x-rapidapi-host': 'fear-and-greed-index.p.rapidapi.com',
      },
      timeout: 10000
    });
    
    return response.data;
  } catch (error) {
    console.error('Error fetching Fear & Greed Index:', error?.response?.data || error.message);
    return null;
  }
}

module.exports = { fetchMarketStatus, fetchFearGreedIndex }; 