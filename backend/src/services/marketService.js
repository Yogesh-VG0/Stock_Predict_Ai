const axios = require('axios');
const { DateTime } = require('luxon');
const redisClient = require('./redisClient');

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

// Helper to fetch US market holidays from a public API (e.g., NYSE holidays from Calendarific)
async function fetchUSHolidays(year) {
  // Try Redis cache first
  const redisKey = `us_holidays_${year}`;
  try {
    if (redisClient.isOpen) {
      const cached = await redisClient.get(redisKey);
      if (cached) {
        return JSON.parse(cached);
      }
    }
  } catch (err) {
    console.error('Redis error (get):', err);
  }
  // Try in-memory cache
  if (holidaysMemoryCache[year]) {
    return holidaysMemoryCache[year];
  }
  // Fetch from API
  try {
    const apiKey = process.env.CALENDARIFIC_API_KEY;
    const url = `https://calendarific.com/api/v2/holidays?&api_key=${apiKey}&country=US&year=${year}&type=national`;
    const { data } = await axios.get(url);
    const holidays = data.response.holidays
      .filter(h => h.locations === 'All' || h.locations.includes('New York Stock Exchange'))
      .map(h => h.date.iso);
    // Cache in Redis (1 year TTL)
    try {
      if (redisClient.isOpen) {
        await redisClient.set(redisKey, JSON.stringify(holidays), { EX: 60 * 60 * 24 * 365 });
      }
    } catch (err) {
      console.error('Redis error (set):', err);
    }
    // Cache in memory
    holidaysMemoryCache[year] = holidays;
    return holidays;
  } catch (error) {
    console.error('Error fetching US holidays:', error?.response?.data || error.message);
    // Fallback to a static list for 2025 if API fails
    return [
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
    ];
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

module.exports = { fetchMarketStatus }; 