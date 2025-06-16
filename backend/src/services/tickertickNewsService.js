const axios = require('axios');

// In-memory rate limit: allow 10 requests per minute
const REQUEST_LIMIT = 10;
const WINDOW_MS = 60 * 1000;
let requestTimestamps = [];

function normalizeTickerTick(story) {
  // Use tags or tickers, ensure uppercase
  let tickers = [];
  if (Array.isArray(story.tickers) && story.tickers.length > 0) {
    tickers = story.tickers.map(t => t.toUpperCase());
  } else if (Array.isArray(story.tags) && story.tags.length > 0) {
    tickers = story.tags.map(t => t.toUpperCase());
  }
  return {
    uuid: `tickertick_${story.id}`,
    title: story.title,
    url: story.url,
    published_at: new Date(story.time).toISOString(),
    source: story.site,
    snippet: story.description || '',
    sentiment: 'neutral',
    sentiment_score: 0,
    tickers,
    industry: '',
    image_url: story.favicon_url || '',
    provider: 'tickertick',
  };
}

function buildTickerTickQuery({ symbols }) {
  let q = [];
  if (symbols) {
    const tickers = symbols.split(',').map(s => s.trim().toLowerCase());
    if (tickers.length === 1) {
      q.push(`tt:${tickers[0]}`);
    } else if (tickers.length > 1) {
      q.push(`(or ${tickers.map(t => `tt:${t}`).join(' ')})`);
    }
  }
  return q.length ? q.join(' ') : 'T:curated';
}

async function getTickerTickNews(params) {
  // Rate limiting: only allow 10 requests per minute
  const now = Date.now();
  requestTimestamps = requestTimestamps.filter(ts => now - ts < WINDOW_MS);
  if (requestTimestamps.length >= REQUEST_LIMIT) {
    console.warn('TickerTick API: Rate limit reached, skipping request.');
    return [];
  }
  requestTimestamps.push(now);

  const { symbols, page = 1, limit = 20 } = params;
  const n = Math.min(limit, 100); // TickerTick allows up to 1000, but keep it reasonable
  const q = buildTickerTickQuery(params);
  const url = `https://api.tickertick.com/feed?q=${encodeURIComponent(q)}&n=${n}`;
  try {
    const { data } = await axios.get(url);
    if (!data || !Array.isArray(data.stories)) return [];
    return data.stories.map(normalizeTickerTick);
  } catch (err) {
    if (err.response) {
      console.error('TickerTick API error:', err.response.status, err.response.data);
    } else {
      console.error('TickerTick API error:', err.message || err);
    }
    return [];
  }
}

module.exports = { getTickerTickNews }; 