let fetch;
try {
  fetch = global.fetch || require('node-fetch');
} catch (e) {
  fetch = require('node-fetch');
}

const FINNHUB_API_KEY = process.env.FINNHUB_API_KEY;
const BASE_URL = 'https://finnhub.io/api/v1';

// Fetch general market news
async function getFinnhubGeneralNews(category = 'general', minId = 0) {
  const url = `${BASE_URL}/news?category=${category}&minId=${minId}&token=${FINNHUB_API_KEY}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Finnhub General News API error: ${res.statusText}`);
  return await res.json();
}

// Fetch company news by symbol and date range
async function getFinnhubCompanyNews(symbol, from, to) {
  const url = `${BASE_URL}/company-news?symbol=${symbol}&from=${from}&to=${to}&token=${FINNHUB_API_KEY}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Finnhub Company News API error: ${res.statusText}`);
  return await res.json();
}

// Normalize Finnhub news to match aggregate structure
function normalizeFinnhub(article) {
  return {
    uuid: article.id ? String(article.id) : article.url,
    title: article.headline,
    url: article.url,
    published_at: article.datetime ? new Date(article.datetime * 1000).toISOString() : '',
    source: article.source || 'Finnhub',
    snippet: article.summary || '',
    sentiment: 'neutral', // Finnhub does not provide sentiment
    sentiment_score: 0,
    tickers: article.related ? article.related.split(',').map(t => t.trim()).filter(Boolean) : [],
    industry: article.category || '',
    image_url: article.image || '',
    provider: 'finnhub',
  };
}

module.exports = {
  getFinnhubGeneralNews,
  getFinnhubCompanyNews,
  normalizeFinnhub,
}; 