const dotenv = require('dotenv');
dotenv.config();

const EODHD_API_KEY = process.env.EODHD_API_KEY || 'DEMO';
const BASE_URL = 'https://eodhd.com/api/news';

async function getEodhdNews(params) {
  if (!params.symbol && !params.tag) throw new Error('Either symbol or tag must be provided');
  const url = new URL(BASE_URL);
  if (params.symbol) url.searchParams.set('s', params.symbol);
  if (params.tag) url.searchParams.set('t', params.tag);
  if (params.from) url.searchParams.set('from', params.from);
  if (params.to) url.searchParams.set('to', params.to);
  url.searchParams.set('limit', String(params.limit ?? 10));
  url.searchParams.set('offset', String(params.offset ?? 0));
  url.searchParams.set('api_token', EODHD_API_KEY);
  url.searchParams.set('fmt', 'json');
  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`EODHD News API error: ${res.statusText}`);
  return await res.json();
}

module.exports = { getEodhdNews }; 