const dotenv = require('dotenv');
dotenv.config();

const MARKETAUX_API_KEY = process.env.MARKETAUX_API_KEY;
let fetch;
try {
  fetch = global.fetch || require('node-fetch');
} catch (e) {
  fetch = require('node-fetch');
}

const vader = require('vader-sentiment');

function normalizeMarketaux(article) {
  const entities = Array.isArray(article.entities) ? article.entities : [];
  const tickers = entities
    .map(e => e.symbol)
    .filter(Boolean)
    .map(s => s.toUpperCase());
  let sentiment = 'neutral';
  if (entities[0] && typeof entities[0].sentiment_score === 'number') {
    if (entities[0].sentiment_score > 0.01) sentiment = 'positive';
    else if (entities[0].sentiment_score < -0.01) sentiment = 'negative';
  }
  return {
    uuid: article.uuid,
    title: article.title,
    url: article.url,
    published_at: article.published_at,
    source: article.source,
    snippet: article.snippet || article.description,
    sentiment,
    tickers,
    industry: entities[0] && entities[0].industry ? entities[0].industry : '',
    image_url: article.image_url || '',
    provider: 'marketaux',
  };
}

async function getAggregateNews(params) {
  if (!MARKETAUX_API_KEY) {
    throw new Error('Marketaux API key not set');
  }

  const { symbols, industries, sentiment, search, page = 1, limit = 20 } = params;

  const marketParams = new URLSearchParams({
    api_token: MARKETAUX_API_KEY,
    group_similar: 'true',
    must_have_entities: 'true',
    filter_entities: 'true',
    limit: limit.toString(),
    page: page.toString(),
    ...(symbols ? { symbols: String(symbols) } : {}),
    ...(industries ? { industries: String(industries) } : {}),
    ...(search ? { search: String(search) } : {}),
  });
  if (sentiment === 'positive') marketParams.set('sentiment_gte', '0.01');
  if (sentiment === 'neutral') { marketParams.set('sentiment_gte', '-0.01'); marketParams.set('sentiment_lte', '0.01'); }
  if (sentiment === 'negative') marketParams.set('sentiment_lte', '-0.01');

  try {
    const marketRes = await fetch(`https://api.marketaux.com/v1/news/all?${marketParams.toString()}`);
    const marketData = await marketRes.json();
    const marketArticles = (marketData.data || []).map(normalizeMarketaux);
    return {
      data: marketArticles.slice(0, limit),
      meta: {
        total: marketArticles.length,
        page,
        limit,
        sources: {
          marketaux: marketArticles.length,
        },
      },
    };
  } catch (error) {
    console.error('Aggregate news error:', error instanceof Error ? error.stack : error);
    throw new Error('Failed to fetch news');
  }
}

async function getNewsApiSectorNews(sector) {
  const NEWSAPI_KEY = process.env.NEWSAPI_KEY;
  if (!NEWSAPI_KEY) throw new Error('Missing NEWSAPI_KEY');
  const query = encodeURIComponent(
    `${sector} sector OR ${sector} industry OR ${sector} stocks OR ${sector}`
  );
  const url = `https://newsapi.org/v2/everything?q=${query}&apiKey=${NEWSAPI_KEY}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('NewsAPI error: ' + res.statusText);
  const data = await res.json();
  return (data.articles || []).map(article => {
    const textToAnalyze = [article.title, article.description].filter(Boolean).join('. ');
    const intensity = vader.SentimentIntensityAnalyzer.polarity_scores(textToAnalyze);
    let sentiment = 'neutral';
    if (intensity.compound > 0.05) sentiment = 'positive';
    else if (intensity.compound < -0.05) sentiment = 'negative';
    return {
      uuid: article.url,
      title: article.title,
      url: article.url,
      published_at: article.publishedAt,
      source: article.source?.name || 'NewsAPI',
      snippet: article.description || '',
      provider: 'newsapi',
      tickers: [],
      image_url: article.urlToImage || '',
      sentiment,
      sentiment_score: intensity.compound,
      industry: sector,
    };
  });
}

module.exports = { getAggregateNews, getNewsApiSectorNews }; 