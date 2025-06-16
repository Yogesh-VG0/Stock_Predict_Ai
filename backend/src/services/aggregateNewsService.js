const { getAggregateNews, getNewsApiSectorNews } = require('./newsService'); // Marketaux + NewsAPI
const { getFinnhubGeneralNews, getFinnhubCompanyNews, normalizeFinnhub: baseNormalizeFinnhub } = require('./finnhubNewsService');
const vader = require('vader-sentiment');
const { getTickerTickNews } = require('./tickertickNewsService');

// Static map of tickers to company name variations
const TICKER_COMPANY_MAP = {
  AAPL: ["Apple Inc.", "Apple", "Apple Incorporated"],
  MSFT: ["Microsoft Corporation", "Microsoft", "Microsoft Corp"],
  TSLA: ["Tesla, Inc.", "Tesla", "Tesla Motors"],
  NVDA: ["NVIDIA Corporation", "NVIDIA", "Nvidia Corp"],
  AMZN: ["Amazon.com, Inc.", "Amazon", "Amazon.com"],
  GOOGL: ["Alphabet Inc.", "Google", "Alphabet"],
  META: ["Meta Platforms, Inc.", "Meta", "Facebook"],
  NFLX: ["Netflix, Inc.", "Netflix"],
  AMD: ["Advanced Micro Devices, Inc.", "AMD", "Advanced Micro Devices"],
  INTC: ["Intel Corporation", "Intel", "Intel Corp"],
  // Add more as needed
};

// Map Finnhub and other categories to user-facing sectors
const CATEGORY_TO_SECTOR_MAP = {
  technology: 'Technology',
  tech: 'Technology',
  business: 'Finance',
  finance: 'Finance',
  healthcare: 'Healthcare',
  health: 'Healthcare',
  energy: 'Energy',
  automotive: 'Automotive',
  auto: 'Automotive',
  consumer: 'Consumer',
  industrials: 'Industrials',
  materials: 'Materials',
  realestate: 'Real Estate',
  utilities: 'Utilities',
  communication: 'Communication',
  communications: 'Communication',
  // Add more mappings as needed
};

function analyzeSentiment(text) {
  if (!text) return { sentiment: 'neutral', score: 0 };
  const intensity = vader.SentimentIntensityAnalyzer.polarity_scores(text);
  let sentiment = 'neutral';
  if (intensity.compound > 0.05) sentiment = 'positive';
  else if (intensity.compound < -0.05) sentiment = 'negative';
  return { sentiment, score: intensity.compound };
}

function normalizeEodhd(article, requestedSymbols) {
  // Sentiment analysis using vader-sentiment
  const textToAnalyze = [article.title, article.content].filter(Boolean).join('. ');
  const sentimentResult = analyzeSentiment(textToAnalyze);
  // Log the text, score, and label for debugging
  console.log('EODHD Sentiment Analysis:', {
    text: textToAnalyze,
    score: sentimentResult.score,
    sentiment: sentimentResult.sentiment
  });
  // If article.sentiment exists and is valid, use it; otherwise use analyzed
  let sentiment = 'neutral';
  let score = 0;
  if (article.sentiment && typeof article.sentiment.normalized === 'number') {
    score = article.sentiment.normalized;
    if (score > 0.01) sentiment = 'positive';
    else if (score < -0.01) sentiment = 'negative';
  } else {
    sentiment = sentimentResult.sentiment;
    score = sentimentResult.score;
  }
  // If article.symbols is empty, use requestedSymbols
  let tickers = article.symbols && article.symbols.length > 0
    ? article.symbols
    : (requestedSymbols ? requestedSymbols.split(',') : []);
  // Normalize EODHD tickers: add both with and without '.US' for matching
  let normalizedTickers = [];
  tickers.forEach(t => {
    normalizedTickers.push(t);
    if (t.endsWith('.US')) {
      normalizedTickers.push(t.replace(/\.US$/, ''));
    }
  });
  // Remove duplicates
  normalizedTickers = [...new Set(normalizedTickers)];
  return {
    uuid: article.link,
    title: article.title,
    url: article.link,
    published_at: article.date,
    source: article.source || 'EODHD',
    snippet: article.content?.slice(0, 200) || '',
    sentiment,
    sentiment_score: score,
    tickers: normalizedTickers,
    industry: article.tags?.[0] || '',
    image_url: '', // EODHD may not have images
    provider: 'eodhd',
  };
}

function toEodhdSymbol(symbol) {
  if (!symbol) return '';
  if (symbol.endsWith('.US')) return symbol;
  return symbol + '.US';
}

function mapCategoryToSector(category) {
  if (!category) return '';
  const key = category.toLowerCase().replace(/\s/g, '');
  return CATEGORY_TO_SECTOR_MAP[key] || category;
}

// Enhance Finnhub normalization with sentiment
function normalizeFinnhub(article) {
  const textToAnalyze = [article.headline, article.summary].filter(Boolean).join('. ');
  const sentimentResult = vader.SentimentIntensityAnalyzer.polarity_scores(textToAnalyze);
  let sentiment = 'neutral';
  if (sentimentResult.compound > 0.05) sentiment = 'positive';
  else if (sentimentResult.compound < -0.05) sentiment = 'negative';
  return {
    ...baseNormalizeFinnhub(article),
    sentiment,
    sentiment_score: sentimentResult.compound,
    industry: mapCategoryToSector(article.category),
  };
}

async function getUnifiedNews(params) {
  const {
    symbols, industries, tags, sentiment, search, from, to, page = 1, limit = 20, source = 'all'
  } = params;

  // Fetch in parallel
  const promises = [];
  let newsApiSectorPromise = null;
  if (symbols) {
    // Fetch all sources for ticker filter
    const today = new Date();
    const fromDate = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000); // last 7 days
    promises.push(getAggregateNews({ symbols, page, limit })); // Marketaux
    promises.push(getFinnhubCompanyNews(symbols.split(',')[0], fromDate.toISOString().slice(0, 10), today.toISOString().slice(0, 10)));
    promises.push(getTickerTickNews({ symbols, page, limit }));
  } else {
    // Fetch all sources as before
    promises.push(getAggregateNews({ symbols, industries, sentiment, search, page, limit }));
    if (source === 'all' || source === 'finnhub') {
      if (symbols) {
        const today = new Date();
        const fromDate = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000); // last 7 days
        promises.push(getFinnhubCompanyNews(symbols.split(',')[0], fromDate.toISOString().slice(0, 10), today.toISOString().slice(0, 10)));
      } else {
        promises.push(getFinnhubGeneralNews());
      }
    } else {
      promises.push(Promise.resolve([]));
    }
    // TickerTick
    promises.push(getTickerTickNews({ symbols, industries, sentiment, search, page, limit }));
    // NewsAPI sector news (only if industries filter is set)
    if (industries) {
      const firstIndustry = industries.split(',')[0];
      console.log('Calling NewsAPI for sector:', firstIndustry);
      newsApiSectorPromise = getNewsApiSectorNews(firstIndustry);
    }
  }

  const [marketauxRes, finnhubRes, tickertickRes] = await Promise.all(promises);
  let newsApiSectorArticles = [];
  if (newsApiSectorPromise) {
    try {
      newsApiSectorArticles = await newsApiSectorPromise;
      console.log('NewsAPI articles fetched:', newsApiSectorArticles.length);
    } catch (e) {
      newsApiSectorArticles = [];
      console.error('NewsAPI fetch error:', e);
    }
  }

  // --- FILTER AND LIMIT EACH SOURCE INDEPENDENTLY ---
  // Marketaux
  let marketauxArticles = (marketauxRes && Array.isArray(marketauxRes.data) ? marketauxRes.data : []).map((a) => ({
    ...a,
    provider: 'marketaux',
    sentiment_score: a.sentiment === 'positive' ? 1 : a.sentiment === 'negative' ? -1 : 0,
  }));
  // Finnhub
  let finnhubArticles = (Array.isArray(finnhubRes) ? finnhubRes : []).map(normalizeFinnhub);
  // TickerTick
  let tickertickArticles = Array.isArray(tickertickRes) ? tickertickRes : [];
  // NewsAPI
  let newsApiArticles = Array.isArray(newsApiSectorArticles) ? newsApiSectorArticles : [];
  console.log('NewsAPI raw articles:', newsApiArticles);

  function filterArticles(articles) {
    let filtered = articles;
    console.log('Filtering articles with sentiment:', sentiment);
    console.log('Before sentiment filter:', articles.map(a => ({ title: a.title, sentiment: a.sentiment })));
    if (sentiment) filtered = filtered.filter(a => a.sentiment && a.sentiment.toLowerCase() === sentiment.toLowerCase());
    console.log('After sentiment filter:', filtered.map(a => ({ title: a.title, sentiment: a.sentiment })));
    if (from) filtered = filtered.filter(a => new Date(a.published_at) >= new Date(from));
    if (to) filtered = filtered.filter(a => new Date(a.published_at) <= new Date(to));
    if (tags) filtered = filtered.filter(a => a.industry?.toLowerCase().includes(tags.toLowerCase()));
    if (symbols) {
      const symbolArr = symbols.split(',').map(s => s.trim().toUpperCase());
      filtered = filtered.filter(a => {
        const mainTicker = a.tickers && a.tickers[0] ? a.tickers[0].toUpperCase() : "";
        return symbolArr.includes(mainTicker);
      });
    }
    if (industries) {
      const industryArr = industries.split(',').map(i => i.trim().toLowerCase());
      filtered = filtered.filter(a => {
        const mapped = mapCategoryToSector(a.industry || '');
        return mapped && industryArr.includes(mapped.toLowerCase());
      });
    }
    if (search) {
      const searchLower = search.toLowerCase();
      filtered = filtered.filter(a =>
        (a.title && a.title.toLowerCase().includes(searchLower)) ||
        (a.snippet && a.snippet.toLowerCase().includes(searchLower))
      );
    }
    return filtered;
  }

  marketauxArticles = filterArticles(marketauxArticles);
  finnhubArticles = filterArticles(finnhubArticles);
  tickertickArticles = filterArticles(tickertickArticles);
  newsApiArticles = filterArticles(newsApiArticles);
  console.log('NewsAPI filtered articles:', newsApiArticles);

  // --- DEDUPE BY EXACT URL (case-insensitive, trimmed) ---
  const seenUrls = new Set();
  const allArticles = [...marketauxArticles, ...finnhubArticles, ...tickertickArticles, ...newsApiArticles].filter(a => {
    if (!a.url) return false;
    const urlKey = a.url.trim().toLowerCase();
    if (seenUrls.has(urlKey)) return false;
    seenUrls.add(urlKey);
    return true;
  });

  // Sort by published_at desc
  allArticles.sort((a, b) => new Date(b.published_at).getTime() - new Date(a.published_at).getTime());

  // Pagination
  const startIndex = (page - 1) * limit;
  const endIndex = startIndex + limit;
  const paged = allArticles.slice(startIndex, endIndex);

  return {
    data: paged,
    meta: {
      total: allArticles.length,
      page,
      limit,
      hasMore: endIndex < allArticles.length,
      sources: {
        marketaux: marketauxArticles.length,
        finnhub: finnhubArticles.length,
        tickertick: tickertickArticles.length,
        newsapi: newsApiArticles.length,
      },
    },
  };
}

module.exports = { getUnifiedNews }; 