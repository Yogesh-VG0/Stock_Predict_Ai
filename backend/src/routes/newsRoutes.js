const { Router } = require('express');
const { getNews } = require('../controllers/newsController');
const { getUnifiedNews } = require('../services/aggregateNewsService');
const { fetchRssNews } = require('../services/rssNewsService');

const router = Router();

router.get('/aggregate', getNews);

// M6 FIX: Validate and sanitize query parameters
const VALID_SYMBOL_RE = /^[A-Z0-9.]{1,10}$/;
const MAX_SYMBOLS = 20;
const MAX_SEARCH_LENGTH = 100;

router.get('/unified', async (req, res) => {
  try {
    const {
      symbols, industries, tags, sentiment, search, from, to, page, limit, source
    } = req.query;
    
    // Validate symbols if provided
    let validatedSymbols = symbols;
    if (symbols) {
      const symbolList = symbols.split(',').slice(0, MAX_SYMBOLS);
      validatedSymbols = symbolList
        .map(s => s.trim().toUpperCase())
        .filter(s => VALID_SYMBOL_RE.test(s))
        .join(',');
    }
    
    // Sanitize search query
    let sanitizedSearch = search;
    if (search && typeof search === 'string') {
      sanitizedSearch = search.slice(0, MAX_SEARCH_LENGTH).replace(/[<>]/g, '');
    }
    
    const news = await getUnifiedNews({
      symbols: validatedSymbols,
      industries,
      tags,
      sentiment,
      search: sanitizedSearch,
      from,
      to,
      page: page ? Number(page) : undefined,
      limit: limit ? Math.min(Number(limit) || 20, 100) : undefined,
      source,
    });
    res.json(news);
  } catch (err) {
    console.error('News API error:', err.message);
    res.status(500).json({ error: 'Failed to fetch news' });
  }
});

router.get('/rss', async (req, res) => {
  const { symbol } = req.query;
  if (!symbol) return res.status(400).json({ error: 'Missing symbol' });
  
  const upperSymbol = symbol.toUpperCase();
  if (!VALID_SYMBOL_RE.test(upperSymbol)) {
    return res.status(400).json({ error: 'Invalid symbol format' });
  }
  
  try {
    const news = await fetchRssNews(upperSymbol);
    res.json({ data: news });
  } catch (err) {
    console.error('RSS news error:', err.message);
    res.status(500).json({ error: 'Failed to fetch RSS news' });
  }
});

module.exports = router;