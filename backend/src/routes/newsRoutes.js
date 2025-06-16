const { Router } = require('express');
const { getNews } = require('../controllers/newsController');
const { getUnifiedNews } = require('../services/aggregateNewsService');
const { fetchRssNews } = require('../services/rssNewsService');

const router = Router();

router.get('/aggregate', getNews);

router.get('/unified', async (req, res) => {
  console.log('Received request to /api/news/unified', req.query);
  try {
    const {
      symbols, industries, tags, sentiment, search, from, to, page, limit, source
    } = req.query;
    const news = await getUnifiedNews({
      symbols,
      industries,
      tags,
      sentiment,
      search,
      from,
      to,
      page: page ? Number(page) : undefined,
      limit: limit ? Number(limit) : undefined,
      source,
    });
    res.json(news);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

router.get('/rss', async (req, res) => {
  const { symbol } = req.query;
  if (!symbol) return res.status(400).json({ error: 'Missing symbol' });
  try {
    const news = await fetchRssNews(symbol.toUpperCase());
    res.json({ data: news });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router; 