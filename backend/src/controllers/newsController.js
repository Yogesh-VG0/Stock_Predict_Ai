const { getAggregateNews } = require('../services/newsService');

async function getNews(req, res) {
  try {
    const { symbols, industries, sentiment, search, page, limit } = req.query;
    const result = await getAggregateNews({
      symbols,
      industries,
      sentiment,
      search,
      page: page ? parseInt(page, 10) : undefined,
      limit: limit ? parseInt(limit, 10) : undefined,
    });
    res.json(result);
  } catch (error) {
    console.error('News controller error:', error);
    res.status(500).json({
      error: 'Failed to fetch news',
      details: error instanceof Error ? error.message : error
    });
  }
}

module.exports = { getNews }; 