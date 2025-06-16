const MarketService = require('../services/marketService');
const redisClient = require('../services/redisClient');

const getMarketStatus = async (req, res) => {
  try {
    const data = await MarketService.fetchMarketStatus();
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch market status' });
  }
};

const getFearGreedIndex = async (req, res) => {
  try {
    const data = await MarketService.fetchFearGreedIndex();
    if (!data) return res.status(500).json({ error: 'Failed to fetch Fear & Greed Index' });
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch Fear & Greed Index' });
  }
};

module.exports = { getMarketStatus, getFearGreedIndex }; 