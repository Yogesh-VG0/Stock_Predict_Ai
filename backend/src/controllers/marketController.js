const MarketService = require('../services/marketService');

const getMarketStatus = async (req, res) => {
  try {
    const data = await MarketService.fetchMarketStatus();
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch market status' });
  }
};

module.exports = { getMarketStatus }; 