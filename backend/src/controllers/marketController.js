const MarketService = require('../services/marketService');
const notificationService = require('../services/notificationService');

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
    if (data) {
      // Check for Fear & Greed notifications (non-blocking)
      notificationService.checkFearGreedNotification(data).catch(() => {});
      res.json(data);
    } else {
      // Return clearly labeled unavailable data if API fails
      res.json({
        fgi: {
          now: { value: null, valueText: 'Unavailable' },
          previousClose: { value: null, valueText: 'Unavailable' },
          oneWeekAgo: { value: null, valueText: 'Unavailable' },
          oneMonthAgo: { value: null, valueText: 'Unavailable' },
          oneYearAgo: { value: null, valueText: 'Unavailable' }
        },
        lastUpdated: new Date().toISOString(),
        source: 'unavailable'
      });
    }
  } catch (error) {
    console.error('Error in getFearGreedIndex:', error);
    res.status(500).json({ error: 'Failed to fetch Fear & Greed Index' });
  }
};

const getMarketSentiment = async (req, res) => {
  try {
    // Get real Fear & Greed Index data
    const fgiData = await MarketService.fetchFearGreedIndex();
    
    if (fgiData && fgiData.fgi) {
      // Return data in format expected by frontend: data.fgi.now.valueText
      return res.json(fgiData);
    }
    
    // Fallback if API fails - return null values instead of fake data
    res.json({
      fgi: {
        now: { value: null, valueText: 'Unavailable' },
        previousClose: { value: null, valueText: 'Unavailable' },
        oneWeekAgo: { value: null, valueText: 'Unavailable' },
        oneMonthAgo: { value: null, valueText: 'Unavailable' },
        oneYearAgo: { value: null, valueText: 'Unavailable' }
      },
      lastUpdated: {
        epochUnixSeconds: Math.floor(Date.now() / 1000),
        humanDate: new Date().toISOString()
      },
      source: 'unavailable'
    });
  } catch (error) {
    console.error('Error in getMarketSentiment:', error);
    res.status(500).json({ error: 'Failed to fetch market sentiment' });
  }
};

module.exports = { getMarketStatus, getFearGreedIndex, getMarketSentiment }; 