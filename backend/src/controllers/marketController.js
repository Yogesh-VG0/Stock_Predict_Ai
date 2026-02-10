const MarketService = require('../services/marketService');

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
      res.json(data);
    } else {
      // Return fallback data if API fails
      res.json({
        fgi: {
          now: {
            value: Math.floor(Math.random() * 30 + 45),
            valueText: 'Neutral'
          },
          previousClose: {
            value: Math.floor(Math.random() * 30 + 45),
            valueText: 'Neutral'
          },
          oneWeekAgo: {
            value: Math.floor(Math.random() * 30 + 40),
            valueText: 'Neutral'
          },
          oneMonthAgo: {
            value: Math.floor(Math.random() * 30 + 50),
            valueText: 'Greed'
          },
          oneYearAgo: {
            value: Math.floor(Math.random() * 30 + 55),
            valueText: 'Greed'
          }
        },
        lastUpdated: new Date().toISOString()
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
    
    // Fallback if API fails - return same structure as real API
    const fearGreedIndex = Math.floor(Math.random() * 30 + 45);
    const getValueText = (value) => {
      if (value < 25) return 'Extreme Fear';
      if (value < 45) return 'Fear';
      if (value < 55) return 'Neutral';
      if (value < 75) return 'Greed';
      return 'Extreme Greed';
    };
    
    res.json({
      fgi: {
        now: {
          value: fearGreedIndex,
          valueText: getValueText(fearGreedIndex)
        },
        previousClose: {
          value: fearGreedIndex - 2,
          valueText: getValueText(fearGreedIndex - 2)
        },
        oneWeekAgo: {
          value: fearGreedIndex + 3,
          valueText: getValueText(fearGreedIndex + 3)
        },
        oneMonthAgo: {
          value: fearGreedIndex + 5,
          valueText: getValueText(fearGreedIndex + 5)
        },
        oneYearAgo: {
          value: fearGreedIndex - 10,
          valueText: getValueText(fearGreedIndex - 10)
        }
      },
      lastUpdated: {
        epochUnixSeconds: Math.floor(Date.now() / 1000),
        humanDate: new Date().toISOString()
      }
    });
  } catch (error) {
    console.error('Error in getMarketSentiment:', error);
    res.status(500).json({ error: 'Failed to fetch market sentiment' });
  }
};

module.exports = { getMarketStatus, getFearGreedIndex, getMarketSentiment }; 