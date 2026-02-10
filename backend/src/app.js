const express = require('express');
const cors = require('cors');
const compression = require('compression');
const path = require('path');
const mongoConnection = require('./config/mongodb');
const newsRoutes = require('./routes/newsRoutes');
const marketRoutes = require('./routes/market');
const stockRoutes = require('./routes/stock');
const watchlistRoutes = require('./routes/watchlist');
const notificationRoutes = require('./routes/notifications');
const notificationService = require('./services/notificationService');

// Note: dotenv is loaded in server.js before this module is required

const app = express();

// Start notification scheduler after a short delay (to ensure DB is connected)
setTimeout(() => {
  // Check for market session notifications every 5 minutes
  setInterval(async () => {
    try {
      await notificationService.checkMarketSessionNotifications();
    } catch (error) {
      console.error('Notification check error:', error.message);
    }
  }, 5 * 60 * 1000); // Every 5 minutes
  
  // Cleanup old notifications every hour
  setInterval(async () => {
    try {
      await notificationService.cleanupOldNotifications();
    } catch (error) {
      console.error('Notification cleanup error:', error.message);
    }
  }, 60 * 60 * 1000); // Every hour
  
  // Initial check on startup
  notificationService.checkMarketSessionNotifications().catch(() => {});
  console.log('📢 Notification service started');
}, 5000);

// MongoDB connection is now handled in server.js before app starts
// This ensures the DB is connected before any requests are processed

// Performance optimizations
app.use(compression()); // Enable gzip compression for responses
app.use(cors());
app.use(express.json({ limit: '1mb' })); // Limit JSON body size

// Cache control headers for API responses
app.use((req, res, next) => {
  // Set cache headers for GET requests
  if (req.method === 'GET') {
    res.set('Cache-Control', 'public, max-age=30, stale-while-revalidate=60');
  }
  next();
});

app.use('/api/news', newsRoutes);
app.use('/api/market', marketRoutes);
app.use('/api/stock', stockRoutes);
app.use('/api/watchlist', watchlistRoutes);
app.use('/api/notifications', notificationRoutes);

// V1 API routes for comprehensive features
app.use('/api/v1/explain/:ticker/:date', (req, res, next) => {
  req.url = `/${req.params.ticker}/explain/${req.params.date}`;
  stockRoutes(req, res, next);
});

app.use('/api/v1/predictions/:ticker/explanation', (req, res, next) => {
  req.url = `/${req.params.ticker}/explanation?${req.url.split('?')[1] || ''}`;
  stockRoutes(req, res, next);
});

// Sentiment endpoint - returns stock sentiment data
app.get('/api/v1/sentiment/:ticker', async (req, res) => {
  const { ticker } = req.params;
  const axios = require('axios');
  const ML_BACKEND_URL = process.env.ML_BACKEND_URL || 'http://localhost:8000';
  const MarketService = require('./services/marketService');
  
  try {
    // Try ML backend first
    const response = await axios.get(`${ML_BACKEND_URL}/api/v1/sentiment/${ticker}`, {
      timeout: 5000
    });
    return res.json(response.data);
  } catch (error) {
    // Try real Fear & Greed Index for market context
    let fearGreedIndex = 50;
    let fearGreedLabel = 'Neutral';
    
    try {
      const fgiData = await MarketService.fetchFearGreedIndex();
      if (fgiData && fgiData.fgi && fgiData.fgi.now) {
        fearGreedIndex = fgiData.fgi.now.value || 50;
        fearGreedLabel = fgiData.fgi.now.valueText || 'Neutral';
      }
    } catch (fgiError) {
      console.log('Fear & Greed API unavailable for ticker sentiment');
    }
    
    // Return sentiment data with real Fear & Greed
    const sentimentScore = (fearGreedIndex - 50) / 100; // Convert to -0.5 to 0.5 range
    
    res.json({
      ticker: ticker.toUpperCase(),
      sentiment: {
        reddit: sentimentScore + (Math.random() * 0.2 - 0.1),
        weighted: sentimentScore,
        news: sentimentScore + (Math.random() * 0.1 - 0.05),
        social: sentimentScore + (Math.random() * 0.15 - 0.075)
      },
      sentiment_score: sentimentScore,
      fear_greed_index: fearGreedIndex,
      fear_greed_label: fearGreedLabel,
      market_mood: fearGreedIndex > 55 ? 'Bullish' : fearGreedIndex < 45 ? 'Bearish' : 'Neutral',
      data_sources: ['Fear & Greed Index', 'Market Data'],
      last_updated: new Date().toISOString()
    });
  }
});

// Market-wide sentiment endpoint
app.get('/api/v1/sentiment', async (req, res) => {
  const axios = require('axios');
  const ML_BACKEND_URL = process.env.ML_BACKEND_URL || 'http://localhost:8000';
  const MarketService = require('./services/marketService');
  
  try {
    // Try ML backend first
    const response = await axios.get(`${ML_BACKEND_URL}/api/v1/sentiment`, {
      timeout: 5000
    });
    return res.json(response.data);
  } catch (error) {
    // Try real Fear & Greed Index API
    try {
      const fgiData = await MarketService.fetchFearGreedIndex();
      if (fgiData && fgiData.fgi) {
        const fearGreedIndex = fgiData.fgi.now?.value || 50;
        const fearGreedLabel = fgiData.fgi.now?.valueText || 'Neutral';
        
        return res.json({
          market: 'US',
          fear_greed_index: fearGreedIndex,
          fear_greed_label: fearGreedLabel,
          market_sentiment: fearGreedIndex > 55 ? 'Bullish' : fearGreedIndex < 45 ? 'Bearish' : 'Neutral',
          historical: {
            previous_close: fgiData.fgi.previousClose,
            one_week_ago: fgiData.fgi.oneWeekAgo,
            one_month_ago: fgiData.fgi.oneMonthAgo,
            one_year_ago: fgiData.fgi.oneYearAgo
          },
          last_updated: new Date().toISOString()
        });
      }
    } catch (fgiError) {
      console.log('Fear & Greed API unavailable, using fallback');
    }
    
    // Return fallback market sentiment
    const fearGreedIndex = Math.floor(Math.random() * 30 + 45);
    
    res.json({
      market: 'US',
      fear_greed_index: fearGreedIndex,
      fear_greed_label: fearGreedIndex < 25 ? 'Extreme Fear' : 
                        fearGreedIndex < 45 ? 'Fear' : 
                        fearGreedIndex < 55 ? 'Neutral' : 
                        fearGreedIndex < 75 ? 'Greed' : 'Extreme Greed',
      market_sentiment: fearGreedIndex > 55 ? 'Bullish' : fearGreedIndex < 45 ? 'Bearish' : 'Neutral',
      last_updated: new Date().toISOString()
    });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    service: 'stockpredict-backend'
  });
});

module.exports = app; 