const express = require('express');
const cors = require('cors');
const compression = require('compression');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const path = require('path');
const mongoConnection = require('./config/mongodb');
const newsRoutes = require('./routes/newsRoutes');
const marketRoutes = require('./routes/market');
const stockRoutes = require('./routes/stock');
const watchlistRoutes = require('./routes/watchlist');
const notificationRoutes = require('./routes/notifications');
const notificationService = require('./services/notificationService');
const { sessionHandler } = require('./middleware/auth');

// Note: dotenv is loaded in server.js before this module is required

const app = express();

// Trust the first proxy (Koyeb, Vercel, etc.) so express-rate-limit
// reads the real client IP from X-Forwarded-For instead of throwing.
app.set('trust proxy', 1);

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

// Security headers via helmet
app.use(helmet({
  contentSecurityPolicy: false, // CSP handled by frontend framework
  crossOriginEmbedderPolicy: false, // Allow embedded TradingView widgets
}));

// Rate limiting: 200 requests per 15-minute window per IP
const apiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 200,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests, please try again later.' },
});
app.use('/api/', apiLimiter);

// CORS: restrict to known origins; never fall back to open in production
const DEFAULT_ORIGINS = ['https://stockpredict.dev', 'https://www.stockpredict.dev'];
const allowedOrigins = process.env.CORS_ORIGINS
  ? process.env.CORS_ORIGINS.split(',').map(s => s.trim())
  : (process.env.NODE_ENV === 'production' ? DEFAULT_ORIGINS : ['http://localhost:3000', 'http://localhost:5000', ...DEFAULT_ORIGINS]);
app.use(cors({ origin: allowedOrigins, credentials: true }));

// Warn on missing security env vars at startup
if (!process.env.JWT_SECRET_KEY) {
  console.warn('⚠️  JWT_SECRET_KEY not set — using insecure default. Set this in production!');
}
if (!process.env.CORS_ORIGINS && process.env.NODE_ENV === 'production') {
  console.warn('⚠️  CORS_ORIGINS not set — using default allowlist (stockpredict.dev)');
}

app.use(express.json({ limit: '1mb' })); // Limit JSON body size

// Ticker/symbol validation regex (used in route handlers)
const VALID_SYMBOL_RE = /^[A-Z0-9.\-]{1,10}$/;

// Cache control headers for API responses
app.use((req, res, next) => {
  // Set cache headers for GET requests
  if (req.method === 'GET') {
    res.set('Cache-Control', 'public, max-age=30, stale-while-revalidate=60');
  }
  next();
});

// Auth endpoint — issue/refresh anonymous session tokens
app.post('/api/auth/session', sessionHandler);

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
        reddit: sentimentScore,
        weighted: sentimentScore,
        news: sentimentScore,
        social: sentimentScore
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
    
    // Return fallback market sentiment with null values (no fake data)
    res.json({
      market: 'US',
      fear_greed_index: null,
      fear_greed_label: 'Unavailable',
      market_sentiment: 'Neutral',
      last_updated: new Date().toISOString(),
      source: 'unavailable'
    });
  }
});

// Health check endpoint — reports real dependency status
app.get('/health', (req, res) => {
  const mongoStatus = mongoConnection.isConnected ? 'connected' : 'disconnected';
  const memUsage = process.memoryUsage();
  const uptimeSec = process.uptime();

  const health = {
    status: mongoStatus === 'connected' ? 'healthy' : 'degraded',
    timestamp: new Date().toISOString(),
    service: 'stockpredict-backend',
    uptime: `${Math.floor(uptimeSec / 3600)}h ${Math.floor((uptimeSec % 3600) / 60)}m`,
    dependencies: {
      mongodb: mongoStatus,
    },
    memory: {
      rss_mb: Math.round(memUsage.rss / 1024 / 1024),
      heap_used_mb: Math.round(memUsage.heapUsed / 1024 / 1024),
      heap_total_mb: Math.round(memUsage.heapTotal / 1024 / 1024),
    },
  };

  const httpStatus = mongoStatus === 'connected' ? 200 : 503;
  res.status(httpStatus).json(health);
});

module.exports = app;