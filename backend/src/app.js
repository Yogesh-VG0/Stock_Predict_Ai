const express = require('express');
const cors = require('cors');
const compression = require('compression');
const dotenv = require('dotenv');
const mongoConnection = require('./config/mongodb');
const newsRoutes = require('./routes/newsRoutes');
const marketRoutes = require('./routes/market');
const stockRoutes = require('./routes/stock');
const watchlistRoutes = require('./routes/watchlist');

dotenv.config();

const app = express();

// Initialize MongoDB connection
mongoConnection.connect().catch(err => {
  console.error('Failed to connect to MongoDB:', err);
});

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

// V1 API routes for comprehensive features
app.use('/api/v1/explain/:ticker/:date', (req, res, next) => {
  req.url = `/${req.params.ticker}/explain/${req.params.date}`;
  stockRoutes(req, res, next);
});

app.use('/api/v1/predictions/:ticker/explanation', (req, res, next) => {
  req.url = `/${req.params.ticker}/explanation?${req.url.split('?')[1] || ''}`;
  stockRoutes(req, res, next);
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