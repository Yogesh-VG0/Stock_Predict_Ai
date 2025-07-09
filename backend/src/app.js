const express = require('express');
const cors = require('cors');
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

app.use(cors());
app.use(express.json());

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