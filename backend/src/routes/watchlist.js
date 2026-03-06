const express = require('express');
const router = express.Router();
const { 
  getWatchlist,
  addToWatchlist,
  removeFromWatchlist,
  getRealtimeUpdates,
  getWebSocketStatus,
  subscribeToUpdates
} = require('../controllers/watchlistController');

// Validate :userId param — alphanumeric + hyphens only (UUID / Firebase UID)
const VALID_USERID_RE = /^[a-zA-Z0-9_\-]{1,128}$/;
router.param('userId', (req, res, next, val) => {
  if (!VALID_USERID_RE.test(val)) {
    return res.status(400).json({ error: 'Invalid userId' });
  }
  next();
});

// Validate :symbol param on watchlist routes
const VALID_SYMBOL_RE = /^[A-Z0-9.\-]{1,10}$/;
router.param('symbol', (req, res, next, val) => {
  if (!VALID_SYMBOL_RE.test(val.toUpperCase())) {
    return res.status(400).json({ error: 'Invalid stock symbol' });
  }
  req.params.symbol = val.toUpperCase();
  next();
});

// Get user's watchlist
router.get('/:userId', getWatchlist);

// Add stock to watchlist
router.post('/:userId/add', addToWatchlist);

// Remove stock from watchlist
router.delete('/:userId/:symbol', removeFromWatchlist);

// Get real-time updates for symbols
router.get('/updates/realtime', getRealtimeUpdates);

// Get WebSocket connection status
router.get('/status/websocket', getWebSocketStatus);

// Subscribe to real-time updates
router.post('/subscribe', subscribeToUpdates);

module.exports = router; 