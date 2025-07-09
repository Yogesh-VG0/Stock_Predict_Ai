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