const express = require('express');
const router = express.Router();
const { requireAuth } = require('../middleware/auth');
const { 
  getWatchlist,
  addToWatchlist,
  removeFromWatchlist,
  getRealtimeUpdates,
  getWebSocketStatus,
  subscribeToUpdates
} = require('../controllers/watchlistController');

// Validate :symbol param on watchlist routes
const VALID_SYMBOL_RE = /^[A-Z0-9.\-]{1,10}$/;
router.param('symbol', (req, res, next, val) => {
  if (!VALID_SYMBOL_RE.test(val.toUpperCase())) {
    return res.status(400).json({ error: 'Invalid stock symbol' });
  }
  req.params.symbol = val.toUpperCase();
  next();
});

// ── Public routes (no auth needed) ──
// Get real-time updates for symbols
router.get('/updates/realtime', getRealtimeUpdates);

// Get WebSocket connection status
router.get('/status/websocket', getWebSocketStatus);

// Subscribe to real-time updates
router.post('/subscribe', subscribeToUpdates);

// ── Protected routes (require valid session token) ──
// Get authenticated user's watchlist
router.get('/me', requireAuth, getWatchlist);

// Add stock to authenticated user's watchlist
router.post('/me/add', requireAuth, addToWatchlist);

// Remove stock from authenticated user's watchlist
router.delete('/me/:symbol', requireAuth, removeFromWatchlist);

// ── Legacy routes for backward compatibility (redirect to /me) ──
// These allow old clients to keep working while they migrate to the new auth flow.
router.get('/:legacyId', requireAuth, getWatchlist);
router.post('/:legacyId/add', requireAuth, addToWatchlist);
router.delete('/:legacyId/:symbol', requireAuth, removeFromWatchlist);

module.exports = router;