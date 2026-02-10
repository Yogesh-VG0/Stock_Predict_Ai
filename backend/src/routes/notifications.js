const express = require('express');
const router = express.Router();
const notificationService = require('../services/notificationService');

// GET /api/notifications - Get recent notifications
router.get('/', async (req, res) => {
  try {
    const limit = Math.min(parseInt(req.query.limit) || 20, 50);
    const since = req.query.since || null;
    
    const notifications = await notificationService.getNotifications(limit, since);
    const unreadCount = await notificationService.getUnreadCount(since);
    
    res.json({
      success: true,
      notifications,
      unreadCount,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error fetching notifications:', error.message);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch notifications',
      notifications: [],
      unreadCount: 0
    });
  }
});

// GET /api/notifications/unread-count - Get just the unread count
router.get('/unread-count', async (req, res) => {
  try {
    const since = req.query.since || null;
    const count = await notificationService.getUnreadCount(since);
    
    res.json({
      success: true,
      unreadCount: count
    });
  } catch (error) {
    console.error('Error getting unread count:', error.message);
    res.status(500).json({
      success: false,
      unreadCount: 0
    });
  }
});

// POST /api/notifications/check-market - Trigger market session check (internal use)
router.post('/check-market', async (req, res) => {
  try {
    const notifications = await notificationService.checkMarketSessionNotifications();
    res.json({
      success: true,
      created: notifications.length,
      notifications
    });
  } catch (error) {
    console.error('Error checking market notifications:', error.message);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

module.exports = router;
