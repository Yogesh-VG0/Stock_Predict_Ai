const express = require('express');
const router = express.Router();
const marketController = require('../controllers/marketController');

// Market status endpoint
router.get('/status', marketController.getMarketStatus);
// Fear & Greed Index (sentiment) endpoint
router.get('/sentiment', marketController.getFearGreedIndex);

module.exports = router; 