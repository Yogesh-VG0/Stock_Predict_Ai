const express = require('express');
const router = express.Router();
const marketController = require('../controllers/marketController');

// Market status endpoint
router.get('/status', marketController.getMarketStatus);

// Fear & Greed Index endpoint
router.get('/fear-greed', marketController.getFearGreedIndex);

// Market sentiment endpoint (used by frontend)
router.get('/sentiment', marketController.getMarketSentiment);

module.exports = router; 