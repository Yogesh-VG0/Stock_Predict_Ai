const express = require('express');
const router = express.Router();
const marketController = require('../controllers/marketController');

// Market status endpoint
router.get('/status', marketController.getMarketStatus);

module.exports = router; 