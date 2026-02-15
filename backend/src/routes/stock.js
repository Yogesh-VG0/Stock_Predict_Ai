const express = require('express');
const router = express.Router();
const {
  getStockDetails,
  getAIAnalysis,
  getComprehensiveExplanation,
  getStoredExplanation,
  generateAIExplanation,
  getBatchExplanationStatus,
  getAvailableStocksWithExplanations,
  getPredictions,
  getTechnicalIndicators,
  searchStocks
} = require('../controllers/stockController');

// Specific routes first to avoid shadowing by /:symbol
// Get enhanced AI analysis (detailed sentiment + factors)
router.get('/:symbol/analysis', getAIAnalysis);

// Get comprehensive AI explanation for a specific date
router.get('/:symbol/explain/:date', getComprehensiveExplanation);

// Get stored explanation from database
router.get('/:symbol/explanation', getStoredExplanation);

// Generate new AI explanation (triggers ML backend)
router.post('/:symbol/explanation/generate', generateAIExplanation);

// Get real ML predictions
router.get('/:symbol/predictions', getPredictions);

// Get technical indicators (RSI, MACD, SMA, EMA)
router.get('/:symbol/indicators', getTechnicalIndicators);

// Get batch explanation status — MUST be before /:symbol catch-all
router.get('/batch/status', getBatchExplanationStatus);

// Get available stocks with explanations — MUST be before /:symbol catch-all
router.get('/batch/available', getAvailableStocksWithExplanations);

// Search stocks by query — MUST be before /:symbol catch-all
router.get('/search/:query', searchStocks);

// Get stock details (company info + basic AI analysis)
// This is LAST because it matches any single parameter
router.get('/:symbol', getStockDetails);

module.exports = router; 