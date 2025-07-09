const WebSocketService = require('../services/websocketService');
const mongoConnection = require('../config/mongodb');

// Initialize WebSocket service
const wsService = new WebSocketService();

// Connect to WebSocket on startup
wsService.connect();

// Company data mapping
const COMPANY_DATA = {
  'AAPL': { name: 'Apple Inc.', sector: 'Technology' },
  'MSFT': { name: 'Microsoft Corporation', sector: 'Technology' },
  'GOOGL': { name: 'Alphabet Inc.', sector: 'Communication Services' },
  'AMZN': { name: 'Amazon.com Inc.', sector: 'Consumer Discretionary' },
  'TSLA': { name: 'Tesla, Inc.', sector: 'Consumer Discretionary' },
  'META': { name: 'Meta Platforms, Inc.', sector: 'Communication Services' },
  'NVDA': { name: 'NVIDIA Corporation', sector: 'Technology' },
  'NFLX': { name: 'Netflix Inc.', sector: 'Communication Services' },
  'JPM': { name: 'JPMorgan Chase & Co.', sector: 'Financial Services' },
  'V': { name: 'Visa Inc.', sector: 'Financial Services' },
  'JNJ': { name: 'Johnson & Johnson', sector: 'Healthcare' },
  'WMT': { name: 'Walmart Inc.', sector: 'Consumer Staples' },
  'PG': { name: 'Procter & Gamble Co.', sector: 'Consumer Staples' },
  'UNH': { name: 'UnitedHealth Group Inc.', sector: 'Healthcare' },
  'HD': { name: 'Home Depot Inc.', sector: 'Consumer Discretionary' },
  'MA': { name: 'Mastercard Incorporated', sector: 'Financial Services' },
  'BAC': { name: 'Bank of America Corp.', sector: 'Financial Services' },
  'XOM': { name: 'Exxon Mobil Corporation', sector: 'Energy' },
  'LLY': { name: 'Eli Lilly & Company', sector: 'Healthcare' },
  'ABBV': { name: 'AbbVie Inc.', sector: 'Healthcare' },
  'COST': { name: 'Costco Wholesale Corporation', sector: 'Consumer Staples' },
  'ORCL': { name: 'Oracle Corporation', sector: 'Technology' },
  'CRM': { name: 'Salesforce, Inc.', sector: 'Technology' },
  'AVGO': { name: 'Broadcom Inc.', sector: 'Technology' },
  'BRK.B': { name: 'Berkshire Hathaway Inc.', sector: 'Financial Services' }
};

// In-memory watchlist storage (in production, use MongoDB)
const userWatchlists = new Map();

// Get user's watchlist
const getWatchlist = async (req, res) => {
  try {
    const { userId } = req.params;
    
    // Get user's watchlist from memory (or MongoDB in production)
    let userWatchlist = userWatchlists.get(userId) || [];
    
    // If no watchlist exists, create a default one
    if (userWatchlist.length === 0) {
      userWatchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX'];
      userWatchlists.set(userId, userWatchlist);
    }
    
    // Get current prices for all watchlist symbols
    const prices = await wsService.getCurrentPrices(userWatchlist);
    
    // Build watchlist with real data
    const watchlist = userWatchlist.map(symbol => {
      const priceData = prices[symbol];
      const companyData = COMPANY_DATA[symbol] || { name: symbol, sector: 'Unknown' };
      
      return {
        symbol: symbol,
        name: companyData.name,
        sector: companyData.sector,
        price: priceData?.price || 0,
        change: priceData?.change || 0,
        changePercent: priceData?.changePercent || 0,
        high: priceData?.high || 0,
        low: priceData?.low || 0,
        open: priceData?.open || 0,
        previousClose: priceData?.previousClose || 0,
        volume: priceData?.volume || 0,
        tradeCount: priceData?.tradeCount || 0,
        timestamp: priceData?.timestamp || Date.now(),
        sparklineData: generateSparklineData(priceData?.price || 100) // Mock sparkline for now
      };
    });
    
    res.json({
      success: true,
      watchlist: watchlist
    });
    
  } catch (error) {
    console.error('Error fetching watchlist:', error);
    res.status(500).json({ 
      success: false, 
      error: 'Failed to fetch watchlist',
      message: error.message 
    });
  }
};

// Add stock to watchlist
const addToWatchlist = async (req, res) => {
  try {
    const { userId } = req.params;
    const { symbol } = req.body;
    
    if (!symbol) {
      return res.status(400).json({ 
        success: false, 
        error: 'Symbol is required' 
      });
    }
    
    const upperSymbol = symbol.toUpperCase();
    
    // Validate symbol exists
    if (!COMPANY_DATA[upperSymbol]) {
      return res.status(400).json({ 
        success: false, 
        error: 'Invalid symbol' 
      });
    }
    
    // Get user's current watchlist
    let userWatchlist = userWatchlists.get(userId) || [];
    
    // Check if symbol is already in watchlist
    if (userWatchlist.includes(upperSymbol)) {
      return res.status(400).json({ 
        success: false, 
        error: `${upperSymbol} is already in your watchlist` 
      });
    }
    
    // Get current price for the symbol
    const priceData = await wsService.getCurrentPrice(upperSymbol);
    
    if (!priceData) {
      return res.status(400).json({ 
        success: false, 
        error: 'Unable to fetch price data for symbol' 
      });
    }
    
    // Add to user's watchlist
    userWatchlist.push(upperSymbol);
    userWatchlists.set(userId, userWatchlist);
    
    const companyData = COMPANY_DATA[upperSymbol];
    const watchlistItem = {
      symbol: upperSymbol,
      name: companyData.name,
      sector: companyData.sector,
      price: priceData.price,
      change: priceData.change,
      changePercent: priceData.changePercent,
      high: priceData.high,
      low: priceData.low,
      open: priceData.open,
      previousClose: priceData.previousClose,
      volume: priceData.volume,
      tradeCount: priceData.tradeCount,
      timestamp: priceData.timestamp,
      sparklineData: generateSparklineData(priceData.price)
    };
    
    res.json({
      success: true,
      message: `${upperSymbol} added to watchlist`,
      item: watchlistItem
    });
    
  } catch (error) {
    console.error('Error adding to watchlist:', error);
    res.status(500).json({ 
      success: false, 
      error: 'Failed to add to watchlist',
      message: error.message 
    });
  }
};

// Remove stock from watchlist
const removeFromWatchlist = async (req, res) => {
  try {
    const { userId, symbol } = req.params;
    
    if (!symbol) {
      return res.status(400).json({ 
        success: false, 
        error: 'Symbol is required' 
      });
    }
    
    const upperSymbol = symbol.toUpperCase();
    
    // Get user's current watchlist
    let userWatchlist = userWatchlists.get(userId) || [];
    
    // Remove symbol from watchlist
    const index = userWatchlist.indexOf(upperSymbol);
    if (index > -1) {
      userWatchlist.splice(index, 1);
      userWatchlists.set(userId, userWatchlist);
    }
    
    res.json({
      success: true,
      message: `${upperSymbol} removed from watchlist`
    });
    
  } catch (error) {
    console.error('Error removing from watchlist:', error);
    res.status(500).json({ 
      success: false, 
      error: 'Failed to remove from watchlist',
      message: error.message 
    });
  }
};

// Get real-time updates for watchlist
const getRealtimeUpdates = async (req, res) => {
  try {
    const { symbols } = req.query;
    
    if (!symbols) {
      return res.status(400).json({ 
        success: false, 
        error: 'Symbols parameter is required' 
      });
    }
    
    const symbolArray = symbols.split(',');
    
    // Get current prices for all symbols
    const prices = await wsService.getCurrentPrices(symbolArray);
    
    res.json({
      success: true,
      updates: prices,
      timestamp: Date.now()
    });
    
  } catch (error) {
    console.error('Error fetching real-time updates:', error);
    res.status(500).json({ 
      success: false, 
      error: 'Failed to fetch real-time updates',
      message: error.message 
    });
  }
};

// Get WebSocket connection status
const getWebSocketStatus = async (req, res) => {
  try {
    res.json({
      success: true,
      connected: wsService.isConnected,
      subscribedSymbols: wsService.getSubscribedSymbols(),
      reconnectAttempts: wsService.reconnectAttempts
    });
  } catch (error) {
    console.error('Error getting WebSocket status:', error);
    res.status(500).json({ 
      success: false, 
      error: 'Failed to get WebSocket status',
      message: error.message 
    });
  }
};

// Subscribe to real-time updates
const subscribeToUpdates = async (req, res) => {
  try {
    const { symbols } = req.body;
    
    if (!symbols || !Array.isArray(symbols)) {
      return res.status(400).json({ 
        success: false, 
        error: 'Symbols array is required' 
      });
    }
    
    // Remove duplicates
    const uniqueSymbols = [...new Set(symbols)];
    
    // Subscribe to each symbol
    uniqueSymbols.forEach(symbol => {
      wsService.subscribe(symbol, (tradeData) => {
        // This callback will be called when real-time data arrives
        // In a real app, you'd emit this to connected clients via Socket.IO
        console.log(`Real-time update for ${symbol}:`, tradeData);
      });
    });
    
    res.json({
      success: true,
      message: `Subscribed to ${uniqueSymbols.length} symbols`,
      subscribedSymbols: wsService.getSubscribedSymbols()
    });
    
  } catch (error) {
    console.error('Error subscribing to updates:', error);
    res.status(500).json({ 
      success: false, 
      error: 'Failed to subscribe to updates',
      message: error.message 
    });
  }
};

// Generate mock sparkline data
const generateSparklineData = (currentPrice) => {
  const data = [];
  const basePrice = currentPrice * 0.98; // Start slightly lower
  
  for (let i = 0; i < 20; i++) {
    const randomChange = (Math.random() - 0.5) * 0.02; // ±1% change
    const price = basePrice * (1 + randomChange);
    data.push(Math.round(price * 100) / 100);
  }
  
  return data;
};

module.exports = {
  getWatchlist,
  addToWatchlist,
  removeFromWatchlist,
  getRealtimeUpdates,
  getWebSocketStatus,
  subscribeToUpdates,
  wsService // Export for use in other parts of the app
}; 