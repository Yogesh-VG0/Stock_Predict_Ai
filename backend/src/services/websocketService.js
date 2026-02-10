const WebSocket = require('ws');
const axios = require('axios');
const path = require('path');

// Ensure dotenv is loaded (safeguard if this module is required early)
require('dotenv').config({ path: path.resolve(__dirname, '..', '..', '.env') });

// Import notification service (lazy loaded to avoid circular dependency)
let notificationService = null;
const getNotificationService = () => {
  if (!notificationService) {
    try {
      notificationService = require('./notificationService');
    } catch (e) {
      // Notification service not available
    }
  }
  return notificationService;
};

// Singleton instance - ensures only ONE WebSocket connection per API key
let instance = null;

class WebSocketService {
  constructor() {
    // Singleton pattern - return existing instance if already created
    if (instance) {
      console.log('📡 Returning existing WebSocketService singleton');
      return instance;
    }
    
    console.log('📡 Creating new WebSocketService singleton');
    instance = this;
    
    this.ws = null;
    this.isConnected = false;
    this.subscribers = new Map(); // Map of symbol -> callback functions
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.baseReconnectDelay = 5000; // Start with 5 seconds
    this.currentReconnectDelay = 5000;
    this.finnhubToken = process.env.FINNHUB_API_KEY;
    this.wsUrl = 'wss://ws.finnhub.io';
    this.reconnectTimer = null;
    this.isShuttingDown = false; // Track if server is shutting down
    
    // Rate limiting
    this.requestQueue = [];
    this.isProcessingQueue = false;
    this.lastRequestTime = 0;
    this.minRequestInterval = 1000; // 1 second between requests
    this.rateLimitedUntil = 0; // Timestamp when rate limit expires
    
    // Volume tracking
    this.volumeData = new Map(); // Map of symbol -> volume data
    
    // Price baseline tracking for alerts
    this.priceBaselines = new Map(); // Map of symbol -> { baselinePrice, lastCheckedPrice }
    
    // Price cache to reduce API calls (60 second TTL)
    this.priceCache = new Map(); // Map of symbol -> { price, expiry }
    this.CACHE_TTL = 60000; // 60 seconds cache (longer to reduce API calls)
    
    // Track if API key is valid
    this.isApiKeyValid = !!this.finnhubToken && this.finnhubToken !== 'undefined' && this.finnhubToken !== 'your_finnhub_api_key_here';
    
    if (!this.isApiKeyValid) {
      console.error('❌ FINNHUB_API_KEY not found or invalid in environment variables');
      console.error('   Get your free API key at https://finnhub.io/register');
      console.error('   Add it to your .env file: FINNHUB_API_KEY=your_key_here');
    } else {
      console.log('✅ Finnhub API key configured');
    }

    // Handle graceful shutdown - don't reconnect if server is stopping
    process.on('SIGTERM', () => {
      console.log('📴 Received SIGTERM - stopping WebSocket reconnection');
      this.isShuttingDown = true;
      this.stopReconnecting();
      this.disconnect();
    });

    process.on('SIGINT', () => {
      console.log('📴 Received SIGINT - stopping WebSocket reconnection');
      this.isShuttingDown = true;
      this.stopReconnecting();
      this.disconnect();
    });
  }

  stopReconnecting() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  // Rate-limited request helper
  async makeRateLimitedRequest(url) {
    return new Promise((resolve, reject) => {
      this.requestQueue.push({ url, resolve, reject });
      this.processQueue();
    });
  }

  async processQueue() {
    if (this.isProcessingQueue || this.requestQueue.length === 0) {
      return;
    }

    this.isProcessingQueue = true;

    while (this.requestQueue.length > 0) {
      const now = Date.now();
      const timeSinceLastRequest = now - this.lastRequestTime;
      
      if (timeSinceLastRequest < this.minRequestInterval) {
        await new Promise(resolve => setTimeout(resolve, this.minRequestInterval - timeSinceLastRequest));
      }

      const { url, resolve, reject } = this.requestQueue.shift();
      this.lastRequestTime = Date.now();

      try {
        const response = await axios.get(url, { timeout: 10000 });
        resolve(response);
      } catch (error) {
        if (error.response?.status === 429) {
          // Rate limit hit, wait longer and retry
          console.log('Rate limit hit, waiting 60 seconds...');
          await new Promise(resolve => setTimeout(resolve, 60000));
          this.requestQueue.unshift({ url, resolve, reject });
        } else {
          reject(error);
        }
      }
    }

    this.isProcessingQueue = false;
  }

  connect() {
    // Don't connect if shutting down
    if (this.isShuttingDown) {
      console.log('📴 Server shutting down - skipping WebSocket connection');
      return;
    }

    if (!this.isApiKeyValid) {
      console.error('❌ Cannot connect to Finnhub WebSocket - API key missing or invalid');
      console.error('   Set FINNHUB_API_KEY in your .env file');
      return;
    }

    // Check if we're rate limited
    const now = Date.now();
    if (now < this.rateLimitedUntil) {
      const waitTime = Math.ceil((this.rateLimitedUntil - now) / 1000);
      console.log(`⏳ Rate limited - waiting ${waitTime}s before connecting`);
      this.reconnectTimer = setTimeout(() => this.connect(), this.rateLimitedUntil - now);
      return;
    }

    try {
      this.ws = new WebSocket(`${this.wsUrl}?token=${this.finnhubToken}`);
      
      this.ws.on('open', () => {
        console.log('✅ Connected to Finnhub WebSocket');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.currentReconnectDelay = this.baseReconnectDelay; // Reset delay on success
        
        // Subscribe to all symbols that have subscribers
        const symbols = Array.from(this.subscribers.keys());
        if (symbols.length > 0) {
          this.subscribeToSymbols(symbols);
        }
      });

      this.ws.on('message', (data) => {
        try {
          const message = JSON.parse(data);
          this.handleMessage(message);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      });

      this.ws.on('error', (error) => {
        // Check for rate limit (429) error
        if (error.message && error.message.includes('429')) {
          console.warn('⚠️ Finnhub rate limit hit (429) - backing off for 60 seconds');
          this.rateLimitedUntil = Date.now() + 60000; // Wait 60 seconds
          this.currentReconnectDelay = 60000; // Set next reconnect to 60s
        } else {
          console.error('WebSocket error:', error.message || error);
        }
        this.isConnected = false;
      });

      this.ws.on('close', () => {
        console.log('WebSocket connection closed');
        this.isConnected = false;
        // Only reconnect if not shutting down
        if (!this.isShuttingDown) {
          this.scheduleReconnect();
        }
      });

    } catch (error) {
      console.error('Error creating WebSocket connection:', error);
      if (!this.isShuttingDown) {
        this.scheduleReconnect();
      }
    }
  }

  scheduleReconnect() {
    // Don't reconnect if shutting down
    if (this.isShuttingDown) {
      console.log('📴 Server shutting down - not scheduling reconnect');
      return;
    }

    // Clear any existing timer
    this.stopReconnecting();

    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      
      // Exponential backoff: 5s, 10s, 20s, 40s, 60s (capped at 60s)
      this.currentReconnectDelay = Math.min(
        this.baseReconnectDelay * Math.pow(2, this.reconnectAttempts - 1),
        60000 // Max 60 seconds
      );
      
      console.log(`⏳ Scheduling WebSocket reconnect attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${this.currentReconnectDelay / 1000}s`);
      
      this.reconnectTimer = setTimeout(() => {
        if (!this.isShuttingDown) {
          this.connect();
        }
      }, this.currentReconnectDelay);
    } else {
      console.log('⚠️ Max WebSocket reconnect attempts reached - will retry on next API request');
      // Reset attempts after a longer cooldown so it can try again later
      setTimeout(() => {
        if (!this.isShuttingDown) {
          this.reconnectAttempts = 0;
          console.log('🔄 Resetting WebSocket reconnect attempts after cooldown');
        }
      }, 300000); // Reset after 5 minutes
    }
  }

  handleMessage(message) {
    if (message.type === 'trade' && message.data) {
      message.data.forEach(trade => {
        const symbol = trade.s;
        const callbacks = this.subscribers.get(symbol);
        
        // Track volume data
        if (!this.volumeData.has(symbol)) {
          this.volumeData.set(symbol, {
            totalVolume: 0,
            lastUpdate: Date.now(),
            tradeCount: 0
          });
        }
        
        const volumeInfo = this.volumeData.get(symbol);
        volumeInfo.totalVolume += trade.v || 0;
        volumeInfo.lastUpdate = Date.now();
        volumeInfo.tradeCount += 1;
        
        // Update price cache with live WebSocket data
        this.updatePriceFromTrade(symbol, trade.p);
        
        // Check for price alerts (non-blocking)
        this.checkPriceAlertForSymbol(symbol, trade.p);
        
        if (callbacks) {
          const tradeData = {
            symbol: trade.s,
            price: trade.p,
            volume: trade.v,
            timestamp: trade.t,
            conditions: trade.c || [],
            totalVolume: volumeInfo.totalVolume,
            tradeCount: volumeInfo.tradeCount
          };
          
          callbacks.forEach(callback => {
            try {
              callback(tradeData);
            } catch (error) {
              console.error(`Error in WebSocket callback for ${symbol}:`, error);
            }
          });
        }
      });
    }
  }

  // Check for significant price movements and create notifications
  checkPriceAlertForSymbol(symbol, currentPrice) {
    if (!currentPrice || !symbol) return;
    
    const ns = getNotificationService();
    if (!ns) return;
    
    // Initialize baseline if not exists
    if (!this.priceBaselines.has(symbol)) {
      this.priceBaselines.set(symbol, {
        baselinePrice: currentPrice,
        lastCheckedPrice: currentPrice,
        lastAlertTime: 0
      });
      return;
    }
    
    const baseline = this.priceBaselines.get(symbol);
    const changePercent = ((currentPrice - baseline.baselinePrice) / baseline.baselinePrice) * 100;
    
    // Only check every 100 price updates to avoid too much processing
    if (Math.abs(currentPrice - baseline.lastCheckedPrice) < 0.01) return;
    baseline.lastCheckedPrice = currentPrice;
    
    // Create notification for significant moves (>3%)
    if (Math.abs(changePercent) >= 3) {
      ns.checkPriceAlert(symbol, currentPrice, baseline.baselinePrice, changePercent)
        .then(() => {
          // Reset baseline after alert so we don't keep alerting
          baseline.baselinePrice = currentPrice;
        })
        .catch(() => {});
    }
  }

  subscribeToSymbols(symbols) {
    if (!this.isConnected || !this.ws) {
      console.log('WebSocket not connected, will subscribe when connected');
      return;
    }

    // Remove duplicates
    const uniqueSymbols = [...new Set(symbols)];

    uniqueSymbols.forEach(symbol => {
      const subscribeMessage = {
        type: 'subscribe',
        symbol: symbol
      };
      
      try {
        this.ws.send(JSON.stringify(subscribeMessage));
        console.log(`✅ Subscribed to ${symbol}`);
      } catch (error) {
        console.error(`Error subscribing to ${symbol}:`, error);
      }
    });
  }

  unsubscribeFromSymbols(symbols) {
    if (!this.isConnected || !this.ws) {
      return;
    }

    symbols.forEach(symbol => {
      const unsubscribeMessage = {
        type: 'unsubscribe',
        symbol: symbol
      };
      
      try {
        this.ws.send(JSON.stringify(unsubscribeMessage));
        console.log(`❌ Unsubscribed from ${symbol}`);
      } catch (error) {
        console.error(`Error unsubscribing from ${symbol}:`, error);
      }
    });
  }

  subscribe(symbol, callback) {
    // Check if symbol already has subscribers
    if (!this.subscribers.has(symbol)) {
      this.subscribers.set(symbol, []);
      // Subscribe to the symbol if WebSocket is connected
      if (this.isConnected) {
        this.subscribeToSymbols([symbol]);
      }
    }
    
    // Check if callback already exists for this symbol
    const callbacks = this.subscribers.get(symbol);
    if (!callbacks.includes(callback)) {
      callbacks.push(callback);
      console.log(`📊 Added subscriber for ${symbol}`);
    } else {
      console.log(`📊 Subscriber already exists for ${symbol}`);
    }
  }

  unsubscribe(symbol, callback) {
    const callbacks = this.subscribers.get(symbol);
    if (callbacks) {
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
        
        // If no more callbacks for this symbol, remove it and unsubscribe
        if (callbacks.length === 0) {
          this.subscribers.delete(symbol);
          if (this.isConnected) {
            this.unsubscribeFromSymbols([symbol]);
          }
        }
        console.log(`📊 Removed subscriber for ${symbol}`);
      }
    }
  }

  getSubscribedSymbols() {
    return Array.from(this.subscribers.keys());
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.isConnected = false;
    this.subscribers.clear();
    console.log('WebSocket service disconnected');
  }

  // Get current price for a symbol (with caching to reduce API calls)
  async getCurrentPrice(symbol) {
    const now = Date.now();
    
    // Check cache first
    const cached = this.priceCache.get(symbol);
    if (cached && now < cached.expiry) {
      return cached.data;
    }
    
    // Don't attempt API call if key is missing/invalid
    if (!this.isApiKeyValid) {
      console.warn(`⚠️ Skipping Finnhub API call for ${symbol} - API key not configured`);
      return cached?.data || null; // Return stale cache if available
    }
    
    // Check if we're rate limited
    if (now < this.rateLimitedUntil) {
      return cached?.data || null; // Return stale cache if available
    }

    try {
      const url = `https://finnhub.io/api/v1/quote?symbol=${symbol}&token=${this.finnhubToken}`;
      const response = await this.makeRateLimitedRequest(url);
      
      // Check for API error response
      if (response.data && response.data.error) {
        console.error(`Finnhub API error for ${symbol}:`, response.data.error);
        return cached?.data || null;
      }
      
      const priceData = {
        symbol: symbol,
        price: response.data.c,
        change: response.data.d,
        changePercent: response.data.dp,
        high: response.data.h,
        low: response.data.l,
        open: response.data.o,
        previousClose: response.data.pc,
        timestamp: now
      };
      
      // Cache the result
      this.priceCache.set(symbol, {
        data: priceData,
        expiry: now + this.CACHE_TTL
      });
      
      return priceData;
    } catch (error) {
      // More specific error logging
      if (error.response?.status === 401) {
        console.error(`❌ Finnhub API 401 Unauthorized for ${symbol} - check your FINNHUB_API_KEY`);
        this.isApiKeyValid = false; // Prevent future calls with invalid key
      } else if (error.response?.status === 429) {
        console.warn(`⚠️ Finnhub rate limit hit for ${symbol} - will retry after cooldown`);
        this.rateLimitedUntil = now + 60000; // Wait 60 seconds
      } else {
        console.error(`Error fetching current price for ${symbol}:`, error.message);
      }
      return cached?.data || null; // Return stale cache if available
    }
  }

  // Get current prices for multiple symbols (with smart caching)
  async getCurrentPrices(symbols) {
    const prices = {};
    const now = Date.now();
    const symbolsToFetch = [];
    
    // First, get all cached prices (even stale ones)
    for (const symbol of symbols) {
      const cached = this.priceCache.get(symbol);
      if (cached) {
        // Always return cached data if we have it
        const volumeInfo = this.volumeData.get(symbol);
        prices[symbol] = {
          ...cached.data,
          volume: volumeInfo?.totalVolume || 0,
          tradeCount: volumeInfo?.tradeCount || 0
        };
        
        // Only add to fetch list if cache is expired AND we're not rate limited
        if (now >= cached.expiry && now >= this.rateLimitedUntil) {
          symbolsToFetch.push(symbol);
        }
      } else if (now >= this.rateLimitedUntil) {
        // No cache at all - prioritize fetching
        symbolsToFetch.unshift(symbol); // Add to front (priority)
      }
    }
    
    // Fetch missing/expired symbols
    // Use higher limit on first request (when cache is empty), lower for updates
    const cacheHitRate = Object.keys(prices).length / symbols.length;
    const maxFetch = cacheHitRate < 0.5 ? 15 : 5; // Fetch more if cache is mostly empty
    const toFetch = symbolsToFetch.slice(0, maxFetch);
    
    // Fetch in parallel (faster) but respect rate limits
    if (toFetch.length > 0 && now >= this.rateLimitedUntil) {
      const fetchPromises = toFetch.map(async (symbol) => {
        const price = await this.getCurrentPrice(symbol);
        return { symbol, price };
      });
      
      const results = await Promise.all(fetchPromises);
      
      for (const { symbol, price } of results) {
        if (price) {
          const volumeInfo = this.volumeData.get(symbol);
          prices[symbol] = {
            ...price,
            volume: volumeInfo?.totalVolume || 0,
            tradeCount: volumeInfo?.tradeCount || 0
          };
        }
      }
    }
    
    return prices;
  }
  
  // Update cache from WebSocket trade data (called from handleMessage)
  updatePriceFromTrade(symbol, price) {
    if (!symbol || !price) return;
    
    const now = Date.now();
    const cached = this.priceCache.get(symbol);
    
    // Need previousClose to calculate change - use cached value or current price
    const previousClose = cached?.data?.previousClose || price;
    const change = price - previousClose;
    const changePercent = previousClose > 0 ? (change / previousClose) * 100 : 0;
    
    // Update or create cache entry with live data
    this.priceCache.set(symbol, {
      data: {
        symbol,
        price,
        change,
        changePercent,
        high: Math.max(price, cached?.data?.high || price),
        low: cached?.data?.low ? Math.min(price, cached.data.low) : price,
        open: cached?.data?.open || price,
        previousClose: previousClose,
        timestamp: now
      },
      expiry: now + this.CACHE_TTL * 2 // WebSocket data gets longer TTL
    });
  }

  // Get volume data for a symbol
  getVolumeData(symbol) {
    return this.volumeData.get(symbol) || {
      totalVolume: 0,
      lastUpdate: Date.now(),
      tradeCount: 0
    };
  }

  // Get volume data for multiple symbols
  getVolumeDataForSymbols(symbols) {
    const volumeData = {};
    symbols.forEach(symbol => {
      volumeData[symbol] = this.getVolumeData(symbol);
    });
    return volumeData;
  }
}

module.exports = WebSocketService; 