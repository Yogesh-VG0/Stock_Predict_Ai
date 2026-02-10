const WebSocket = require('ws');
const axios = require('axios');
const path = require('path');

// Ensure dotenv is loaded (safeguard if this module is required early)
require('dotenv').config({ path: path.resolve(__dirname, '..', '..', '.env') });

class WebSocketService {
  constructor() {
    this.ws = null;
    this.isConnected = false;
    this.subscribers = new Map(); // Map of symbol -> callback functions
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 5000; // 5 seconds
    this.finnhubToken = process.env.FINNHUB_API_KEY;
    this.wsUrl = 'wss://ws.finnhub.io';
    
    // Rate limiting
    this.requestQueue = [];
    this.isProcessingQueue = false;
    this.lastRequestTime = 0;
    this.minRequestInterval = 1000; // 1 second between requests
    
    // Volume tracking
    this.volumeData = new Map(); // Map of symbol -> volume data
    
    // Track if API key is valid
    this.isApiKeyValid = !!this.finnhubToken && this.finnhubToken !== 'undefined' && this.finnhubToken !== 'your_finnhub_api_key_here';
    
    if (!this.isApiKeyValid) {
      console.error('❌ FINNHUB_API_KEY not found or invalid in environment variables');
      console.error('   Get your free API key at https://finnhub.io/register');
      console.error('   Add it to your .env file: FINNHUB_API_KEY=your_key_here');
    } else {
      console.log('✅ Finnhub API key configured');
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
    if (!this.isApiKeyValid) {
      console.error('❌ Cannot connect to Finnhub WebSocket - API key missing or invalid');
      console.error('   Set FINNHUB_API_KEY in your .env file');
      return;
    }

    try {
      this.ws = new WebSocket(`${this.wsUrl}?token=${this.finnhubToken}`);
      
      this.ws.on('open', () => {
        console.log('✅ Connected to Finnhub WebSocket');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        
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
        console.error('WebSocket error:', error);
        this.isConnected = false;
      });

      this.ws.on('close', () => {
        console.log('WebSocket connection closed');
        this.isConnected = false;
        this.scheduleReconnect();
      });

    } catch (error) {
      console.error('Error creating WebSocket connection:', error);
      this.scheduleReconnect();
    }
  }

  scheduleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Scheduling WebSocket reconnect attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${this.reconnectDelay}ms`);
      
      setTimeout(() => {
        this.connect();
      }, this.reconnectDelay);
    } else {
      console.error('Max WebSocket reconnect attempts reached');
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

  // Get current price for a symbol (fallback to REST API)
  async getCurrentPrice(symbol) {
    // Don't attempt API call if key is missing/invalid
    if (!this.isApiKeyValid) {
      console.warn(`⚠️ Skipping Finnhub API call for ${symbol} - API key not configured`);
      return null;
    }

    try {
      const url = `https://finnhub.io/api/v1/quote?symbol=${symbol}&token=${this.finnhubToken}`;
      const response = await this.makeRateLimitedRequest(url);
      
      // Check for API error response
      if (response.data && response.data.error) {
        console.error(`Finnhub API error for ${symbol}:`, response.data.error);
        return null;
      }
      
      return {
        symbol: symbol,
        price: response.data.c,
        change: response.data.d,
        changePercent: response.data.dp,
        high: response.data.h,
        low: response.data.l,
        open: response.data.o,
        previousClose: response.data.pc,
        timestamp: Date.now()
      };
    } catch (error) {
      // More specific error logging
      if (error.response?.status === 401) {
        console.error(`❌ Finnhub API 401 Unauthorized for ${symbol} - check your FINNHUB_API_KEY`);
        this.isApiKeyValid = false; // Prevent future calls with invalid key
      } else if (error.response?.status === 429) {
        console.warn(`⚠️ Finnhub rate limit hit for ${symbol} - will retry after cooldown`);
      } else {
        console.error(`Error fetching current price for ${symbol}:`, error.message);
      }
      return null;
    }
  }

  // Get current prices for multiple symbols
  async getCurrentPrices(symbols) {
    const prices = {};
    
    for (const symbol of symbols) {
      const price = await this.getCurrentPrice(symbol);
      if (price) {
        // Add volume data if available
        const volumeInfo = this.volumeData.get(symbol);
        if (volumeInfo) {
          price.volume = volumeInfo.totalVolume;
          price.tradeCount = volumeInfo.tradeCount;
        } else {
          price.volume = 0;
          price.tradeCount = 0;
        }
        prices[symbol] = price;
      }
    }
    
    return prices;
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