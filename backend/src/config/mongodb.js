const mongoose = require('mongoose');
const path = require('path');

// Load .env from the backend directory
require('dotenv').config({ path: path.resolve(__dirname, '..', '..', '.env') });

// MongoDB connection string from environment variable
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/stock_predictor';

// Normalize ticker for MongoDB queries: the ML pipeline stores BRK-B (hyphen)
// but Finnhub/frontend uses BRK.B (dot). Convert dots to hyphens for DB lookups.
function normalizeTickerForDB(ticker) {
  if (!ticker) return ticker;
  return ticker.toUpperCase().replace(/\./g, '-');
}

class MongoDBConnection {
  constructor() {
    this.connection = null;
    this.db = null;
    this.isConnected = false;
  }

  async connect() {
    try {
      // Log connection attempt (mask credentials in URI)
      const maskedUri = MONGODB_URI.replace(/:\/\/[^:]+:[^@]+@/, '://*****:*****@');
      console.log(`📡 Connecting to MongoDB: ${maskedUri}`);

      this.connection = await mongoose.connect(MONGODB_URI, {
        useNewUrlParser: true,
        useUnifiedTopology: true,
        serverSelectionTimeoutMS: 10000, // 10 second timeout
        connectTimeoutMS: 10000,
      });

      this.db = this.connection.connection.db;
      this.isConnected = true;

      // Set up connection event handlers
      mongoose.connection.on('disconnected', () => {
        console.warn('⚠️ MongoDB disconnected');
        this.isConnected = false;
      });

      mongoose.connection.on('reconnected', () => {
        console.log('✅ MongoDB reconnected');
        this.isConnected = true;
      });

      mongoose.connection.on('error', (err) => {
        console.error('❌ MongoDB connection error:', err.message);
        this.isConnected = false;
      });

      console.log('✅ Connected to MongoDB (Node.js Backend)');
      return this.connection;
    } catch (error) {
      this.isConnected = false;
      console.error('❌ MongoDB connection error:', error.message);

      // Provide helpful error messages
      if (error.message.includes('ECONNREFUSED')) {
        console.error('   💡 Is MongoDB running? Try: mongod --dbpath /path/to/data');
      } else if (error.message.includes('Authentication failed')) {
        console.error('   💡 Check your MONGODB_URI credentials in .env file');
      } else if (error.message.includes('getaddrinfo ENOTFOUND')) {
        console.error('   💡 Cannot resolve MongoDB host - check your MONGODB_URI');
      }

      throw error;
    }
  }

  async disconnect() {
    if (this.connection) {
      await this.connection.connection.close();
      console.log('MongoDB connection closed');
    }
  }

  async getStoredExplanation(ticker, window = 'comprehensive') {
    try {
      if (!this.isConnected || !this.db) {
        console.warn(`⚠️ MongoDB not connected - cannot retrieve explanation for ${ticker}`);
        throw new Error('MongoDB not connected');
      }

      const collection = this.db.collection('prediction_explanations');
      const dbTicker = normalizeTickerForDB(ticker);

      const explanation = await collection.findOne(
        { ticker: dbTicker, window: window },
        { sort: { timestamp: -1 } }
      );

      return explanation;
    } catch (error) {
      console.error(`Error getting stored explanation for ${ticker}:`, error.message);
      return null;
    }
  }

  async getAvailableStocks() {
    try {
      if (!this.isConnected || !this.db) {
        console.warn('⚠️ MongoDB not connected - cannot retrieve available stocks');
        return [];
      }

      const collection = this.db.collection('prediction_explanations');

      const stocks = await collection.distinct('ticker', { window: 'comprehensive' });

      return stocks;
    } catch (error) {
      console.error('Error getting available stocks:', error.message);
      return [];
    }
  }

  async getBatchStatus() {
    try {
      if (!this.isConnected || !this.db) {
        console.warn('⚠️ MongoDB not connected - cannot retrieve batch status');
        return {
          with_explanations: 0,
          without_explanations: 0,
          total_tickers: 0,
          coverage_percentage: 0,
          available_tickers: [],
          error: 'MongoDB not connected'
        };
      }

      const collection = this.db.collection('prediction_explanations');

      // Get all tickers with explanations
      const stocksWithExplanations = await collection.distinct('ticker', { window: 'comprehensive' });

      // Define all target tickers (same as ML backend)
      const allTickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
        'JPM', 'V', 'JNJ', 'WMT', 'PG', 'UNH', 'HD', 'MA', 'BAC', 'XOM',
        'LLY', 'ABBV', 'AVGO', 'COST', 'CRM', 'ORCL', 'BRK-B'
      ];

      const coverage = stocksWithExplanations.length;
      const total = allTickers.length;
      const coveragePercentage = Math.round((coverage / total) * 100);

      return {
        with_explanations: coverage,
        without_explanations: total - coverage,
        total_tickers: total,
        coverage_percentage: coveragePercentage,
        available_tickers: stocksWithExplanations
      };
    } catch (error) {
      console.error('Error getting batch status:', error);
      return {
        with_explanations: 0,
        without_explanations: 0,
        total_tickers: 0,
        coverage_percentage: 0,
        available_tickers: []
      };
    }
  }

  async getLatestPredictions(ticker) {
    try {
      if (!this.isConnected || !this.db) {
        console.warn(`⚠️ MongoDB not connected - cannot retrieve predictions for ${ticker}`);
        return null;
      }

      const dbTicker = normalizeTickerForDB(ticker);
      console.log(`🔍 MongoDB: Fetching latest predictions for ${dbTicker}`);

      const collection = this.db.collection('stock_predictions');

      const latestDoc = await collection.findOne(
        { ticker: dbTicker },
        { sort: { timestamp: -1 } }
      );

      if (!latestDoc) {
        console.log(`❌ MongoDB: No predictions found for ${dbTicker}`);
        return null;
      }

      console.log(`✅ MongoDB: Found latest doc for ${dbTicker} at ${latestDoc.timestamp}`);

      const cursor = collection.find({
        ticker: dbTicker,
        timestamp: latestDoc.timestamp
      });

      const predictions = {};
      const docs = await cursor.toArray();

      for (const doc of docs) {
        const window = doc.window;
        if (window === '_meta') continue; // Skip metadata entries

        predictions[window] = {
          predicted_price: doc.predicted_price,
          price_change: doc.price_change,
          current_price: doc.current_price,
          confidence: doc.confidence || 0,
          price_range: doc.price_range || {},
          model_predictions: doc.model_predictions || {},
          ensemble_weights: doc.ensemble_weights || {},
          timestamp: doc.timestamp
        };
      }

      const count = Object.keys(predictions).length;
      console.log(`✅ MongoDB: Retrieved ${count} prediction windows for ${ticker}`);
      return count > 0 ? predictions : null;

    } catch (error) {
      console.error(`Error getting predictions for ${ticker}:`, error.message);
      return null;
    }
  }
}

// Create singleton instance
const mongoConnection = new MongoDBConnection();

module.exports = mongoConnection;
module.exports.normalizeTickerForDB = normalizeTickerForDB;