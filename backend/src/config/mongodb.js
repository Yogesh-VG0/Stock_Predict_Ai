const mongoose = require('mongoose');
require('dotenv').config();

// MongoDB connection string from environment variable
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/stock_predictor';

class MongoDBConnection {
  constructor() {
    this.connection = null;
    this.db = null;
  }

  async connect() {
    try {
      this.connection = await mongoose.connect(MONGODB_URI, {
        useNewUrlParser: true,
        useUnifiedTopology: true,
      });
      
      this.db = this.connection.connection.db;
      console.log('✅ Connected to MongoDB Atlas (Node.js Backend)');
      return this.connection;
    } catch (error) {
      console.error('❌ MongoDB connection error:', error);
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
      if (!this.db) {
        throw new Error('MongoDB not connected');
      }

      const collection = this.db.collection('prediction_explanations');
      
      const explanation = await collection.findOne(
        { ticker: ticker.toUpperCase(), window: window },
        { sort: { timestamp: -1 } } // Get most recent
      );

      return explanation;
    } catch (error) {
      console.error(`Error getting stored explanation for ${ticker}:`, error);
      return null;
    }
  }

  async getAvailableStocks() {
    try {
      if (!this.db) {
        throw new Error('MongoDB not connected');
      }

      const collection = this.db.collection('prediction_explanations');
      
      const stocks = await collection.distinct('ticker', { window: 'comprehensive' });
      
      return stocks;
    } catch (error) {
      console.error('Error getting available stocks:', error);
      return [];
    }
  }

  async getBatchStatus() {
    try {
      if (!this.db) {
        throw new Error('MongoDB not connected');
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
}

// Create singleton instance
const mongoConnection = new MongoDBConnection();

module.exports = mongoConnection; 