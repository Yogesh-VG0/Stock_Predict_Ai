import { MongoClient } from 'mongodb';
import fetch from 'node-fetch';
import * as tf from '@tensorflow/tfjs-node';

// Cache models in memory (Vercel functions persist for some time)
const modelCache = new Map();
const CACHE_DURATION = 30 * 60 * 1000; // 30 minutes

export default async function handler(req, res) {
  const { ticker } = req.query;
  const { window = 'next_day' } = req.query;

  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  try {
    console.log(`Processing prediction request for ${ticker} - ${window}`);

    // 1. Check cache for recent prediction
    const cacheKey = `prediction:${ticker}:${window}`;
    const cachedResult = await getCachedPrediction(cacheKey);
    if (cachedResult) {
      console.log('Returning cached prediction');
      return res.json(cachedResult);
    }

    // 2. Get latest market data and sentiment
    const [marketData, sentimentData] = await Promise.all([
      getLatestMarketData(ticker),
      getSentimentData(ticker)
    ]);

    if (!marketData) {
      return res.status(404).json({ error: 'Market data not found' });
    }

    // 3. Load or download model
    const model = await loadModel(ticker, window);
    
    if (!model) {
      return res.status(500).json({ error: 'Model not available' });
    }

    // 4. Prepare features
    const features = prepareFeatures(marketData, sentimentData, window);

    // 5. Make prediction
    const prediction = await model.predict(features).data();
    
    // 6. Calculate confidence and format result
    const result = {
      ticker,
      window,
      prediction: prediction[0],
      current_price: marketData.close,
      sentiment_score: sentimentData?.overall_sentiment || 0,
      confidence: calculateConfidence(prediction, marketData),
      timestamp: new Date().toISOString(),
      features_used: Object.keys(features).length
    };

    // 7. Cache result for 15 minutes
    await cachePrediction(cacheKey, result, 15 * 60);

    console.log(`Prediction completed for ${ticker}: ${result.prediction}`);
    res.json(result);

  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({ 
      error: 'Prediction failed',
      details: error.message 
    });
  }
}

async function loadModel(ticker, window) {
  const modelKey = `${ticker}-${window}`;
  
  // Check memory cache first
  if (modelCache.has(modelKey)) {
    const cached = modelCache.get(modelKey);
    if (Date.now() - cached.timestamp < CACHE_DURATION) {
      console.log('Using cached model');
      return cached.model;
    }
  }

  try {
    // Download model from GitHub LFS
    const modelUrl = `https://github.com/${process.env.GITHUB_REPO}/raw/main/models/${ticker}/model_${ticker}_${window}_lstm.h5`;
    
    console.log(`Downloading model from: ${modelUrl}`);
    const response = await fetch(modelUrl);
    
    if (!response.ok) {
      console.log('Model not found, using fallback');
      return await loadFallbackModel(window);
    }

    const modelBuffer = await response.buffer();
    
    // Load TensorFlow model
    const model = await tf.loadLayersModel(tf.io.fromMemory({
      modelTopology: null, // Will be loaded from .h5 file
      weightSpecs: null,
      weightData: modelBuffer
    }));

    // Cache model in memory
    modelCache.set(modelKey, {
      model,
      timestamp: Date.now()
    });

    console.log(`Model loaded successfully for ${ticker}-${window}`);
    return model;

  } catch (error) {
    console.error('Error loading model:', error);
    return await loadFallbackModel(window);
  }
}

async function loadFallbackModel(window) {
  // Use a simple linear model as fallback
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [50], units: 1, activation: 'linear' })
    ]
  });
  
  console.log('Using fallback linear model');
  return model;
}

async function getLatestMarketData(ticker) {
  try {
    const client = new MongoClient(process.env.MONGODB_URI);
    await client.connect();
    
    const db = client.db('stockpredict');
    const collection = db.collection('historical_data');
    
    const latestData = await collection
      .findOne(
        { ticker },
        { sort: { date: -1 } }
      );
    
    await client.close();
    
    return latestData ? {
      open: latestData.Open,
      high: latestData.High,
      low: latestData.Low,
      close: latestData.Close,
      volume: latestData.Volume,
      date: latestData.date
    } : null;

  } catch (error) {
    console.error('Error fetching market data:', error);
    return null;
  }
}

async function getSentimentData(ticker) {
  try {
    // Call sentiment aggregation service
    const sentimentUrl = `${process.env.SENTIMENT_API_URL}/sentiment/${ticker}`;
    const response = await fetch(sentimentUrl);
    
    if (response.ok) {
      return await response.json();
    }
    
    // Fallback to neutral sentiment
    return { overall_sentiment: 0.0 };

  } catch (error) {
    console.error('Error fetching sentiment:', error);
    return { overall_sentiment: 0.0 };
  }
}

function prepareFeatures(marketData, sentimentData, window) {
  // Simple feature preparation - in production, use your feature engineering
  const features = tf.tensor2d([[
    marketData.open / marketData.close,     // Open/Close ratio
    marketData.high / marketData.close,     // High/Close ratio  
    marketData.low / marketData.close,      // Low/Close ratio
    Math.log(marketData.volume || 1000000), // Log volume
    sentimentData.overall_sentiment || 0,   // Sentiment score
    // Add more features as needed...
    ...Array(45).fill(0) // Pad to expected input size
  ]]);

  return features;
}

function calculateConfidence(prediction, marketData) {
  // Simple confidence calculation based on prediction vs current price
  const predictionValue = prediction[0];
  const currentPrice = marketData.close;
  const percentChange = Math.abs((predictionValue - currentPrice) / currentPrice);
  
  // Higher confidence for smaller changes (more realistic)
  return Math.max(0.1, 1 - percentChange * 2);
}

async function getCachedPrediction(key) {
  try {
    // Use Upstash Redis or simple in-memory cache
    if (process.env.UPSTASH_REDIS_URL) {
      const Redis = require('@upstash/redis');
      const redis = Redis.fromEnv();
      const cached = await redis.get(key);
      return cached ? JSON.parse(cached) : null;
    }
    return null;
  } catch (error) {
    console.error('Cache read error:', error);
    return null;
  }
}

async function cachePrediction(key, data, ttlSeconds) {
  try {
    if (process.env.UPSTASH_REDIS_URL) {
      const Redis = require('@upstash/redis');
      const redis = Redis.fromEnv();
      await redis.setex(key, ttlSeconds, JSON.stringify(data));
    }
  } catch (error) {
    console.error('Cache write error:', error);
  }
} 