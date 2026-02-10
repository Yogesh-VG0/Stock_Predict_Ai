// Redis is OPTIONAL - only connect if REDIS_URL is explicitly set
// Otherwise, use a mock client that does nothing

const path = require('path');

// Ensure dotenv is loaded
require('dotenv').config({ path: path.resolve(__dirname, '..', '..', '.env') });

let redisClient = null;
let isRedisAvailable = false;

// Only attempt Redis connection if REDIS_URL is explicitly provided
if (process.env.REDIS_URL) {
  const { createClient } = require('redis');
  
  redisClient = createClient({
    url: process.env.REDIS_URL,
    socket: {
      connectTimeout: 5000, // 5 second connection timeout
      reconnectStrategy: (retries) => {
        if (retries > 2) {
          console.log('⚠️ Redis max retries reached - disabling caching');
          isRedisAvailable = false;
          return false; // Stop reconnecting
        }
        return Math.min(retries * 1000, 3000); // Exponential backoff, max 3s
      }
    }
  });

  redisClient.on('error', (err) => {
    // Log first error, then suppress
    if (isRedisAvailable) {
      console.log('⚠️ Redis error - caching disabled:', err.message);
    }
    isRedisAvailable = false;
  });

  redisClient.on('connect', () => {
    console.log('✅ Redis connected');
    isRedisAvailable = true;
  });

  // Attempt connection with timeout
  (async () => {
    try {
      const connectPromise = redisClient.connect();
      const timeoutPromise = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Connection timeout')), 5000)
      );
      await Promise.race([connectPromise, timeoutPromise]);
    } catch (err) {
      console.log('⚠️ Redis not available - caching disabled:', err.message);
      isRedisAvailable = false;
    }
  })();
} else {
  console.log('ℹ️ Redis URL not configured - caching disabled');
}

// Mock Redis client for when Redis is not available
const mockRedisClient = {
  isOpen: false,
  get: async () => null,
  set: async () => null,
  del: async () => null,
  exists: async () => 0,
  connect: async () => {},
  disconnect: async () => {},
};

module.exports = redisClient || mockRedisClient; 