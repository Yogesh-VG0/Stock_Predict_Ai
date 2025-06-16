const { createClient } = require('redis');

const redisClient = createClient({
  url: process.env.REDIS_URL || 'redis://localhost:6379',
});

redisClient.on('error', (err) => console.error('Redis Client Error', err));

(async () => {
  if (!redisClient.isOpen) {
    await redisClient.connect();
  }
})();

module.exports = redisClient; 