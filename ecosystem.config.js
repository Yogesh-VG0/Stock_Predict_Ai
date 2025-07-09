module.exports = {
  apps: [
    {
      name: 'frontend',
      script: 'npm',
      args: 'start',
      cwd: '/app',
      env: {
        PORT: 3000,
        NODE_ENV: 'production',
        NEXT_PUBLIC_NODE_BACKEND_URL: 'http://localhost:5000'
      }
    },
    {
      name: 'backend',
      script: 'src/server.js',
      cwd: '/app/backend',
      env: {
        PORT: 5000,
        NODE_ENV: 'production',
        ML_BACKEND_URL: process.env.ML_BACKEND_URL || 'https://stockpredict-ai-ml-api.onrender.com',
        FINNHUB_API_KEY: process.env.FINNHUB_API_KEY,
        MARKETAUX_API_KEY: process.env.MARKETAUX_API_KEY,
        NEWSAPI_KEY: process.env.NEWSAPI_KEY,
        CALENDARIFIC_API_KEY: process.env.CALENDARIFIC_API_KEY,
        RAPIDAPI_KEY: process.env.RAPIDAPI_KEY,
        EODHD_API_KEY: process.env.EODHD_API_KEY,
        REDIS_URL: process.env.REDIS_URL
      }
    }
  ]
}; 