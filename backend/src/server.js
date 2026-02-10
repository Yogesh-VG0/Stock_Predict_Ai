const path = require('path');

// Load .env from the backend directory (one level up from src)
const envPath = path.resolve(__dirname, '..', '.env');
require('dotenv').config({ path: envPath });

// Validate required environment variables before starting
function validateEnvironment() {
  const warnings = [];

  console.log('\n📋 Environment Configuration:');
  console.log(`   .env path: ${envPath}`);
  
  // Check Finnhub API Key (required for stock quotes)
  if (!process.env.FINNHUB_API_KEY) {
    console.error('❌ FINNHUB_API_KEY: NOT SET');
    console.error('   Get yours at https://finnhub.io/register');
  } else {
    // Mask the key for security (show first 4 chars only)
    const maskedKey = process.env.FINNHUB_API_KEY.substring(0, 4) + '****';
    console.log(`✅ FINNHUB_API_KEY: ${maskedKey}`);
  }

  // Check MongoDB URI (required for data storage)
  if (!process.env.MONGODB_URI) {
    warnings.push('MONGODB_URI not set - using default: mongodb://localhost:27017/stock_predictor');
    console.log('⚠️  MONGODB_URI: Using default (localhost)');
  } else {
    // Mask credentials in URI
    const maskedUri = process.env.MONGODB_URI.replace(/:\/\/[^:]+:[^@]+@/, '://*****:*****@');
    console.log(`✅ MONGODB_URI: ${maskedUri}`);
  }

  // Check Redis
  if (process.env.REDIS_URL) {
    console.log('✅ REDIS_URL: Configured');
  } else {
    console.log('ℹ️  REDIS_URL: Not set (caching disabled)');
  }

  // Check Calendarific API Key (optional)
  if (!process.env.CALENDARIFIC_API_KEY) {
    console.log('ℹ️  CALENDARIFIC_API_KEY: Not set (using static holidays)');
  } else {
    console.log('✅ CALENDARIFIC_API_KEY: Configured');
  }

  console.log('');
}

// Validate environment first
validateEnvironment();

const app = require('./app');
const mongoConnection = require('./config/mongodb');

const PORT = process.env.PORT || 5000;

// Start server - MongoDB connection is optional for some features
async function startServer() {
  let mongoConnected = false;
  
  try {
    // Try to connect to MongoDB (with timeout)
    await mongoConnection.connect();
    mongoConnected = true;
    console.log('✅ MongoDB connection established');
  } catch (error) {
    console.warn('⚠️ MongoDB connection failed:', error.message);
    console.warn('   Some features will be unavailable (stored explanations, watchlists)');
    console.warn('   Stock quotes and market data will still work\n');
  }

  // Start server regardless of MongoDB status
  app.listen(PORT, () => {
    console.log(`\n🚀 Server is running on port ${PORT}`);
    console.log(`📊 Health check: http://localhost:${PORT}/health`);
    console.log(`📈 API base URL: http://localhost:${PORT}/api`);
    if (!mongoConnected) {
      console.log('⚠️ Running in limited mode (no MongoDB)\n');
    } else {
      console.log('');
    }
  });
}

startServer(); 