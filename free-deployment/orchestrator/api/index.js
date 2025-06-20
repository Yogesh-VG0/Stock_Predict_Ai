/**
 * Smart API Orchestrator for Free Stock Prediction System
 * Routes requests to appropriate free platforms based on load and availability
 */

// Platform endpoints - replace with your actual deployments
const PLATFORMS = {
  vercel: {
    base: 'https://your-app.vercel.app',
    endpoints: {
      predictions: '/api/predict',
      sentiment_light: '/api/sentiment'
    },
    limits: { concurrent: 100, daily: 100000 }
  },
  netlify: {
    base: 'https://your-app.netlify.app',
    endpoints: {
      sentiment_scraping: '/.netlify/functions/sentiment-finviz',
      sentiment_reddit: '/.netlify/functions/sentiment-reddit'
    },
    limits: { concurrent: 1000, monthly: 125000 }
  },
  railway: {
    base: 'https://your-app.railway.app',
    endpoints: {
      sentiment_heavy: '/sentiment/batch',
      model_training: '/models/train'
    },
    limits: { concurrent: 50, monthly: 'unlimited' }
  },
  render: {
    base: 'https://your-app.onrender.com',
    endpoints: {
      fallback: '/api',
      analytics: '/analytics'
    },
    limits: { concurrent: 25, monthly: 'unlimited' }
  }
};

// Request routing logic
const ROUTING_RULES = {
  predictions: {
    primary: 'vercel',
    fallback: ['netlify', 'railway']
  },
  sentiment: {
    light: 'vercel',      // RSS feeds, simple APIs
    scraping: 'netlify',  // FinViz, web scraping
    heavy: 'railway',     // Reddit, SEC, batch processing
    fallback: 'render'
  },
  training: {
    primary: 'railway',
    fallback: 'render'
  }
};

// Request tracking for load balancing
const requestTracker = new Map();

export default async function handler(req, res) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const { action, ticker, ...params } = req.query;
  const body = req.body;

  try {
    console.log(`Orchestrating request: ${action} for ${ticker}`);

    let result;
    
    switch (action) {
      case 'predict':
        result = await handlePredictionRequest(ticker, params);
        break;
        
      case 'sentiment':
        result = await handleSentimentRequest(ticker, params);
        break;
        
      case 'sentiment-batch':
        result = await handleBatchSentimentRequest(body);
        break;
        
      case 'train':
        result = await handleTrainingRequest(ticker, params);
        break;
        
      case 'health':
        result = await checkSystemHealth();
        break;
        
      default:
        return res.status(400).json({ error: 'Invalid action' });
    }

    res.json(result);

  } catch (error) {
    console.error('Orchestration error:', error);
    res.status(500).json({ 
      error: 'Service temporarily unavailable',
      details: error.message 
    });
  }
}

async function handlePredictionRequest(ticker, params) {
  const { window = 'next_day', use_cache = 'true' } = params;
  
  // Try primary platform (Vercel)
  try {
    const platform = PLATFORMS.vercel;
    const url = `${platform.base}${platform.endpoints.predictions}/${ticker}?window=${window}`;
    
    const response = await fetchWithTimeout(url, { timeout: 25000 });
    
    if (response.ok) {
      const data = await response.json();
      return {
        ...data,
        platform: 'vercel',
        cached: data.cached || false
      };
    }
  } catch (error) {
    console.warn('Vercel prediction failed, trying fallback:', error.message);
  }
  
  // Try fallback platforms
  for (const platformName of ROUTING_RULES.predictions.fallback) {
    try {
      const platform = PLATFORMS[platformName];
      const endpoint = platform.endpoints.predictions || platform.endpoints.fallback;
      const url = `${platform.base}${endpoint}/${ticker}?window=${window}`;
      
      const response = await fetchWithTimeout(url, { timeout: 15000 });
      
      if (response.ok) {
        const data = await response.json();
        return {
          ...data,
          platform: platformName,
          fallback: true
        };
      }
    } catch (error) {
      console.warn(`${platformName} prediction failed:`, error.message);
      continue;
    }
  }
  
  throw new Error('All prediction services unavailable');
}

async function handleSentimentRequest(ticker, params) {
  const { sources = 'all', priority = 'speed' } = params;
  
  if (sources === 'all' || sources.includes(',')) {
    // Multiple sources - use heavy processing platform
    return await routeToHeavyProcessing('/sentiment/batch', {
      method: 'POST',
      body: JSON.stringify({
        tickers: [ticker],
        sources: sources === 'all' ? ['finviz', 'reddit', 'marketaux', 'seeking_alpha'] : sources.split(',')
      })
    });
  }
  
  // Single source - route to appropriate platform
  const source = sources;
  let platform, endpoint;
  
  switch (source) {
    case 'finviz':
      platform = PLATFORMS.netlify;
      endpoint = platform.endpoints.sentiment_scraping;
      break;
      
    case 'reddit':
      platform = PLATFORMS.railway;
      endpoint = platform.endpoints.sentiment_heavy;
      break;
      
    default:
      platform = PLATFORMS.vercel;
      endpoint = platform.endpoints.sentiment_light;
  }
  
  const url = `${platform.base}${endpoint}?ticker=${ticker}`;
  const response = await fetchWithTimeout(url, { timeout: 15000 });
  
  if (!response.ok) {
    throw new Error(`Sentiment service error: ${response.status}`);
  }
  
  return await response.json();
}

async function handleBatchSentimentRequest(body) {
  return await routeToHeavyProcessing('/sentiment/batch', {
    method: 'POST',
    body: JSON.stringify(body)
  });
}

async function handleTrainingRequest(ticker, params) {
  return await routeToHeavyProcessing(`/models/train/${ticker}`, {
    method: 'POST',
    body: JSON.stringify(params)
  });
}

async function routeToHeavyProcessing(endpoint, options = {}) {
  // Try Railway first (primary heavy processing platform)
  try {
    const platform = PLATFORMS.railway;
    const url = `${platform.base}${endpoint}`;
    
    const response = await fetchWithTimeout(url, {
      ...options,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      }
    });
    
    if (response.ok) {
      const data = await response.json();
      return { ...data, platform: 'railway' };
    }
  } catch (error) {
    console.warn('Railway heavy processing failed:', error.message);
  }
  
  // Try Render as fallback
  try {
    const platform = PLATFORMS.render;
    const url = `${platform.base}${endpoint}`;
    
    const response = await fetchWithTimeout(url, {
      ...options,
      timeout: 45000,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      }
    });
    
    if (response.ok) {
      const data = await response.json();
      return { ...data, platform: 'render', fallback: true };
    }
  } catch (error) {
    console.warn('Render fallback failed:', error.message);
  }
  
  throw new Error('Heavy processing services unavailable');
}

async function checkSystemHealth() {
  const health = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    platforms: {},
    overall_status: 'operational'
  };
  
  // Check each platform
  const healthChecks = Object.entries(PLATFORMS).map(async ([name, platform]) => {
    try {
      const url = `${platform.base}/health`;
      const response = await fetchWithTimeout(url, { timeout: 5000 });
      
      health.platforms[name] = {
        status: response.ok ? 'healthy' : 'degraded',
        response_time: response.headers.get('x-response-time') || 'unknown',
        last_check: new Date().toISOString()
      };
    } catch (error) {
      health.platforms[name] = {
        status: 'unhealthy',
        error: error.message,
        last_check: new Date().toISOString()
      };
    }
  });
  
  await Promise.allSettled(healthChecks);
  
  // Determine overall status
  const healthyCount = Object.values(health.platforms).filter(p => p.status === 'healthy').length;
  const totalCount = Object.keys(health.platforms).length;
  
  if (healthyCount === totalCount) {
    health.overall_status = 'operational';
  } else if (healthyCount >= totalCount / 2) {
    health.overall_status = 'degraded';
  } else {
    health.overall_status = 'major_outage';
  }
  
  return health;
}

async function fetchWithTimeout(url, options = {}) {
  const { timeout = 10000, ...fetchOptions } = options;
  
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(url, {
      ...fetchOptions,
      signal: controller.signal,
      headers: {
        'User-Agent': 'StockPredict-Orchestrator/1.0',
        ...fetchOptions.headers
      }
    });
    
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

// Track request patterns for optimization
function trackRequest(platform, endpoint, duration, success) {
  const key = `${platform}:${endpoint}`;
  const tracker = requestTracker.get(key) || {
    count: 0,
    avg_duration: 0,
    success_rate: 0,
    last_used: null
  };
  
  tracker.count += 1;
  tracker.avg_duration = (tracker.avg_duration * (tracker.count - 1) + duration) / tracker.count;
  tracker.success_rate = (tracker.success_rate * (tracker.count - 1) + (success ? 1 : 0)) / tracker.count;
  tracker.last_used = new Date().toISOString();
  
  requestTracker.set(key, tracker);
}

// Export configuration for use in other parts of the system
export const config = {
  api: {
    bodyParser: {
      sizeLimit: '1mb',
    },
  },
} 