const fetch = require('node-fetch');
const cheerio = require('cheerio');

// Simple sentiment analysis without heavy ML dependencies
const sentimentWords = {
  positive: ['bullish', 'buy', 'strong', 'growth', 'profit', 'up', 'gain', 'rise', 'increase', 'upgrade', 'beat', 'positive', 'good', 'excellent'],
  negative: ['bearish', 'sell', 'weak', 'loss', 'down', 'fall', 'decrease', 'downgrade', 'miss', 'negative', 'bad', 'poor', 'decline']
};

exports.handler = async (event, context) => {
  // Set CORS headers
  const headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS'
  };

  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 200, headers, body: '' };
  }

  try {
    const { ticker } = event.queryStringParameters || JSON.parse(event.body || '{}');
    
    if (!ticker) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({ error: 'Ticker parameter required' })
      };
    }

    console.log(`Processing FinViz sentiment for ${ticker}`);

    // Check cache first
    const cacheKey = `finviz:${ticker}`;
    const cached = await getCache(cacheKey);
    if (cached) {
      console.log('Returning cached FinViz sentiment');
      return {
        statusCode: 200,
        headers,
        body: JSON.stringify(cached)
      };
    }

    // Scrape FinViz news
    const sentiment = await scrapeFinVizSentiment(ticker);
    
    // Cache for 1 hour
    await setCache(cacheKey, sentiment, 3600);

    return {
      statusCode: 200,
      headers,
      body: JSON.stringify(sentiment)
    };

  } catch (error) {
    console.error('FinViz sentiment error:', error);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ 
        error: 'Failed to fetch FinViz sentiment',
        details: error.message 
      })
    };
  }
};

async function scrapeFinVizSentiment(ticker) {
  const sentimentScores = [];
  const headlines = [];
  
  try {
    const url = `https://finviz.com/quote.ashx?t=${ticker}`;
    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
      },
      timeout: 10000
    });

    if (!response.ok) {
      throw new Error(`FinViz returned status ${response.status}`);
    }

    const html = await response.text();
    const $ = cheerio.load(html);
    
    // Find news table
    const newsTable = $('.fullview-news-outer');
    if (!newsTable.length) {
      console.log('No news table found');
      return {
        sentiment: 0.0,
        volume: 0,
        confidence: 0.0,
        headlines: [],
        source: 'finviz'
      };
    }

    // Extract headlines
    newsTable.find('tr').slice(0, 15).each((i, row) => {
      const cells = $(row).find('td');
      if (cells.length >= 2) {
        const headline = $(cells[1]).text().trim();
        const dateText = $(cells[0]).text().trim();
        
        if (headline && isRecentDate(dateText)) {
          headlines.push(headline);
          const score = calculateSentiment(headline);
          sentimentScores.push(score);
        }
      }
    });

    const avgSentiment = sentimentScores.length > 0 
      ? sentimentScores.reduce((a, b) => a + b, 0) / sentimentScores.length 
      : 0.0;

    const result = {
      sentiment: avgSentiment,
      volume: sentimentScores.length,
      confidence: Math.min(sentimentScores.length / 10, 1.0),
      headlines: headlines.slice(0, 5), // Return top 5 headlines
      source: 'finviz',
      timestamp: new Date().toISOString()
    };

    console.log(`FinViz sentiment for ${ticker}: ${avgSentiment.toFixed(3)} (${sentimentScores.length} headlines)`);
    return result;

  } catch (error) {
    console.error('FinViz scraping error:', error);
    return {
      sentiment: 0.0,
      volume: 0,
      confidence: 0.0,
      headlines: [],
      source: 'finviz',
      error: error.message,
      timestamp: new Date().toISOString()
    };
  }
}

function calculateSentiment(text) {
  const words = text.toLowerCase().split(/\s+/);
  let score = 0;
  let wordCount = 0;

  words.forEach(word => {
    if (sentimentWords.positive.includes(word)) {
      score += 1;
      wordCount++;
    } else if (sentimentWords.negative.includes(word)) {
      score -= 1;
      wordCount++;
    }
  });

  // Normalize score between -1 and 1
  return wordCount > 0 ? Math.max(-1, Math.min(1, score / Math.sqrt(wordCount))) : 0;
}

function isRecentDate(dateText) {
  try {
    const now = new Date();
    const sevenDaysAgo = new Date(now - 7 * 24 * 60 * 60 * 1000);
    
    // Parse various date formats from FinViz
    let date;
    if (dateText.includes('Today')) {
      date = now;
    } else if (dateText.includes('Yesterday')) {
      date = new Date(now - 24 * 60 * 60 * 1000);
    } else {
      date = new Date(dateText);
    }
    
    return date >= sevenDaysAgo;
  } catch {
    return true; // Include if we can't parse the date
  }
}

// Simple in-memory cache for serverless functions
const cache = new Map();

async function getCache(key) {
  const entry = cache.get(key);
  if (entry && Date.now() - entry.timestamp < entry.ttl * 1000) {
    return entry.data;
  }
  cache.delete(key);
  return null;
}

async function setCache(key, data, ttlSeconds) {
  cache.set(key, {
    data,
    timestamp: Date.now(),
    ttl: ttlSeconds
  });
  
  // Clean up old entries to prevent memory leaks
  if (cache.size > 100) {
    const now = Date.now();
    for (const [k, v] of cache.entries()) {
      if (now - v.timestamp > v.ttl * 1000) {
        cache.delete(k);
      }
    }
  }
} 