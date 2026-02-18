const axios = require('axios');
const mongoConnection = require('../config/mongodb');
const { normalizeTickerForDB } = require('../config/mongodb');
const massiveService = require('../services/massiveService');

// Company overview data mapping for all 25 tickers
const COMPANY_DATA = {
  'AAPL': {
    name: 'Apple Inc.',
    sector: 'Technology',
    industry: 'Consumer Electronics',
    description: 'Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.',
    headquarters: 'Cupertino, California',
    founded: 1976,
    employees: 154000,
    ceo: 'Tim Cook',
    website: 'www.apple.com'
  },
  'MSFT': {
    name: 'Microsoft Corporation',
    sector: 'Technology',
    industry: 'Software',
    description: 'Microsoft Corporation develops, licenses, and supports software, services, devices, and solutions worldwide.',
    headquarters: 'Redmond, Washington',
    founded: 1975,
    employees: 221000,
    ceo: 'Satya Nadella',
    website: 'www.microsoft.com'
  },
  'NVDA': {
    name: 'NVIDIA Corporation',
    sector: 'Technology',
    industry: 'Semiconductors',
    description: 'NVIDIA Corporation operates as a computing company worldwide. The company operates in two segments, Graphics and Compute & Networking. It offers graphics processing units (GPUs) for gaming and professional markets, as well as system on chip units (SoCs) for the mobile computing and automotive market.',
    headquarters: 'Santa Clara, California',
    founded: 1993,
    employees: 29600,
    ceo: 'Jensen Huang',
    website: 'www.nvidia.com'
  },
  'AMZN': {
    name: 'Amazon.com Inc.',
    sector: 'Consumer Discretionary',
    industry: 'Internet Retail',
    description: 'Amazon.com, Inc. engages in the retail sale of consumer products and subscriptions in North America and internationally. The company operates through three segments: North America, International, and Amazon Web Services (AWS).',
    headquarters: 'Seattle, Washington',
    founded: 1994,
    employees: 1541000,
    ceo: 'Andy Jassy',
    website: 'www.amazon.com'
  },
  'GOOGL': {
    name: 'Alphabet Inc.',
    sector: 'Communication Services',
    industry: 'Internet Content & Information',
    description: 'Alphabet Inc. provides various products and platforms in the United States, Europe, the Middle East, Africa, the Asia-Pacific, Canada, and Latin America. It operates through Google Services, Google Cloud, and Other Bets segments.',
    headquarters: 'Mountain View, California',
    founded: 1998,
    employees: 182502,
    ceo: 'Sundar Pichai',
    website: 'www.alphabet.com'
  },
  'META': {
    name: 'Meta Platforms Inc.',
    sector: 'Communication Services',
    industry: 'Internet Content & Information',
    description: 'Meta Platforms, Inc. develops products that enable people to connect and share with friends and family through mobile devices, personal computers, virtual reality headsets, wearables, and in-home devices worldwide.',
    headquarters: 'Menlo Park, California',
    founded: 2004,
    employees: 77805,
    ceo: 'Mark Zuckerberg',
    website: 'www.meta.com'
  },
  'BRK.B': {
    name: 'Berkshire Hathaway Inc.',
    sector: 'Financial Services',
    industry: 'Insurance',
    description: 'Berkshire Hathaway Inc., through its subsidiaries, provides insurance and reinsurance, utilities and energy, freight rail transportation, manufacturing, retailing, and various other products and services worldwide.',
    headquarters: 'Omaha, Nebraska',
    founded: 1965,
    employees: 383000,
    ceo: 'Warren Buffett',
    website: 'www.berkshirehathaway.com'
  },
  'TSLA': {
    name: 'Tesla Inc.',
    sector: 'Consumer Discretionary',
    industry: 'Auto Manufacturers',
    description: 'Tesla, Inc. designs, develops, manufactures, leases, and sells electric vehicles, and energy generation and storage systems in the United States, China, and internationally.',
    headquarters: 'Austin, Texas',
    founded: 2003,
    employees: 140473,
    ceo: 'Elon Musk',
    website: 'www.tesla.com'
  },
  'AVGO': {
    name: 'Broadcom Inc.',
    sector: 'Technology',
    industry: 'Semiconductors',
    description: 'Broadcom Inc. designs, develops, and supplies various semiconductor devices with a focus on complex digital and mixed signal complementary metal oxide semiconductor based devices and analog III-V based products worldwide.',
    headquarters: 'San Jose, California',
    founded: 1961,
    employees: 24000,
    ceo: 'Hock Tan',
    website: 'www.broadcom.com'
  },
  'LLY': {
    name: 'Eli Lilly & Company',
    sector: 'Healthcare',
    industry: 'Drug Manufacturers',
    description: 'Eli Lilly and Company discovers, develops, and markets human pharmaceuticals worldwide. It offers Basaglar, Humalog, Humalog Mix 75/25, Humalog U-100, Humalog U-200, Humalog Mix 50/50, insulin lispro, insulin lispro protamine, insulin lispro mix 75/25, Humulin, Humulin 70/30, Humulin N, Humulin R, and Humulin U-500 for diabetes.',
    headquarters: 'Indianapolis, Indiana',
    founded: 1876,
    employees: 43000,
    ceo: 'David Ricks',
    website: 'www.lilly.com'
  },
  'WMT': {
    name: 'Walmart Inc.',
    sector: 'Consumer Staples',
    industry: 'Discount Stores',
    description: 'Walmart Inc. engages in the operation of retail, wholesale, and other units worldwide. The company operates through three segments: Walmart U.S., Walmart International, and Sam\'s Club.',
    headquarters: 'Bentonville, Arkansas',
    founded: 1962,
    employees: 2100000,
    ceo: 'Doug McMillon',
    website: 'www.walmart.com'
  },
  'JPM': {
    name: 'JPMorgan Chase & Co.',
    sector: 'Financial Services',
    industry: 'Banks',
    description: 'JPMorgan Chase & Co. operates as a financial services company worldwide. It operates through four segments: Consumer & Community Banking (CCB), Corporate & Investment Bank (CIB), Commercial Banking (CB), and Asset & Wealth Management (AWM).',
    headquarters: 'New York, New York',
    founded: 1799,
    employees: 293723,
    ceo: 'Jamie Dimon',
    website: 'www.jpmorganchase.com'
  },
  'V': {
    name: 'Visa Inc.',
    sector: 'Financial Services',
    industry: 'Credit Services',
    description: 'Visa Inc. operates as a payments technology company worldwide. The company facilitates digital payments among consumers, merchants, financial institutions, businesses, strategic partners, and government entities.',
    headquarters: 'San Francisco, California',
    founded: 1958,
    employees: 26500,
    ceo: 'Ryan McInerney',
    website: 'www.visa.com'
  },
  'MA': {
    name: 'Mastercard Incorporated',
    sector: 'Financial Services',
    industry: 'Credit Services',
    description: 'Mastercard Incorporated, a technology company, provides transaction processing and other payment-related products and services in the United States and internationally.',
    headquarters: 'Purchase, New York',
    founded: 1966,
    employees: 33000,
    ceo: 'Michael Miebach',
    website: 'www.mastercard.com'
  },
  'NFLX': {
    name: 'Netflix Inc.',
    sector: 'Communication Services',
    industry: 'Entertainment',
    description: 'Netflix, Inc. provides entertainment services. It offers TV series, documentaries, feature films, and mobile games across a wide variety of genres and languages to members in over 190 countries.',
    headquarters: 'Los Gatos, California',
    founded: 1997,
    employees: 15000,
    ceo: 'Reed Hastings',
    website: 'www.netflix.com'
  },
  'XOM': {
    name: 'Exxon Mobil Corporation',
    sector: 'Energy',
    industry: 'Oil & Gas Integrated',
    description: 'Exxon Mobil Corporation explores for and produces crude oil and natural gas in the United States and internationally. It operates through Upstream, Downstream, and Chemical segments.',
    headquarters: 'Irving, Texas',
    founded: 1870,
    employees: 62000,
    ceo: 'Darren Woods',
    website: 'www.exxonmobil.com'
  },
  'COST': {
    name: 'Costco Wholesale Corporation',
    sector: 'Consumer Staples',
    industry: 'Discount Stores',
    description: 'Costco Wholesale Corporation, together with its subsidiaries, engages in the operation of membership warehouses in the United States, Puerto Rico, Canada, Mexico, Japan, Korea, Taiwan, Australia, Spain, France, Iceland, China, and Sweden.',
    headquarters: 'Issaquah, Washington',
    founded: 1976,
    employees: 304000,
    ceo: 'Craig Jelinek',
    website: 'www.costco.com'
  },
  'ORCL': {
    name: 'Oracle Corporation',
    sector: 'Technology',
    industry: 'Software',
    description: 'Oracle Corporation provides products and services that address enterprise information technology environments worldwide. Its Oracle cloud software as a service offering include various cloud software applications, including Oracle Fusion cloud enterprise resource planning (ERP), Oracle Fusion cloud enterprise performance management, Oracle Fusion cloud supply chain & manufacturing management, Oracle Fusion cloud human capital management, Oracle Fusion cloud advertising and customer experience, and others.',
    headquarters: 'Austin, Texas',
    founded: 1977,
    employees: 164000,
    ceo: 'Safra Catz',
    website: 'www.oracle.com'
  },
  'PG': {
    name: 'The Procter & Gamble Company',
    sector: 'Consumer Staples',
    industry: 'Household & Personal Products',
    description: 'The Procter & Gamble Company provides branded consumer packaged goods to consumers in North and Latin America, Europe, the Asia Pacific, Greater China, India, the Middle East, and Africa.',
    headquarters: 'Cincinnati, Ohio',
    founded: 1837,
    employees: 118000,
    ceo: 'Jon Moeller',
    website: 'www.pg.com'
  },
  'JNJ': {
    name: 'Johnson & Johnson',
    sector: 'Healthcare',
    industry: 'Drug Manufacturers',
    description: 'Johnson & Johnson researches, develops, manufactures, and sells various products in the healthcare field worldwide. The company\'s MedTech segment offers orthopaedic products, general surgery, cardiovascular, women\'s health, and other products.',
    headquarters: 'New Brunswick, New Jersey',
    founded: 1886,
    employees: 152700,
    ceo: 'Joaquin Duato',
    website: 'www.jnj.com'
  },
  'UNH': {
    name: 'UnitedHealth Group Incorporated',
    sector: 'Healthcare',
    industry: 'Healthcare Plans',
    description: 'UnitedHealth Group Incorporated operates as a diversified health care company in the United States. It operates through four segments: UnitedHealthcare, Optum Health, Optum Insight, and Optum Rx.',
    headquarters: 'Minnetonka, Minnesota',
    founded: 1977,
    employees: 400000,
    ceo: 'Andrew Witty',
    website: 'www.unitedhealthgroup.com'
  },
  'HD': {
    name: 'The Home Depot Inc.',
    sector: 'Consumer Discretionary',
    industry: 'Home Improvement Retail',
    description: 'The Home Depot, Inc. operates as a home improvement retailer. It operates The Home Depot stores that sell various building materials, home improvement products, lawn and garden products, and décor products, as well as facilities maintenance, repair, and operations products.',
    headquarters: 'Atlanta, Georgia',
    founded: 1978,
    employees: 514000,
    ceo: 'Craig Menear',
    website: 'www.homedepot.com'
  },
  'ABBV': {
    name: 'AbbVie Inc.',
    sector: 'Healthcare',
    industry: 'Drug Manufacturers',
    description: 'AbbVie Inc. discovers, develops, manufactures, and sells pharmaceuticals in the worldwide. The company offers HUMIRA for the treatment of rheumatoid arthritis, psoriatic arthritis, Crohn\'s disease, ulcerative colitis, plaque psoriasis, hidradenitis suppurativa, and uveitis.',
    headquarters: 'North Chicago, Illinois',
    founded: 2013,
    employees: 50000,
    ceo: 'Richard Gonzalez',
    website: 'www.abbvie.com'
  },
  'KO': {
    name: 'The Coca-Cola Company',
    sector: 'Consumer Staples',
    industry: 'Beverages',
    description: 'The Coca-Cola Company, a beverage company, manufactures, markets, and sells various nonalcoholic beverages worldwide. The company provides sparkling soft drinks; flavored and enhanced water, and sports drinks; juice, dairy and plant-based beverages; tea and coffee; and energy drinks.',
    headquarters: 'Atlanta, Georgia',
    founded: 1892,
    employees: 82500,
    ceo: 'James Quincey',
    website: 'www.coca-cola.com'
  },
  'CRM': {
    name: 'Salesforce Inc.',
    sector: 'Technology',
    industry: 'Software',
    description: 'Salesforce, Inc. provides Customer Relationship Management (CRM) technology that brings companies and customers together worldwide. Its Customer 360 platform empowers its customers to work together to deliver connected experiences for their customers.',
    headquarters: 'San Francisco, California',
    founded: 1999,
    employees: 79390,
    ceo: 'Marc Benioff',
    website: 'www.salesforce.com'
  }
};

const ML_BACKEND_URL = process.env.ML_BACKEND_URL || 'http://localhost:8000';

const getStockDetails = async (req, res) => {
  try {
    const { symbol } = req.params;
    const upperSymbol = symbol.toUpperCase();

    // Check if this is a tracked stock with hardcoded data
    const companyData = COMPANY_DATA[upperSymbol];
    
    let stockDetails;
    
    if (companyData) {
      // Tracked stock — use hardcoded data + AI analysis
      let aiAnalysis = null;
      try {
        const explanationResponse = await axios.get(`${ML_BACKEND_URL}/api/v1/predictions/${upperSymbol}/explanation`, {
          timeout: 10000
        });
        if (explanationResponse.data && explanationResponse.data.explanation) {
          const explanation = explanationResponse.data.explanation;
          aiAnalysis = {
            positiveFactors: explanation.positive_factors || [
              "Strong technical indicators show bullish momentum",
              "Market sentiment analysis indicates positive investor outlook"
            ],
            negativeFactors: explanation.negative_factors || [
              "Market volatility increases investment risk",
              "Macroeconomic uncertainties affect sector performance"
            ]
          };
        }
      } catch (mlError) {
        console.log(`ML Backend not available for ${upperSymbol}, using fallback`);
      }

      if (!aiAnalysis) {
        aiAnalysis = {
          positiveFactors: [
            "Strong technical indicators show bullish momentum",
            "Market sentiment analysis indicates positive investor outlook"
          ],
          negativeFactors: [
            "Market volatility increases investment risk",
            "Macroeconomic uncertainties affect sector performance"
          ]
        };
      }

      stockDetails = {
        symbol: upperSymbol,
        ...companyData,
        isTracked: true,
        aiAnalysis
      };
    } else {
      // Non-tracked stock — fetch from Finnhub Company Profile API
      console.log(`📊 Fetching Finnhub profile for untracked stock: ${upperSymbol}`);
      try {
        const finnhubKey = process.env.FINNHUB_API_KEY;
        const profileRes = await axios.get(
          `https://finnhub.io/api/v1/stock/profile2?symbol=${upperSymbol}&token=${finnhubKey}`,
          { timeout: 10000 }
        );
        const profile = profileRes.data;
        if (profile && profile.name) {
          stockDetails = {
            symbol: upperSymbol,
            name: profile.name,
            sector: profile.finnhubIndustry || 'Unknown',
            industry: profile.finnhubIndustry || 'Unknown',
            description: `${profile.name} (${upperSymbol}) is listed on ${profile.exchange || 'N/A'}. Market cap: $${profile.marketCapitalization ? (profile.marketCapitalization / 1000).toFixed(1) + 'B' : 'N/A'}. Country: ${profile.country || 'N/A'}.`,
            headquarters: profile.country || 'N/A',
            founded: 0,
            employees: 0,
            ceo: 'N/A',
            website: profile.weburl || 'N/A',
            isTracked: false,
            aiAnalysis: { positiveFactors: [], negativeFactors: [] }
          };
        } else {
          // Finnhub returned empty profile — still allow viewing with minimal data
          stockDetails = {
            symbol: upperSymbol,
            name: upperSymbol,
            sector: 'Unknown',
            industry: 'Unknown',
            description: `View the live chart for ${upperSymbol} below.`,
            headquarters: 'N/A',
            founded: 0,
            employees: 0,
            ceo: 'N/A',
            website: 'N/A',
            isTracked: false,
            aiAnalysis: { positiveFactors: [], negativeFactors: [] }
          };
        }
      } catch (finnhubError) {
        console.log(`⚠️ Finnhub profile failed for ${upperSymbol}: ${finnhubError.message}`);
        // Minimal fallback so the page still renders
        stockDetails = {
          symbol: upperSymbol,
          name: upperSymbol,
          sector: 'Unknown',
          industry: 'Unknown',
          description: `View the live chart for ${upperSymbol} below.`,
          headquarters: 'N/A',
          founded: 0,
          employees: 0,
          ceo: 'N/A',
          website: 'N/A',
          isTracked: false,
          aiAnalysis: { positiveFactors: [], negativeFactors: [] }
        };
      }
    }

    // Add real-time price from latest MongoDB prediction (reliable source)
    try {
      const latestPrediction = await mongoConnection.getLatestPredictions(upperSymbol);
      if (latestPrediction) {
        // Find the most recent current_price from any prediction window
        for (const window of ['next_day', '1_day', '7_day', '30_day']) {
          if (latestPrediction[window]?.current_price) {
            stockDetails.price = latestPrediction[window].current_price;
            break;
          }
        }
      }
    } catch (priceError) {
      console.log(`⚠️ Could not fetch price for ${upperSymbol}:`, priceError.message);
    }

    res.json(stockDetails);

  } catch (error) {
    console.error('Error fetching stock details:', error);
    res.status(500).json({ error: 'Failed to fetch stock details' });
  }
};

const getAIAnalysis = async (req, res) => {
  try {
    const { symbol } = req.params;
    const upperSymbol = symbol.toUpperCase();

    // Get AI analysis from ML backend
    try {
      const [explanationResponse, sentimentResponse] = await Promise.all([
        axios.get(`${ML_BACKEND_URL}/api/v1/predictions/${upperSymbol}/explanation`, { timeout: 10000 }),
        axios.get(`${ML_BACKEND_URL}/api/v1/sentiment/${upperSymbol}`, { timeout: 10000 })
      ]);

      const explanation = explanationResponse.data?.explanation || {};
      const sentiment = sentimentResponse.data || {};

      const aiAnalysis = {
        positiveFactors: explanation.positive_factors || [
          "Technical indicators suggest bullish trend continuation",
          "Sentiment analysis shows improving market perception",
          "Volume patterns indicate strong institutional support",
          "Price momentum remains above key moving averages"
        ],
        negativeFactors: explanation.negative_factors || [
          "Market volatility presents near-term risks",
          "Technical resistance levels may cap upside potential",
          "Sector rotation concerns affect performance outlook",
          "Macroeconomic headwinds create uncertainty"
        ],
        sentimentScore: sentiment.sentiment_score || 0,
        confidence: explanation.confidence || 0.5,
        lastUpdated: new Date().toISOString()
      };

      res.json(aiAnalysis);

    } catch (mlError) {
      console.log(`ML Backend not available for ${upperSymbol}, using fallback`);

      // Fallback AI analysis
      res.json({
        positiveFactors: [
          "Technical indicators suggest bullish trend continuation",
          "Market sentiment analysis shows positive momentum",
          "Historical performance metrics indicate strength",
          "Sector fundamentals remain supportive"
        ],
        negativeFactors: [
          "Market volatility presents near-term risks",
          "Technical resistance levels may limit gains",
          "Macroeconomic uncertainty affects outlook",
          "High valuation metrics suggest caution"
        ],
        sentimentScore: 0.6,
        confidence: 0.5,
        lastUpdated: new Date().toISOString()
      });
    }

  } catch (error) {
    console.error('Error fetching AI analysis:', error);
    res.status(500).json({ error: 'Failed to fetch AI analysis' });
  }
};

// Get comprehensive AI explanation
const getComprehensiveExplanation = async (req, res) => {
  try {
    const { symbol, date } = req.params;
    const ticker = symbol.toUpperCase();

    // Look up stored explanation in MongoDB (generated by CI/CD pipeline)
    const storedData = await mongoConnection.getStoredExplanation(ticker, 'comprehensive');

    if (storedData && storedData.explanation_data) {
      console.log(`✅ Found stored comprehensive explanation for ${ticker}`);
      const explanationData = storedData.explanation_data;

      return res.json({
        ticker,
        date: explanationData.explanation_date || date,
        explanation: explanationData.ai_explanation,
        data_summary: {
          blended_sentiment: explanationData.sentiment_summary?.blended_sentiment || 0,
          total_data_points: (
            (explanationData.sentiment_summary?.finviz_articles || 0) +
            (explanationData.sentiment_summary?.reddit_posts || 0) +
            (explanationData.sentiment_summary?.rss_articles || 0) +
            (explanationData.sentiment_summary?.marketaux_articles || 0)
          ),
          finviz_articles: explanationData.sentiment_summary?.finviz_articles || 0,
          reddit_posts: explanationData.sentiment_summary?.reddit_posts || 0,
          rss_articles: explanationData.sentiment_summary?.rss_articles || 0,
          marketaux_articles: explanationData.sentiment_summary?.marketaux_articles || 0
        },
        prediction_summary: {
          '1_day': {
            predicted_price: explanationData.prediction_data?.['1_day']?.predicted_price || explanationData.prediction_data?.next_day?.predicted_price || 0,
            confidence: explanationData.prediction_data?.['1_day']?.confidence || explanationData.prediction_data?.next_day?.confidence || 0,
            price_change: explanationData.prediction_data?.['1_day']?.price_change || explanationData.prediction_data?.next_day?.price_change || 0,
            price_range: explanationData.prediction_data?.['1_day']?.price_range || explanationData.prediction_data?.next_day?.price_range || {},
            model_predictions: explanationData.prediction_data?.['1_day']?.model_predictions || explanationData.prediction_data?.next_day?.model_predictions || {},
            ensemble_weights: explanationData.prediction_data?.['1_day']?.ensemble_weights || explanationData.prediction_data?.next_day?.ensemble_weights || {}
          },
          next_day: {
            predicted_price: explanationData.prediction_data?.next_day?.predicted_price || explanationData.prediction_data?.['1_day']?.predicted_price || 0,
            confidence: explanationData.prediction_data?.next_day?.confidence || explanationData.prediction_data?.['1_day']?.confidence || 0,
            price_change: explanationData.prediction_data?.next_day?.price_change || explanationData.prediction_data?.['1_day']?.price_change || 0,
            price_range: explanationData.prediction_data?.next_day?.price_range || explanationData.prediction_data?.['1_day']?.price_range || {},
            model_predictions: explanationData.prediction_data?.next_day?.model_predictions || explanationData.prediction_data?.['1_day']?.model_predictions || {},
            ensemble_weights: explanationData.prediction_data?.next_day?.ensemble_weights || explanationData.prediction_data?.['1_day']?.ensemble_weights || {}
          },
          '7_day': {
            predicted_price: explanationData.prediction_data?.['7_day']?.predicted_price || 0,
            confidence: explanationData.prediction_data?.['7_day']?.confidence || 0,
            price_change: explanationData.prediction_data?.['7_day']?.price_change || 0,
            price_range: explanationData.prediction_data?.['7_day']?.price_range || {},
            model_predictions: explanationData.prediction_data?.['7_day']?.model_predictions || {},
            ensemble_weights: explanationData.prediction_data?.['7_day']?.ensemble_weights || {}
          },
          '30_day': {
            predicted_price: explanationData.prediction_data?.['30_day']?.predicted_price || 0,
            confidence: explanationData.prediction_data?.['30_day']?.confidence || 0,
            price_change: explanationData.prediction_data?.['30_day']?.price_change || 0,
            price_range: explanationData.prediction_data?.['30_day']?.price_range || {},
            model_predictions: explanationData.prediction_data?.['30_day']?.model_predictions || {},
            ensemble_weights: explanationData.prediction_data?.['30_day']?.ensemble_weights || {}
          }
        },
        technical_summary: {
          rsi: explanationData.technical_indicators?.RSI || 0,
          macd_signal: (explanationData.technical_indicators?.MACD_Signal || 0) > 0 ? 'Bullish' : 'Bearish',
          bollinger_position: explanationData.technical_indicators?.Close > explanationData.technical_indicators?.Bollinger_Upper ? 'Upper Band' :
            explanationData.technical_indicators?.Close < explanationData.technical_indicators?.Bollinger_Lower ? 'Lower Band' : 'Mid-range',
          volume_trend: (explanationData.technical_indicators?.Volume || 0) > (explanationData.technical_indicators?.Volume_SMA || 0) ? 'High' : 'Normal'
        },
        metadata: {
          data_sources: explanationData.data_sources_used || [],
          quality_score: 0.95,
          processing_time: explanationData.timestamp || storedData.timestamp,
          api_version: 'v2.5-stored'
        }
      });
    }

    console.log(`❌ No stored explanation found for ${ticker} on ${date}`);
    res.status(404).json({
      error: 'No explanation available',
      message: `No AI explanation has been generated for ${ticker} yet. Explanations are generated daily by the CI/CD pipeline.`,
      ticker,
      date
    });
  } catch (error) {
    console.error('Error fetching comprehensive explanation:', error);
    res.status(500).json({
      error: 'Failed to fetch comprehensive explanation',
      message: error.message
    });
  }
};

// Get stored explanation from database
const getStoredExplanation = async (req, res) => {
  try {
    const { symbol } = req.params;
    const { window = 'comprehensive' } = req.query;
    const ticker = symbol.toUpperCase();

    console.log(`📥 Retrieving stored explanation for ${ticker}-${window} from MongoDB`);

    // Get directly from MongoDB
    const storedData = await mongoConnection.getStoredExplanation(ticker, window);

    if (storedData && storedData.explanation_data) {
      console.log(`✅ Found stored explanation for ${ticker} (${storedData.explanation_data.explanation_length || 0} chars)`);

      // Transform MongoDB data to frontend format
      const explanationData = storedData.explanation_data;

      const transformedData = {
        ticker: ticker,
        date: explanationData.explanation_date || new Date().toISOString().split('T')[0],
        explanation: explanationData.ai_explanation,
        data_summary: {
          blended_sentiment: explanationData.sentiment_summary?.blended_sentiment || 0,
          total_data_points: (
            (explanationData.sentiment_summary?.finviz_articles || 0) +
            (explanationData.sentiment_summary?.reddit_posts || 0) +
            (explanationData.sentiment_summary?.rss_articles || 0) +
            (explanationData.sentiment_summary?.marketaux_articles || 0)
          ),
          finviz_articles: explanationData.sentiment_summary?.finviz_articles || 0,
          reddit_posts: explanationData.sentiment_summary?.reddit_posts || 0,
          rss_articles: explanationData.sentiment_summary?.rss_articles || 0,
          marketaux_articles: explanationData.sentiment_summary?.marketaux_articles || 0
        },
        prediction_summary: {
          '1_day': {
            predicted_price: explanationData.prediction_data?.['1_day']?.predicted_price || explanationData.prediction_data?.next_day?.predicted_price || 0,
            confidence: explanationData.prediction_data?.['1_day']?.confidence || explanationData.prediction_data?.next_day?.confidence || 0,
            price_change: explanationData.prediction_data?.['1_day']?.price_change || explanationData.prediction_data?.next_day?.price_change || 0,
            price_range: explanationData.prediction_data?.['1_day']?.price_range || explanationData.prediction_data?.next_day?.price_range || {},
            model_predictions: explanationData.prediction_data?.['1_day']?.model_predictions || explanationData.prediction_data?.next_day?.model_predictions || {},
            ensemble_weights: explanationData.prediction_data?.['1_day']?.ensemble_weights || explanationData.prediction_data?.next_day?.ensemble_weights || {}
          },
          // Keep next_day for backward compatibility if needed by other components
          next_day: {
            predicted_price: explanationData.prediction_data?.next_day?.predicted_price || explanationData.prediction_data?.['1_day']?.predicted_price || 0,
            confidence: explanationData.prediction_data?.next_day?.confidence || explanationData.prediction_data?.['1_day']?.confidence || 0,
            price_change: explanationData.prediction_data?.next_day?.price_change || explanationData.prediction_data?.['1_day']?.price_change || 0,
            price_range: explanationData.prediction_data?.next_day?.price_range || explanationData.prediction_data?.['1_day']?.price_range || {},
            model_predictions: explanationData.prediction_data?.next_day?.model_predictions || explanationData.prediction_data?.['1_day']?.model_predictions || {},
            ensemble_weights: explanationData.prediction_data?.next_day?.ensemble_weights || explanationData.prediction_data?.['1_day']?.ensemble_weights || {}
          },
          '7_day': {
            predicted_price: explanationData.prediction_data?.['7_day']?.predicted_price || 0,
            confidence: explanationData.prediction_data?.['7_day']?.confidence || 0,
            price_change: explanationData.prediction_data?.['7_day']?.price_change || 0,
            price_range: explanationData.prediction_data?.['7_day']?.price_range || {},
            model_predictions: explanationData.prediction_data?.['7_day']?.model_predictions || {},
            ensemble_weights: explanationData.prediction_data?.['7_day']?.ensemble_weights || {}
          },
          '30_day': {
            predicted_price: explanationData.prediction_data?.['30_day']?.predicted_price || 0,
            confidence: explanationData.prediction_data?.['30_day']?.confidence || 0,
            price_change: explanationData.prediction_data?.['30_day']?.price_change || 0,
            price_range: explanationData.prediction_data?.['30_day']?.price_range || {},
            model_predictions: explanationData.prediction_data?.['30_day']?.model_predictions || {},
            ensemble_weights: explanationData.prediction_data?.['30_day']?.ensemble_weights || {}
          }
        },
        technical_summary: {
          rsi: explanationData.technical_indicators?.RSI || 0,
          macd: explanationData.technical_indicators?.MACD || 0,
          macd_signal: (explanationData.technical_indicators?.MACD_Signal || 0) > 0 ? 'Bullish' : 'Bearish',
          bollinger_position: explanationData.technical_indicators?.Close > explanationData.technical_indicators?.Bollinger_Upper ? 'Upper Band' :
            explanationData.technical_indicators?.Close < explanationData.technical_indicators?.Bollinger_Lower ? 'Lower Band' : 'Mid-range',
          volume_trend: (explanationData.technical_indicators?.Volume || 0) > (explanationData.technical_indicators?.Volume_SMA || 0) ? 'High' : 'Normal',
          // Extended technical data
          bollinger_upper: explanationData.technical_indicators?.Bollinger_Upper || 0,
          bollinger_lower: explanationData.technical_indicators?.Bollinger_Lower || 0,
          sma_20: explanationData.technical_indicators?.SMA_20 || 0,
          sma_50: explanationData.technical_indicators?.SMA_50 || 0,
          ema_12: explanationData.technical_indicators?.EMA_12 || 0,
          ema_26: explanationData.technical_indicators?.EMA_26 || 0,
          volume: explanationData.technical_indicators?.Volume || 0,
          volume_sma: explanationData.technical_indicators?.Volume_SMA || 0,
          close_price: explanationData.technical_indicators?.Close || 0
        },
        metadata: {
          data_sources: explanationData.data_sources_used || [],
          quality_score: 0.95,
          processing_time: "Retrieved from MongoDB (Node.js Backend)",
          api_version: "v2.5-stored-backend",
          explanation_length: explanationData.explanation_length || 0,
          timestamp: explanationData.timestamp || storedData.timestamp
        }
      };

      return res.json(transformedData);
    }

    console.log(`❌ No stored explanation found for ${ticker}-${window}`);
    res.status(404).json({
      error: 'No stored explanation found',
      message: `No explanation available for ${ticker} with window ${window}`
    });
  } catch (error) {
    console.error('Error fetching stored explanation:', error);
    res.status(500).json({
      error: 'Failed to fetch stored explanation',
      message: error.message
    });
  }
};

// Generate new AI explanation (triggers ML backend)
const generateAIExplanation = async (req, res) => {
  try {
    const { symbol } = req.params;
    const { date } = req.body;
    const ticker = symbol.toUpperCase();
    const targetDate = date || new Date().toISOString().split('T')[0];

    // Call ML backend to generate comprehensive explanation
    try {
      const response = await axios.get(`${ML_BACKEND_URL}/api/v1/explain/${ticker}/${targetDate}`, {
        timeout: 30000 // 30 seconds for AI generation
      });

      if (response.data) {
        return res.json({
          status: 'success',
          explanation: response.data,
          message: `AI explanation generated for ${ticker} on ${targetDate}`
        });
      } else {
        throw new Error('Empty response from ML backend');
      }
    } catch (mlError) {
      console.error(`ML Backend error for ${ticker}:`, mlError.message);

      // Return fallback response
      return res.status(503).json({
        status: 'fallback',
        error: 'ML backend unavailable',
        message: `Could not generate AI explanation for ${ticker}. ML backend may be down.`,
        fallback_available: true
      });
    }
  } catch (error) {
    console.error('Error generating AI explanation:', error);
    res.status(500).json({
      status: 'error',
      error: 'Failed to generate AI explanation',
      message: error.message
    });
  }
};

// Get batch explanation status
const getBatchExplanationStatus = async (req, res) => {
  try {
    console.log('📊 Getting batch explanation status from MongoDB');

    const status = await mongoConnection.getBatchStatus();

    console.log(`✅ Batch status: ${status.with_explanations}/${status.total_tickers} (${status.coverage_percentage}%)`);

    res.json(status);
  } catch (error) {
    console.error('Error getting batch status:', error);
    res.status(500).json({
      error: 'Failed to get batch status',
      message: error.message
    });
  }
};

// Get available stocks with explanations
const getAvailableStocksWithExplanations = async (req, res) => {
  try {
    const mongoClient = req.app.locals.mongoClient;
    const result = await mongoClient.getAvailableStocksWithExplanations();

    res.json(result);
  } catch (error) {
    console.error('Error getting available stocks with explanations:', error);
    res.status(500).json({ error: 'Failed to get available stocks' });
  }
};

// Get real ML predictions from ML backend or MongoDB
const getPredictions = async (req, res) => {
  try {
    const { symbol } = req.params;
    const upperSymbol = symbol.toUpperCase();

    // Try to get predictions from ML backend first (live)
    try {
      const mlResponse = await axios.get(`${ML_BACKEND_URL}/api/v1/predictions/${upperSymbol}`, {
        timeout: 5000
      });

      if (mlResponse.data) {
        // Support both { windows: {...} } and legacy { next_day: {...} } formats
        const predictions = mlResponse.data.windows || mlResponse.data;
        const transformedData = {};

        // Transform prediction data structure
        for (const [window, predData] of Object.entries(predictions)) {
          // Skip _meta — it's metadata, not a prediction window
          if (window === '_meta') continue;
          if (predData && typeof predData === 'object') {
            // Standardize 'next_day' to '1_day' for frontend compatibility
            const normalizedWindow = window === 'next_day' ? '1_day' : window;

            transformedData[normalizedWindow] = {
              predicted_price: predData.predicted_price || 0,
              predicted_change: predData.price_change || 0,
              current_price: predData.current_price || 0,
              confidence: predData.confidence || 0,
              price_change: predData.price_change || 0,
              model_predictions: predData.model_predictions || {},
              ensemble_weights: predData.ensemble_weights || {},
              // Alpha fields (market-neutral model)
              alpha_pct: predData.alpha_pct ?? null,
              alpha_implied_price: predData.alpha_implied_price ?? null,
              prob_positive: predData.prob_positive ?? null,
              is_market_neutral: predData.is_market_neutral ?? false
            };
          }
        }

        return res.json({ [upperSymbol]: transformedData });
      }
    } catch (mlError) {
      console.log(`ML backend unavailable for ${upperSymbol}, trying MongoDB...`);
    }

    // Fallback: Read stored predictions from MongoDB
    try {
      console.log(`🔍 Controller: Attempting to fetch stored predictions for ${upperSymbol}`);
      const storedPredictions = await mongoConnection.getLatestPredictions(upperSymbol);

      if (storedPredictions) {
        console.log(`✅ Controller: Found stored predictions for ${upperSymbol}`);

        const transformedData = {};

        for (const [window, predData] of Object.entries(storedPredictions)) {
          // Skip _meta — it's metadata, not a prediction window
          if (window === '_meta') continue;
          if (predData && typeof predData === 'object') {
            // Standardize 'next_day' to '1_day' for frontend compatibility
            const normalizedWindow = window === 'next_day' ? '1_day' : window;

            transformedData[normalizedWindow] = {
              predicted_price: predData.predicted_price || 0,
              predicted_change: predData.price_change || 0,
              current_price: predData.current_price || 0,
              confidence: predData.confidence || 0,
              price_change: predData.price_change || 0,
              model_predictions: predData.model_predictions || {},
              ensemble_weights: predData.ensemble_weights || {},
              // Alpha fields (market-neutral model)
              alpha_pct: predData.alpha_pct ?? null,
              alpha_implied_price: predData.alpha_implied_price ?? null,
              prob_positive: predData.prob_positive ?? null,
              is_market_neutral: predData.is_market_neutral ?? false
            };
          }
        }

        console.log(`✅ Serving stored predictions for ${upperSymbol} from MongoDB`);
        return res.json({ [upperSymbol]: transformedData });
      }
    } catch (mongoError) {
      console.log(`MongoDB predictions unavailable for ${upperSymbol}:`, mongoError.message);
    }

    // No predictions available from any source
    res.status(404).json({ error: `No predictions available for ${upperSymbol}` });
  } catch (error) {
    console.error('Error getting predictions:', error);
    res.status(500).json({ error: 'Failed to get predictions' });
  }
};

// Get technical indicators for a stock
const getTechnicalIndicators = async (req, res) => {
  try {
    const { symbol } = req.params;
    const upperSymbol = symbol.toUpperCase();

    console.log(`📊 Fetching technical indicators for ${upperSymbol}`);

    // PRIORITY 1: Read pre-computed indicators from MongoDB (instant, no API calls)
    let indicators = null;
    try {
      const explanation = await mongoConnection.getStoredExplanation(upperSymbol, 'comprehensive');
      const tech = explanation?.explanation_data?.technical_indicators;
      // Only use MongoDB data if the object has at least one real indicator value
      const hasRealData = tech && (tech.RSI !== undefined || tech.MACD !== undefined || tech.SMA_20 !== undefined || tech.Close !== undefined);
      if (hasRealData) {
        const close = tech.Close || 0;
        const sma20 = tech.SMA_20 || null;
        const sma50 = tech.SMA_50 || null;
        const ema12 = tech.EMA_12 || null;
        const ema26 = tech.EMA_26 || null;
        const rsiVal = tech.RSI || null;
        const macdVal = tech.MACD || null;
        const macdSig = tech.MACD_Signal || null;
        const histogram = macdVal !== null && macdSig !== null ? macdVal - macdSig : null;

        // RSI interpretation
        let rsiSignal = 'Neutral';
        let rsiInterp = 'Neutral - No clear direction';
        if (rsiVal !== null) {
          if (rsiVal >= 70) { rsiSignal = 'Overbought'; rsiInterp = 'Overbought - Consider taking profits'; }
          else if (rsiVal <= 30) { rsiSignal = 'Oversold'; rsiInterp = 'Oversold - Consider buying opportunity'; }
          else if (rsiVal >= 60) { rsiInterp = 'Bullish momentum - Uptrend likely'; }
          else if (rsiVal <= 40) { rsiInterp = 'Bearish momentum - Downtrend likely'; }
        }

        // MACD interpretation
        const macdTrend = histogram !== null ? (histogram > 0 ? 'Bullish' : 'Bearish') : 'Neutral';
        let macdInterp = 'Neutral - Watch for crossover';
        if (histogram !== null && macdVal !== null && macdSig !== null) {
          if (histogram > 0 && macdVal > macdSig) macdInterp = 'Bullish momentum increasing';
          else if (histogram > 0) macdInterp = 'Bullish momentum weakening';
          else if (histogram < 0 && macdVal < macdSig) macdInterp = 'Bearish momentum increasing';
          else macdInterp = 'Bearish momentum weakening';
        }

        // Summary signal
        let bullish = 0, bearish = 0;
        if (rsiVal !== null) {
          if (rsiVal < 30) bullish += 2;
          else if (rsiVal < 40) bullish += 1;
          else if (rsiVal > 70) bearish += 2;
          else if (rsiVal > 60) bearish += 1;
        }
        if (macdTrend === 'Bullish') bullish += 2; else if (macdTrend === 'Bearish') bearish += 2;
        if (sma20 && sma50) { if (sma20 > sma50) bullish += 1; else bearish += 1; }

        const total = bullish + bearish;
        const bullPct = total > 0 ? (bullish / total) * 100 : 50;
        let summarySignal = 'Neutral', summaryDesc = 'Mixed signals - wait for confirmation', summaryStr = 50;
        if (bullPct >= 70) { summarySignal = 'Strong Buy'; summaryStr = bullPct; summaryDesc = 'Multiple indicators confirm bullish trend'; }
        else if (bullPct >= 55) { summarySignal = 'Buy'; summaryStr = bullPct; summaryDesc = 'Technical indicators lean bullish'; }
        else if (bullPct <= 30) { summarySignal = 'Strong Sell'; summaryStr = 100 - bullPct; summaryDesc = 'Multiple indicators confirm bearish trend'; }
        else if (bullPct <= 45) { summarySignal = 'Sell'; summaryStr = 100 - bullPct; summaryDesc = 'Technical indicators lean bearish'; }

        indicators = {
          symbol: upperSymbol,
          timestamp: explanation.timestamp || new Date().toISOString(),
          indicators: {
            rsi: { value: rsiVal, signal: rsiSignal, window: 14, interpretation: rsiInterp },
            macd: { value: macdVal, signal: macdSig, histogram, trend: macdTrend, interpretation: macdInterp },
            sma: {
              sma20, sma50,
              trend: sma20 && sma50 ? (sma20 > sma50 ? 'Bullish (Golden Cross potential)' : 'Bearish (Death Cross potential)') : 'Unknown'
            },
            ema: {
              ema12, ema26,
              trend: ema12 && ema26 ? (ema12 > ema26 ? 'Bullish' : 'Bearish') : 'Unknown'
            }
          },
          summary: { signal: summarySignal, strength: summaryStr, description: summaryDesc },
          source: 'mongodb_pipeline',
          cached: true
        };
        console.log(`✅ Technical indicators for ${upperSymbol} loaded from MongoDB (instant)`);
      }
    } catch (mongoErr) {
      console.warn(`⚠️ MongoDB technical indicators lookup failed for ${upperSymbol}:`, mongoErr.message);
    }

    // PRIORITY 2: Fall back to financialdata.net calculated indicators
    if (!indicators) {
      console.log(`⚠️ No MongoDB indicators for ${upperSymbol}, falling back to calculated indicators...`);
      indicators = await massiveService.getAllIndicators(upperSymbol);
    }

    res.json({
      success: true,
      data: indicators
    });
  } catch (error) {
    console.error(`Error fetching technical indicators for ${req.params.symbol}:`, error.message);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch technical indicators',
      message: error.message
    });
  }
};

// ── Search stocks using Finnhub Symbol Search (works for ANY stock) ──
const searchStocks = async (req, res) => {
  try {
    const { query } = req.params;
    if (!query || query.length < 1) {
      return res.json({ results: [] });
    }

    const upperQuery = query.toUpperCase();

    // First check local tracked stocks for instant results
    const localMatches = Object.entries(COMPANY_DATA)
      .filter(([symbol, data]) =>
        symbol.includes(upperQuery) ||
        data.name.toUpperCase().includes(upperQuery)
      )
      .map(([symbol, data]) => ({
        symbol,
        name: data.name,
        isTracked: true
      }));

    // Then search Finnhub for broader results
    let finnhubResults = [];
    try {
      const finnhubKey = process.env.FINNHUB_API_KEY;
      const response = await axios.get(
        `https://finnhub.io/api/v1/search?q=${encodeURIComponent(query)}&token=${finnhubKey}`,
        { timeout: 8000 }
      );
      if (response.data?.result) {
        finnhubResults = response.data.result
          .filter(r => (r.type === 'Common Stock' || r.type === 'ETP') && !r.symbol.includes('.'))
          .slice(0, 10)
          .map(r => ({
            symbol: r.symbol,
            name: r.description,
            isTracked: !!COMPANY_DATA[r.symbol]
          }));
      }
    } catch (e) {
      console.log(`⚠️ Finnhub search failed: ${e.message}`);
    }

    // Merge: local matches first, then Finnhub results (deduped)
    const seenSymbols = new Set(localMatches.map(m => m.symbol));
    const merged = [...localMatches];
    for (const r of finnhubResults) {
      if (!seenSymbols.has(r.symbol)) {
        seenSymbols.add(r.symbol);
        merged.push(r);
      }
    }

    res.json({ results: merged.slice(0, 15) });
  } catch (error) {
    console.error('Error searching stocks:', error.message);
    res.json({ results: [] });
  }
};

module.exports = {
  getStockDetails,
  getAIAnalysis,
  getComprehensiveExplanation,
  getStoredExplanation,
  generateAIExplanation,
  getBatchExplanationStatus,
  getAvailableStocksWithExplanations,
  getPredictions,
  getTechnicalIndicators,
  searchStocks
}; 