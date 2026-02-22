const axios = require('axios');
const path = require('path');
const NodeCache = require('node-cache');
const redisClient = require('./redisClient');

// Ensure dotenv is loaded
require('dotenv').config({ path: path.resolve(__dirname, '..', '..', '.env') });

const FMP_API_KEY = process.env.FMP_API_KEY;
const BASE_URL = 'https://financialmodelingprep.com/api/v3';
const STABLE_URL = 'https://financialmodelingprep.com/stable';

// Cache responses for 12 hours to respect the 250 calls/day limit
const fmpCache = new NodeCache({ stdTTL: 43200 });

// Active requests map to prevent concurrent stampedes
const activeRequests = new Map();

/**
 * Fetches the raw income statement TTM for a given symbol
 */
const getIncomeStatementTTM = async (symbol) => {
    const cacheKey = `fmp_raw_ttm_${symbol}`;

    // 1. Fast in-memory check only. Raw data doesn't go to Redis to save space (30MB limit).
    // The final built Sankey is what goes to Redis.
    const cached = fmpCache.get(cacheKey);
    if (cached) return cached;

    if (!FMP_API_KEY) {
        console.warn('⚠️ FMP_API_KEY not configured.');
        return null;
    }

    try {
        const url = `${BASE_URL}/income-statement/${symbol}?period=annual&limit=1&apikey=${FMP_API_KEY}`;
        const response = await axios.get(url, { timeout: 10000 });

        if (response.data && response.data.length > 0) {
            const data = response.data[0];
            fmpCache.set(cacheKey, data);
            return data;
        }
        return null;
    } catch (error) {
        console.error(`Error fetching FMP Income Statement for ${symbol}:`, error.message);
        return null;
    }
};

/**
 * Fetches the product segmentation revenue for a given symbol
 */
const getProductSegmentation = async (symbol) => {
    const cacheKey = `fmp_raw_prod_seg_${symbol}`;
    const negativeCacheKey = `sankey:v1:${symbol}:segments:none`;

    // 1. Fast in-memory check
    const cached = fmpCache.get(cacheKey);
    if (cached) return cached;

    // 2. Persistent negative cache check
    try {
        if (redisClient.isOpen) {
            const isNegative = await redisClient.get(negativeCacheKey);
            if (isNegative) {
                fmpCache.set(cacheKey, {});
                return {};
            }
        }
    } catch (err) { }

    if (!FMP_API_KEY) {
        return null;
    }

    try {
        const url = `${BASE_URL}/revenue-product-segmentation?symbol=${symbol}&structure=flat&period=annual&apikey=${FMP_API_KEY}`;
        const response = await axios.get(url, { timeout: 10000 });

        if (response.data && response.data.length > 0 && response.data[0].data) {
            const data = response.data[0].data;
            fmpCache.set(cacheKey, data);
            return data;
        }

        // Cache negative result (empty mapping) for 7 days in memory and Redis
        fmpCache.set(cacheKey, {}, 604800);
        try {
            if (redisClient.isOpen) {
                await redisClient.set(negativeCacheKey, '1', { EX: 604800 });
            }
        } catch (err) { }

        return null;
    } catch (error) {
        console.warn(`Segmentation not available or failed for ${symbol}:`, error.message);
        return null;
    }
};

/**
 * Generates the Sankey chart data schema mapping
 */
const getSankeyData = async (symbol) => {
    // Check if there is already an active generation happening for this symbol
    if (activeRequests.has(symbol)) {
        return activeRequests.get(symbol);
    }

    const generatePromise = (async () => {
        // Versioned key for easy schema invalidation later
        const sankeyCacheKey = `sankey:v1:${symbol}:ttm`;
        const lockKey = `lock:sankey:v1:${symbol}:ttm`;

        // Check caches for fully built sankey data
        const memoryCached = fmpCache.get(sankeyCacheKey);
        if (memoryCached) return memoryCached;

        try {
            if (redisClient.isOpen) {
                const redisCached = await redisClient.get(sankeyCacheKey);
                if (redisCached) {
                    const parsed = JSON.parse(redisCached);
                    fmpCache.set(sankeyCacheKey, parsed);
                    return parsed;
                }
            }
        } catch (err) { }

        // Cross-instance Redis locking to prevent stampedes
        let holdingLock = false;
        try {
            if (redisClient.isOpen) {
                const acquired = await redisClient.set(lockKey, '1', { NX: true, EX: 30 });
                if (!acquired) {
                    // Another instance is generating this data. Poll briefly.
                    for (let i = 0; i < 15; i++) {
                        await new Promise(r => setTimeout(r, 1000));
                        const recheck = await redisClient.get(sankeyCacheKey);
                        if (recheck) {
                            const parsed = JSON.parse(recheck);
                            fmpCache.set(sankeyCacheKey, parsed);
                            return parsed;
                        }
                    }
                    throw new Error("Timeout waiting for other instance to compile financial data");
                }
                holdingLock = true;
            }
        } catch (err) { }

        try {
            const [incomeData, segmentationData] = await Promise.all([
                getIncomeStatementTTM(symbol),
                getProductSegmentation(symbol)
            ]);

            if (!incomeData) {
                throw new Error(`Financial data unavailable for ${symbol}`);
            }

            const output = {
                nodes: [],
                links: []
            };

            // Helper to add unique nodes
            const addNode = (id, label) => {
                if (!output.nodes.find(n => n.id === id)) {
                    output.nodes.push({
                        id: id,
                        nodeColor: 'hsl(210, 70%, 50%)' // default
                    });
                }
            };

            // Helper to add links
            const addLink = (source, target, value, color) => {
                // Avoid 0 or negative values which break Sankey
                if (value > 0) {
                    output.links.push({
                        source,
                        target,
                        value,
                        color: color || '#22c55e' // Default green for standard flows
                    });
                }
            };

            // 1. Revenue Sources -> Total Revenue
            addNode('Total Revenue');

            let totalSegmentedRevenue = 0;

            // If we have segmentation, connect those nodes to Total Revenue
            if (segmentationData && Object.keys(segmentationData).length > 0) {
                for (const [segmentName, revenueValue] of Object.entries(segmentationData)) {
                    // Clean up extremely long segment names or map known ones
                    let cleanName = segmentName.replace(' Segment', '').trim();
                    addNode(cleanName);
                    addLink(cleanName, 'Total Revenue', revenueValue, '#3b82f6'); // Blue for incoming revenue sources
                    totalSegmentedRevenue += revenueValue;
                }

                // If segmented revenue doesn't perfectly match total revenue, make an "Other Revenue" bucket
                const diff = incomeData.revenue - totalSegmentedRevenue;
                if (diff > (incomeData.revenue * 0.05)) { // If discrepancy is > 5%
                    addNode('Other Revenue');
                    addLink('Other Revenue', 'Total Revenue', diff, '#3b82f6');
                }
            } else {
                // Fallback if no segmentation: Just have an abstract 'Sales' node feeding Revenue,
                // so the graph has a starting point (Sankey needs roots)
                addNode('Sales/Operations');
                addLink('Sales/Operations', 'Total Revenue', incomeData.revenue, '#3b82f6');
            }

            // 2. Total Revenue -> Cost of Revenue & Gross Profit
            addNode('Cost of Revenue');
            addNode('Gross Profit');
            addLink('Total Revenue', 'Cost of Revenue', incomeData.costOfRevenue, '#ef4444'); // Red for expense
            addLink('Total Revenue', 'Gross Profit', incomeData.grossProfit, '#22c55e'); // Green for profit

            // 3. Gross Profit -> Operating Expenses & Operating Income
            addNode('Operating Expenses');

            // Explicitly map nested Operating Expenses if available (R&D, SG&A, etc.)
            // Otherwise just map the total
            const rd = incomeData.researchAndDevelopmentExpenses || 0;
            const sga = incomeData.sellingGeneralAndAdministrativeExpenses || 0;
            const otherOpex = incomeData.operatingExpenses - (rd + sga);

            if (rd > 0 || sga > 0) {
                if (rd > 0) {
                    addNode('R&D');
                    addLink('Gross Profit', 'R&D', rd, '#ef4444');
                }
                if (sga > 0) {
                    addNode('SG&A');
                    addLink('Gross Profit', 'SG&A', sga, '#ef4444');
                }
                if (otherOpex > 0) {
                    addNode('Other OpEx');
                    addLink('Gross Profit', 'Other OpEx', otherOpex, '#ef4444');
                }
            } else {
                addLink('Gross Profit', 'Operating Expenses', incomeData.operatingExpenses, '#ef4444');
            }

            // Map the remaining Gross Profit to Operating Income
            addNode('Operating Income');
            // Ensure operating income doesn't mathematically break the flow (FMP data can be weird)
            const opIncomeVal = Math.max(0, incomeData.operatingIncome);
            addLink('Gross Profit', 'Operating Income', opIncomeVal, '#22c55e');

            // 4. Operating Income (and Other Income) -> Income Before Tax
            addNode('Income Before Tax');

            // Handling non-operating income/expenses (Interest, etc)
            const otherIncome = incomeData.totalOtherIncomeExpensesNet || 0;
            if (otherIncome < 0) {
                addNode('Other/Interest Expense');
                addLink('Operating Income', 'Other/Interest Expense', Math.abs(otherIncome), '#ef4444');
                addLink('Operating Income', 'Income Before Tax', Math.max(0, incomeData.incomeBeforeTax), '#22c55e');
            } else if (otherIncome > 0) {
                addNode('Other/Interest Income');
                addLink('Other/Interest Income', 'Income Before Tax', otherIncome, '#22c55e');
                addLink('Operating Income', 'Income Before Tax', opIncomeVal, '#22c55e');
            } else {
                addLink('Operating Income', 'Income Before Tax', Math.max(0, incomeData.incomeBeforeTax), '#22c55e');
            }

            // 5. Income Before Tax -> Taxes & Net Income
            addNode('Net Income');

            const taxes = incomeData.incomeTaxExpense || 0;
            if (taxes > 0) {
                addNode('Taxes');
                addLink('Income Before Tax', 'Taxes', taxes, '#ef4444');
            }

            addLink('Income Before Tax', 'Net Income', Math.max(0, incomeData.netIncome), '#22c55e');


            const finalData = {
                financials: {
                    revenue: incomeData.revenue,
                    netIncome: incomeData.netIncome,
                    grossProfitMargin: incomeData.grossProfitRatio,
                    date: incomeData.date,
                    period: incomeData.period,
                    fiscalYear: incomeData.fiscalYear
                },
                sankey: output,
                meta: {
                    hasSegments: !!(segmentationData && Object.keys(segmentationData).length > 0),
                    asOfDate: incomeData.date,
                    version: 'v1'
                }
            };

            // Cache the fully built sankey data explicitly
            fmpCache.set(sankeyCacheKey, finalData);
            try {
                if (redisClient.isOpen) {
                    // TTM TTL = 14 days (1,209,600 seconds)
                    await redisClient.set(sankeyCacheKey, JSON.stringify(finalData), { EX: 1209600 });
                }
            } catch (err) { }

            return finalData;
        } finally {
            // Release the cross-instance lock if we held it
            if (holdingLock && redisClient.isOpen) {
                try {
                    await redisClient.del(lockKey);
                } catch (err) { }
            }
        }
    })();

    // Store the promise in the active requests map to prevent stampedes
    activeRequests.set(symbol, generatePromise);
    try {
        return await generatePromise;
    } finally {
        activeRequests.delete(symbol);
    }
};

module.exports = {
    getIncomeStatementTTM,
    getProductSegmentation,
    getSankeyData
};
