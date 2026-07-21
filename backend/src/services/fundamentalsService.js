const axios = require('axios');
const Parser = require('rss-parser');

const parser = new Parser({ timeout: 8000 });

const SEC_USER_AGENT = process.env.SEC_USER_AGENT || 'StockPredictAI/1.0 contact@stockpredict.dev';
const CACHE_TTL_MS = 6 * 60 * 60 * 1000;
const cache = new Map();

const COMMON_CIKS = {
  AAPL: '0000320193',
  MSFT: '0000789019',
  GOOGL: '0001652044',
  GOOG: '0001652044',
  AMZN: '0001018724',
  TSLA: '0001318605',
  NVDA: '0001045810',
  META: '0001326801',
  NFLX: '0001065280',
  JPM: '0000019617',
  V: '0001403161',
  JNJ: '0000200406',
  WMT: '0000104169',
  PG: '0000080424',
  UNH: '0000731766',
  HD: '0000354950',
  MA: '0001141391',
  BAC: '0000070858',
  XOM: '0000034088',
  LLY: '0000059478',
  ABBV: '0001551152',
  COST: '0000909832',
  ORCL: '0001341439',
  KO: '0000021344',
  CRM: '0001108524',
  AVGO: '0001730168',
  BRK_B: '0001067983',
  BRK_A: '0001067983',
};

const COMPANY_RSS_FEEDS = {
  AAPL: [
    { source: 'Apple Newsroom', url: 'https://www.apple.com/newsroom/rss-feed.rss' },
  ],
  MSFT: [
    { source: 'Microsoft Blog', url: 'https://blogs.microsoft.com/feed/' },
  ],
  GOOGL: [
    { source: 'Google Blog', url: 'https://blog.google/rss/' },
  ],
  GOOG: [
    { source: 'Google Blog', url: 'https://blog.google/rss/' },
  ],
  META: [
    { source: 'Meta Newsroom', url: 'https://about.fb.com/news/feed/' },
  ],
};

const FACT_DEFINITIONS = {
  revenue: ['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues', 'SalesRevenueNet'],
  grossProfit: ['GrossProfit'],
  operatingIncome: ['OperatingIncomeLoss'],
  netIncome: ['NetIncomeLoss', 'ProfitLoss'],
  assets: ['Assets'],
  liabilities: ['Liabilities'],
  equity: ['StockholdersEquity'],
  operatingCashFlow: ['NetCashProvidedByUsedInOperatingActivities'],
};

const IMPORTANT_FORMS = new Set(['10-K', '10-Q', '8-K', 'DEF 14A', 'S-8', '4']);

function cacheGet(key) {
  const entry = cache.get(key);
  if (!entry || Date.now() > entry.expiresAt) {
    cache.delete(key);
    return null;
  }
  return entry.value;
}

function cacheSet(key, value, ttl = CACHE_TTL_MS) {
  cache.set(key, { value, expiresAt: Date.now() + ttl });
  return value;
}

function normalizeSymbol(symbol) {
  return String(symbol || '')
    .trim()
    .toUpperCase()
    .replace(/[^A-Z0-9.\-]/g, '')
    .slice(0, 10);
}

function cikLookupKey(symbol) {
  return normalizeSymbol(symbol).replace(/[.\-]/g, '_');
}

function secHeaders() {
  return {
    'User-Agent': SEC_USER_AGENT,
    Accept: 'application/json,text/plain,*/*',
  };
}

async function fetchJson(url, timeout = 10000) {
  const response = await axios.get(url, { headers: secHeaders(), timeout });
  return response.data;
}

async function getCikForSymbol(symbol) {
  const normalized = normalizeSymbol(symbol);
  if (!normalized) throw new Error('Invalid symbol');

  const direct = COMMON_CIKS[cikLookupKey(normalized)];
  if (direct) return direct;

  const cacheKey = `sec-company-tickers`;
  let companyTickers = cacheGet(cacheKey);
  if (!companyTickers) {
    companyTickers = await fetchJson('https://www.sec.gov/files/company_tickers.json');
    cacheSet(cacheKey, companyTickers, 24 * 60 * 60 * 1000);
  }

  const rows = Object.values(companyTickers || {});
  const match = rows.find(row => String(row.ticker || '').toUpperCase() === normalized);
  if (!match?.cik_str) throw new Error(`No SEC CIK found for ${normalized}`);

  return String(match.cik_str).padStart(10, '0');
}

function getFactUnits(companyFacts, factNames, unit = 'USD') {
  const usGaap = companyFacts?.facts?.['us-gaap'] || {};
  for (const factName of factNames) {
    const units = usGaap[factName]?.units;
    if (Array.isArray(units?.[unit])) return { factName, items: units[unit] };
  }
  return { factName: null, items: [] };
}

function parseQuarterFromFrame(frame) {
  const match = String(frame || '').match(/^CY(\d{4})Q([1-4])$/);
  if (!match) return null;
  return { fiscalYear: Number(match[1]), fiscalPeriod: `Q${match[2]}`, period: `${match[1]} Q${match[2]}` };
}

function normalizeFactItem(item, factName) {
  const quarter = parseQuarterFromFrame(item.frame);
  return {
    factName,
    value: Number(item.val),
    end: item.end,
    filed: item.filed,
    form: item.form,
    fiscalYear: quarter?.fiscalYear || item.fy || null,
    fiscalPeriod: quarter?.fiscalPeriod || item.fp || null,
    period: quarter?.period || `${item.fy || ''} ${item.fp || ''}`.trim(),
  };
}

function latestAnnualMetric(companyFacts, factNames) {
  const { factName, items } = getFactUnits(companyFacts, factNames);
  const annual = items
    .filter(item => Number.isFinite(Number(item.val)) && (item.form === '10-K' || item.fp === 'FY'))
    .map(item => normalizeFactItem(item, factName))
    .sort((a, b) => new Date(b.filed || b.end).getTime() - new Date(a.filed || a.end).getTime());

  return annual[0] || null;
}

function quarterlyMetricMap(companyFacts, factNames) {
  const { factName, items } = getFactUnits(companyFacts, factNames);
  const byKey = new Map();

  for (const item of items) {
    if (!Number.isFinite(Number(item.val))) continue;
    const parsed = parseQuarterFromFrame(item.frame);
    if (!parsed && !(item.form === '10-Q' && /^Q[1-3]$/.test(String(item.fp || '')))) continue;

    const normalized = normalizeFactItem(item, factName);
    const key = normalized.end || normalized.period;
    if (!key) continue;

    const existing = byKey.get(key);
    if (!existing || new Date(normalized.filed || normalized.end).getTime() > new Date(existing.filed || existing.end).getTime()) {
      byKey.set(key, normalized);
    }
  }

  return byKey;
}

function buildQuarterlySeries(companyFacts) {
  const revenue = quarterlyMetricMap(companyFacts, FACT_DEFINITIONS.revenue);
  const netIncome = quarterlyMetricMap(companyFacts, FACT_DEFINITIONS.netIncome);
  const keys = [...new Set([...revenue.keys(), ...netIncome.keys()])]
    .sort((a, b) => new Date(a).getTime() - new Date(b).getTime())
    .slice(-8);

  return keys.map(key => {
    const rev = revenue.get(key);
    const net = netIncome.get(key);
    const base = rev || net;
    return {
      date: key,
      period: base?.period || key,
      fiscalYear: base?.fiscalYear || null,
      fiscalPeriod: base?.fiscalPeriod || null,
      revenue: rev?.value ?? null,
      netIncome: net?.value ?? null,
    };
  });
}

function buildAnnualMetrics(companyFacts) {
  const metrics = {};
  for (const [key, factNames] of Object.entries(FACT_DEFINITIONS)) {
    const item = latestAnnualMetric(companyFacts, factNames);
    metrics[key] = item
      ? {
          value: item.value,
          period: item.period,
          fiscalYear: item.fiscalYear,
          fiscalPeriod: item.fiscalPeriod,
          end: item.end,
          filed: item.filed,
          factName: item.factName,
        }
      : null;
  }
  return metrics;
}

function normalizeFiling(cik, recent, index) {
  const accessionNumber = recent.accessionNumber?.[index];
  const primaryDocument = recent.primaryDocument?.[index];
  const accessionNoDashes = String(accessionNumber || '').replace(/-/g, '');
  const cikNoLeadingZeros = String(Number(cik));

  return {
    form: recent.form?.[index],
    filingDate: recent.filingDate?.[index],
    reportDate: recent.reportDate?.[index],
    accessionNumber,
    primaryDocument,
    description: recent.primaryDocDescription?.[index] || recent.form?.[index] || 'SEC filing',
    url: accessionNumber && primaryDocument
      ? `https://www.sec.gov/Archives/edgar/data/${cikNoLeadingZeros}/${accessionNoDashes}/${primaryDocument}`
      : `https://www.sec.gov/edgar/browse/?CIK=${cikNoLeadingZeros}`,
  };
}

function buildRecentFilings(cik, submissions) {
  const recent = submissions?.filings?.recent || {};
  const forms = recent.form || [];
  const filings = [];

  for (let i = 0; i < forms.length && filings.length < 10; i += 1) {
    if (!IMPORTANT_FORMS.has(forms[i])) continue;
    filings.push(normalizeFiling(cik, recent, i));
  }

  return filings;
}

function fallbackFeeds(symbol) {
  const clean = encodeURIComponent(normalizeSymbol(symbol));
  return [
    { source: 'Yahoo Finance', url: `https://feeds.finance.yahoo.com/rss/2.0/headline?s=${clean}&region=US&lang=en-US` },
  ];
}

async function fetchCompanyUpdates(symbol) {
  const normalized = normalizeSymbol(symbol);
  const feeds = [...(COMPANY_RSS_FEEDS[normalized] || []), ...fallbackFeeds(normalized)];
  const settled = await Promise.allSettled(
    feeds.map(async feed => ({ feed, parsed: await parser.parseURL(feed.url) }))
  );

  const seen = new Set();
  return settled
    .flatMap(result => {
      if (result.status !== 'fulfilled') return [];
      const { feed, parsed } = result.value;
      return (parsed.items || []).map(item => ({
        title: item.title || 'Company update',
        url: item.link || feed.url,
        published_at: item.isoDate || item.pubDate || new Date().toISOString(),
        source: feed.source,
        snippet: item.contentSnippet || item.content || '',
      }));
    })
    .filter(item => {
      const key = `${item.title}|${item.url}`.toLowerCase();
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .sort((a, b) => new Date(b.published_at).getTime() - new Date(a.published_at).getTime())
    .slice(0, 6);
}

async function getFundamentalsOverview(symbol) {
  const normalized = normalizeSymbol(symbol);
  if (!normalized) throw new Error('Invalid symbol');

  const cacheKey = `fundamentals:${normalized}`;
  const cached = cacheGet(cacheKey);
  if (cached) return cached;

  const cik = await getCikForSymbol(normalized);
  const [companyFacts, submissions, updates] = await Promise.all([
    fetchJson(`https://data.sec.gov/api/xbrl/companyfacts/CIK${cik}.json`),
    fetchJson(`https://data.sec.gov/submissions/CIK${cik}.json`),
    fetchCompanyUpdates(normalized).catch(() => []),
  ]);

  const overview = {
    symbol: normalized,
    cik,
    companyName: submissions?.name || companyFacts?.entityName || normalized,
    source: 'sec_companyfacts_submissions_rss',
    annualMetrics: buildAnnualMetrics(companyFacts),
    quarterly: buildQuarterlySeries(companyFacts),
    filings: buildRecentFilings(cik, submissions),
    updates,
    lastUpdated: new Date().toISOString(),
  };

  return cacheSet(cacheKey, overview);
}

module.exports = {
  getFundamentalsOverview,
  getCikForSymbol,
};