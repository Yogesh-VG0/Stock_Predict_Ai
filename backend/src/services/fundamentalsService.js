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
  revenue: [
    'RevenueFromContractWithCustomerExcludingAssessedTax',
    'Revenues',
    'RevenuesNetOfInterestExpense',
    'SalesRevenueNet',
  ],
  grossProfit: ['GrossProfit'],
  operatingIncome: ['OperatingIncomeLoss'],
  netIncome: ['NetIncomeLoss', 'ProfitLoss'],
  assets: ['Assets'],
  liabilities: ['Liabilities'],
  equity: ['StockholdersEquity'],
  operatingCashFlow: ['NetCashProvidedByUsedInOperatingActivities'],
};

const IMPORTANT_FORMS = new Set(['10-K', '10-Q', '8-K', 'DEF 14A', 'S-8', '4']);
const DAY_MS = 24 * 60 * 60 * 1000;

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

function selectUnit(units, preferredUnit = 'USD') {
  if (!units) return null;
  if (Array.isArray(units?.[preferredUnit])) return preferredUnit;

  const unitKeys = Object.keys(units).filter(key => Array.isArray(units[key]));
  const preferred = preferredUnit.toUpperCase();
  return unitKeys.find(key => key.toUpperCase() === preferred)
    || unitKeys.find(key => key.toUpperCase().includes(preferred) && !key.includes('/'))
    || unitKeys[0]
    || null;
}

function getFactUnitCandidates(companyFacts, factNames, preferredUnit = 'USD') {
  const usGaap = companyFacts?.facts?.['us-gaap'] || {};
  return factNames.flatMap((factName, priority) => {
    const units = usGaap[factName]?.units;
    const unit = selectUnit(units, preferredUnit);
    if (!unit) return [];
    return [{ factName, unit, priority, items: units[unit] }];
  });
}

function compatibleFactGroup(factName) {
  if (factName === 'RevenuesNetOfInterestExpense') return 'bank-net-revenue';
  if (['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues', 'SalesRevenueNet'].includes(factName)) return 'standard-revenue';
  if (['NetIncomeLoss', 'ProfitLoss'].includes(factName)) return 'net-income';
  return factName || 'unknown';
}

function parseQuarterFromFrame(frame) {
  const match = String(frame || '').match(/^CY(\d{4})Q([1-4])$/);
  if (!match) return null;
  return { fiscalYear: Number(match[1]), fiscalPeriod: `Q${match[2]}`, period: `${match[1]} Q${match[2]}` };
}

function dateTime(value) {
  const time = new Date(value || '').getTime();
  return Number.isFinite(time) ? time : null;
}

function durationDays(item) {
  const start = dateTime(item.start);
  const end = dateTime(item.end);
  if (start === null || end === null || end <= start) return null;
  return Math.round((end - start) / DAY_MS);
}

function isQuarterDuration(item) {
  const days = durationDays(item);
  if (days !== null) return days >= 55 && days <= 125;
  return Boolean(parseQuarterFromFrame(item.frame));
}

function isAnnualDurationOrInstant(item) {
  const days = durationDays(item);
  if (days === null) return true;
  return days >= 250 && days <= 400;
}

function normalizeFactItem(item, factName, unit = 'USD') {
  const quarter = parseQuarterFromFrame(item.frame);
  const fp = String(item.fp || '').toUpperCase();
  const fy = Number(item.fy);
  const quarterlyDuration = isQuarterDuration(item);
  const fiscalPeriod = /^Q[1-4]$/.test(fp)
    ? fp
    : quarter?.fiscalPeriod || (item.form === '10-K' && fp === 'FY' && quarterlyDuration ? 'Q4' : null);
  const fiscalYear = Number.isFinite(fy) ? fy : quarter?.fiscalYear || null;

  return {
    factName,
    unit,
    value: Number(item.val),
    start: item.start,
    end: item.end,
    filed: item.filed,
    form: item.form,
    frame: item.frame,
    fiscalYear,
    fiscalPeriod,
    period: fiscalYear && fiscalPeriod ? `${fiscalYear} ${fiscalPeriod}` : quarter?.period || `${item.fy || ''} ${item.fp || ''}`.trim(),
    durationDays: durationDays(item),
  };
}

function latestAnnualMetric(companyFacts, factNames) {
  const candidates = getFactUnitCandidates(companyFacts, factNames);
  const annual = candidates
    .flatMap(({ factName, unit, priority, items }) => items
      .filter(item => Number.isFinite(Number(item.val)) && (item.form === '10-K' || item.fp === 'FY') && isAnnualDurationOrInstant(item))
      .map(item => ({ ...normalizeFactItem(item, factName, unit), priority })))
    .sort((a, b) => {
      const endDiff = (dateTime(b.end) || 0) - (dateTime(a.end) || 0);
      if (endDiff !== 0) return endDiff;
      const filedDiff = (dateTime(b.filed) || 0) - (dateTime(a.filed) || 0);
      if (filedDiff !== 0) return filedDiff;
      return a.priority - b.priority;
    });

  return annual[0] || null;
}

function annualMetricMap(companyFacts, factNames) {
  const candidates = getFactUnitCandidates(companyFacts, factNames);
  const byYearFactAndUnit = new Map();

  for (const { factName, unit, priority, items } of candidates) {
    for (const item of items) {
      if (!Number.isFinite(Number(item.val))) continue;
      if (!(item.form === '10-K' || String(item.fp || '').toUpperCase() === 'FY')) continue;
      if (!isAnnualDurationOrInstant(item)) continue;

      const normalized = { ...normalizeFactItem(item, factName, unit), priority };
      if (!normalized.fiscalYear) continue;

      const key = `${normalized.fiscalYear}-${compatibleFactGroup(normalized.factName)}-${normalized.unit}`;
      const existing = byYearFactAndUnit.get(key);
      const candidate = {
        ...normalized,
        score: 100 + Math.max(0, factNames.length - priority),
      };

      if (isBetterQuarterlyCandidate(candidate, existing)) {
        byYearFactAndUnit.set(key, candidate);
      }
    }
  }

  return byYearFactAndUnit;
}

function quarterlyCandidateScore(item) {
  let score = 0;
  const days = durationDays(item);
  const fp = String(item.fp || '').toUpperCase();

  if (days !== null && days >= 55 && days <= 125) score += 50;
  if (parseQuarterFromFrame(item.frame)) score += 20;
  if (/^Q[1-4]$/.test(fp)) score += 15;
  if (item.form === '10-Q') score += 10;
  if (item.form === '10-K') score += 6;
  if (item.filed) score += 2;
  return score;
}

function isBetterQuarterlyCandidate(next, existing) {
  if (!existing) return true;
  if (next.score !== existing.score) return next.score > existing.score;

  const nextEnd = dateTime(next.end) || 0;
  const existingEnd = dateTime(existing.end) || 0;
  if (nextEnd !== existingEnd) return nextEnd > existingEnd;

  return (dateTime(next.filed) || 0) > (dateTime(existing.filed) || 0);
}

function quarterlyKey(item) {
  if (item.fiscalYear && /^Q[1-4]$/.test(String(item.fiscalPeriod || ''))) {
    return `${item.fiscalYear}-${item.fiscalPeriod}`;
  }

  return item.end || item.period || item.frame || null;
}

function quarterlyMetricMap(companyFacts, factNames) {
  const candidates = getFactUnitCandidates(companyFacts, factNames);
  const byKey = new Map();

  for (const { factName, unit, priority, items } of candidates) {
    for (const item of items) {
      if (!Number.isFinite(Number(item.val))) continue;
      if (String(item.fp || '').toUpperCase() === 'FY') continue;
      if (!isQuarterDuration(item)) continue;

      const normalized = normalizeFactItem(item, factName, unit);
      const key = quarterlyKey(normalized);
      if (!key) continue;

      const existing = byKey.get(key);
      const candidate = {
        ...normalized,
        priority,
        score: quarterlyCandidateScore(item) + Math.max(0, factNames.length - priority),
      };
      if (isBetterQuarterlyCandidate(candidate, existing)) {
        byKey.set(key, candidate);
      }
    }
  }

  return byKey;
}

function annualKeyForQuarter(item) {
  if (!item?.fiscalYear || !item.factName || !item.unit) return null;
  return `${item.fiscalYear}-${compatibleFactGroup(item.factName)}-${item.unit}`;
}

function addDerivedFourthQuarters(quarterly, annualByFactAndUnit) {
  const fiscalYears = new Set(
    [...quarterly.values()]
      .map(item => item.fiscalYear)
      .filter(Boolean)
  );

  for (const fiscalYear of fiscalYears) {
    const q4Key = `${fiscalYear}-Q4`;
    if (quarterly.has(q4Key)) continue;

    const q1 = quarterly.get(`${fiscalYear}-Q1`);
    const q2 = quarterly.get(`${fiscalYear}-Q2`);
    const q3 = quarterly.get(`${fiscalYear}-Q3`);
    if (!q1 || !q2 || !q3) continue;

    const q1AnnualKey = annualKeyForQuarter(q1);
    const q2AnnualKey = annualKeyForQuarter(q2);
    const q3AnnualKey = annualKeyForQuarter(q3);
    if (!q1AnnualKey || q1AnnualKey !== q2AnnualKey || q1AnnualKey !== q3AnnualKey) continue;

    const annualItem = annualByFactAndUnit.get(q1AnnualKey);
    if (!annualItem) continue;

    const derivedValue = annualItem.value - q1.value - q2.value - q3.value;
    if (!Number.isFinite(derivedValue)) continue;

    // Some SEC facts are restated or use non-comparable units. Avoid showing
    // nonsensical derived Q4 values if the annual and quarterly values clearly
    // do not reconcile.
    const annualAbs = Math.abs(annualItem.value) || 1;
    if (Math.abs(derivedValue) > annualAbs * 1.5) continue;

    quarterly.set(q4Key, {
      ...annualItem,
      value: derivedValue,
      fiscalPeriod: 'Q4',
      period: `${fiscalYear} Q4`,
      derived: true,
      derivation: 'FY minus Q1-Q3 SEC XBRL facts',
      score: 85,
    });
  }

  return quarterly;
}

function quarterSortValue(row) {
  const fiscalYear = Number(row.fiscalYear) || 0;
  const fiscalQuarter = Number(String(row.fiscalPeriod || '').replace('Q', '')) || 0;
  if (fiscalYear && fiscalQuarter) {
    // Return a date-like value, not a small ordinal, so mixed rows with only
    // end dates still sort correctly against fiscal-year/quarter rows.
    return Date.UTC(fiscalYear, fiscalQuarter * 3 - 1, 1);
  }

  const byEnd = dateTime(row.end || row.date);
  return byEnd !== null ? byEnd : 0;
}

function buildQuarterlySeries(companyFacts) {
  const revenue = addDerivedFourthQuarters(
    quarterlyMetricMap(companyFacts, FACT_DEFINITIONS.revenue),
    annualMetricMap(companyFacts, FACT_DEFINITIONS.revenue),
  );
  const netIncome = addDerivedFourthQuarters(
    quarterlyMetricMap(companyFacts, FACT_DEFINITIONS.netIncome),
    annualMetricMap(companyFacts, FACT_DEFINITIONS.netIncome),
  );
  const keys = [...new Set([...revenue.keys(), ...netIncome.keys()])]
    .map(key => {
      const rev = revenue.get(key);
      const net = netIncome.get(key);
      const base = rev || net;
      return { key, rev, net, base };
    })
    .sort((a, b) => quarterSortValue({ ...a.base, date: a.key }) - quarterSortValue({ ...b.base, date: b.key }))
    .slice(-8);

  return keys.map(({ key, rev, net, base }) => {
    return {
      date: base?.end || key,
      period: base?.period || key,
      fiscalYear: base?.fiscalYear || null,
      fiscalPeriod: base?.fiscalPeriod || null,
      end: base?.end || null,
      filed: base?.filed || null,
      revenueFactName: rev?.factName || null,
      netIncomeFactName: net?.factName || null,
      revenueUnit: rev?.unit || null,
      netIncomeUnit: net?.unit || null,
      revenueDerived: Boolean(rev?.derived),
      netIncomeDerived: Boolean(net?.derived),
      revenueDerivation: rev?.derivation || null,
      netIncomeDerivation: net?.derivation || null,
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
          unit: item.unit,
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