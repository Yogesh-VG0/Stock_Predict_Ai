/**
 * Shared company data for all tracked stocks.
 * M1 FIX: Extracted from stockController.js and watchlistController.js to prevent data drift.
 */

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
    description: 'NVIDIA Corporation operates as a computing company worldwide. The company operates in two segments, Graphics and Compute & Networking.',
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
    description: 'Amazon.com, Inc. engages in the retail sale of consumer products and subscriptions in North America and internationally.',
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
    description: 'Alphabet Inc. provides various products and platforms in the United States, Europe, the Middle East, Africa, the Asia-Pacific, Canada, and Latin America.',
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
    description: 'Eli Lilly and Company discovers, develops, and markets human pharmaceuticals worldwide.',
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
    description: 'Walmart Inc. engages in the operation of retail, wholesale, and other units worldwide.',
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
    description: 'JPMorgan Chase & Co. operates as a financial services company worldwide.',
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
    description: 'Visa Inc. operates as a payments technology company worldwide.',
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
    description: 'Netflix, Inc. provides entertainment services. It offers TV series, documentaries, feature films, and mobile games across a wide variety of genres and languages.',
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
    description: 'Exxon Mobil Corporation explores for and produces crude oil and natural gas in the United States and internationally.',
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
    description: 'Costco Wholesale Corporation, together with its subsidiaries, engages in the operation of membership warehouses.',
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
    description: 'Oracle Corporation provides products and services that address enterprise information technology environments worldwide.',
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
    description: 'The Procter & Gamble Company provides branded consumer packaged goods worldwide.',
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
    description: 'Johnson & Johnson researches, develops, manufactures, and sells various products in the healthcare field worldwide.',
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
    description: 'UnitedHealth Group Incorporated operates as a diversified health care company in the United States.',
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
    description: 'The Home Depot, Inc. operates as a home improvement retailer.',
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
    description: 'AbbVie Inc. discovers, develops, manufactures, and sells pharmaceuticals worldwide.',
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
    description: 'The Coca-Cola Company, a beverage company, manufactures, markets, and sells various nonalcoholic beverages worldwide.',
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
    description: 'Salesforce, Inc. provides Customer Relationship Management (CRM) technology that brings companies and customers together worldwide.',
    headquarters: 'San Francisco, California',
    founded: 1999,
    employees: 79390,
    ceo: 'Marc Benioff',
    website: 'www.salesforce.com'
  },
  'BAC': {
    name: 'Bank of America Corp.',
    sector: 'Financial Services',
    industry: 'Banks',
    description: 'Bank of America Corporation, through its subsidiaries, provides banking and financial products and services worldwide.',
    headquarters: 'Charlotte, North Carolina',
    founded: 1904,
    employees: 213000,
    ceo: 'Brian Moynihan',
    website: 'www.bankofamerica.com'
  }
};

// List of all tracked symbols
const ALL_TRACKED_SYMBOLS = Object.keys(COMPANY_DATA);

// Priority symbols for faster loading
const PRIORITY_SYMBOLS = [
  'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'BRK.B', 'TSLA', 'AVGO', 'LLY'
];

// Secondary symbols
const SECONDARY_SYMBOLS = ALL_TRACKED_SYMBOLS.filter(s => !PRIORITY_SYMBOLS.includes(s));

// Default watchlist for new users
const DEFAULT_WATCHLIST = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX'];

module.exports = {
  COMPANY_DATA,
  ALL_TRACKED_SYMBOLS,
  PRIORITY_SYMBOLS,
  SECONDARY_SYMBOLS,
  DEFAULT_WATCHLIST
};
