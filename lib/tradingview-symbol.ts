const KNOWN_EXCHANGE_PREFIXES = [
  "NASDAQ",
  "NYSE",
  "AMEX",
  "NYSEARCA",
  "BATS",
  "OTCMKTS",
  "OTC",
  "CBOE",
  "INDEX",
  "BINANCE",
  "OANDA",
  "CAPITALCOM",
] as const

const NASDAQ_SYMBOLS = new Set([
  "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "AMD", "INTC",
  "CSCO", "ADBE", "QCOM", "TXN", "INTU", "AMAT", "PYPL", "PLTR",
  "AMZN", "TSLA", "NFLX", "SBUX", "BKNG", "GILD", "ISRG", "CMCSA",
  "CHTR", "TMUS", "WMT", "COST",
])

const NYSE_SYMBOLS = new Set([
  "JPM", "BAC", "WFC", "GS", "MS", "AXP", "C", "BRK.B", "V", "MA",
  "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "AMGN", "CVS", "PG",
  "KO", "PEP", "MDLZ", "HD", "LOW", "NKE", "MCD", "DIS", "TGT",
  "XOM", "CVX", "CAT", "HON", "BA", "RTX", "LMT", "DE", "GE",
  "FDX", "UPS", "VZ", "T", "LIN", "NEE", "AMT", "IBM", "CRM", "ORCL",
])

export function normalizeTradingViewSymbol(symbol: string): string {
  return symbol
    .trim()
    .toUpperCase()
    .replace(/\//g, ".")
    .replace(/-/g, (match, offset, full) => {
      if (full.includes(":")) return match
      return "."
    })
}

export function mapExchangeToTradingViewPrefix(exchange?: string | null): string | null {
  if (!exchange) return null

  const normalized = exchange.trim().toUpperCase()
  if (!normalized) return null

  if (KNOWN_EXCHANGE_PREFIXES.includes(normalized as (typeof KNOWN_EXCHANGE_PREFIXES)[number])) return normalized
  if (normalized.includes("NASDAQ")) return "NASDAQ"
  if (normalized.includes("NYSE ARCA") || normalized.includes("ARCA")) return "NYSEARCA"
  if (normalized.includes("AMEX") || normalized.includes("NYSE AMERICAN") || normalized.includes("NYSE MKT")) return "AMEX"
  if (normalized.includes("NYSE")) return "NYSE"
  if (normalized.includes("BATS") || normalized.includes("CBOE BZX")) return "BATS"
  if (normalized.includes("OTC")) return "OTCMKTS"
  if (normalized.includes("CBOE")) return "CBOE"

  return null
}

export function getPreferredTradingViewSymbol(symbol: string, exchange?: string | null): string {
  const trimmed = symbol.trim().toUpperCase()
  if (!trimmed) return symbol
  if (trimmed.includes(":")) return trimmed

  const normalizedSymbol = normalizeTradingViewSymbol(trimmed)
  const exchangePrefix = mapExchangeToTradingViewPrefix(exchange)

  if (exchangePrefix) return `${exchangePrefix}:${normalizedSymbol}`
  if (NASDAQ_SYMBOLS.has(normalizedSymbol)) return `NASDAQ:${normalizedSymbol}`
  if (NYSE_SYMBOLS.has(normalizedSymbol)) return `NYSE:${normalizedSymbol}`

  return normalizedSymbol
}