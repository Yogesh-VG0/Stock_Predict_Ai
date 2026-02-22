import { MetadataRoute } from "next"

// Full S&P 100 constituent list (as of 2026)
const SP100_TICKERS = [
  "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
  "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C",
  "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
  "CVX", "DE", "DHR", "DIS", "DUK", "EMR", "EXC", "F", "FDX", "GD",
  "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC",
  "INTU", "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA",
  "MCD", "MDLZ", "MDT", "MET", "META", "MMC", "MMM", "MO", "MRK", "MS",
  "MSFT", "NEE", "NFLX", "NKE", "NOW", "NVDA", "ORCL", "OXY", "PANW", "PEP",
  "PFE", "PG", "PM", "PYPL", "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG",
  "T", "TGT", "TMO", "TSLA", "TXN", "UNH", "UNP", "UPS", "USB", "V",
  "VZ", "WFC", "WMT", "XOM",
]

export default function sitemap(): MetadataRoute.Sitemap {
  const baseUrl = "https://stockpredict.dev"
  const now = new Date()

  // Homepage: weekly (content is relatively stable) - Monday-start week
  const homepageDate = new Date(now)
  const day = homepageDate.getDay() // 0=Sun, 1=Mon, ..., 6=Sat
  const diffToMonday = (day + 6) % 7 // Days to subtract to get to Monday
  homepageDate.setDate(homepageDate.getDate() - diffToMonday)
  homepageDate.setHours(0, 0, 0, 0)

  // Dashboard/Predictions: daily (predictions update daily)
  const dailyDate = new Date(now)
  dailyDate.setHours(0, 0, 0, 0) // Today at midnight

  // News: hourly (if it truly updates hourly)
  const hourlyDate = new Date(now)
  hourlyDate.setMinutes(0, 0, 0) // Current hour

  // Static pages
  const staticPages: MetadataRoute.Sitemap = [
    {
      url: baseUrl,
      lastModified: homepageDate,
      changeFrequency: "weekly",
      priority: 1,
    },
    {
      url: `${baseUrl}/dashboard`,
      lastModified: dailyDate,
      changeFrequency: "daily",
      priority: 0.9,
    },
    {
      url: `${baseUrl}/predictions`,
      lastModified: dailyDate,
      changeFrequency: "daily",
      priority: 0.9,
    },
    {
      url: `${baseUrl}/news`,
      lastModified: hourlyDate,
      changeFrequency: "hourly",
      priority: 0.8,
    },
    {
      url: `${baseUrl}/fundamentals`,
      lastModified: dailyDate,
      changeFrequency: "daily",
      priority: 0.7,
    },
    {
      url: `${baseUrl}/how-it-works`,
      lastModified: homepageDate,
      changeFrequency: "monthly",
      priority: 0.6,
    },
    {
      url: `${baseUrl}/methodology`,
      lastModified: homepageDate,
      changeFrequency: "monthly",
      priority: 0.6,
    },
    {
      url: `${baseUrl}/disclaimer`,
      lastModified: homepageDate,
      changeFrequency: "yearly",
      priority: 0.3,
    },
  ]

  // Individual stock pages — highest SEO value (people search "[TICKER] AI stock prediction")
  const stockPages: MetadataRoute.Sitemap = SP100_TICKERS.map((ticker) => ({
    url: `${baseUrl}/stocks/${ticker}`,
    lastModified: dailyDate,
    changeFrequency: "daily" as const,
    priority: 0.8,
  }))

  return [...staticPages, ...stockPages]
}
