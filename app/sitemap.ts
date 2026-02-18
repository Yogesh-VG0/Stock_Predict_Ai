import { MetadataRoute } from "next"

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

  return [
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
      priority: 0.8,
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
      url: `${baseUrl}/watchlist`,
      lastModified: dailyDate,
      changeFrequency: "daily",
      priority: 0.7,
    },
  ]
}
