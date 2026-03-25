import type { Metadata } from "next"
import NewsPage from "@/views/news"

export const metadata: Metadata = {
  title: "Stock Market News | StockPredict AI",
  description: "Aggregated stock market news from Finnhub, Marketaux, TickerTick, and NewsAPI with AI sentiment analysis. Filter by ticker, sector, or sentiment.",
  alternates: { canonical: "/news" },
}

export default function Page() {
  return <NewsPage />
}
