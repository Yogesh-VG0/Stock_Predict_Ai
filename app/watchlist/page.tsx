import type { Metadata } from "next"
import WatchlistPage from "@/views/watchlist"

export const metadata: Metadata = {
  title: "Watchlist | StockPredict AI",
  description: "Track your favorite stocks with real-time prices, AI predictions, and alerts.",
  robots: { index: false, follow: false },
}

export default function Page() {
  return <WatchlistPage />
}
