import type { Metadata } from "next"
import HomePage from "@/views/home"

export const metadata: Metadata = {
  title: "Dashboard | StockPredict AI",
  description: "Real-time AI stock dashboard with market heatmap, economic calendar, hotlists, and live price data for S&P 100 companies.",
  alternates: { canonical: "/dashboard" },
}

export default function Page() {
  return <HomePage />
}
