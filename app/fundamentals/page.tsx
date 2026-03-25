import type { Metadata } from "next"
import FundamentalsPage from "@/views/fundamentals"

export const metadata: Metadata = {
  title: "Stock Fundamentals | StockPredict AI",
  description: "Explore fundamental data for S&P 100 companies including financials, valuation metrics, and company profiles alongside AI predictions.",
  alternates: { canonical: "/fundamentals" },
}

export default function Page() {
  return <FundamentalsPage />
}
