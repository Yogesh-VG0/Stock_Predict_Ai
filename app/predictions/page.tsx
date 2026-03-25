import type { Metadata } from "next"
import PredictionsPage from "@/views/predictions"

export const metadata: Metadata = {
  title: "AI Stock Predictions | StockPredict AI",
  description: "Browse daily AI-generated stock predictions for S&P 100 companies. LightGBM forecasts for 1-day, 7-day, and 30-day horizons with confidence scores.",
  alternates: { canonical: "/predictions" },
}

export default function Page() {
  return <PredictionsPage />
}
