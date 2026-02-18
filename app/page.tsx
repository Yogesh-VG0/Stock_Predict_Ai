import type { Metadata } from "next"
import LandingPage from "@/views/landing"

export const metadata: Metadata = {
  title: "AI-Powered Stock Predictions & Analysis | StockPredict AI",
  description:
    "Daily AI-powered stock predictions for S&P 100 companies using LightGBM ML models, SHAP explainability, and Gemini AI. Real-time market data, sentiment analysis, and technical indicators.",
  openGraph: {
    title: "StockPredict AI — AI-Powered Stock Predictions & Analysis",
    description:
      "Daily AI-powered stock predictions for S&P 100 companies using LightGBM ML models, SHAP explainability, and Gemini AI.",
    url: "https://stockpredict.dev",
    siteName: "StockPredict AI",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "StockPredict AI — AI-Powered Stock Predictions",
      },
    ],
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "StockPredict AI — AI-Powered Stock Predictions & Analysis",
    description:
      "Daily AI-powered stock predictions for S&P 100 companies using LightGBM ML models, SHAP explainability, and Gemini AI.",
    images: ["/og-image.png"],
  },
  alternates: {
    canonical: "https://stockpredict.dev",
  },
}

export default function Page() {
  return <LandingPage />
}
