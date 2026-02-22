import type React from "react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import { WebSocketProvider } from "@/hooks/use-websocket-context"
import { Analytics } from "@vercel/analytics/next"
import { SpeedInsights } from "@vercel/speed-insights/next"
import Layout from "@/components/layout/layout"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: {
    default: "StockPredict AI — AI-Powered Stock Predictions & Analysis",
    template: "%s | StockPredict AI",
  },
  description:
    "Free AI stock predictions for S&P 100 companies. Daily LightGBM forecasts with SHAP explainability, multi-source sentiment analysis, and Gemini AI narratives. 1-day, 5-day, and 20-day price direction signals.",
  keywords: [
    "stock prediction",
    "AI stock analysis",
    "machine learning stocks",
    "S&P 100 predictions",
    "LightGBM",
    "SHAP explainability",
    "stock market AI",
    "financial predictions",
    "stock sentiment analysis",
    "TradingView charts",
  ],
  authors: [{ name: "Yogesh Vadivel" }],
  creator: "Yogesh Vadivel",
  publisher: "Yogesh Vadivel",
  metadataBase: new URL("https://stockpredict.dev"),
  alternates: {
    canonical: "/",
  },
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://stockpredict.dev",
    siteName: "StockPredict AI",
    title: "StockPredict AI — AI-Powered Stock Predictions & Analysis",
    description:
      "Free AI stock predictions for S&P 100 companies. Daily LightGBM forecasts, SHAP explainability, and Gemini AI narratives for smarter investing.",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "StockPredict AI — AI-Powered Stock Predictions",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "StockPredict AI — AI-Powered Stock Predictions & Analysis",
    description:
      "Free AI stock predictions for S&P 100 companies. Daily LightGBM forecasts with SHAP explainability and Gemini AI narratives.",
    images: ["/og-image.png"],
    creator: "@YogeshVadivel",
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
  // Favicon for Google/search and browsers (favicon.ico is required for crawlers)
  icons: {
    icon: [
      { url: "/favicon.ico?v=2", sizes: "any" },
      { url: "/favicon-32x32.png?v=2", type: "image/png", sizes: "32x32" },
    ],
    shortcut: "/favicon.ico?v=2",
    apple: "/apple-touch-icon.png?v=2",
  },
  themeColor: "#0f172a",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className} suppressHydrationWarning>
        <WebSocketProvider>
          <Layout>{children}</Layout>
        </WebSocketProvider>
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  )
}
