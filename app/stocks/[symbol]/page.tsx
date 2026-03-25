import type { Metadata } from "next"
import StockDetail from "@/views/stock-detail"

interface Props {
  params: Promise<{ symbol: string }>
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { symbol } = await params
  const ticker = symbol.toUpperCase()
  const title = `${ticker} AI Stock Prediction & Analysis`
  const description = `Free AI-powered prediction for ${ticker}. Daily LightGBM forecasts with SHAP explainability, sentiment analysis, and technical indicators.`

  return {
    title,
    description,
    alternates: { canonical: `/stocks/${ticker}` },
    openGraph: {
      title,
      description,
      url: `https://stockpredict.dev/stocks/${ticker}`,
      images: [{ url: "/opengraph-image", width: 1200, height: 630, alt: `${ticker} AI Stock Prediction` }],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
      images: ["/opengraph-image"],
    },
  }
}

export default function Page() {
  return <StockDetail />
}
