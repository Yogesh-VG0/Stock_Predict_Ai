import { ImageResponse } from "next/og"

export const runtime = "edge"
export const alt = "StockPredict AI — AI-Powered Stock Predictions"
export const size = { width: 1200, height: 630 }
export const contentType = "image/png"

export default async function Image() {
  return new ImageResponse(
    (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          width: "100%",
          height: "100%",
          background: "linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)",
          color: "white",
          fontFamily: "sans-serif",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "16px",
            marginBottom: "24px",
          }}
        >
          <span style={{ fontSize: "56px" }}>📈</span>
          <span style={{ fontSize: "64px", fontWeight: 800, letterSpacing: "-2px" }}>
            StockPredict AI
          </span>
        </div>
        <div
          style={{
            fontSize: "28px",
            color: "#94a3b8",
            textAlign: "center",
            maxWidth: "800px",
            lineHeight: 1.4,
            display: "flex",
            flexDirection: "column",
          }}
        >
          <span>Free AI stock predictions for S&P 100 companies.</span>
          <span>Daily LightGBM forecasts with SHAP explainability.</span>
        </div>
        <div
          style={{
            display: "flex",
            gap: "32px",
            marginTop: "40px",
            fontSize: "20px",
            color: "#10b981",
          }}
        >
          <span>75+ Tickers</span>
          <span style={{ color: "#64748b" }}>|</span>
          <span>ML Powered</span>
          <span style={{ color: "#64748b" }}>|</span>
          <span>Daily Updates</span>
        </div>
      </div>
    ),
    { ...size }
  )
}
