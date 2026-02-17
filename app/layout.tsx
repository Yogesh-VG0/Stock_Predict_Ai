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
  title: "StockPredict AI",
  description:
    "Daily AI predictions for the top 100 US stocks with confidence scores and clear explanations.",
  icons: {
    icon: "/icon.svg",
  },
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
