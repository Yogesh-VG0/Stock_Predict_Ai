import type React from "react"
import { Inter } from "next/font/google"
import "./globals.css"
import { WebSocketProvider } from "@/hooks/use-websocket-context"
import { Analytics } from "@vercel/analytics/next"
import { SpeedInsights } from "@vercel/speed-insights/next"

const inter = Inter({ subsets: ["latin"] })

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className} suppressHydrationWarning>
        <WebSocketProvider>
          {children}
        </WebSocketProvider>
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  )
}
