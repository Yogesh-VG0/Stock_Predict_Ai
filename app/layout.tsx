import type React from "react"
import { Inter } from "next/font/google"
import "./globals.css"
import { WebSocketProvider } from "@/hooks/use-websocket-context"

const inter = Inter({ subsets: ["latin"] })

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className={inter.className} suppressHydrationWarning={true}>
        <WebSocketProvider>
          {children}
        </WebSocketProvider>
      </body>
    </html>
  )
}
