 "use client"

import type { AppProps } from "next/app"
import "@/app/globals.css"
import { WebSocketProvider } from "@/hooks/use-websocket-context"
import App from "@/app_v0modified"

export default function MyApp(_: AppProps) {
  return (
    <WebSocketProvider>
      <App />
    </WebSocketProvider>
  )
}


