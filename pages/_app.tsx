import type { AppProps } from "next/app"
import "@/app/globals.css"
import { WebSocketProvider } from "@/hooks/use-websocket-context"

export default function MyApp({ Component, pageProps }: AppProps) {
  return (
    <WebSocketProvider>
      <Component {...pageProps} />
    </WebSocketProvider>
  )
}

