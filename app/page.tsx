"use client"

import { useEffect, useState } from "react"
import App from "@/app_v0modified"

export default function Page() {
  const [isClient, setIsClient] = useState(false)

  useEffect(() => {
    setIsClient(true)
  }, [])

  // Only render on client side - BrowserRouter needs document
  if (!isClient) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-black">
        <div className="flex flex-col items-center gap-4">
          <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
          <p className="text-zinc-500 text-sm">Loading...</p>
        </div>
      </div>
    )
  }

  return <App />
}
