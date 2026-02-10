"use client"

import { useEffect } from "react"
import { useRouter } from "next/router"

// Temporary redirect so direct /stocks/:symbol URLs don't 404.
// The main experience still lives in the SPA router on the homepage.
export default function StockSymbolRedirectPage() {
  const router = useRouter()

  useEffect(() => {
    // Send users back to the app shell; the SPA router will handle the symbol.
    router.replace("/")
  }, [router])

  return null
}

