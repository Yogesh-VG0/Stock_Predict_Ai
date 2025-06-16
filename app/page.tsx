"use client"

import dynamic from "next/dynamic"

// Dynamically import the App component with no SSR
const AppWithNoSSR = dynamic(() => import("@/app_v0modified"), { ssr: false })

export default function Page() {
  return <AppWithNoSSR />
}
