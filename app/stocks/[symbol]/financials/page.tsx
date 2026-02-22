"use client"

import SankeyView from "@/views/sankey-view"
import { useParams } from "next/navigation"

export default function FinancialsPage() {
    const params = useParams<{ symbol: string }>()
    const symbol = params?.symbol || "AAPL"

    return <SankeyView symbol={symbol} />
}
