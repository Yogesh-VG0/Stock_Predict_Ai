import SankeyView from "@/views/sankey-view"

function sanitizeSymbol(value?: string) {
    return (value || "AAPL").trim().toUpperCase().replace(/[^A-Z0-9.\-]/g, "").slice(0, 10) || "AAPL"
}

export default async function SankeyPage({
    searchParams,
}: {
    searchParams?: Promise<{ symbol?: string | string[] }>
}) {
    const params = searchParams ? await searchParams : {}
    const symbolParam = Array.isArray(params.symbol) ? params.symbol[0] : params.symbol

    return <SankeyView symbol={sanitizeSymbol(symbolParam)} />
}
