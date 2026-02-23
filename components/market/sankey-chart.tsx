"use client";

import React, { useMemo, useEffect, useState } from "react";
import { ResponsiveSankey, SankeyNodeDatum } from "@nivo/sankey";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import { ZoomIn, ZoomOut, RotateCcw, Move } from "lucide-react";

type NodeKind = "segment" | "revenue" | "expense" | "profit" | "tax" | "neutral";

type SankeyNode = {
    id: string;
    kind?: NodeKind;
    displayValue?: number;
};

type SankeyLink = {
    source: string;
    target: string;
    value: number;
    color?: string;
};

function formatMoney(v: number) {
    const abs = Math.abs(v || 0);
    if (abs >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
    if (abs >= 1e9) return `$${(v / 1e9).toFixed(2)}B`;
    if (abs >= 1e6) return `$${(v / 1e6).toFixed(2)}M`;
    if (abs >= 1e3) return `$${(v / 1e3).toFixed(2)}K`;
    return `$${(v || 0).toFixed(0)}`;
}

const NEON = {
    blue: "#2EA0FF",
    green: "#2DFF9A",
    yellow: "#FFD54A",
    red: "#FF4D6D",
    gray: "#A7B0C0",
    bg: "transparent",
    panel: "rgba(10, 12, 20, 0.78)",
    border: "rgba(255,255,255,0.10)",
};

const SEGMENT_PALETTE = ["#2EA0FF", "#FFD54A", "#A7B0C0", "#9B5CFF", "#FF7A1A", "#00E5FF"];

function hashColor(id: string) {
    let h = 0;
    for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0;
    return SEGMENT_PALETTE[h % SEGMENT_PALETTE.length];
}

function kindColor(kind: NodeKind, id?: string) {
    switch (kind) {
        case "segment":
            return id ? hashColor(id) : NEON.gray;
        case "revenue":
            return NEON.blue;
        case "profit":
            return NEON.green;
        case "expense":
            return NEON.red;
        case "tax":
            return NEON.yellow;
        default:
            return NEON.gray;
    }
}

function truncate(s: string, max: number) {
    if (!s) return "";
    if (s.length <= max) return s;
    return s.slice(0, max - 1) + "…";
}

export default function SankeyChart({
    data,
    height = 600,
    symbol,
}: {
    data: { nodes: SankeyNode[]; links: SankeyLink[] };
    height?: number;
    symbol?: string;
}) {
    const [isMobile, setIsMobile] = useState(false);

    useEffect(() => {
        const onResize = () => setIsMobile(window.innerWidth < 640);
        onResize();
        window.addEventListener("resize", onResize);
        return () => window.removeEventListener("resize", onResize);
    }, []);

    const enriched = useMemo(() => {
        const nodeMap = new Map<string, SankeyNode>();
        data.nodes.forEach((n) => nodeMap.set(n.id, n));

        return {
            nodes: data.nodes.map((n) => ({
                ...n,
                color: kindColor((n.kind || "neutral") as NodeKind, n.id),
            })),
            links: data.links.map((l) => {
                const targetNode = nodeMap.get(String(l.target));
                const inferred = kindColor(((targetNode?.kind || "neutral") as NodeKind));
                return { ...l, color: l.color || inferred };
            }),
        };
    }, [data]);

    const margin = isMobile
        ? { top: 26, right: 18, bottom: 26, left: 18 }
        : { top: 28, right: 52, bottom: 28, left: 120 };

    const linkOpacity = isMobile ? 0.65 : 0.55;

    const keyNodes = new Set([
        "Total Revenue",
        "Gross Profit",
        "Operating Expenses",
        "Operating Income",
        "Income Before Tax",
        "Net Income",
        "Cost of Revenue",
        "Taxes",
    ]);

    const NodeCard = (nodeProps: any) => {
        const node = nodeProps.node as SankeyNodeDatum<any, any> & {
            kind?: NodeKind;
            displayValue?: number;
            color?: string;
        };

        const x = node.x;
        const y = node.y;
        const w = node.width;
        const h = node.height;

        const kind = (node as any).kind || "neutral";
        const color = (node as any).color || NEON.gray;

        const labelMax = isMobile ? 12 : 18;
        const label = truncate(String(node.id), labelMax);

        const cardW = isMobile ? 180 : 200;
        const cardH = isMobile ? 42 : 48;

        const value = (node as any).displayValue ?? node.value ?? 0;

        // Show fewer cards on mobile to avoid clutter
        const shouldShowCard = !isMobile || keyNodes.has(String(node.id)) || kind === "segment";

        // Place card
        const isLeft = x < 140;
        const rawX = isLeft ? x - (cardW + 10) : x + w + 10;
        const rawY = y + h / 2 - cardH / 2;

        // Clamp
        const PAD = 10;
        const viewportW = typeof window !== "undefined" ? window.innerWidth : 1200;
        let safeX = rawX;
        if (safeX < PAD) safeX = x + w + 10;
        if (safeX + cardW > viewportW - PAD) safeX = x - (cardW + 10);

        let safeY = Math.max(PAD, rawY);
        safeY = Math.min(safeY, height - cardH - PAD);

        const showLogo = String(node.id) === "Total Revenue" && symbol;

        const icon =
            kind === "profit" ? "✅" :
                kind === "expense" ? "🧾" :
                    kind === "tax" ? "🧮" :
                        kind === "revenue" ? "💰" :
                            kind === "segment" ? "◼" :
                                "•";

        return (
            <g>
                <rect x={x} y={y} width={w} height={h} rx={4} ry={4} fill={color} opacity={0.98} />
                <rect
                    x={x - 2}
                    y={y - 2}
                    width={w + 4}
                    height={h + 4}
                    rx={6}
                    ry={6}
                    fill="none"
                    stroke={color}
                    strokeOpacity={0.22}
                    strokeWidth={4}
                />

                {shouldShowCard && (
                    <g transform={`translate(${safeX}, ${safeY})`}>
                        <rect width={cardW} height={cardH} rx={12} ry={12} fill={NEON.panel} stroke={NEON.border} strokeWidth={1} />
                        <rect x={10} y={cardH / 2 - 12} width={4} height={24} rx={2} fill={color} opacity={0.95} />

                        {showLogo ? (
                            <image
                                href={`https://raw.githubusercontent.com/davidepalazzo/ticker-logos/main/ticker_icons/${symbol}.png`}
                                x={22}
                                y={cardH / 2 - 13}
                                width={26}
                                height={26}
                                opacity={0.95}
                                preserveAspectRatio="xMidYMid meet"
                            />
                        ) : (
                            <text x={24} y={cardH / 2 + 6} fontSize={14} fill="#E6EAF2">
                                {icon}
                            </text>
                        )}

                        <text x={52} y={cardH / 2 - 2} fontSize={12} fill="#E6EAF2" fontWeight={750}>
                            {label}
                        </text>
                        <text x={52} y={cardH / 2 + 14} fontSize={12} fill="#A7B0C0" fontWeight={650}>
                            {formatMoney(Number(value))}
                        </text>
                    </g>
                )}
            </g>
        );
    };

    return (
        <div className="relative w-full" style={{ height }}>
            {/* Background */}
            <div
                className="absolute inset-0 rounded-2xl border border-zinc-800/60 pointer-events-none"
                style={{
                    background:
                        "radial-gradient(1200px 420px at 20% 10%, rgba(46,160,255,0.08), transparent 55%)," +
                        "radial-gradient(900px 380px at 70% 20%, rgba(45,255,154,0.07), transparent 55%)," +
                        "radial-gradient(900px 380px at 70% 80%, rgba(255,77,109,0.06), transparent 55%)," +
                        "linear-gradient(180deg, rgba(0,0,0,0.25), rgba(0,0,0,0.35))",
                }}
            />

            <TransformWrapper
                minScale={0.5}
                maxScale={2.5}
                initialScale={isMobile ? 0.75 : 1}
                centerOnInit={true}
                limitToBounds={false}
                wheel={{ step: 0.12 }}
                pinch={{ step: 6 }}
                doubleClick={{ disabled: true }}
            >
                {({ zoomIn, zoomOut, resetTransform }) => (
                    <>
                        {/* Controls */}
                        <div className="absolute right-3 top-3 z-10 flex items-center gap-2 rounded-xl border border-zinc-800 bg-black/60 p-2 backdrop-blur hover:bg-black/80 transition-colors">
                            <div className="hidden sm:flex items-center gap-2 pr-2 border-r border-zinc-800">
                                <Move className="h-4 w-4 text-zinc-300" />
                                <span className="text-xs text-zinc-400 font-medium tracking-wide uppercase">Drag</span>
                            </div>
                            <button onClick={() => zoomOut()} className="p-1.5 rounded-lg hover:bg-white/10 transition" aria-label="Zoom out">
                                <ZoomOut className="h-4 w-4 text-zinc-200" />
                            </button>
                            <button onClick={() => zoomIn()} className="p-1.5 rounded-lg hover:bg-white/10 transition" aria-label="Zoom in">
                                <ZoomIn className="h-4 w-4 text-zinc-200" />
                            </button>
                            <button onClick={() => resetTransform()} className="p-1.5 rounded-lg hover:bg-white/10 transition" aria-label="Reset">
                                <RotateCcw className="h-4 w-4 text-zinc-200" />
                            </button>
                        </div>

                        {/* Chart */}
                        <TransformComponent wrapperStyle={{ width: "100%", height }} contentStyle={{ width: "100%", height }}>
                            <div style={{ width: "100%", height }}>
                                <ResponsiveSankey
                                    data={enriched as any}
                                    margin={margin}
                                    align="justify"
                                    sort="auto"
                                    nodeThickness={isMobile ? 12 : 18}
                                    nodeSpacing={isMobile ? 12 : 24}
                                    nodeBorderWidth={0}
                                    linkOpacity={linkOpacity}
                                    linkHoverOpacity={0.9}
                                    linkContract={isMobile ? 1 : 2}
                                    enableLinkGradient={true}
                                    linkBlendMode="normal"
                                    // Disable default labels & tooltips (we draw our own)
                                    // @ts-ignore - Nivo generic mappings fail strict TS bounds depending on version
                                    nodeLabel={() => ""}
                                    labelTextColor="transparent"
                                    nodeTooltip={() => null}
                                    linkTooltip={() => null}
                                    theme={{
                                        background: "transparent",
                                        text: { fill: "#E6EAF2" },
                                        tooltip: { container: { display: "none" } },
                                    }}
                                    // @ts-ignore
                                    nodeComponent={NodeCard}
                                />
                            </div>
                        </TransformComponent>
                    </>
                )}
            </TransformWrapper>
        </div>
    );
}
