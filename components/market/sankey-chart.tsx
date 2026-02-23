"use client";

import React, { useEffect, useMemo, useState } from "react";
import { ResponsiveSankey } from "@nivo/sankey";
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

// “Infographic solid” palette (closer to your 2nd image)
const PALETTE = {
    revenue: "#2F6BFF", // solid blue
    profit: "#22C55E",  // solid green
    expense: "#EF4444", // solid red
    tax: "#F59E0B",     // solid amber
    neutral: "#94A3B8", // slate/gray
    panel: "rgba(10, 12, 20, 0.78)",
    border: "rgba(255,255,255,0.10)",
    text: "#E6EAF2",
    subtext: "#A7B0C0",
};

const SEGMENT_PALETTE = [
    "#2F6BFF", // blue
    "#F59E0B", // amber
    "#94A3B8", // gray
    "#8B5CF6", // purple
    "#F97316", // orange
    "#06B6D4", // cyan
];

function hashColor(id: string) {
    let h = 0;
    for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0;
    return SEGMENT_PALETTE[h % SEGMENT_PALETTE.length];
}

function kindColor(kind: NodeKind, id?: string) {
    switch (kind) {
        case "segment":
            return id ? hashColor(id) : PALETTE.neutral;
        case "revenue":
            return PALETTE.revenue;
        case "profit":
            return PALETTE.profit;
        case "expense":
            return PALETTE.expense;
        case "tax":
            return PALETTE.tax;
        default:
            return PALETTE.neutral;
    }
}

function truncate(s: string, max: number) {
    if (!s) return "";
    if (s.length <= max) return s;
    return s.slice(0, max - 1) + "…";
}

export default function SankeyChart({
    data,
    height = 680,
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

    // Choose which nodes get cards on mobile (to avoid clutter)
    const keyNodes = useMemo(
        () =>
            new Set([
                "Total Revenue",
                "Gross Profit",
                "Operating Expenses",
                "Operating Income",
                "Income Before Tax",
                "Net Income",
                "Cost of Revenue",
                "Taxes",
            ]),
        []
    );

    // Enrich nodes/links with colors (solid)
    const enriched = useMemo(() => {
        const nodeMap = new Map<string, SankeyNode>();
        data.nodes.forEach((n) => nodeMap.set(n.id, n));

        return {
            nodes: data.nodes.map((n) => ({
                ...n,
                color: kindColor((n.kind || "neutral") as NodeKind, n.id),
            })),
            links: data.links.map((l) => {
                const target = nodeMap.get(String(l.target));
                const inferred = kindColor((target?.kind || "neutral") as NodeKind, target?.id);
                return { ...l, color: l.color || inferred };
            }),
        };
    }, [data]);

    // Bigger right margin prevents “end labels cut off”
    const margin = isMobile
        ? { top: 18, right: 24, bottom: 18, left: 18 }
        : { top: 22, right: 220, bottom: 22, left: 160 };

    const nodeThickness = isMobile ? 12 : 18;
    const nodeSpacing = isMobile ? 12 : 20;

    // Custom layer: draw “infographic cards”
    const CardsLayer = (layerProps: any) => {
        const { nodes, width, height: innerH } = layerProps;

        // chart inner area is 0..width / 0..innerH
        const PAD = 10;
        const cardW = isMobile ? 176 : 210;
        const cardH = isMobile ? 42 : 48;

        return (
            <g>
                {nodes.map((node: any) => {
                    const kind: NodeKind = node.kind || "neutral";
                    const color = node.color || PALETTE.neutral;

                    const shouldShowCard = !isMobile || keyNodes.has(String(node.id)) || kind === "segment";
                    if (!shouldShowCard) return null;

                    const labelMax = isMobile ? 12 : 18;
                    const label = truncate(String(node.id), labelMax);
                    const value = node.displayValue ?? node.value ?? 0;

                    // Card placement:
                    // - segments on the left -> card to the left if possible, otherwise right
                    // - mid/right nodes -> card to the right if possible, otherwise left
                    const preferLeft = node.x < 140;
                    let x = preferLeft ? node.x - (cardW + 12) : node.x + node.width + 12;
                    let y = node.y + node.height / 2 - cardH / 2;

                    // Clamp to inner chart bounds
                    if (x < PAD) x = node.x + node.width + 12;
                    if (x + cardW > width - PAD) x = node.x - (cardW + 12);
                    x = Math.max(PAD, Math.min(x, width - cardW - PAD));

                    y = Math.max(PAD, Math.min(y, innerH - cardH - PAD));

                    const showLogo = String(node.id) === "Total Revenue" && symbol;
                    const icon =
                        kind === "profit" ? "✅" :
                            kind === "expense" ? "🧾" :
                                kind === "tax" ? "🧮" :
                                    kind === "revenue" ? "💰" :
                                        kind === "segment" ? "◼" : "•";

                    return (
                        <g key={node.id}>
                            {/* accent glow around node bar */}
                            <rect
                                x={node.x - 2}
                                y={node.y - 2}
                                width={node.width + 4}
                                height={node.height + 4}
                                rx={6}
                                ry={6}
                                fill="none"
                                stroke={color}
                                strokeOpacity={0.18}
                                strokeWidth={4}
                            />

                            {/* card */}
                            <g transform={`translate(${x}, ${y})`}>
                                <rect
                                    width={cardW}
                                    height={cardH}
                                    rx={12}
                                    ry={12}
                                    fill={PALETTE.panel}
                                    stroke={PALETTE.border}
                                    strokeWidth={1}
                                />
                                <rect x={10} y={cardH / 2 - 12} width={4} height={24} rx={2} fill={color} opacity={0.98} />

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
                                    <text x={24} y={cardH / 2 + 6} fontSize={14} fill={PALETTE.text}>
                                        {icon}
                                    </text>
                                )}

                                <text x={52} y={cardH / 2 - 2} fontSize={12} fill={PALETTE.text} fontWeight={800}>
                                    {label}
                                </text>
                                <text x={52} y={cardH / 2 + 14} fontSize={12} fill={PALETTE.subtext} fontWeight={650}>
                                    {formatMoney(Number(value))}
                                </text>
                            </g>
                        </g>
                    );
                })}
            </g>
        );
    };

    return (
        <div className="relative w-full" style={{ height }}>
            {/* Background like infographic */}
            <div
                className="absolute inset-0 rounded-2xl border border-zinc-800/60 pointer-events-none"
                style={{
                    background:
                        "radial-gradient(1200px 420px at 20% 10%, rgba(47,107,255,0.10), transparent 55%)," +
                        "radial-gradient(900px 380px at 70% 20%, rgba(34,197,94,0.08), transparent 55%)," +
                        "radial-gradient(900px 380px at 70% 80%, rgba(239,68,68,0.07), transparent 55%)," +
                        "linear-gradient(180deg, rgba(0,0,0,0.25), rgba(0,0,0,0.35))",
                }}
            />

            <TransformWrapper
                minScale={0.55}
                maxScale={2.75}
                initialScale={isMobile ? 0.72 : 1}
                centerOnInit
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

                        {/* IMPORTANT: overflow visible so right-side cards are NOT clipped */}
                        <TransformComponent
                            wrapperStyle={{ width: "100%", height, overflow: "visible" }}
                            contentStyle={{ width: "100%", height, overflow: "visible" }}
                        >
                            <div style={{ width: "100%", height }}>
                                <ResponsiveSankey
                                    data={enriched as any}
                                    margin={margin}
                                    align="justify"
                                    sort="auto"
                                    nodeThickness={nodeThickness}
                                    nodeSpacing={nodeSpacing}
                                    nodeBorderWidth={0}
                                    nodeBorderColor={{ from: "color", modifiers: [["darker", 0.2]] }}
                                    // Solid, more “infographic” look:
                                    enableLinkGradient={false}
                                    linkBlendMode="normal"
                                    linkOpacity={isMobile ? 0.70 : 0.60}
                                    linkHoverOpacity={0.92}
                                    linkContract={isMobile ? 1 : 2}
                                    // No default tooltips/labels (we render cards)
                                    nodeTooltip={() => null}
                                    linkTooltip={() => null}
                                    theme={{
                                        background: "transparent",
                                        text: { fill: PALETTE.text },
                                        tooltip: { container: { display: "none" } },
                                    }}
                                    // Custom layer draws the cards AFTER nodes/links
                                    layers={["links", "nodes", CardsLayer]}
                                />
                            </div>
                        </TransformComponent>
                    </>
                )}
            </TransformWrapper>
        </div>
    );
}
