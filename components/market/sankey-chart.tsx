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
    value?: number;
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

// Premium Dark Infographic palette
const PALETTE = {
    background: "#08090D",
    revenue: "#3B82F6", // solid blue
    profit: "#22C55E",  // solid green
    expense: "#EF4444", // solid red
    tax: "#F59E0B",     // solid amber
    neutral: "#9CA3AF", // slate gray
    panel: "rgba(17, 20, 28, 0.95)", // dark card
    border: "rgba(255, 255, 255, 0.12)",
    text: "#F3F4F6",
    subtext: "#9CA3AF",
    linkOpacity: 0.98,
};

const SEGMENT_PALETTE = [
    "#3B82F6", // blue
    "#F59E0B", // amber
    "#A78BFA", // soft purple
    "#F97316", // orange
    "#06B6D4", // cyan
    "#94A3B8", // slate
];

function hashColor(id: string) {
    let h = 0;
    for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0;
    return SEGMENT_PALETTE[h % SEGMENT_PALETTE.length];
}

function inferKind(id: string): NodeKind {
    const s = id.toLowerCase();
    if (s.includes("revenue") || s.includes("sales") || s.includes("operations")) return "revenue";
    if (s.includes("gross profit") || s.includes("operating income") || s.includes("net income") || s.includes("profit")) return "profit";
    if (s.includes("cost") || s.includes("expense") || s.includes("opex") || s.includes("r&d") || s.includes("sg&a")) return "expense";
    if (s.includes("tax")) return "tax";
    return "neutral";
}

function kindColor(kind: NodeKind, id?: string) {
    switch (kind) {
        case "segment":
            return id ? hashColor(id) : PALETTE.revenue;
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
    const [isMounted, setIsMounted] = useState(false);

    useEffect(() => {
        setIsMounted(true);
        const onResize = () => setIsMobile(window.innerWidth < 640);
        onResize();
        window.addEventListener("resize", onResize);
        return () => window.removeEventListener("resize", onResize);
    }, []);

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

    const enriched = useMemo(() => {
        // 1) Compute enriched nodes (with inferred kind)
        const enrichedNodes = data.nodes.map((n) => {
            let k = n.kind as NodeKind | undefined;
            if (!k) {
                const isSourceOfRevenue = data.links.some(
                    (l) => String(l.source) === n.id && String(l.target) === "Total Revenue"
                );
                k = isSourceOfRevenue ? "segment" : inferKind(n.id);
            }
            return {
                ...n,
                kind: k,
                color: kindColor(k, n.id),
            };
        });

        // 2) Build map from enriched nodes
        const nodeMap = new Map<string, (typeof enrichedNodes)[number]>();
        enrichedNodes.forEach((n) => nodeMap.set(n.id, n));

        // 3) Color links using enriched target kind
        const enrichedLinks = data.links.map((l) => {
            const target = nodeMap.get(String(l.target));
            const k = (target?.kind ?? inferKind(String(l.target))) as NodeKind;
            return { ...l, color: l.color || kindColor(k, target?.id) };
        });

        return { nodes: enrichedNodes, links: enrichedLinks };
    }, [data]);

    const totalRev = useMemo(() => {
        const revNode = enriched.nodes.find((n) => n.id === "Total Revenue");
        return revNode?.value || enriched.links.filter(l => l.target === "Total Revenue").reduce((sum, l) => sum + l.value, 0) || 1;
    }, [enriched]);

    const margin = isMobile
        ? { top: 20, right: 60, bottom: 20, left: 60 }
        : { top: 30, right: 280, bottom: 30, left: 180 };

    const CardsLayer = (layerProps: any) => {
        const { nodes, width, height: innerH } = layerProps;

        const PAD = 12;
        const cardW = isMobile ? 180 : 200;
        const cardH = isMobile ? 40 : 52;

        return (
            <g>
                {nodes.map((node: any) => {
                    const kind: NodeKind = node.kind || "neutral";
                    const color = node.color || PALETTE.neutral;

                    // Mobile Filter: hide minor segments (<12% of rev)
                    const isSegment = kind === "segment";
                    const isSignificant = (node.value / totalRev) >= (isMobile ? 0.12 : 0.08);

                    // Tighten segments on mobile: only show significant ones on the left
                    const isLeftSide = node.x < width * 0.33;
                    const shouldShowCard =
                        !isMobile ||
                        keyNodes.has(String(node.id)) ||
                        (isSegment && isSignificant && isLeftSide);

                    if (!shouldShowCard) return null;

                    const labelMax = isMobile ? 14 : 22;
                    const label = truncate(String(node.id), labelMax);
                    const value = node.displayValue ?? node.value ?? 0;

                    const isLeftMost = node.x < width * 0.4;
                    let x = isLeftMost ? node.x - (cardW + 15) : node.x + node.width + 15;
                    let y = node.y + node.height / 2 - cardH / 2;

                    // Clamping to inner bounds
                    if (x < PAD) x = node.x + node.width + 15;
                    if (x + cardW > width - PAD) x = node.x - (cardW + 15);

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
                            <rect
                                x={node.x - 2}
                                y={node.y - 2}
                                width={node.width + 4}
                                height={node.height + 4}
                                rx={6}
                                ry={6}
                                fill="none"
                                stroke={color}
                                strokeOpacity={0.25}
                                strokeWidth={4}
                            />

                            <g transform={`translate(${x}, ${y})`}>
                                <rect
                                    width={cardW}
                                    height={cardH}
                                    rx={10}
                                    ry={10}
                                    fill={PALETTE.panel}
                                    stroke={PALETTE.border}
                                    strokeWidth={1}
                                />
                                <rect x={12} y={cardH / 2 - 14} width={4} height={28} rx={2} fill={color} />

                                {showLogo ? (
                                    <image
                                        href={`https://raw.githubusercontent.com/davidepalazzo/ticker-logos/main/ticker_icons/${symbol}.png`}
                                        x={24}
                                        y={cardH / 2 - 14}
                                        width={28}
                                        height={28}
                                        opacity={0.95}
                                        preserveAspectRatio="xMidYMid meet"
                                    />
                                ) : (
                                    <text x={26} y={cardH / 2 + 7} fontSize={16} fill={PALETTE.text}>
                                        {icon}
                                    </text>
                                )}

                                <text x={60} y={cardH / 2 - 2} fontSize={13} fill={PALETTE.text} fontWeight={700}>
                                    {label}
                                </text>
                                <text x={60} y={cardH / 2 + 16} fontSize={13} fill={PALETTE.subtext} fontWeight={500}>
                                    {formatMoney(Number(value))}
                                </text>
                            </g>
                        </g>
                    );
                })}
            </g>
        );
    };

    if (!isMounted) return <div style={{ height }} />;

    const mobileCanvasWidth = 980;
    const canvasWidth = isMobile ? mobileCanvasWidth : "100%";

    return (
        <div className="relative w-full rounded-2xl overflow-hidden border border-white/5 shadow-2xl" style={{ height, background: PALETTE.background }}>
            <TransformWrapper
                minScale={0.55}
                maxScale={3}
                initialScale={isMobile ? 1.15 : 1}
                initialPositionX={isMobile ? -180 : 0}
                initialPositionY={isMobile ? -20 : 0}
                centerOnInit={!isMobile}
                limitToBounds={true}
                wheel={{ step: 0.1 }}
                pinch={{ step: 5 }}
                doubleClick={{ disabled: true }}
            >
                {({ zoomIn, zoomOut, resetTransform }) => (
                    <>
                        <div className="absolute right-4 top-4 z-10 flex items-center gap-2 rounded-xl border border-white/10 bg-black/60 p-2 backdrop-blur shadow-lg">
                            <div className="hidden sm:flex items-center gap-2 pr-2 border-r border-white/10">
                                <Move className="h-3.5 w-3.5 text-white/40" />
                                <span className="text-[10px] text-white/40 font-bold tracking-widest uppercase">Pan</span>
                            </div>
                            <button onClick={() => zoomOut()} className="p-1.5 rounded-lg hover:bg-white/10 transition" aria-label="Zoom out">
                                <ZoomOut className="h-4 w-4 text-white/70" />
                            </button>
                            <button onClick={() => zoomIn()} className="p-1.5 rounded-lg hover:bg-white/10 transition" aria-label="Zoom in">
                                <ZoomIn className="h-4 w-4 text-white/70" />
                            </button>
                            <button onClick={() => resetTransform()} className="p-1.5 rounded-lg hover:bg-white/10 transition" aria-label="Reset">
                                <RotateCcw className="h-4 w-4 text-white/70" />
                            </button>
                        </div>

                        <TransformComponent
                            wrapperStyle={{ width: "100%", height }}
                            contentStyle={{ width: canvasWidth, height }}
                        >
                            <div style={{ width: canvasWidth, height }}>
                                <ResponsiveSankey
                                    data={enriched as any}
                                    margin={margin}
                                    align="justify"
                                    sort="auto"
                                    nodeThickness={isMobile ? 16 : 20}
                                    nodeSpacing={isMobile ? 22 : 24}
                                    nodeBorderWidth={0}
                                    nodeOpacity={1}
                                    linkOpacity={PALETTE.linkOpacity}
                                    linkHoverOpacity={1}
                                    linkContract={isMobile ? 1 : 2}
                                    enableLinkGradient={false}
                                    linkBlendMode="normal"
                                    nodeTooltip={() => null}
                                    linkTooltip={() => null}
                                    // @ts-ignore
                                    nodeLabel={() => ""}
                                    theme={{
                                        background: "transparent",
                                        text: { fill: PALETTE.text, fontSize: 12 },
                                        tooltip: { container: { display: "none" } },
                                    }}
                                    // @ts-ignore
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
