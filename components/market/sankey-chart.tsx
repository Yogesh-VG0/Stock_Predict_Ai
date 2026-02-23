"use client";

import React, { useMemo, useEffect, useState } from "react";
import { ResponsiveSankey, SankeyNodeDatum, SankeyLinkDatum } from "@nivo/sankey";
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
    panel: "rgba(10, 12, 20, 0.75)",
    border: "rgba(255,255,255,0.10)",
};

function kindColor(kind: NodeKind) {
    switch (kind) {
        case "segment":
            return NEON.gray;
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

// Truncate but keep it readable
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
                color: kindColor((n.kind || "neutral") as NodeKind),
            })),
            links: data.links.map((l) => {
                const target = typeof l.target === "string" ? nodeMap.get(l.target) : nodeMap.get((l.target as SankeyNode).id);
                const inferred = kindColor(((target?.kind || "neutral") as NodeKind));
                return { ...l, color: l.color || inferred };
            }),
        };
    }, [data]);

    // More left padding on desktop; on mobile we rely on zoom + truncation
    const margin = isMobile
        ? { top: 28, right: 18, bottom: 28, left: 18 }
        : { top: 28, right: 42, bottom: 28, left: 110 };

    // Render "node cards" like the infographic (label + value + icon)
    const NodeCard = (nodeProps: any) => {
        const node = nodeProps.node as SankeyNodeDatum<any, any> & { kind?: NodeKind; displayValue?: number; color?: string };
        const x = node.x;
        const y = node.y;
        const w = node.width;
        const h = node.height;

        const kind = (node as any).kind || "neutral";
        const color = (node as any).color || NEON.gray;

        const labelMax = isMobile ? 12 : 18;
        const label = truncate(String(node.id), labelMax);

        // Make sure tiny nodes still show a card
        const cardW = Math.max(w + 140, isMobile ? 170 : 190);
        const cardH = Math.max(h + 14, isMobile ? 40 : 46);

        // Place label card slightly outside node
        const isLeft = x < 120;
        const cardX = isLeft ? x - (cardW + 10) : x + w + 10;
        const cardY = y + h / 2 - cardH / 2;

        const showLogo = String(node.id).toLowerCase() === "total revenue" && symbol;

        const icon =
            kind === "profit" ? "✅" :
                kind === "expense" ? "🧾" :
                    kind === "tax" ? "🧮" :
                        kind === "revenue" ? "💰" :
                            kind === "segment" ? "📱" :
                                "•";

        const value = (node as any).displayValue ?? node.value ?? 0;

        return (
            <g>
                {/* The actual node bar */}
                <rect
                    x={x}
                    y={y}
                    width={w}
                    height={h}
                    rx={3}
                    ry={3}
                    fill={color}
                    opacity={0.98}
                />

                {/* Glow (subtle) */}
                <rect
                    x={x - 2}
                    y={y - 2}
                    width={w + 4}
                    height={h + 4}
                    rx={6}
                    ry={6}
                    fill="none"
                    stroke={color}
                    strokeOpacity={0.25}
                    strokeWidth={4}
                />

                {/* Label card */}
                <g transform={`translate(${cardX}, ${cardY})`}>
                    <rect
                        width={cardW}
                        height={cardH}
                        rx={10}
                        ry={10}
                        fill={NEON.panel}
                        stroke={NEON.border}
                        strokeWidth={1}
                    />

                    {/* left accent */}
                    <rect x={10} y={cardH / 2 - 12} width={4} height={24} rx={2} fill={color} opacity={0.95} />

                    {/* icon / logo */}
                    {showLogo ? (
                        <image
                            href={`https://raw.githubusercontent.com/davidepalazzo/ticker-logos/main/ticker_icons/${symbol}.png`}
                            x={22}
                            y={cardH / 2 - 13}
                            width={26}
                            height={26}
                            opacity={0.95}
                            preserveAspectRatio="xMidYMid meet"
                            onError={(e) => {
                                // Ignore if image fails to load
                                (e.target as any).href.baseVal = "";
                            }}
                        />
                    ) : (
                        <text x={24} y={cardH / 2 + 5} fontSize={14} fill="#E6EAF2">
                            {icon}
                        </text>
                    )}

                    {/* label */}
                    <text x={50} y={cardH / 2 - 2} fontSize={12} fill="#E6EAF2" fontWeight={700}>
                        {label}
                    </text>

                    {/* value */}
                    <text x={50} y={cardH / 2 + 14} fontSize={12} fill="#A7B0C0" fontWeight={600}>
                        {formatMoney(Number(value))}
                    </text>
                </g>
            </g>
        );
    };

    // Make links thicker & "solid neon"
    const linkOpacity = isMobile ? 0.45 : 0.35;

    return (
        <div className="relative w-full h-full overflow-hidden">
            {/* Zoom controls */}
            <div className="absolute right-3 top-3 z-10 flex items-center gap-2 rounded-xl border border-zinc-800 bg-black/60 p-2 backdrop-blur hover:bg-black/80 transition-colors cursor-default">
                <div className="hidden sm:flex items-center gap-2 pr-2 border-r border-zinc-800">
                    <Move className="h-4 w-4 text-zinc-300" />
                    <span className="text-xs text-zinc-400 font-medium tracking-wide uppercase">Drag</span>
                </div>
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
                            <button
                                onClick={() => zoomOut()}
                                className="p-1.5 rounded-lg hover:bg-white/10 transition"
                                aria-label="Zoom out"
                            >
                                <ZoomOut className="h-4 w-4 text-zinc-200" />
                            </button>
                            <button
                                onClick={() => zoomIn()}
                                className="p-1.5 rounded-lg hover:bg-white/10 transition"
                                aria-label="Zoom in"
                            >
                                <ZoomIn className="h-4 w-4 text-zinc-200" />
                            </button>
                            <button
                                onClick={() => resetTransform()}
                                className="p-1.5 rounded-lg hover:bg-white/10 transition"
                                aria-label="Reset"
                            >
                                <RotateCcw className="h-4 w-4 text-zinc-200" />
                            </button>

                            <div className="w-full h-full absolute top-0 left-0 -z-10 bg-transparent flex items-center justify-center">
                                <TransformComponent wrapperStyle={{ width: "100%", height }} contentStyle={{ width: "100%", height }}>
                                    <div style={{ width: "100%", height }}>
                                        {/* @ts-ignore - Nivo generic component mappings often fail strict TS bounds */}
                                        <ResponsiveSankey
                                            data={enriched as any}
                                            margin={margin}
                                            align="justify"
                                            sort="auto"
                                            nodeThickness={isMobile ? 12 : 18}
                                            nodeSpacing={isMobile ? 12 : 24}
                                            nodeBorderWidth={0}
                                            linkOpacity={linkOpacity}
                                            linkHoverOpacity={0.85}
                                            linkContract={isMobile ? 1 : 2}
                                            enableLinkGradient={true}
                                            linkBlendMode="screen"
                                            // We draw labels ourselves via nodeComponent, so hide default labels
                                            labelPosition="outside"
                                            labelOrientation="horizontal"
                                            labelPadding={0}
                                            labelTextColor="transparent"
                                            label={() => ""}
                                            nodeTooltip={() => null}
                                            linkTooltip={() => null}
                                            theme={{
                                                background: NEON.bg,
                                                text: { fill: "#E6EAF2" },
                                                tooltip: { container: { display: "none" } },
                                            }}
                                            // @ts-ignore - nodeComponent is valid in Nivo but missing in some typedefs
                                            nodeComponent={NodeCard}
                                        />
                                    </div>
                                </TransformComponent>
                            </div>
                        </>
                    )}
                </TransformWrapper>
            </div>

            {/* Background panel */}
            <div
                className="w-full rounded-2xl border border-zinc-800/60 absolute top-0 left-0 pointer-events-none -z-20"
                style={{
                    height,
                    background:
                        "radial-gradient(1200px 400px at 20% 10%, rgba(46,160,255,0.06), transparent 55%)," +
                        "radial-gradient(900px 380px at 70% 20%, rgba(45,255,154,0.05), transparent 55%)," +
                        "radial-gradient(900px 380px at 70% 80%, rgba(255,77,109,0.04), transparent 55%)",
                }}
            />
        </div>
    );
}
