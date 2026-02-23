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

function svgDataUri(svg: string) {
    const encoded = encodeURIComponent(svg)
        .replace(/'/g, "%27")
        .replace(/"/g, "%22");
    return `data:image/svg+xml,${encoded}`;
}

const ICONS: Record<string, string> = {
    "Cost of Revenue": svgDataUri(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#F3F4F6"><path d="M6 2h9l3 3v17H6z"/><path fill="#08090D" d="M8 8h8v2H8zm0 4h8v2H8zm0 4h6v2H8z"/></svg>`),
    "R&D": svgDataUri(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#F3F4F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/></svg>`),
    "SG&A": svgDataUri(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#F3F4F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M3 12h18M3 18h18"/></svg>`),
    "Other OpEx": svgDataUri(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#F3F4F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z"/></svg>`),
    "Taxes": svgDataUri(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#F3F4F6"><path d="M7 2h10v4H7z"/><path d="M5 6h14v16H5z"/><path fill="#08090D" d="M8 10h8v2H8zm0 4h8v2H8z"/></svg>`),
    "Operating Expenses": svgDataUri(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#F3F4F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/></svg>`),
    "Gross Profit": svgDataUri(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#22C55E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6"/></svg>`),
    "Operating Income": svgDataUri(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#22C55E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>`),
    "Income Before Tax": svgDataUri(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#22C55E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6"/></svg>`),
    "Net Income": svgDataUri(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#22C55E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><path d="M22 4L12 14.01l-3-3"/></svg>`),
    "Total Revenue": svgDataUri(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6"/></svg>`),
    "Other/Interest Expense": svgDataUri(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#F3F4F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8v4m0 4h.01"/></svg>`),
    "Other/Interest Income": svgDataUri(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#F3F4F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8v4m0 4h.01"/></svg>`),
};

function SankeyTooltip({ title, value, color }: { title: string; value: string; color: string }) {
    return (
        <div
            style={{
                background: "rgba(17, 20, 28, 0.95)",
                border: "1px solid rgba(255,255,255,0.12)",
                padding: "10px 12px",
                borderRadius: 12,
                color: "#F3F4F6",
                fontSize: 12,
            }}
        >
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                <span style={{ width: 10, height: 10, borderRadius: 3, background: color, display: "inline-block" }} />
                <strong style={{ fontSize: 13 }}>{title}</strong>
            </div>
            <div style={{ color: "#9CA3AF" }}>{value}</div>
        </div>
    );
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
            const id = String(n.id);
            let k = n.kind as NodeKind | undefined;
            if (!k) {
                const isSourceOfRevenue = data.links.some(
                    (l) => String(l.source) === id && String(l.target) === "Total Revenue"
                );
                k = isSourceOfRevenue ? "segment" : inferKind(id);
            }
            return {
                ...n,
                kind: k,
                color: kindColor(k, id),
            };
        });

        // 2) Build map from enriched nodes (normalize keys for reliable lookup)
        const nodeMap = new Map<string, (typeof enrichedNodes)[number]>();
        enrichedNodes.forEach((n) => nodeMap.set(String(n.id), n));

        // 3) Color links using enriched target kind
        const enrichedLinks = data.links.map((l) => {
            const target = nodeMap.get(String(l.target));
            const k = (target?.kind ?? inferKind(String(l.target))) as NodeKind;
            return { ...l, color: l.color || kindColor(k, target?.id) };
        });

        return { nodes: enrichedNodes, links: enrichedLinks };
    }, [data]);

    const totalRev = useMemo(() => {
        const revNode: any = enriched.nodes.find((n) => String(n.id) === "Total Revenue");
        const v = Number(revNode?.displayValue ?? revNode?.value ?? 0);
        return Number.isFinite(v) && v > 0 ? v : 1;
    }, [enriched]);

    const topSegments = useMemo(() => {
        const segs = enriched.nodes.filter((n: any) => n.kind === "segment");
        segs.sort((a: any, b: any) => (Number(b.displayValue ?? b.value ?? 0) - Number(a.displayValue ?? a.value ?? 0)));
        return new Set(segs.slice(0, isMobile ? 4 : 8).map((n: any) => String(n.id)));
    }, [enriched, isMobile]);

    const cardW = isMobile ? 190 : 220;
    // On desktop, right margin must fit 2 columns: cardW*2 + 28px gap + 12px outer pad
    const margin = isMobile
        ? { top: 54, right: 18, bottom: 24, left: 18 }
        : { top: 54, right: cardW * 2 + 56, bottom: 36, left: cardW + 90 };

    const totalRevenueValue = useMemo(() => {
        const vFromLinks = enriched.links
            .filter((l: any) => String(l.target) === "Total Revenue")
            .reduce((acc: number, l: any) => acc + Number(l.value || 0), 0);
        if (vFromLinks > 0) return vFromLinks;
        const n: any = enriched.nodes.find((x: any) => String(x.id) === "Total Revenue");
        return Number(n?.displayValue ?? n?.value ?? 0);
    }, [enriched]);

    const CardsLayer = (layerProps: any) => {
        const { nodes, width, height: innerH } = layerProps;

        const PAD = 12;
        const GAP = 8;
        const cardH = isMobile ? 44 : 56;
        const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(v, max));

        type CardCandidate = {
            id: string;
            node: any;
            side: "L" | "R";
            x: number;
            desiredY: number;
            y: number;
        };

        const minX = nodes.reduce((m: number, n: any) => Math.min(m, Number(n.x ?? Infinity)), Infinity);

        // Two right rails: outer (rail 1) and inner (rail 2)
        const LEFT_RAIL_X = 12;
        const RIGHT_RAIL_1 = width - cardW - 12;        // outer right column
        const RIGHT_RAIL_2 = width - (cardW * 2) - 28;  // inner right column

        const candidates: CardCandidate[] = [];
        let rightIndex = 0;

        for (const node of nodes) {
            const id = String(node.id);
            if (id === "Total Revenue") continue;

            const kind: NodeKind = node.kind || "neutral";
            const nx = Number(node.x);
            const ny = Number(node.y);
            const nw = Number(node.width);
            const nh = Number(node.height);
            if (!Number.isFinite(nx) || !Number.isFinite(ny) || !Number.isFinite(nw) || !Number.isFinite(nh)) continue;

            const nodeVal = Number(node.displayValue ?? node.value ?? 0);
            const isSegment = kind === "segment";
            const isSignificant = (nodeVal / totalRev) >= (isMobile ? 0.12 : 0.08);

            const shouldShowCard =
                !isMobile ||
                keyNodes.has(id) ||
                (isSegment && isSignificant && (isMobile ? topSegments.has(id) : true));

            if (!shouldShowCard) continue;

            const rawDepth =
                (Number.isFinite(Number(node.depth)) ? Number(node.depth) : undefined) ??
                (Number.isFinite(Number(node.layer)) ? Number(node.layer) : undefined);

            const isFirstColumn =
                rawDepth === 0 ||
                Math.abs(nx - minX) < 2;

            const side: "L" | "R" = isSegment || isFirstColumn ? "L" : "R";

            const centerY = ny + nh / 2;
            const desiredY = clamp(centerY - cardH / 2, PAD, innerH - cardH - PAD);

            // On desktop, alternate right-side cards between two columns to reduce crowding
            let x: number;
            if (side === "L") {
                x = LEFT_RAIL_X;
            } else if (isMobile) {
                x = RIGHT_RAIL_1;
            } else {
                x = (rightIndex % 2 === 0) ? RIGHT_RAIL_1 : RIGHT_RAIL_2;
                rightIndex++;
            }

            candidates.push({ id, node, side, x, desiredY, y: desiredY });
        }

        // Resolve stacking per rail (unique x position) to avoid overlaps within each column
        const resolveRail = (railX: number) => {
            const arr = candidates
                .filter((c) => c.x === railX)
                .sort((a, b) => a.desiredY - b.desiredY);

            let cursor = PAD;
            for (const c of arr) {
                c.y = Math.max(c.desiredY, cursor);
                c.y = Math.min(c.y, innerH - cardH - PAD);
                cursor = c.y + cardH + GAP;
            }

            if (arr.length) {
                const last = arr[arr.length - 1];
                const overflow = last.y + cardH + PAD - innerH;
                if (overflow > 0) {
                    for (let i = arr.length - 1; i >= 0; i--) {
                        arr[i].y = clamp(arr[i].y - overflow, PAD, innerH - cardH - PAD);
                        if (i > 0) {
                            arr[i - 1].y = Math.min(arr[i - 1].y, arr[i].y - cardH - GAP);
                            arr[i - 1].y = clamp(arr[i - 1].y, PAD, innerH - cardH - PAD);
                        }
                    }
                }
            }
        };

        resolveRail(LEFT_RAIL_X);
        resolveRail(RIGHT_RAIL_1);
        if (!isMobile) resolveRail(RIGHT_RAIL_2);

        return (
            <g>
                {candidates.map(({ id, node, side, x, y }) => {
                    const color = node.color || PALETTE.neutral;

                    const nx = Number(node.x);
                    const ny = Number(node.y);
                    const nw = Number(node.width);
                    const nh = Number(node.height);

                    const anchorX = side === "L" ? x + cardW : x;
                    const anchorY = y + cardH / 2;
                    const nodeCX = nx + nw / 2;
                    const nodeCY = ny + nh / 2;

                    const labelMax = isMobile ? 14 : 22;
                    const label = truncate(id, labelMax);
                    const value = node.displayValue ?? node.value ?? 0;
                    const iconHref = ICONS[id];

                    return (
                        <g key={id}>
                            <rect
                                x={nx - 2}
                                y={ny - 2}
                                width={nw + 4}
                                height={nh + 4}
                                rx={6}
                                ry={6}
                                fill="none"
                                stroke={color}
                                strokeOpacity={0.25}
                                strokeWidth={4}
                            />

                            <path
                                d={`M ${nodeCX} ${nodeCY} C ${nodeCX + (side === "L" ? -40 : 40)} ${nodeCY}, ${anchorX + (side === "L" ? 40 : -40)} ${anchorY}, ${anchorX} ${anchorY}`}
                                fill="none"
                                stroke={color}
                                strokeOpacity={0.35}
                                strokeWidth={1.25}
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

                                {iconHref ? (
                                    <image
                                        href={iconHref}
                                        x={24}
                                        y={cardH / 2 - 14}
                                        width={28}
                                        height={28}
                                        opacity={0.95}
                                        preserveAspectRatio="xMidYMid meet"
                                    />
                                ) : (
                                    <circle cx={38} cy={cardH / 2} r={6} fill={color} opacity={0.9} />
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

    const canvasWidth = isMobile ? 1100 : 1400;
    const linkColorFn = (link: any) => link.color;

    return (
        <div className="sankeyWrap relative w-full overflow-hidden rounded-2xl border border-white/5 shadow-2xl" style={{ height }}>
            <div className="absolute inset-0 rounded-2xl" style={{ background: PALETTE.background }} />

            {symbol && (
                <div className="absolute left-4 top-4 z-20">
                    <div
                        style={{
                            width: isMobile ? 240 : 280,
                            borderRadius: 14,
                            background: PALETTE.panel,
                            border: `1px solid ${PALETTE.border}`,
                            padding: "10px 12px",
                            display: "flex",
                            gap: 10,
                            alignItems: "center",
                            boxShadow: "0 10px 30px rgba(0,0,0,0.55)",
                        }}
                    >
                        <img
                            src={`https://raw.githubusercontent.com/davidepalazzo/ticker-logos/main/ticker_icons/${symbol}.png`}
                            alt={symbol}
                            style={{ width: 32, height: 32, borderRadius: 8, objectFit: "contain" }}
                            onError={(e) => {
                                const target = e.target as HTMLImageElement;
                                target.style.display = "none";
                            }}
                        />
                        <div style={{ lineHeight: 1.1 }}>
                            <div style={{ color: PALETTE.text, fontWeight: 800, fontSize: 13 }}>Total Revenue</div>
                            <div style={{ color: PALETTE.subtext, fontWeight: 600, fontSize: 13 }}>
                                {formatMoney(totalRevenueValue)}
                            </div>
                        </div>
                    </div>
                </div>
            )}

            <div className="relative w-full h-full rounded-2xl">
            <TransformWrapper
                    minScale={0.55}
                    maxScale={3}
                    initialScale={isMobile ? 1.05 : 1}
                    {...(isMobile ? { initialPositionX: -140 } : {})}
                    initialPositionY={0}
                    centerOnInit={!isMobile}
                    limitToBounds={true}
                    panning={{ excluded: [] }}
                    alignmentAnimation={{ disabled: true }}
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
                            wrapperStyle={{
                                width: "100%",
                                height,
                                overflow: "hidden",
                                touchAction: "none",
                                userSelect: "none",
                            }}
                            contentStyle={{
                                width: canvasWidth,
                                height,
                                overflow: "hidden",
                            }}
                        >
                            <div style={{ width: canvasWidth, height, overflow: "hidden" }}>
                                <ResponsiveSankey
                                    {...({
                                        data: enriched,
                                        margin,
                                        align: "justify" as const,
                                        sort: "auto" as const,
                                        colors: (node: any) => node.color,
                                        linkColor: linkColorFn,
                                        nodeThickness: isMobile ? 16 : 24,
                                        nodeSpacing: isMobile ? 22 : 24,
                                        nodeBorderWidth: 0,
                                        nodeOpacity: 1,
                                        linkOpacity: isMobile ? 0.9 : 0.95,
                                        linkHoverOpacity: 1,
                                        linkContract: isMobile ? 1 : 3,
                                        enableLinkGradient: false,
                                        linkBlendMode: "normal" as const,
                                        nodeTooltip: ({ node }: any) => (
                                            <SankeyTooltip
                                                title={String(node?.id ?? node?.label ?? "Unknown")}
                                                value={formatMoney(Number(node.value ?? node.displayValue ?? 0))}
                                                color={node.color || PALETTE.neutral}
                                            />
                                        ),
                                        linkTooltip: ({ link }: any) => (
                                            <SankeyTooltip
                                                title={`${link?.source?.id ?? link?.source ?? "?"} → ${link?.target?.id ?? link?.target ?? "?"}`}
                                                value={formatMoney(Number(link.value ?? 0))}
                                                color={link.color || PALETTE.neutral}
                                            />
                                        ),
                                        nodeLabel: () => "",
                                        theme: {
                                            background: "transparent",
                                            text: { fill: PALETTE.text, fontSize: 12 },
                                            tooltip: { container: { zIndex: 9999 } },
                                        },
                                        layers: ["links", "nodes", CardsLayer],
                                    } as any)}
                                />
                            </div>
                        </TransformComponent>
                    </>
                )}
            </TransformWrapper>
            </div>
        </div>
    );
}
