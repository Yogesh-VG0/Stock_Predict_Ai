"use client";

import React, { useEffect, useMemo, useState, useCallback } from "react";
import { createPortal } from "react-dom";
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

type ComputedNode = {
    id: string;
    x: number;
    y: number;
    width: number;
    height: number;
    color: string;
    value: number;
    displayValue?: number;
    kind?: NodeKind;
    depth?: number;
};

function formatMoney(v: number) {
    const abs = Math.abs(v || 0);
    if (abs >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
    if (abs >= 1e9) return `$${(v / 1e9).toFixed(2)}B`;
    if (abs >= 1e6) return `$${(v / 1e6).toFixed(2)}M`;
    if (abs >= 1e3) return `$${(v / 1e3).toFixed(2)}K`;
    return `$${(v || 0).toFixed(0)}`;
}

const PALETTE = {
    background: "#08090D",
    revenue: "#3B82F6",
    profit: "#22C55E",
    expense: "#EF4444",
    tax: "#F59E0B",
    neutral: "#9CA3AF",
    panel: "rgba(17, 20, 28, 0.95)",
    border: "rgba(255, 255, 255, 0.12)",
    text: "#F3F4F6",
    subtext: "#9CA3AF",
};

const SEGMENT_PALETTE = ["#3B82F6", "#F59E0B", "#A78BFA", "#F97316", "#06B6D4", "#94A3B8"];

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
        case "segment": return id ? hashColor(id) : PALETTE.revenue;
        case "revenue": return PALETTE.revenue;
        case "profit": return PALETTE.profit;
        case "expense": return PALETTE.expense;
        case "tax": return PALETTE.tax;
        default: return PALETTE.neutral;
    }
}

function truncate(s: string, max: number) {
    if (!s) return "";
    if (s.length <= max) return s;
    return s.slice(0, max - 1) + "…";
}

function svgDataUri(svg: string) {
    const encoded = encodeURIComponent(svg).replace(/'/g, "%27").replace(/"/g, "%22");
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

// Tooltip component - will be portal'd to document.body
function SankeyTooltip({ 
    title, 
    value, 
    color,
    position 
}: { 
    title: string; 
    value: string; 
    color: string;
    position: { x: number; y: number };
}) {
    return (
        <div style={{
            position: "fixed",
            left: position.x + 15,
            top: position.y + 15,
            background: "rgba(17, 20, 28, 0.97)",
            border: "1px solid rgba(255,255,255,0.12)",
            padding: "10px 14px",
            borderRadius: 10,
            color: "#F3F4F6",
            fontSize: 13,
            maxWidth: 360,
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
            zIndex: 999999,
            boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
            pointerEvents: "none",
        }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                <span style={{ width: 10, height: 10, borderRadius: 3, background: color, display: "inline-block", flexShrink: 0 }} />
                <strong style={{ fontSize: 13 }}>{title}</strong>
            </div>
            <div style={{ color: "#9CA3AF", fontSize: 13 }}>{value}</div>
        </div>
    );
}

// Node card for HTML overlay
function OverlayCard({
    node,
    cardX,
    cardY,
    cardW,
    cardH,
    onMouseEnter,
    onMouseLeave,
}: {
    node: ComputedNode;
    cardX: number;
    cardY: number;
    cardW: number;
    cardH: number;
    onMouseEnter: (e: React.MouseEvent, node: ComputedNode) => void;
    onMouseLeave: () => void;
}) {
    const label = truncate(String(node.id), 18);
    const value = node.displayValue ?? node.value ?? 0;
    const iconHref = ICONS[String(node.id)];

    return (
        <div
            style={{
                position: "absolute",
                left: cardX,
                top: cardY,
                width: cardW,
                height: cardH,
                background: PALETTE.panel,
                border: `1px solid ${PALETTE.border}`,
                borderRadius: 7,
                display: "flex",
                alignItems: "center",
                padding: "0 10px",
                gap: 8,
                cursor: "pointer",
                pointerEvents: "auto",
                boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
                transition: "transform 0.15s ease, box-shadow 0.15s ease",
                zIndex: 50,
            }}
            onMouseEnter={(e) => onMouseEnter(e, node)}
            onMouseLeave={onMouseLeave}
            onMouseDown={(e) => e.stopPropagation()}
        >
            {/* Color bar */}
            <div style={{
                width: 3,
                height: 20,
                borderRadius: 1.5,
                background: node.color,
                flexShrink: 0,
            }} />
            
            {/* Icon or circle */}
            {iconHref ? (
                <img
                    src={iconHref}
                    alt=""
                    style={{ width: 20, height: 20, objectFit: "contain", opacity: 0.85, flexShrink: 0 }}
                />
            ) : (
                <div style={{
                    width: 8,
                    height: 8,
                    borderRadius: "50%",
                    background: node.color,
                    opacity: 0.8,
                    flexShrink: 0,
                }} />
            )}
            
            {/* Text content */}
            <div style={{ minWidth: 0, flex: 1, overflow: "hidden" }}>
                <div style={{
                    color: PALETTE.text,
                    fontSize: 11.5,
                    fontWeight: 700,
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    lineHeight: 1.2,
                }}>
                    {label}
                </div>
                <div style={{
                    color: PALETTE.subtext,
                    fontSize: 10.5,
                    fontWeight: 500,
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    lineHeight: 1.2,
                    marginTop: 2,
                }}>
                    {formatMoney(Number(value))}
                </div>
            </div>
        </div>
    );
}

// SVG Connector line component
function ConnectorLine({
    nodeCX,
    nodeCY,
    anchorX,
    anchorY,
    color,
}: {
    nodeCX: number;
    nodeCY: number;
    anchorX: number;
    anchorY: number;
    color: string;
}) {
    const dx = anchorX - nodeCX;
    const ctrl = Math.max(Math.abs(dx) * 0.3, 15);

    return (
        <path
            d={`M ${nodeCX} ${nodeCY} C ${nodeCX + ctrl} ${nodeCY}, ${anchorX - ctrl} ${anchorY}, ${anchorX} ${anchorY}`}
            fill="none"
            stroke={color}
            strokeOpacity={0.25}
            strokeWidth={1}
        />
    );
}

// Custom layer to capture node positions
function NodeCaptureLayer({
    nodes,
    onNodesCaptured,
}: {
    nodes: any[];
    onNodesCaptured: (nodes: ComputedNode[]) => void;
}) {
    useEffect(() => {
        if (nodes && nodes.length > 0) {
            const computedNodes: ComputedNode[] = nodes.map((n: any) => ({
                id: String(n.id),
                x: Number(n.x) || 0,
                y: Number(n.y) || 0,
                width: Number(n.width) || 0,
                height: Number(n.height) || 0,
                color: n.color || PALETTE.neutral,
                value: Number(n.value) || 0,
                displayValue: n.displayValue,
                kind: n.kind,
            }));
            onNodesCaptured(computedNodes);
        }
    }, [nodes, onNodesCaptured]);

    // This layer doesn't render anything - it just captures node positions
    return null;
}

export default function SankeyChart({
    data,
    height: propHeight = 680,
    symbol,
}: {
    data: { nodes: SankeyNode[]; links: SankeyLink[] };
    height?: number;
    symbol?: string;
}) {
    const [isMobile, setIsMobile] = useState(false);
    const [isMounted, setIsMounted] = useState(false);
    const [computedNodes, setComputedNodes] = useState<ComputedNode[]>([]);
    const [tooltip, setTooltip] = useState<{
        visible: boolean;
        title: string;
        value: string;
        color: string;
        position: { x: number; y: number };
    } | null>(null);

    useEffect(() => {
        setIsMounted(true);
        const onResize = () => setIsMobile(window.innerWidth < 640);
        onResize();
        window.addEventListener("resize", onResize);
        return () => window.removeEventListener("resize", onResize);
    }, []);

    const handleNodesCaptured = useCallback((nodes: ComputedNode[]) => {
        setComputedNodes(nodes);
    }, []);

    const handleMouseEnter = useCallback((e: React.MouseEvent, node: ComputedNode) => {
        setTooltip({
            visible: true,
            title: String(node.id),
            value: formatMoney(Number(node.displayValue ?? node.value ?? 0)),
            color: node.color || PALETTE.neutral,
            position: { x: e.clientX, y: e.clientY },
        });
    }, []);

    const handleMouseLeave = useCallback(() => {
        setTooltip(null);
    }, []);

    const keyNodes = useMemo(
        () => new Set([
            "Total Revenue", "Gross Profit", "Operating Expenses", "Operating Income",
            "Income Before Tax", "Net Income", "Cost of Revenue", "Taxes",
        ]),
        []
    );

    const enriched = useMemo(() => {
        const enrichedNodes = data.nodes.map((n) => {
            const id = String(n.id);
            let k = n.kind as NodeKind | undefined;
            if (!k) {
                const isSourceOfRevenue = data.links.some(
                    (l) => String(l.source) === id && String(l.target) === "Total Revenue"
                );
                k = isSourceOfRevenue ? "segment" : inferKind(id);
            }
            return { ...n, kind: k, color: kindColor(k, id) };
        });
        const nodeMap = new Map<string, (typeof enrichedNodes)[number]>();
        enrichedNodes.forEach((n) => nodeMap.set(String(n.id), n));
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
        return new Set(segs.slice(0, isMobile ? 3 : 8).map((n: any) => String(n.id)));
    }, [enriched, isMobile]);

    // Card dimensions
    const cardW = isMobile ? 140 : 190;
    const cardH = isMobile ? 40 : 48;
    const GAP = isMobile ? 4 : 5;
    const PAD = 8;

    // Count right-side nodes for height calculation
    const rightNodeCount = useMemo(() => {
        return enriched.nodes.filter((n: any) => {
            const id = String(n.id);
            if (id === "Total Revenue") return false;
            const k = n.kind as NodeKind;
            return k !== "segment";
        }).length;
    }, [enriched]);

    // Height: ensure enough space for right-side cards in a single column
    const height = useMemo(() => {
        const minForCards = rightNodeCount * (cardH + GAP) + 140;
        return Math.max(propHeight, isMobile ? propHeight : minForCards);
    }, [propHeight, rightNodeCount, cardH, GAP, isMobile]);

    // Margins: on desktop, reserve space for left + right card columns
    const margin = useMemo(() => {
        if (isMobile) return { top: 10, right: 10, bottom: 10, left: 10 };
        return { top: 40, right: cardW + 40, bottom: 30, left: cardW + 40 };
    }, [isMobile, cardW]);

    const totalRevenueValue = useMemo(() => {
        const vFromLinks = enriched.links
            .filter((l: any) => String(l.target) === "Total Revenue")
            .reduce((acc: number, l: any) => acc + Number(l.value || 0), 0);
        if (vFromLinks > 0) return vFromLinks;
        const n: any = enriched.nodes.find((x: any) => String(x.id) === "Total Revenue");
        return Number(n?.displayValue ?? n?.value ?? 0);
    }, [enriched]);

    const canvasWidth = isMobile ? 900 : 1400;

    // Calculate card positions based on computed nodes
    const cardPositions = useMemo(() => {
        if (!computedNodes.length) return [];

        const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(v, hi));

        type RailKey = "L" | "R";
        type CardCandidate = {
            id: string;
            node: ComputedNode;
            side: "L" | "R";
            rail: RailKey;
            x: number;
            desiredY: number;
            y: number;
        };

        const minNodeX = computedNodes.reduce((m, n) => Math.min(m, Number(n.x ?? Infinity)), Infinity);

        // SVG visible bounds from layer coordinate perspective
        const svgLeft = -margin.left;
        const svgRight = canvasWidth - margin.left;

        // Card placement bounds
        const safeLeft = svgLeft + PAD;
        const safeRight = svgRight - cardW - PAD;
        const safeTop = -margin.top + PAD;
        const safeBottom = height + margin.bottom - cardH - PAD;

        // Rail X positions
        const LEFT_X = isMobile ? PAD : safeLeft;
        const RIGHT_X = isMobile
            ? (canvasWidth - margin.left - cardW + PAD)
            : (canvasWidth - margin.left + 20);

        const candidates: CardCandidate[] = [];

        for (const node of computedNodes) {
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
            const isSignificant = (nodeVal / totalRev) >= (isMobile ? 0.15 : 0.08);

            const shouldShowCard = isMobile
                ? (keyNodes.has(id) || (isSegment && isSignificant && topSegments.has(id)))
                : true;

            if (!shouldShowCard) continue;

            const rawDepth = Number(node.depth) || 0;
            const isFirstColumn = rawDepth === 0 || Math.abs(nx - minNodeX) < 2;
            const side: "L" | "R" = isSegment || isFirstColumn ? "L" : "R";

            const centerY = ny + nh / 2;
            const desiredY = clamp(centerY - cardH / 2, safeTop, safeBottom);

            const x = clamp(side === "L" ? LEFT_X : RIGHT_X, safeLeft, safeRight);
            const rail: RailKey = side === "L" ? "L" : "R";

            candidates.push({ id, node, side, rail, x, desiredY, y: desiredY });
        }

        // De-dupe
        const seen = new Set<string>();
        const unique = candidates.filter((c) => {
            if (seen.has(c.id)) return false;
            seen.add(c.id);
            return true;
        });

        // Stack per rail
        const resolveRail = (rail: RailKey) => {
            const arr = unique.filter((c) => c.rail === rail).sort((a, b) => a.desiredY - b.desiredY);
            if (!arr.length) return;

            let cursor = safeTop;
            for (const c of arr) {
                c.y = Math.max(c.desiredY, cursor);
                c.y = Math.min(c.y, safeBottom);
                cursor = c.y + cardH + GAP;
            }

            const last = arr[arr.length - 1];
            if (last.y > safeBottom) {
                let bc = safeBottom;
                for (let i = arr.length - 1; i >= 0; i--) {
                    arr[i].y = Math.min(arr[i].y, bc);
                    arr[i].y = Math.max(arr[i].y, safeTop);
                    bc = arr[i].y - cardH - GAP;
                }
            }

            if (arr.length > 0) {
                const stackTop = arr[0].y;
                const stackBottom = arr[arr.length - 1].y + cardH;
                const used = stackBottom - stackTop;
                const available = safeBottom - safeTop + cardH;
                const slack = available - used;
                if (slack > 20) {
                    const shift = slack * 0.4;
                    for (const c of arr) {
                        c.y = clamp(c.y + shift, safeTop, safeBottom);
                    }
                }
            }
        };

        resolveRail("L");
        resolveRail("R");

        return unique;
    }, [computedNodes, margin, cardW, cardH, GAP, PAD, height, canvasWidth, isMobile, totalRev, keyNodes, topSegments]);

    // Build connector paths for SVG layer
    const connectorPaths = useMemo(() => {
        return cardPositions.map(({ id, node, side, x, y }) => {
            const nx = Number(node.x);
            const ny = Number(node.y);
            const nw = Number(node.width);
            const nh = Number(node.height);

            const nodeCX = nx + nw / 2;
            const nodeCY = ny + nh / 2;
            const anchorX = side === "L" ? x + cardW : x;
            const anchorY = y + cardH / 2;

            const dx = anchorX - nodeCX;
            const ctrl = Math.max(Math.abs(dx) * 0.3, 15) * (dx < 0 ? -1 : 1);

            return {
                id,
                path: `M ${nodeCX} ${nodeCY} C ${nodeCX + ctrl} ${nodeCY}, ${anchorX - ctrl} ${anchorY}, ${anchorX} ${anchorY}`,
                color: node.color,
                nodeX: nx,
                nodeY: ny,
                nodeWidth: nw,
                nodeHeight: nh,
            };
        });
    }, [cardPositions, cardW, cardH]);

    if (!isMounted) return <div style={{ height }} />;

    const linkColorFn = (link: any) => link.color;

    return (
        <div className="sankeyWrap relative w-full overflow-hidden rounded-2xl border border-white/5 shadow-2xl" style={{ height }}>
            <div className="absolute inset-0 rounded-2xl" style={{ background: PALETTE.background }} />

            {/* Revenue badge */}
            {symbol && (
                <div className="absolute left-3 top-3 z-20">
                    <div style={{
                        width: isMobile ? 180 : 220,
                        borderRadius: 10,
                        background: PALETTE.panel,
                        border: `1px solid ${PALETTE.border}`,
                        padding: "7px 10px",
                        display: "flex",
                        gap: 8,
                        alignItems: "center",
                        boxShadow: "0 6px 20px rgba(0,0,0,0.5)",
                    }}>
                        <img
                            src={`https://raw.githubusercontent.com/davidepalazzo/ticker-logos/main/ticker_icons/${symbol}.png`}
                            alt={symbol}
                            style={{ width: 26, height: 26, borderRadius: 5, objectFit: "contain" }}
                            onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                        />
                        <div style={{ lineHeight: 1.1 }}>
                            <div style={{ color: PALETTE.text, fontWeight: 800, fontSize: 11 }}>Total Revenue</div>
                            <div style={{ color: PALETTE.subtext, fontWeight: 600, fontSize: 11 }}>
                                {formatMoney(totalRevenueValue)}
                            </div>
                        </div>
                    </div>
                </div>
            )}

            <div className="relative w-full h-full rounded-2xl">
                <TransformWrapper
                    minScale={0.4}
                    maxScale={3}
                    initialScale={isMobile ? 0.85 : 0.82}
                    initialPositionX={isMobile ? -50 : 0}
                    initialPositionY={0}
                    centerOnInit={true}
                    limitToBounds={false}
                    panning={{ excluded: [] }}
                    alignmentAnimation={{ disabled: true }}
                    wheel={{ step: 0.08 }}
                    pinch={{ step: 5 }}
                    doubleClick={{ disabled: true }}
                >
                    {({ zoomIn, zoomOut, resetTransform }) => (
                        <>
                            {/* Zoom controls */}
                            <div className="absolute right-3 top-3 z-10 flex items-center gap-1 rounded-lg border border-white/10 bg-black/70 p-1 backdrop-blur shadow-lg">
                                <div className="hidden sm:flex items-center gap-1 pr-1.5 border-r border-white/10">
                                    <Move className="h-3 w-3 text-white/40" />
                                    <span className="text-[8px] text-white/40 font-bold tracking-widest uppercase">Pan</span>
                                </div>
                                <button onClick={() => zoomOut()} className="p-1 rounded hover:bg-white/10 transition" aria-label="Zoom out">
                                    <ZoomOut className="h-3.5 w-3.5 text-white/70" />
                                </button>
                                <button onClick={() => zoomIn()} className="p-1 rounded hover:bg-white/10 transition" aria-label="Zoom in">
                                    <ZoomIn className="h-3.5 w-3.5 text-white/70" />
                                </button>
                                <button onClick={() => resetTransform()} className="p-1 rounded hover:bg-white/10 transition" aria-label="Reset">
                                    <RotateCcw className="h-3.5 w-3.5 text-white/70" />
                                </button>
                            </div>

                            <TransformComponent
                                wrapperStyle={{ width: "100%", height, overflow: "hidden", touchAction: "none", userSelect: "none" }}
                                contentStyle={{ width: canvasWidth, height, overflow: "hidden" }}
                            >
                                <div style={{ width: canvasWidth, height, overflow: "hidden", position: "relative" }}>
                                    {/* SVG Sankey - only links and nodes */}
                                    <ResponsiveSankey
                                        {...({
                                            data: enriched,
                                            margin,
                                            align: "justify" as const,
                                            sort: "auto" as const,
                                            colors: (node: any) => node.color,
                                            linkColor: linkColorFn,
                                            nodeThickness: isMobile ? 14 : 20,
                                            nodeSpacing: isMobile ? 18 : 20,
                                            nodeBorderWidth: 0,
                                            nodeOpacity: 1,
                                            linkOpacity: isMobile ? 0.85 : 0.92,
                                            linkHoverOpacity: 1,
                                            linkContract: isMobile ? 1 : 2,
                                            enableLinkGradient: false,
                                            linkBlendMode: "normal" as const,
                                            nodeTooltip: () => null,
                                            linkTooltip: () => null,
                                            nodeLabel: () => "",
                                            theme: {
                                                background: "transparent",
                                                text: { fill: PALETTE.text, fontSize: 11 },
                                                tooltip: { container: { zIndex: 9999 } },
                                            },
                                            layers: [
                                                "links", 
                                                "nodes",
                                                // Custom layer to capture node positions
                                                (layerProps: any) => (
                                                    <NodeCaptureLayer
                                                        key="node-capture"
                                                        nodes={layerProps.nodes}
                                                        onNodesCaptured={handleNodesCaptured}
                                                    />
                                                ),
                                            ],
                                        } as any)}
                                    />

                                    {/* SVG Connectors - drawn in SVG for crisp lines */}
                                    <svg
                                        style={{
                                            position: "absolute",
                                            top: 0,
                                            left: 0,
                                            width: "100%",
                                            height: "100%",
                                            pointerEvents: "none",
                                            overflow: "visible",
                                        }}
                                    >
                                        {connectorPaths.map((conn) => (
                                            <g key={conn.id}>
                                                {/* Node highlight */}
                                                <rect
                                                    x={conn.nodeX - 1}
                                                    y={conn.nodeY - 1}
                                                    width={conn.nodeWidth + 2}
                                                    height={conn.nodeHeight + 2}
                                                    rx={4}
                                                    ry={4}
                                                    fill="none"
                                                    stroke={conn.color}
                                                    strokeOpacity={0.15}
                                                    strokeWidth={2}
                                                />
                                                {/* Connector */}
                                                <path
                                                    d={conn.path}
                                                    fill="none"
                                                    stroke={conn.color}
                                                    strokeOpacity={0.25}
                                                    strokeWidth={1}
                                                />
                                            </g>
                                        ))}
                                    </svg>

                                    {/* HTML Overlay Cards - moves with pan/zoom */}
                                    <div
                                        style={{
                                            position: "absolute",
                                            top: 0,
                                            left: 0,
                                            width: "100%",
                                            height: "100%",
                                            pointerEvents: "none",
                                            overflow: "visible",
                                        }}
                                    >
                                        {cardPositions.map(({ id, node, side, x, y }) => (
                                            <OverlayCard
                                                key={id}
                                                node={node}
                                                cardX={x}
                                                cardY={y}
                                                cardW={cardW}
                                                cardH={cardH}
                                                onMouseEnter={handleMouseEnter}
                                                onMouseLeave={handleMouseLeave}
                                            />
                                        ))}
                                    </div>
                                </div>
                            </TransformComponent>
                        </>
                    )}
                </TransformWrapper>
            </div>

            {/* Portal'd tooltip - never clips */}
            {tooltip?.visible && typeof document !== "undefined" && createPortal(
                <SankeyTooltip
                    title={tooltip.title}
                    value={tooltip.value}
                    color={tooltip.color}
                    position={tooltip.position}
                />,
                document.body
            )}
        </div>
    );
}
