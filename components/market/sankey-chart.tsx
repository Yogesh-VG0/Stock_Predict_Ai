"use client";

import React, { useEffect, useMemo, useState, useCallback, useRef } from "react";
import ReactECharts from "echarts-for-react";
import { Download, ZoomIn, ZoomOut, RotateCcw, Move } from "lucide-react";

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
    const [isTablet, setIsTablet] = useState(false);
    const [windowWidth, setWindowWidth] = useState(0);
    const [isMounted, setIsMounted] = useState(false);
    const chartRef = useRef<ReactECharts>(null);

    // On mobile, give the chart a generous fixed width so it can breathe.
    // The outer container will scroll horizontally.
    const mobileChartWidth = useMemo(() => {
        if (!isMobile) return undefined; // let it fill 100%
        const nodeCount = data?.nodes?.length ?? 0;
        // Base 600px, scale up with nodes
        return Math.max(700, Math.min(1100, nodeCount * 70));
    }, [isMobile, data?.nodes?.length]);

    // Responsive height
    const chartHeight = useMemo(() => {
        if (isMobile) return Math.max(propHeight, 560);
        if (isTablet) return Math.max(propHeight, 600);
        return propHeight;
    }, [propHeight, isMobile, isTablet]);

    useEffect(() => {
        setIsMounted(true);
        const onResize = () => {
            const width = window.innerWidth;
            setWindowWidth(width);
            setIsMobile(width < 480);
            setIsTablet(width < 768 && width >= 480);
        };
        onResize();
        window.addEventListener("resize", onResize);
        return () => window.removeEventListener("resize", onResize);
    }, []);

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

    const totalRevenueValue = useMemo(() => {
        const vFromLinks = enriched.links
            .filter((l: any) => String(l.target) === "Total Revenue")
            .reduce((acc: number, l: any) => acc + Number(l.value || 0), 0);
        if (vFromLinks > 0) return vFromLinks;
        const n: any = enriched.nodes.find((x: any) => String(x.id) === "Total Revenue");
        return Number(n?.displayValue ?? n?.value ?? 0);
    }, [enriched]);

    // Transform data for ECharts
    const chartOption = useMemo(() => {
        const nodes = enriched.nodes.map((n: any) => ({
            name: String(n.id),
            itemStyle: {
                color: n.color || PALETTE.neutral,
            },
        }));

        const links = enriched.links.map((l: any) => ({
            source: String(l.source),
            target: String(l.target),
            value: Number(l.value || 0),
            lineStyle: {
                color: l.color || PALETTE.neutral,
                opacity: 0.4,
            },
        }));

        // Responsive margins — mobile uses wider canvas (scrollable), so we can be generous
        const topMargin = isMobile ? 55 : isTablet ? 70 : 60;
        const bottomMargin = isMobile ? 20 : isTablet ? 30 : 40;
        const sideMargin = isMobile ? 90 : isTablet ? 100 : 200;
        const nodeThickness = isMobile ? 14 : isTablet ? 16 : 20;
        const nodeSpacing = isMobile ? 16 : isTablet ? 16 : 16;
        const labelFontSize = isMobile ? 10 : isTablet ? 10 : 12;

        return {
            backgroundColor: "transparent",
            tooltip: {
                trigger: "item",
                triggerOn: "mousemove",
                backgroundColor: "rgba(17, 20, 28, 0.97)",
                borderColor: "rgba(255, 255, 255, 0.12)",
                borderWidth: 1,
                padding: [10, 14],
                textStyle: {
                    color: PALETTE.text,
                    fontSize: isMobile ? 11 : 13,
                },
                formatter: (params: any) => {
                    if (params.dataType === "edge") {
                        return `<div style="display:flex;align-items:center;gap:8px;">
                            <span style="width:10px;height:10px;border-radius:3px;background:${params.color};display:inline-block;"></span>
                            <strong>${params.data.source} → ${params.data.target}</strong>
                            <span style="color:#9CA3AF;">${formatMoney(params.value)}</span>
                        </div>`;
                    }
                    return `<div style="display:flex;align-items:center;gap:8px;">
                        <span style="width:10px;height:10px;border-radius:3px;background:${params.color};display:inline-block;"></span>
                        <strong>${params.name}</strong>
                        <span style="color:#9CA3AF;">${formatMoney(params.value)}</span>
                    </div>`;
                },
            },
            series: [{
                type: "sankey",
                layout: "none",
                emphasis: {
                    focus: "adjacency",
                },
                data: nodes,
                links: links,
                top: topMargin,
                bottom: bottomMargin,
                left: sideMargin,
                right: sideMargin,
                nodeWidth: nodeThickness,
                nodeGap: nodeSpacing,
                layoutIterations: 32,
                draggable: false,
                label: {
                    show: true,
                    color: PALETTE.text,
                    fontSize: labelFontSize,
                    fontWeight: 500,
                    formatter: (params: any) => {
                        const name = params.name as string;
                        const value = params.value;
                        // Wrap long labels: if more than 3 words, split into lines of ~3 words
                        const words = name.split(/\s+/);
                        let wrappedName = name;
                        if (words.length > 3) {
                            const lines: string[] = [];
                            for (let i = 0; i < words.length; i += 3) {
                                lines.push(words.slice(i, i + 3).join(" "));
                            }
                            wrappedName = lines.join("\n");
                        }
                        if (isMobile) return wrappedName;
                        if (!value) return wrappedName;
                        return `${wrappedName}\n${formatMoney(value)}`;
                    },
                },
                lineStyle: {
                    color: "gradient",
                    curveness: 0.5,
                    opacity: 0.4,
                },
                itemStyle: {
                    borderWidth: 0,
                },
            }],
            grid: {
                top: 0,
                bottom: 0,
                left: 0,
                right: 0,
            },
        };
    }, [enriched, isMobile, isTablet]);

    // Export to PNG
    const handleExportPng = useCallback(() => {
        const chart = chartRef.current?.getEchartsInstance();
        if (!chart) return;

        const url = chart.getDataURL({
            type: "png",
            pixelRatio: 2,
            backgroundColor: PALETTE.background,
        });

        const link = document.createElement("a");
        link.download = `${symbol || "sankey"}-chart-${Date.now()}.png`;
        link.href = url;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }, [symbol]);

    if (!isMounted) return <div style={{ height: chartHeight }} />;

    // Inner chart width: on mobile use the computed wider width, otherwise 100%
    const innerWidth = isMobile && mobileChartWidth ? mobileChartWidth : undefined;

    return (
        <div className="relative w-full rounded-2xl border border-white/5 shadow-2xl" style={{ background: PALETTE.background }}>
            {/* Sticky overlay controls — stay visible even when scrolling */}
            <div className="pointer-events-none absolute inset-x-0 top-0 z-30 flex items-start justify-between p-3">
                {/* Revenue badge */}
                {symbol && (
                    <div className="pointer-events-auto" style={{ marginTop: isMobile ? 4 : 0 }}>
                        <div style={{
                            width: isMobile ? 150 : isTablet ? 180 : 220,
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

                {/* Export button */}
                <div className="pointer-events-auto flex items-center gap-1 rounded-lg border border-white/10 bg-black/70 p-1 backdrop-blur shadow-lg">
                    <button
                        onClick={handleExportPng}
                        className="flex items-center gap-1 px-2 py-1 rounded hover:bg-white/10 transition"
                        aria-label="Export PNG"
                        title="Download PNG"
                    >
                        <Download className="h-3.5 w-3.5 text-white/70" />
                        <span className="text-[8px] text-white/70 font-bold tracking-widest uppercase">PNG</span>
                    </button>
                </div>
            </div>

            {/* Mobile scroll hint */}
            {isMobile && (
                <div className="pointer-events-none absolute bottom-3 inset-x-0 z-20 flex justify-center">
                    <div className="flex items-center gap-1.5 rounded-full bg-white/10 backdrop-blur px-3 py-1 text-[10px] text-white/60 font-medium">
                        <Move className="h-3 w-3" />
                        Swipe to explore
                    </div>
                </div>
            )}

            {/* Scrollable chart area — only scrolls horizontally on mobile */}
            <div
                className="w-full"
                style={{
                    overflowX: isMobile ? "auto" : "hidden",
                    overflowY: "hidden",
                    WebkitOverflowScrolling: "touch",
                    height: chartHeight,
                }}
            >
                <div style={{ width: innerWidth ?? "100%", minWidth: "100%", height: chartHeight }}>
                    <ReactECharts
                        ref={chartRef}
                        option={chartOption}
                        style={{
                            width: "100%",
                            height: chartHeight,
                        }}
                        opts={{
                            renderer: "canvas",
                            width: innerWidth,
                        }}
                    />
                </div>
            </div>
        </div>
    );
}
