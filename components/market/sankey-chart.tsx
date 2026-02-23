"use client";

import React, { useMemo, useState } from "react";
import { ResponsiveSankey, SankeyNodeDatum, SankeyLinkDatum } from "@nivo/sankey";

type NodeKind = "segment" | "revenue" | "expense" | "profit" | "tax" | "neutral";

type SankeyNode = {
    id: string;
    kind?: NodeKind;
};

type SankeyLink = {
    source: string;
    target: string;
    value: number;
    color?: string;
};

export default function SankeyChart({
    data,
    height = 560,
}: {
    data: { nodes: SankeyNode[]; links: SankeyLink[] };
    height?: number;
}) {
    const kindColor = (kind: NodeKind) => {
        switch (kind) {
            case "segment":
                return "#3b82f6"; // neon blue
            case "revenue":
                return "#eab308"; // solid neon yellow
            case "profit":
                return "#22c55e"; // neon green
            case "expense":
                return "#ef4444"; // neon red
            case "tax":
                return "#f97316"; // neon orange
            default:
                return "#9ca3af"; // solid gray
        }
    };

    const getKindIcon = (kind: NodeKind) => {
        switch (kind) {
            case "revenue": return "💰";
            case "expense": return "🧾";
            case "profit": return "✅";
            case "tax": return "🧮";
            case "segment": return "📊";
            default: return "";
        }
    };

    // truncate for labels (prevents left overflow)
    const formatNodeLabel = (id: string, max = 18) => {
        if (!id) return "";
        let cleanId = id.replace(" Segment", "").trim();
        if (cleanId.length <= max) return cleanId;
        return cleanId.slice(0, max - 1) + "…";
    };

    const enriched = useMemo(() => {
        // Need to protect against empty data throws in Nivo
        if (!data || !data.nodes || !data.links) return { nodes: [], links: [] };

        const nodeMap = new Map<string, SankeyNode>();
        for (const n of data.nodes) nodeMap.set(n.id, n);

        return {
            nodes: data.nodes.map((n) => ({
                ...n,
                color: kindColor((n.kind || "neutral") as NodeKind),
            })),
            links: data.links.map((l) => {
                const t = typeof l.target === 'string' ? nodeMap.get(l.target) : nodeMap.get((l.target as SankeyNode).id);
                // If link color already set by backend, keep it. Else infer from target kind.
                const inferred =
                    t?.kind === "expense" ? "#ef4444" :
                        t?.kind === "tax" ? "#f59e0b" :
                            t?.kind === "profit" ? "#10b981" :
                                "#60a5fa";

                return { ...l, color: l.color || inferred };
            }),
        };
    }, [data]);

    // Responsive margins — give left side more room on desktop
    // On mobile we *truncate* more and reduce margins to fit.
    const isClient = typeof window !== "undefined";
    const vw = isClient ? window.innerWidth : 1200;
    const isMobile = vw < 640;

    const margin = isMobile
        ? { top: 24, right: 18, bottom: 24, left: 18 }
        : { top: 28, right: 40, bottom: 28, left: 90 };

    const labelMax = isMobile ? 12 : 22;

    if (enriched.nodes.length === 0) return null;

    const [zoom, setZoom] = useState(1);

    return (
        <div className="relative w-full flex flex-col">
            {/* Zoom Controls */}
            <div className="absolute top-2 right-2 z-10 flex items-center gap-1 bg-zinc-900/80 p-1 rounded-md border border-zinc-700/50 backdrop-blur-sm">
                <button onClick={() => setZoom(z => Math.max(0.5, z - 0.1))} className="p-1.5 hover:bg-zinc-800 rounded text-zinc-400 hover:text-white transition-colors" title="Zoom Out">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /><line x1="8" y1="11" x2="14" y2="11" /></svg>
                </button>
                <button onClick={() => setZoom(1)} className="p-1.5 hover:bg-zinc-800 rounded text-zinc-400 hover:text-white transition-colors" title="Reset Zoom">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3" /></svg>
                </button>
                <button onClick={() => setZoom(z => Math.min(2.5, z + 0.1))} className="p-1.5 hover:bg-zinc-800 rounded text-zinc-400 hover:text-white transition-colors" title="Zoom In">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /><line x1="11" y1="8" x2="11" y2="14" /><line x1="8" y1="11" x2="14" y2="11" /></svg>
                </button>
            </div>

            <div className="w-full overflow-auto rounded-lg hide-scrollbar">
                <div style={{ height: height * zoom, minWidth: isMobile ? 800 * zoom : `${100 * zoom}%`, transform: `scale(${zoom})`, transformOrigin: 'top left', transition: 'width 0.2s, height 0.2s, transform 0.2s ease-out' }}>
                    <ResponsiveSankey
                        data={enriched}
                        margin={margin}
                        align="justify"
                        sort="auto"
                        // "Premium feel" geometry
                        nodeThickness={isMobile ? 12 : 16}
                        nodeSpacing={isMobile ? 12 : 16}
                        nodeBorderWidth={1}
                        nodeBorderColor={{ from: "color", modifiers: [["darker", 0.6]] }}
                        // Make links "flowy"
                        linkOpacity={0.35}
                        linkHoverOpacity={0.85}
                        linkContract={2}
                        enableLinkGradient={true}
                        linkBlendMode="screen"
                        // Labels
                        labelPosition="outside"
                        labelOrientation="horizontal"
                        labelPadding={14}
                        labelTextColor="#ffffff" // bright white for readability
                        label={(node: any) => `${getKindIcon(node.kind || 'neutral')} ${formatNodeLabel(String(node.id), labelMax)}`}
                        // Dark theme polish
                        theme={{
                            text: {
                                fill: "#ffffff",
                                fontSize: isMobile ? 11 : 13,
                                fontWeight: 600,
                            },
                            tooltip: {
                                container: {
                                    background: "rgba(9, 9, 11, 0.95)",
                                    border: "1px solid rgba(63, 63, 70, 0.8)",
                                    borderRadius: "12px",
                                    boxShadow: "0 10px 30px rgba(0,0,0,0.6)",
                                    color: "#e4e4e7",
                                },
                            },
                            grid: { line: { stroke: "rgba(63,63,70,0.25)" } },
                        }}
                        // Tooltips (full names shown here)
                        nodeTooltip={({ node }) => {
                            const id = String(node.id);
                            const val = node.value || 0;
                            const kind = (node as any).kind || "neutral";
                            const icon = getKindIcon(kind);

                            return (
                                <div style={{ padding: 10 }}>
                                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                                        <span
                                            style={{
                                                width: 10,
                                                height: 10,
                                                borderRadius: 999,
                                                background: (node as any).color,
                                                display: "inline-block",
                                            }}
                                        />
                                        <div style={{ fontWeight: 700, fontSize: 12 }}>{icon} {id}</div>
                                    </div>
                                    <div style={{ fontSize: 12, color: "#a1a1aa" }}>
                                        {String(kind).toUpperCase()}
                                    </div>
                                    <div style={{ fontWeight: 800, marginTop: 6 }}>
                                        {formatMoney(val)}
                                    </div>
                                </div>
                            );
                        }}
                        linkTooltip={({ link }: any) => {
                            let sourceId = typeof link.source === 'string' ? link.source : link.source.id;
                            let targetId = typeof link.target === 'string' ? link.target : link.target.id;
                            const source = String(sourceId);
                            const target = String(targetId);
                            return (
                                <div style={{ padding: 10 }}>
                                    <div style={{ fontWeight: 700, fontSize: 12, marginBottom: 6 }}>
                                        {source} → {target}
                                    </div>
                                    <div style={{ fontWeight: 800 }}>{formatMoney(link.value)}</div>
                                </div>
                            );
                        }}
                    />
                </div>
            </div>
        </div>
    );
}

function formatMoney(v: number) {
    const abs = Math.abs(v || 0);
    if (abs >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
    if (abs >= 1e9) return `$${(v / 1e9).toFixed(2)}B`;
    if (abs >= 1e6) return `$${(v / 1e6).toFixed(2)}M`;
    if (abs >= 1e3) return `$${(v / 1e3).toFixed(2)}K`;
    return `$${(v || 0).toFixed(0)}`;
}
