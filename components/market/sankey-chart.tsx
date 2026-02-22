"use client"

import { ResponsiveSankey } from '@nivo/sankey'
import { useTheme } from 'next-themes'
import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { DollarSign } from 'lucide-react'

interface SankeyNode {
    id: string;
    nodeColor?: string;
}

interface SankeyLink {
    source: string;
    target: string;
    value: number;
    color?: string;
}

interface SankeyData {
    nodes: SankeyNode[];
    links: SankeyLink[];
}

interface SankeyChartProps {
    data: SankeyData;
    height?: number;
}

// Custom format for large numbers (billions/millions)
const formatCurrency = (value: number) => {
    if (value >= 1e9) {
        return `$${(value / 1e9).toFixed(2)}B`;
    }
    if (value >= 1e6) {
        return `$${(value / 1e6).toFixed(2)}M`;
    }
    return `$${value.toLocaleString()}`;
};

export default function SankeyChart({ data, height = 600 }: SankeyChartProps) {
    const { theme } = useTheme();
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    if (!mounted) return (
        <div className={`w-full h-[${height}px] bg-zinc-900/50 rounded-lg animate-pulse flex items-center justify-center`}>
            <div className="text-zinc-600 flex items-center gap-2">
                <DollarSign className="h-5 w-5 animate-spin" /> Rendering Financial Flow...
            </div>
        </div>
    );

    const isDark = theme === 'dark' || true; // Force dark for now based on app aesthetics

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            style={{ height }}
            className="w-full relative"
        >
            <ResponsiveSankey
                data={data}
                margin={{ top: 20, right: 120, bottom: 20, left: 120 }}
                align="justify"
                colors={(node) => node.nodeColor || '#3b82f6'}
                linkBlendMode="multiply"
                nodeOpacity={0.85}
                nodeHoverOthersOpacity={0.15}
                nodeThickness={14}
                nodeSpacing={24}
                nodeBorderWidth={0}
                nodeBorderRadius={3}
                linkOpacity={0.4}
                linkHoverOthersOpacity={0.1}
                enableLinkGradient={true}
                labelPosition="outside"
                labelOrientation="horizontal"
                labelPadding={16}
                labelTextColor={isDark ? '#e4e4e7' : '#27272a'}
                valueFormat={(value) => formatCurrency(value)}
                theme={{
                    text: {
                        fontSize: 12,
                        fill: isDark ? '#e4e4e7' : '#27272a',
                        fontFamily: 'inherit',
                    },
                    tooltip: {
                        container: {
                            background: isDark ? '#18181b' : '#ffffff',
                            color: isDark ? '#f4f4f5' : '#18181b',
                            fontSize: 13,
                            borderRadius: '8px',
                            border: `1px solid ${isDark ? '#27272a' : '#e4e4e7'}`,
                            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)',
                        },
                    },
                }}
                // Custom link color mapping
                // @ts-ignore - nivo typing can be strict
                linkColor={(link: any) => {
                    if (link.color) return link.color;
                    if (link.source.id === 'Gross Profit' && link.target.id === 'Operating Expenses') return '#ef4444';
                    return '#10b981';
                }}
                // Custom Node Rendering for clean badges
                nodeTooltip={({ node }) => (
                    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-3 shadow-xl max-w-xs">
                        <div className="flex items-center gap-2 mb-1">
                            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: node.color }}></div>
                            <strong className="text-zinc-100">{node.id}</strong>
                        </div>
                        <div className="text-zinc-400 text-sm ml-5">
                            Value: <span className="text-white font-medium">{formatCurrency(node.value)}</span>
                        </div>
                    </div>
                )}
                linkTooltip={({ link }) => {
                    const isLoss = link.color === '#ef4444' || link.color === 'red';
                    return (
                        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-3 shadow-xl flex items-center gap-3">
                            <div className="flex flex-col items-end">
                                <span className="text-xs text-zinc-500 truncate max-w-[120px]">{link.source.id}</span>
                                <div className="w-16 h-1 mt-1 rounded-full bg-gradient-to-r from-zinc-700 to-transparent"></div>
                            </div>
                            <div className={`text-lg font-bold ${isLoss ? 'text-red-400' : 'text-emerald-400'}`}>
                                {formatCurrency(link.value)}
                            </div>
                            <div className="flex flex-col items-start">
                                <span className="text-xs text-zinc-500 truncate max-w-[120px]">{link.target.id}</span>
                                <div className="w-16 h-1 mt-1 rounded-full bg-gradient-to-l from-zinc-700 to-transparent"></div>
                            </div>
                        </div>
                    )
                }}
            />
        </motion.div>
    )
}
