"use client"

import { useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  Brain,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Sparkles,
  ChevronDown,
  ChevronUp,
  Loader2,
  RefreshCw,
  Shield,
  Target,
  Zap,
  Newspaper,
  Users,
  BarChart3,
  Globe,
  Activity,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  Database,
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { AIExplanation, getComprehensiveAIExplanation, getStoredAIExplanation, generateAIExplanation } from "@/lib/api"
import { cn } from "@/lib/utils"
import { getCachedData, setCachedData } from "@/hooks/use-prefetch"

interface AIExplanationWidgetProps {
  ticker: string;
  currentPrice: number;
}

interface ParsedExplanation {
  outlook: string;
  summary: string;
  whatThisMeans: string;
  keyDrivers: { text: string; type: 'bullish' | 'bearish' }[];
  newsImpact: string;
  keyLevels: string;
  bottomLine: string;
  raw: string;
}

export default function AIExplanationWidget({ ticker, currentPrice }: AIExplanationWidgetProps) {
  const [explanation, setExplanation] = useState<AIExplanation | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [isExpanded, setIsExpanded] = useState(true)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (ticker) {
      loadAIExplanation()
    }
  }, [ticker])

  const loadAIExplanation = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const cached = getCachedData<AIExplanation>(`explanation-${ticker}`)
      let aiExplanation = cached || await getComprehensiveAIExplanation(ticker)
      if (!cached && aiExplanation) setCachedData(`explanation-${ticker}`, aiExplanation)

      if (!aiExplanation) {
        const stored = await getStoredAIExplanation(ticker)
        if (stored) aiExplanation = stored
      }
      if (!aiExplanation) {
        setError('No AI analysis available. Click generate to create one.')
      } else {
        setError(null)
      }
      setExplanation(aiExplanation)
      setLastUpdated(new Date())
    } catch (err) {
      console.error('Error loading AI explanation:', err)
      setError('Failed to load AI analysis.')
    } finally {
      setIsLoading(false)
    }
  }

  const generateFreshExplanation = async () => {
    setIsGenerating(true)
    setError(null)
    try {
      const isLocalDev = typeof window !== 'undefined' && window.location.hostname === 'localhost'
      if (isLocalDev) {
        try {
          const targetDate = new Date().toISOString().split('T')[0]
          const response = await fetch(`http://127.0.0.1:8000/api/v1/explain/${ticker}/${targetDate}`)
          if (response.ok) {
            const result = await response.json()
            if (result.ai_explanation) {
              setExplanation({
                ticker, date: targetDate,
                explanation: result.ai_explanation,
                data_summary: result.sentiment_summary || {} as any,
                prediction_summary: result.prediction_data || {} as any,
                technical_summary: result.technical_indicators || {} as any,
                metadata: { data_sources: result.data_sources_used || [], quality_score: 0.95, processing_time: "Gemini 2.5 Flash", api_version: "v2.5-live" }
              })
              setLastUpdated(new Date())
              setIsGenerating(false)
              return
            }
          }
        } catch { /* fallback below */ }
      }
      const fresh = await generateAIExplanation(ticker)
      if (fresh) { setExplanation(fresh); setLastUpdated(new Date()); setError(null); }
      else { throw new Error('No explanation returned') }
    } catch (err) {
      console.error('Error generating AI explanation:', err)
      setError('ML backend unavailable.')
    } finally {
      setIsGenerating(false)
    }
  }

  // ── Helpers ──
  const nextDay = explanation?.prediction_summary?.next_day || explanation?.prediction_summary?.['1_day']
  const sevenDay = explanation?.prediction_summary?.['7_day']
  const thirtyDay = explanation?.prediction_summary?.['30_day']
  const confidence = nextDay?.confidence ?? 0
  const blended = explanation?.data_summary?.blended_sentiment ?? 0
  const dataSources = explanation?.metadata?.data_sources ?? []

  const getOutlookColor = (outlook: string) => {
    const lower = outlook.toLowerCase()
    if (lower.includes('bullish')) return 'text-emerald-400'
    if (lower.includes('bearish')) return 'text-red-400'
    return 'text-amber-400'
  }

  const getOutlookBg = (outlook: string) => {
    const lower = outlook.toLowerCase()
    if (lower.includes('bullish')) return 'from-emerald-500/15 to-emerald-500/5 border-emerald-500/30'
    if (lower.includes('bearish')) return 'from-red-500/15 to-red-500/5 border-red-500/30'
    return 'from-amber-500/15 to-amber-500/5 border-amber-500/30'
  }

  const getOutlookIcon = (outlook: string) => {
    const lower = outlook.toLowerCase()
    if (lower.includes('bullish')) return <ArrowUpRight className="h-4 w-4 text-emerald-400" />
    if (lower.includes('bearish')) return <ArrowDownRight className="h-4 w-4 text-red-400" />
    return <Minus className="h-4 w-4 text-amber-400" />
  }

  const getConfidenceGradient = (c: number) =>
    c >= 0.7 ? 'from-purple-500/20 to-violet-500/5 border-purple-500/30' :
    c >= 0.5 ? 'from-blue-500/20 to-indigo-500/5 border-blue-500/30' :
    'from-zinc-500/20 to-zinc-500/5 border-zinc-500/30'

  const parseExplanation = (text: string): ParsedExplanation => {
    const clean = text
      .replace(/Of course[\s\S]*?investment decisions\./g, '')
      .replace(/^```[\s\S]*?```$/gm, '')
      .replace(/---+/g, '')
      .replace(/\*\*/g, '')
      .replace(/#{1,4}\s*/g, '')
      .replace(/\n\s*\n\s*\n/g, '\n\n')
      .trim()

    let outlook = ''
    let summary = ''
    let whatThisMeans = ''
    let newsImpact = ''
    let keyLevels = ''
    let bottomLine = ''
    const keyDrivers: { text: string; type: 'bullish' | 'bearish' }[] = []

    // Parse structured sections
    const sectionPatterns: [RegExp, string][] = [
      [/OVERALL_OUTLOOK:\s*/i, 'outlook'],
      [/SUMMARY:\s*/i, 'summary'],
      [/WHAT_THIS_MEANS:\s*/i, 'whatThisMeans'],
      [/KEY_DRIVERS:\s*/i, 'keyDrivers'],
      [/NEWS_IMPACT:\s*/i, 'newsImpact'],
      [/KEY_LEVELS:\s*/i, 'keyLevels'],
      [/BOTTOM_LINE:\s*/i, 'bottomLine'],
    ]

    // Try structured parsing first
    for (const [pattern, key] of sectionPatterns) {
      const match = clean.match(new RegExp(pattern.source + '([\\s\\S]*?)(?=' +
        sectionPatterns.filter(([_, k]) => k !== key).map(([p]) => p.source).join('|') + '|$)', 'i'))
      if (match && match[1]) {
        const content = match[1].trim()
        switch (key) {
          case 'outlook': outlook = content.split('\n')[0].trim(); break
          case 'summary': summary = content; break
          case 'whatThisMeans': whatThisMeans = content; break
          case 'newsImpact': newsImpact = content; break
          case 'keyLevels': keyLevels = content; break
          case 'bottomLine': bottomLine = content; break
          case 'keyDrivers':
            for (const line of content.split('\n')) {
              const trimmed = line.trim()
              if (!trimmed) continue
              const cleanLine = trimmed.replace(/^[+•\-*]\s*/, '').trim()
              if (cleanLine.length < 5) continue
              if (trimmed.startsWith('+') || trimmed.startsWith('•')) {
                keyDrivers.push({ text: cleanLine, type: 'bullish' })
              } else if (trimmed.startsWith('-') || trimmed.startsWith('*')) {
                keyDrivers.push({ text: cleanLine, type: 'bearish' })
              } else if (cleanLine.length > 10) {
                keyDrivers.push({ text: cleanLine, type: cleanLine.toLowerCase().includes('risk') || cleanLine.toLowerCase().includes('bear') || cleanLine.toLowerCase().includes('weak') || cleanLine.toLowerCase().includes('down') ? 'bearish' : 'bullish' })
              }
            }
            break
        }
      }
    }

    // Fallback: try older format parsing
    if (!summary && !outlook) {
      let currentSection = ''
      for (const line of clean.split('\n')) {
        const trimmed = line.trim()
        if (!trimmed) continue
        const stripped = trimmed.replace(/\*\*/g, '').replace(/^\*\s+/, '').replace(/^#{1,4}\s*/, '')
        const lower = stripped.toLowerCase()

        if (lower.startsWith('summary') || lower.startsWith('overview')) {
          currentSection = 'summary'
          const after = stripped.replace(/^(summary|overview):?\s*/i, '').trim()
          if (after) summary = after
          continue
        }
        if (lower.includes('bullish') && stripped.length < 40) { currentSection = 'bullish'; continue }
        if ((lower.includes('bearish') || lower.includes('risk')) && stripped.length < 40) { currentSection = 'bearish'; continue }
        if (lower.startsWith('outlook')) {
          outlook = stripped.replace(/^outlook:?\s*/i, '').trim()
          continue
        }
        if (lower.startsWith('levels') || lower.startsWith('key levels') || lower.startsWith('support')) {
          keyLevels = stripped.replace(/^(key\s+)?levels:?\s*/i, '').trim()
          continue
        }

        const cleanContent = stripped.replace(/^[+•\-*]\s*/, '').trim()
        if (trimmed.startsWith('+ ') || trimmed.startsWith('• ')) {
          keyDrivers.push({ text: cleanContent, type: 'bullish' })
        } else if (trimmed.startsWith('- ') && currentSection !== 'bullish') {
          keyDrivers.push({ text: cleanContent, type: 'bearish' })
        } else if (currentSection === 'bullish') {
          keyDrivers.push({ text: cleanContent, type: 'bullish' })
        } else if (currentSection === 'bearish') {
          keyDrivers.push({ text: cleanContent, type: 'bearish' })
        } else if (!summary && cleanContent.length > 20) {
          summary = cleanContent
        }
      }
    }

    return { outlook, summary, whatThisMeans, keyDrivers, newsImpact, keyLevels, bottomLine, raw: clean }
  }

  // ── Loading ──
  if (isLoading) {
    return (
      <Card className="overflow-hidden">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-sm">
            <div className="h-7 w-7 rounded-lg bg-gradient-to-br from-purple-500 to-violet-600 flex items-center justify-center">
              <Brain className="h-4 w-4 text-white" />
            </div>
            AI Market Intelligence
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-10 gap-3">
            <div className="relative">
              <div className="h-10 w-10 rounded-full border-2 border-purple-500/30 border-t-purple-500 animate-spin" />
              <Brain className="h-4 w-4 text-purple-400 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
            </div>
            <span className="text-xs text-zinc-500">Analyzing {ticker}...</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  // ── Empty State ──
  if (!explanation) {
    return (
      <Card className="overflow-hidden">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-sm">
            <div className="h-7 w-7 rounded-lg bg-gradient-to-br from-purple-500 to-violet-600 flex items-center justify-center">
              <Brain className="h-4 w-4 text-white" />
            </div>
            AI Market Intelligence
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <AlertTriangle className="h-8 w-8 text-amber-500/60 mx-auto mb-3" />
            <p className="text-sm text-zinc-400 mb-4">{error || 'No AI analysis available'}</p>
            <div className="flex gap-2 justify-center">
              <Button onClick={loadAIExplanation} variant="outline" size="sm" className="border-zinc-700 hover:border-zinc-600 text-xs">
                <RefreshCw className="h-3.5 w-3.5 mr-1.5" /> Reload
              </Button>
              <Button onClick={generateFreshExplanation} size="sm" disabled={isGenerating}
                className="bg-gradient-to-r from-purple-600 to-violet-600 hover:from-purple-500 hover:to-violet-500 border-0 text-xs">
                <Sparkles className={`h-3.5 w-3.5 mr-1.5 ${isGenerating ? 'animate-pulse' : ''}`} />
                {isGenerating ? 'Generating...' : 'Generate'}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  // ── Parse the explanation text ──
  const parsed = parseExplanation(explanation.explanation)
  const bullishDrivers = parsed.keyDrivers.filter(d => d.type === 'bullish')
  const bearishDrivers = parsed.keyDrivers.filter(d => d.type === 'bearish')

  // Determine overall direction from predictions
  const avgPredChange = [nextDay, sevenDay, thirtyDay]
    .filter(Boolean)
    .reduce((sum, p) => sum + (p?.price_change || 0), 0) / 3

  // ── Main Widget ──
  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
      <Card className="overflow-hidden">
        {/* Header */}
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-sm">
              <div className="h-7 w-7 rounded-lg bg-gradient-to-br from-purple-500 to-violet-600 flex items-center justify-center shadow-lg shadow-purple-500/20">
                <Brain className="h-4 w-4 text-white" />
              </div>
              <span>AI Market Intelligence</span>
              <span className="text-[10px] text-zinc-500 font-normal ml-1">— {ticker}</span>
            </CardTitle>
            <div className="flex items-center gap-1">
              <Button variant="ghost" size="sm" onClick={generateFreshExplanation} disabled={isGenerating || isLoading}
                className="h-7 w-7 p-0 hover:bg-purple-500/10" title="Generate fresh analysis">
                <Sparkles className={cn("h-3.5 w-3.5", isGenerating && "animate-pulse text-purple-400")} />
              </Button>
              <Button variant="ghost" size="sm" onClick={loadAIExplanation} disabled={isLoading || isGenerating}
                className="h-7 w-7 p-0 hover:bg-zinc-800" title="Refresh">
                <RefreshCw className={cn("h-3.5 w-3.5", isLoading && "animate-spin")} />
              </Button>
              <Button variant="ghost" size="sm" onClick={() => setIsExpanded(!isExpanded)} className="h-7 w-7 p-0 hover:bg-zinc-800">
                {isExpanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
              </Button>
            </div>
          </div>
          {isGenerating && (
            <div className="flex items-center gap-1.5 text-[11px] text-purple-400 mt-1">
              <Loader2 className="h-3 w-3 animate-spin" /> Generating fresh analysis...
            </div>
          )}
        </CardHeader>

        <CardContent className="space-y-3 pt-0">
          {/* ── Outlook Badge ── */}
          {parsed.outlook && (
            <div className={cn("rounded-lg p-3 border bg-gradient-to-br transition-all", getOutlookBg(parsed.outlook))}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {getOutlookIcon(parsed.outlook)}
                  <span className="text-[10px] text-zinc-400 uppercase tracking-wider">Overall Outlook</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className={cn("text-sm font-bold", getOutlookColor(parsed.outlook))}>
                    {parsed.outlook}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* ── Quick Stats Row ── */}
          <div className="grid grid-cols-3 gap-2">
            {/* Confidence */}
            <div className={cn("rounded-lg p-2.5 border bg-gradient-to-br transition-all", getConfidenceGradient(confidence))}>
              <div className="flex items-center gap-1.5 mb-1">
                <Shield className="h-3.5 w-3.5 text-purple-400" />
                <span className="text-[10px] text-zinc-400 uppercase tracking-wider">Confidence</span>
              </div>
              <div className="text-lg font-bold text-white leading-none">
                {(confidence * 100).toFixed(0)}%
              </div>
              <div className="mt-1.5 h-1 bg-zinc-800 rounded-full overflow-hidden">
                <div className="h-full rounded-full bg-gradient-to-r from-purple-500 to-violet-400 transition-all duration-700"
                  style={{ width: `${confidence * 100}%` }} />
              </div>
            </div>

            {/* Prediction Direction */}
            <div className={cn("rounded-lg p-2.5 border bg-gradient-to-br transition-all",
              avgPredChange > 0 ? 'from-emerald-500/15 to-emerald-500/5 border-emerald-500/30' :
              avgPredChange < 0 ? 'from-red-500/15 to-red-500/5 border-red-500/30' :
              'from-zinc-500/15 to-zinc-500/5 border-zinc-700/50')}>
              <div className="flex items-center gap-1.5 mb-1">
                <Activity className="h-3.5 w-3.5 text-blue-400" />
                <span className="text-[10px] text-zinc-400 uppercase tracking-wider">Prediction</span>
              </div>
              <div className={cn("text-lg font-bold leading-none",
                avgPredChange > 0 ? 'text-emerald-400' : avgPredChange < 0 ? 'text-red-400' : 'text-zinc-300')}>
                {avgPredChange >= 0 ? '+' : ''}{avgPredChange.toFixed(2)}%
              </div>
              <div className="text-[10px] text-zinc-500 mt-0.5">avg. predicted move</div>
            </div>

            {/* Sources */}
            <div className="rounded-lg p-2.5 border border-zinc-700/50 bg-gradient-to-br from-zinc-800/50 to-zinc-900/50">
              <div className="flex items-center gap-1.5 mb-1">
                <Database className="h-3.5 w-3.5 text-amber-400" />
                <span className="text-[10px] text-zinc-400 uppercase tracking-wider">Sources</span>
              </div>
              <div className="text-lg font-bold text-white leading-none">
                {dataSources.length}
              </div>
              <div className="text-[10px] text-zinc-500 mt-0.5 truncate">
                {dataSources.filter(s => !s.includes('Google')).slice(0, 2).join(', ')}
              </div>
            </div>
          </div>

          {/* ── Expandable Analysis ── */}
          <AnimatePresence>
            {isExpanded && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.25 }}
              >
                <div className="space-y-3 pt-1">
                  {/* Summary */}
                  {parsed.summary && (
                    <div>
                      <p className="text-sm text-zinc-200 leading-relaxed">{parsed.summary}</p>
                    </div>
                  )}

                  {/* What This Means */}
                  {parsed.whatThisMeans && (
                    <div className="rounded-lg p-3 bg-blue-500/5 border border-blue-500/15">
                      <div className="text-[10px] text-blue-400 uppercase tracking-wider mb-1.5 flex items-center gap-1">
                        <Target className="h-3 w-3" /> What This Means
                      </div>
                      <p className="text-xs text-zinc-300 leading-relaxed">{parsed.whatThisMeans}</p>
                    </div>
                  )}

                  {/* Key Drivers */}
                  {parsed.keyDrivers.length > 0 && (
                    <div className="grid grid-cols-2 gap-2">
                      {bullishDrivers.length > 0 && (
                        <div className="rounded-lg p-2.5 bg-emerald-500/5 border border-emerald-500/15">
                          <div className="text-[10px] text-emerald-400 uppercase tracking-wider mb-1.5 flex items-center gap-1">
                            <TrendingUp className="h-3 w-3" /> Bullish Factors
                          </div>
                          <ul className="space-y-1.5">
                            {bullishDrivers.map((item, i) => (
                              <li key={i} className="text-xs text-zinc-300 flex items-start gap-1.5">
                                <span className="text-emerald-500 mt-0.5 shrink-0 font-bold">+</span>
                                <span className="leading-relaxed">{item.text}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {bearishDrivers.length > 0 && (
                        <div className="rounded-lg p-2.5 bg-red-500/5 border border-red-500/15">
                          <div className="text-[10px] text-red-400 uppercase tracking-wider mb-1.5 flex items-center gap-1">
                            <TrendingDown className="h-3 w-3" /> Bearish Factors
                          </div>
                          <ul className="space-y-1.5">
                            {bearishDrivers.map((item, i) => (
                              <li key={i} className="text-xs text-zinc-300 flex items-start gap-1.5">
                                <span className="text-red-500 mt-0.5 shrink-0 font-bold">&minus;</span>
                                <span className="leading-relaxed">{item.text}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}

                  {/* News Impact */}
                  {parsed.newsImpact && (
                    <div className="rounded-lg p-3 bg-amber-500/5 border border-amber-500/15">
                      <div className="text-[10px] text-amber-400 uppercase tracking-wider mb-1.5 flex items-center gap-1">
                        <Newspaper className="h-3 w-3" /> News & Sentiment Impact
                      </div>
                      <p className="text-xs text-zinc-300 leading-relaxed">{parsed.newsImpact}</p>
                    </div>
                  )}

                  {/* Key Levels */}
                  {parsed.keyLevels && (
                    <div className="flex flex-wrap gap-2 text-[11px]">
                      <span className="px-2.5 py-1.5 rounded-md bg-zinc-800/80 border border-zinc-700/40 text-zinc-300 flex items-center gap-1.5">
                        <BarChart3 className="h-3 w-3 text-zinc-500" />
                        {parsed.keyLevels}
                      </span>
                    </div>
                  )}

                  {/* Bottom Line */}
                  {parsed.bottomLine && (
                    <div className="rounded-lg p-3 bg-gradient-to-br from-purple-500/10 to-violet-500/5 border border-purple-500/20">
                      <div className="text-[10px] text-purple-400 uppercase tracking-wider mb-1.5 flex items-center gap-1">
                        <Zap className="h-3 w-3" /> Bottom Line
                      </div>
                      <p className="text-sm text-zinc-200 font-medium leading-relaxed">{parsed.bottomLine}</p>
                    </div>
                  )}

                  {/* Fallback: if structured parsing failed, show raw */}
                  {!parsed.summary && !parsed.keyDrivers.length && !parsed.bottomLine && (
                    <p className="text-xs text-zinc-400 leading-relaxed whitespace-pre-line">
                      {parsed.raw}
                    </p>
                  )}

                  {/* Data Sources Footer */}
                  {dataSources.length > 0 && (
                    <div className="pt-2 border-t border-zinc-800/50">
                      <div className="flex flex-wrap gap-1.5">
                        {dataSources.map((source, i) => (
                          <span key={i} className="text-[9px] px-1.5 py-0.5 rounded bg-zinc-800/60 text-zinc-500 border border-zinc-700/30">
                            {source}
                          </span>
                        ))}
                      </div>
                      {lastUpdated && (
                        <div className="text-[9px] text-zinc-600 mt-1.5">
                          Last updated: {lastUpdated.toLocaleString()}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {error && (
            <div className="rounded-md bg-red-500/10 border border-red-500/20 px-3 py-2 text-[11px] text-red-400">
              {error}
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  )
}