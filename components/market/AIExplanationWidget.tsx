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
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { AIExplanation, getComprehensiveAIExplanation, getStoredAIExplanation, generateAIExplanation } from "@/lib/api"
import { cn } from "@/lib/utils"

interface AIExplanationWidgetProps {
  ticker: string;
  currentPrice: number;
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
      let aiExplanation = await getComprehensiveAIExplanation(ticker)
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
  const nextDay = explanation?.prediction_summary?.next_day
  const confidence = nextDay?.confidence ?? 0
  const blended = explanation?.data_summary?.blended_sentiment ?? 0
  const dataSources = explanation?.metadata?.data_sources ?? []

  const getSentimentLabel = (v: number) => v > 0.1 ? 'Bullish' : v < -0.1 ? 'Bearish' : 'Neutral'
  const getSentimentGradient = (v: number) =>
    v > 0.1 ? 'from-emerald-500/20 to-emerald-500/5 border-emerald-500/30' :
    v < -0.1 ? 'from-red-500/20 to-red-500/5 border-red-500/30' :
    'from-amber-500/20 to-amber-500/5 border-amber-500/30'
  const getSentimentIcon = (v: number) =>
    v > 0.1 ? <TrendingUp className="h-4 w-4 text-emerald-400" /> :
    v < -0.1 ? <TrendingDown className="h-4 w-4 text-red-400" /> :
    <Target className="h-4 w-4 text-amber-400" />
  const getConfidenceGradient = (c: number) =>
    c >= 0.7 ? 'from-purple-500/20 to-violet-500/5 border-purple-500/30' :
    c >= 0.5 ? 'from-blue-500/20 to-indigo-500/5 border-blue-500/30' :
    'from-zinc-500/20 to-zinc-500/5 border-zinc-500/30'

  const parseExplanation = (text: string) => {
    // Strip common AI preambles and markdown artifacts
    const clean = text
      .replace(/Of course[\s\S]*?investment decisions\./g, '')
      .replace(/^```[\s\S]*?```$/gm, '')
      .replace(/---+/g, '')
      .replace(/\n\s*\n\s*\n/g, '\n\n')
      .trim()

    const bullish: string[] = []
    const bearish: string[] = []
    let summary = ''
    let outlook = ''
    let levels = ''

    // Track section context for markdown-style headers
    let currentSection = ''

    for (const line of clean.split('\n')) {
      const trimmed = line.trim()
      if (!trimmed) continue

      // Strip markdown formatting for matching
      const stripped = trimmed.replace(/\*\*/g, '').replace(/^\*\s+/, '').replace(/^#{1,4}\s*/, '')

      // Detect section headers (markdown or plain)
      const lowerStripped = stripped.toLowerCase()
      if (lowerStripped.startsWith('summary') || lowerStripped.startsWith('overview')) {
        currentSection = 'summary'
        const afterColon = stripped.replace(/^(summary|overview):?\s*/i, '').trim()
        if (afterColon) summary = afterColon
        continue
      }
      if (lowerStripped.includes('bullish') || lowerStripped.includes('positive') || lowerStripped.includes('upside')) {
        currentSection = 'bullish'
        continue
      }
      if (lowerStripped.includes('bearish') || lowerStripped.includes('negative') || lowerStripped.includes('risk') || lowerStripped.includes('downside')) {
        currentSection = 'bearish'
        continue
      }
      if (lowerStripped.startsWith('outlook')) {
        currentSection = 'outlook'
        const afterColon = stripped.replace(/^outlook:?\s*/i, '').trim()
        if (afterColon) outlook = afterColon
        continue
      }
      if (lowerStripped.startsWith('levels') || lowerStripped.startsWith('key levels') || lowerStripped.startsWith('support')) {
        currentSection = 'levels'
        const afterColon = stripped.replace(/^(key\s+)?levels:?\s*/i, '').trim()
        if (afterColon) levels = afterColon
        continue
      }

      // Process content based on prefixes or current section
      const cleanContent = stripped.replace(/^[+•\-*]\s*/, '').trim()

      if (trimmed.startsWith('+ ') || trimmed.startsWith('• ')) {
        bullish.push(cleanContent)
      } else if ((trimmed.startsWith('- ') || trimmed.startsWith('* ')) && currentSection === 'bearish') {
        bearish.push(cleanContent)
      } else if ((trimmed.startsWith('- ') || trimmed.startsWith('* ')) && currentSection === 'bullish') {
        bullish.push(cleanContent)
      } else if (trimmed.startsWith('- ') && !trimmed.startsWith('---')) {
        bearish.push(cleanContent)
      } else if (trimmed.toLowerCase().startsWith('summary:')) {
        summary = trimmed.replace(/^summary:\s*/i, '')
      } else if (trimmed.toLowerCase().startsWith('outlook:')) {
        outlook = trimmed.replace(/^outlook:\s*/i, '')
      } else if (trimmed.toLowerCase().startsWith('levels:')) {
        levels = trimmed.replace(/^levels:\s*/i, '')
      } else if (currentSection === 'bullish') {
        bullish.push(cleanContent)
      } else if (currentSection === 'bearish') {
        bearish.push(cleanContent)
      } else if (currentSection === 'outlook') {
        outlook = outlook ? `${outlook} ${cleanContent}` : cleanContent
      } else if (currentSection === 'levels') {
        levels = levels ? `${levels} ${cleanContent}` : cleanContent
      } else if (!summary && !trimmed.includes(':') && cleanContent.length > 10) {
        summary = cleanContent
      }
    }

    return { summary, bullish, bearish, outlook, levels, raw: clean }
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

  // ── Parse the explanation text into structured sections ──
  const parsed = parseExplanation(explanation.explanation)

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
              <span className="text-[10px] text-zinc-600 font-normal ml-1">Gemini 2.5</span>
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
          {/* ── Metric Cards ── */}
          <div className="grid grid-cols-3 gap-2">
            {/* Sentiment */}
            <div className={cn("rounded-lg p-2.5 border bg-gradient-to-br transition-all", getSentimentGradient(blended))}>
              <div className="flex items-center gap-1.5 mb-1">
                {getSentimentIcon(blended)}
                <span className="text-[10px] text-zinc-400 uppercase tracking-wider">Sentiment</span>
              </div>
              <div className="text-lg font-bold text-white leading-none">
                {getSentimentLabel(blended)}
              </div>
              <div className="text-[10px] text-zinc-500 mt-0.5">
                Score: {(blended * 100).toFixed(0)}%
              </div>
            </div>

            {/* Confidence */}
            <div className={cn("rounded-lg p-2.5 border bg-gradient-to-br transition-all", getConfidenceGradient(confidence))}>
              <div className="flex items-center gap-1.5 mb-1">
                <Shield className="h-4 w-4 text-purple-400" />
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

            {/* Sources */}
            <div className="rounded-lg p-2.5 border border-zinc-700/50 bg-gradient-to-br from-zinc-800/50 to-zinc-900/50">
              <div className="flex items-center gap-1.5 mb-1">
                <Zap className="h-4 w-4 text-amber-400" />
                <span className="text-[10px] text-zinc-400 uppercase tracking-wider">Sources</span>
              </div>
              <div className="text-lg font-bold text-white leading-none">
                {dataSources.length}
              </div>
              <div className="text-[10px] text-zinc-500 mt-0.5 truncate">
                {dataSources.slice(0, 2).join(', ')}
              </div>
            </div>
          </div>

          {/* ── Model Breakdown ── */}
          {nextDay?.model_predictions && (
            <div className="rounded-lg p-3 bg-zinc-900/60 border border-zinc-800/50">
              <h4 className="text-[11px] text-zinc-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Brain className="h-3 w-3 text-purple-400" /> Model Predictions
              </h4>
              <div className="grid grid-cols-3 gap-2">
                {Object.entries(nextDay.model_predictions).map(([model, prediction]) => {
                  const val = prediction as number
                  const weights = nextDay?.ensemble_weights
                  const weight = weights ? weights[model as keyof typeof weights] : undefined
                  return (
                    <div key={model} className="rounded-md bg-zinc-800/60 p-2 text-center border border-zinc-700/30">
                      <div className="text-[10px] text-zinc-500 mb-0.5">{model.toUpperCase()}</div>
                      <div className={cn("text-sm font-bold", val >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                        {val >= 0 ? '+' : ''}{val.toFixed(2)}%
                      </div>
                      {weight !== undefined && (
                        <div className="text-[9px] text-zinc-600 mt-0.5">w: {(weight * 100).toFixed(0)}%</div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* ── Expandable Analysis ── */}
          <AnimatePresence>
            {isExpanded && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.25 }}
              >
                <div className="space-y-2.5 pt-1">
                  {/* Summary */}
                  {parsed.summary && (
                    <p className="text-sm text-zinc-200 leading-relaxed">{parsed.summary}</p>
                  )}

                  {/* Bullish / Bearish columns */}
                  {(parsed.bullish.length > 0 || parsed.bearish.length > 0) && (
                    <div className="grid grid-cols-2 gap-2">
                      {parsed.bullish.length > 0 && (
                        <div className="rounded-lg p-2.5 bg-emerald-500/5 border border-emerald-500/15">
                          <div className="text-[10px] text-emerald-400 uppercase tracking-wider mb-1.5 flex items-center gap-1">
                            <TrendingUp className="h-3 w-3" /> Bullish
                          </div>
                          <ul className="space-y-1">
                            {parsed.bullish.map((item, i) => (
                              <li key={i} className="text-xs text-zinc-300 flex items-start gap-1.5">
                                <span className="text-emerald-500 mt-0.5 shrink-0">+</span>
                                <span>{item}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {parsed.bearish.length > 0 && (
                        <div className="rounded-lg p-2.5 bg-red-500/5 border border-red-500/15">
                          <div className="text-[10px] text-red-400 uppercase tracking-wider mb-1.5 flex items-center gap-1">
                            <TrendingDown className="h-3 w-3" /> Bearish
                          </div>
                          <ul className="space-y-1">
                            {parsed.bearish.map((item, i) => (
                              <li key={i} className="text-xs text-zinc-300 flex items-start gap-1.5">
                                <span className="text-red-500 mt-0.5 shrink-0">&minus;</span>
                                <span>{item}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Outlook + Levels */}
                  {(parsed.outlook || parsed.levels) && (
                    <div className="flex flex-wrap gap-2 text-[11px]">
                      {parsed.outlook && (
                        <span className="px-2 py-1 rounded-md bg-zinc-800/80 border border-zinc-700/40 text-zinc-300">
                          {parsed.outlook}
                        </span>
                      )}
                      {parsed.levels && (
                        <span className="px-2 py-1 rounded-md bg-zinc-800/80 border border-zinc-700/40 text-zinc-300">
                          {parsed.levels}
                        </span>
                      )}
                    </div>
                  )}

                  {/* Fallback: if parsing found nothing structured, show raw cleaned text */}
                  {!parsed.summary && !parsed.bullish.length && !parsed.bearish.length && (
                    <p className="text-xs text-zinc-400 leading-relaxed whitespace-pre-line">
                      {parsed.raw}
                    </p>
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