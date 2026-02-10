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
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { AIExplanation, getComprehensiveAIExplanation, getStoredAIExplanation, generateMockAIExplanation, generateAIExplanation } from "@/lib/api"

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
      // PRIORITY 1: Get stored explanation from MongoDB (fast, no regeneration)
      let aiExplanation = await getComprehensiveAIExplanation(ticker)
      
      // PRIORITY 2: Fallback to older stored explanation method if needed
      if (!aiExplanation) {
        console.log(`No stored explanation available for ${ticker}, trying legacy stored...`)
        const stored = await getStoredAIExplanation(ticker)
        if (stored) {
          aiExplanation = stored
        }
      }
      
      // PRIORITY 3: Final fallback to mock data for demo (don't generate fresh automatically)
      if (!aiExplanation) {
        console.log(`No stored explanation available for ${ticker}, using mock data. Click ✨ to generate real AI analysis.`)
        aiExplanation = await generateMockAIExplanation(ticker)
        setError('No AI analysis found in database. Click ✨ to generate fresh analysis with Gemini 2.5 Pro.')
      } else {
        setError(null)
      }
      
      setExplanation(aiExplanation)
      setLastUpdated(new Date())
    } catch (err) {
      console.error('Error loading AI explanation:', err)
      setError('Failed to load AI analysis')
      // Still show mock data on error
      const mockExplanation = await generateMockAIExplanation(ticker)
      setExplanation(mockExplanation)
      setLastUpdated(new Date())
    } finally {
      setIsLoading(false)
    }
  }

  const generateFreshExplanation = async () => {
    setIsGenerating(true)
    setError(null)
    
    try {
      console.log(`🔥 Generating fresh AI explanation for ${ticker} using Gemini 2.5 Pro...`)
      
      // Only available in development mode (ML backend runs locally)
      const isLocalDev = typeof window !== 'undefined' && window.location.hostname === 'localhost'
      if (!isLocalDev) {
        setError('AI explanation generation is only available in development mode')
        setIsGenerating(false)
        return
      }
      
      // Call the GENERATION endpoint (not the stored retrieval endpoint)
      const targetDate = new Date().toISOString().split('T')[0]
      const response = await fetch(`http://127.0.0.1:8000/api/v1/explain/${ticker}/${targetDate}`)
      
      if (response.ok) {
        const result = await response.json()
        
        if (result.ai_explanation) {
          const freshExplanation = {
            ticker: ticker,
            date: targetDate,
            explanation: result.ai_explanation,
            data_summary: result.sentiment_summary || {},
            prediction_summary: result.prediction_data || {},
            technical_summary: result.technical_indicators || {},
            metadata: {
              data_sources: result.data_sources_used || [],
              quality_score: 0.95,
              processing_time: "Gemini 2.5 Pro - Fresh Generated",
              api_version: "v2.5-live"
            }
          }
          
          setExplanation(freshExplanation)
          setLastUpdated(new Date())
          setError(null)
        } else {
          throw new Error('No explanation in response')
        }
      } else {
        // Fallback to Node.js backend
        const freshExplanation = await generateAIExplanation(ticker)
        if (freshExplanation) {
          setExplanation(freshExplanation)
          setLastUpdated(new Date())
          setError(null)
        } else {
          throw new Error('Failed to generate fresh explanation')
        }
      }
    } catch (err) {
      console.error('Error generating fresh AI explanation:', err)
      setError('Failed to generate fresh AI analysis. Check if ML backend is running.')
    } finally {
      setIsGenerating(false)
    }
  }

  const refreshExplanation = async () => {
    await loadAIExplanation()
  }

  const formatPercentage = (value: number) => {
    const sign = value >= 0 ? '+' : ''
    return `${sign}${value.toFixed(2)}%`
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-emerald-500'
    if (confidence >= 0.6) return 'text-amber-500'
    return 'text-red-500'
  }

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.8) return 'High'
    if (confidence >= 0.6) return 'Medium'
    return 'Low'
  }

  const getSentimentColor = (sentiment: number) => {
    if (sentiment > 0.1) return 'text-emerald-500'
    if (sentiment < -0.1) return 'text-red-500'
    return 'text-amber-500'
  }

  const parseMarkdown = (text: string) => {
    // Clean up the text first - remove disclaimers and unnecessary formatting
    let cleanText = text
      // Remove the disclaimer paragraph (use [\s\S] instead of . with s flag)
      .replace(/Of course\. Here is a comprehensive analysis[\s\S]*?Please conduct your own due diligence before making any investment decisions\./g, '')
      // Remove horizontal rules and markdown artifacts
      .replace(/---+/g, '')
      .replace(/#+\s*/g, '')
      // Clean up excessive line breaks
      .replace(/\n\s*\n\s*\n/g, '\n\n')
      .trim()

    // Simple markdown parsing for headers and bold text
    return cleanText
      .replace(/## (.*$)/gim, '<h3 class="text-lg font-semibold mb-3 text-white">$1</h3>')
      .replace(/\*\*(.*)\*\*/gim, '<strong class="font-semibold text-white">$1</strong>')
      .replace(/- (.*$)/gim, '<li class="ml-4 text-zinc-300">• $1</li>')
      .replace(/\n/g, '<br/>')
  }

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-purple-500" />
            AI Market Intelligence
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-purple-500" />
            <span className="ml-2 text-zinc-400">Analyzing market data...</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!explanation) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-purple-500" />
            AI Market Intelligence
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <AlertTriangle className="h-8 w-8 text-amber-500 mx-auto mb-2" />
            <p className="text-zinc-400 mb-4">No AI analysis available</p>
            <Button onClick={loadAIExplanation} variant="outline" size="sm">
              <RefreshCw className="h-4 w-4 mr-2" />
              Try Again
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-purple-500" />
              AI Market Intelligence
            </CardTitle>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={generateFreshExplanation}
                disabled={isGenerating || isLoading}
                title="Generate fresh AI analysis"
              >
                <Sparkles className={`h-4 w-4 ${isGenerating ? 'animate-pulse text-purple-500' : ''}`} />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={refreshExplanation}
                disabled={isLoading || isGenerating}
                title="Refresh current analysis"
              >
                <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsExpanded(!isExpanded)}
              >
                {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
              </Button>
            </div>
          </div>
          
          {isGenerating && (
            <div className="flex items-center gap-1 text-xs text-purple-400">
              <Loader2 className="h-3 w-3 animate-spin" />
              Generating fresh analysis...
            </div>
          )}
        </CardHeader>

        <CardContent className="space-y-4">
          {/* Intelligent Summary */}
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div className="bg-gradient-to-br from-blue-500/10 to-purple-500/5 rounded-lg p-3 border border-blue-500/20">
              <div className="text-xs text-blue-400 mb-1">Sentiment</div>
              <div className={`text-sm font-bold ${getSentimentColor(explanation.data_summary.blended_sentiment)}`}>
                {(explanation.data_summary.blended_sentiment * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-zinc-400">
                {explanation.data_summary.total_data_points.toLocaleString()} points
              </div>
            </div>

            <div className="bg-gradient-to-br from-amber-500/10 to-orange-500/5 rounded-lg p-3 border border-amber-500/20">
              <div className="text-xs text-amber-400 mb-1">Confidence</div>
              <div className={`text-sm font-bold ${getConfidenceColor(explanation.prediction_summary.next_day.confidence)}`}>
                {getConfidenceLabel(explanation.prediction_summary.next_day.confidence)}
              </div>
              <div className="text-xs text-zinc-400">
                {(explanation.prediction_summary.next_day.confidence * 100).toFixed(0)}%
              </div>
            </div>

            <div className="bg-gradient-to-br from-emerald-500/10 to-teal-500/5 rounded-lg p-3 border border-emerald-500/20">
              <div className="text-xs text-emerald-400 mb-1">Data Sources</div>
              <div className="text-sm font-bold text-white">
                {explanation.metadata.data_sources.length}
              </div>
            </div>
          </div>

          {/* Model Breakdown (if available) */}
          {explanation.prediction_summary.next_day.model_predictions && (
            <div className="bg-zinc-900 rounded-lg p-4 border border-zinc-800">
              <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
                <Brain className="h-4 w-4 text-purple-500" />
                AI Model Breakdown
              </h4>
              
              <div className="grid grid-cols-3 gap-3">
                {Object.entries(explanation.prediction_summary.next_day.model_predictions).map(([model, prediction]) => (
                  <div key={model} className="bg-zinc-800 rounded-lg p-3 text-center">
                    <div className="text-xs text-zinc-400 mb-1">{model.toUpperCase()}</div>
                    <div className={`text-sm font-bold ${(prediction as number) >= 0 ? 'text-emerald-500' : 'text-red-500'}`}>
                      {((prediction as number) >= 0 ? '+' : '')}{(prediction as number).toFixed(2)}%
                    </div>
                    {explanation.prediction_summary.next_day.ensemble_weights && (
                      <div className="text-xs text-zinc-500 mt-1">
                        Weight: {(explanation.prediction_summary.next_day.ensemble_weights[model as keyof typeof explanation.prediction_summary.next_day.ensemble_weights] * 100).toFixed(0)}%
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Clean up expandable section only - remove metadata clutter */}

          {/* Expandable Detailed Analysis */}
          <AnimatePresence>
            {isExpanded && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.3 }}
              >
                <Separator className="my-4" />
                
                <div className="space-y-4">
                  <h4 className="text-sm font-medium flex items-center gap-2">
                    <Sparkles className="h-4 w-4 text-purple-500" />
                    Comprehensive Analysis
                  </h4>
                  
                  <div 
                    className="prose prose-invert prose-sm max-w-none text-zinc-300 leading-relaxed"
                    dangerouslySetInnerHTML={{ 
                      __html: parseMarkdown(explanation.explanation) 
                    }}
                  />
                  

                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {error && (
            <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 text-red-400 text-xs">
              {error}
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  )
} 