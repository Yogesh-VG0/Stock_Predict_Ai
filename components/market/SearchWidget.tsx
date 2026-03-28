"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { Search, TrendingUp, TrendingDown } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"
import { useRouter } from "next/navigation"

interface SearchResult {
  symbol: string
  name: string
  isTracked?: boolean
  price?: number
  change?: number
  changePercent?: number
}

export default function SearchWidget() {
  const [query, setQuery] = useState("")
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [showResults, setShowResults] = useState(false)
  const [activeIndex, setActiveIndex] = useState(-1)
  const searchRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLUListElement>(null)
  const router = useRouter()

  // Search function using API
  const searchStocks = async (searchQuery: string): Promise<SearchResult[]> => {
    if (!searchQuery || searchQuery.length < 2) return []
    
    try {
      // Import the API function dynamically to avoid circular dependency
      const { searchStocks: apiSearchStocks, getStockPrice } = await import("@/lib/api")
      const searchResults = await apiSearchStocks(searchQuery)
      
      if (!searchResults) return []
      
      // Enhance tracked results with price data (skip untracked to avoid unnecessary API calls)
      const enhancedResults = await Promise.all(
        searchResults.map(async (stock) => {
          if (!stock.isTracked) return stock
          const priceData = await getStockPrice(stock.symbol)
          return {
            ...stock,
            price: priceData?.price,
            change: priceData?.change,
            changePercent: priceData?.changePercent
          }
        })
      )
      
      return enhancedResults
    } catch (error) {
      console.error('Error searching stocks:', error)
      return []
    }
  }

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
        setShowResults(false)
      }
    }

    document.addEventListener("mousedown", handleClickOutside)
    return () => document.removeEventListener("mousedown", handleClickOutside)
  }, [])

  useEffect(() => {
    const delayedSearch = setTimeout(async () => {
      if (query) {
        setLoading(true)
        const searchResults = await searchStocks(query)
        setResults(searchResults)
        setShowResults(true)
        setLoading(false)
      } else {
        setResults([])
        setShowResults(false)
      }
    }, 300)

    return () => clearTimeout(delayedSearch)
  }, [query])

  const handleStockSelect = useCallback((symbol: string) => {
    setQuery("")
    setShowResults(false)
    setActiveIndex(-1)
    router.push(`/stocks/${symbol}`)
  }, [router])

  // Reset active index when results change
  useEffect(() => {
    setActiveIndex(-1)
  }, [results])

  // Keyboard navigation handler
  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!showResults || results.length === 0) return

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault()
        setActiveIndex(prev => (prev < results.length - 1 ? prev + 1 : 0))
        break
      case 'ArrowUp':
        e.preventDefault()
        setActiveIndex(prev => (prev > 0 ? prev - 1 : results.length - 1))
        break
      case 'Enter':
        e.preventDefault()
        if (activeIndex >= 0 && activeIndex < results.length) {
          handleStockSelect(results[activeIndex].symbol)
        }
        break
      case 'Escape':
        setShowResults(false)
        setActiveIndex(-1)
        break
    }
  }, [showResults, results, activeIndex, handleStockSelect])

  // Scroll active item into view
  useEffect(() => {
    if (activeIndex >= 0 && listRef.current) {
      const activeItem = listRef.current.children[activeIndex] as HTMLElement
      if (activeItem) {
        activeItem.scrollIntoView({ block: 'nearest' })
      }
    }
  }, [activeIndex])

  const listboxId = "search-results-listbox"

  return (
    <div ref={searchRef} className="relative w-full max-w-md" role="combobox" aria-expanded={showResults} aria-haspopup="listbox" aria-owns={listboxId}>
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-zinc-400" aria-hidden="true" />
        <Input
          ref={inputRef}
          type="text"
          placeholder="Search stocks..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          className="pl-10 bg-zinc-900 border-zinc-700 text-white placeholder-zinc-400 focus:border-blue-500"
          onFocus={() => query && setShowResults(true)}
          role="searchbox"
          aria-label="Search stocks"
          aria-autocomplete="list"
          aria-controls={showResults ? listboxId : undefined}
          aria-activedescendant={activeIndex >= 0 ? `search-option-${activeIndex}` : undefined}
        />
      </div>
      
      {showResults && (
        <Card className="absolute top-full left-0 right-0 mt-1 z-50 bg-zinc-900 border-zinc-700 max-h-80 overflow-y-auto">
          <CardContent className="p-0">
            {loading ? (
              <div className="p-4 text-center text-zinc-400" role="status" aria-live="polite">Searching...</div>
            ) : results.length > 0 ? (
              <ul 
                ref={listRef}
                id={listboxId}
                role="listbox" 
                aria-label="Stock search results"
                className="divide-y divide-zinc-700"
              >
                {results.map((stock, index) => (
                  <li
                    key={stock.symbol}
                    id={`search-option-${index}`}
                    role="option"
                    aria-selected={index === activeIndex}
                    onClick={() => handleStockSelect(stock.symbol)}
                    onMouseEnter={() => setActiveIndex(index)}
                    className={cn(
                      "p-3 cursor-pointer transition-colors",
                      index === activeIndex ? "bg-zinc-800" : "hover:bg-zinc-800"
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-1.5">
                          <span className="font-medium text-white">{stock.symbol}</span>
                          {stock.isTracked && (
                            <span className="text-[9px] bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded-full border border-emerald-500/30">AI</span>
                          )}
                        </div>
                        <div className="text-sm text-zinc-400 truncate">{stock.name}</div>
                      </div>
                      {stock.price && (
                        <div className="text-right ml-4">
                          <div className="font-medium text-white">${stock.price.toFixed(2)}</div>
                          {stock.change && (
                            <div className={cn(
                              "text-sm flex items-center gap-1",
                              stock.change >= 0 ? "text-emerald-500" : "text-red-500"
                            )}>
                              {stock.change >= 0 ? (
                                <TrendingUp className="h-3 w-3" />
                              ) : (
                                <TrendingDown className="h-3 w-3" />
                              )}
                              {stock.changePercent?.toFixed(2)}%
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="p-4 text-center text-zinc-400" role="status">No results found</div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}