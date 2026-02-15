"use client"

import React, { createContext, useContext, useEffect, useState, useCallback, useRef, useMemo } from 'react'

// Use relative URLs in production, localhost in development
const getApiBaseUrl = () => {
  if (typeof window === 'undefined') return '';
  return window.location.hostname === 'localhost' ? 'http://localhost:5000' : '';
};

interface StockPrice {
  symbol: string
  price: number
  change: number
  changePercent: number
  volume: number
  timestamp: number
  lastUpdated: string
}

interface WebSocketContextType {
  stockPrices: Record<string, StockPrice>
  isConnected: boolean
  lastUpdate: string
  subscribeToStock: (symbol: string) => void
  unsubscribeFromStock: (symbol: string) => void
  getStockPrice: (symbol: string) => StockPrice | null
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined)

// All stocks to track across the application - ORDERED to match sidebar
// First 10 are PRIORITY stocks that load first
const PRIORITY_STOCKS = [
  'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'BRK.B', 'TSLA', 'AVGO', 'LLY'
]
const SECONDARY_STOCKS = [
  'WMT', 'JPM', 'V', 'MA', 'NFLX', 'XOM', 'COST', 'ORCL', 'PG', 'JNJ', 
  'UNH', 'HD', 'ABBV', 'KO', 'CRM', 'BAC'
]
const ALL_TRACKED_STOCKS = [...PRIORITY_STOCKS, ...SECONDARY_STOCKS]

interface WebSocketProviderProps {
  children: React.ReactNode
}

export function WebSocketProvider({ children }: WebSocketProviderProps) {
  const [stockPrices, setStockPrices] = useState<Record<string, StockPrice>>({})
  const [isConnected, setIsConnected] = useState(false)
  const [lastUpdate, setLastUpdate] = useState('')
  const [subscribedStocks, setSubscribedStocks] = useState<Set<string>>(new Set())
  
  // Cache management
  const cacheRef = useRef<Record<string, { data: StockPrice; expiry: number }>>({})
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const lastFetchRef = useRef<number>(0)
  
  const CACHE_DURATION = 60 * 1000 // 1 minute cache
  const POLL_INTERVAL = 5000 // Poll every 5 seconds (reduced to avoid rate limits)
  const MIN_FETCH_INTERVAL = 2000 // Minimum 2 seconds between API calls

  // Initialize with empty placeholder data (real prices loaded via API)
  const initializeMockData = useCallback(() => {
    const placeholderPrices: Record<string, StockPrice> = {}
    ALL_TRACKED_STOCKS.forEach(symbol => {
      placeholderPrices[symbol] = {
        symbol,
        price: 0,
        change: 0,
        changePercent: 0,
        volume: 0,
        timestamp: Date.now(),
        lastUpdated: new Date().toLocaleTimeString()
      }
    })
    setStockPrices(placeholderPrices)
  }, [])

  // Check WebSocket backend status
  const checkWebSocketStatus = useCallback(async () => {
    try {
      const response = await fetch(`${getApiBaseUrl()}/api/watchlist/status/websocket`)
      
      if (!response.ok) {
        setIsConnected(false)
        return false
      }
      
      const contentType = response.headers.get('content-type')
      if (!contentType || !contentType.includes('application/json')) {
        setIsConnected(false)
        return false
      }
      
      const data = await response.json()
      const connected = data.success && data.connected
      setIsConnected(connected)
      return connected
    } catch (error) {
      console.warn('WebSocket backend not available, using mock data')
      setIsConnected(false)
      return false
    }
  }, [])

  // Subscribe to stocks via backend
  const subscribeToStocks = useCallback(async (symbols: string[]) => {
    try {
      const response = await fetch(`${getApiBaseUrl()}/api/watchlist/subscribe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols })
      })
      
      if (response.ok) {
        const data = await response.json()
        return data.success
      }
    } catch (error) {
      console.warn('Failed to subscribe to WebSocket updates:', error)
    }
    return false
  }, [])

  // Fetch real-time prices with caching
  const fetchRealTimePrices = useCallback(async (symbols: string[]) => {
    const now = Date.now()
    
    // Rate limiting - don't fetch too frequently
    if (now - lastFetchRef.current < MIN_FETCH_INTERVAL) {
      return
    }
    lastFetchRef.current = now

    // Check cache first
    const cachedData: Record<string, StockPrice> = {}
    const symbolsToFetch: string[] = []
    
    symbols.forEach(symbol => {
      const cached = cacheRef.current[symbol]
      if (cached && now < cached.expiry) {
        cachedData[symbol] = cached.data
      } else {
        symbolsToFetch.push(symbol)
      }
    })

    // Use cached data immediately, but only if there are actual changes
    if (Object.keys(cachedData).length > 0) {
      setStockPrices(prev => {
        const hasChanges = Object.entries(cachedData).some(([symbol, data]) => 
          !prev[symbol] || prev[symbol].price !== data.price
        )
        return hasChanges ? { ...prev, ...cachedData } : prev
      })
    }

    // Fetch fresh data for non-cached symbols
    if (symbolsToFetch.length === 0) return

    try {
      const symbolsParam = symbolsToFetch.join(',')
      const response = await fetch(`${getApiBaseUrl()}/api/watchlist/updates/realtime?symbols=${symbolsParam}`)
      
      if (!response.ok) return
      
      const contentType = response.headers.get('content-type')
      if (!contentType || !contentType.includes('application/json')) return
      
      const data = await response.json()
      
      if (data.success && data.updates) {
        const newPrices: Record<string, StockPrice> = {}
        const updateTime = new Date().toLocaleTimeString()
        let hasNewData = false
        
        Object.entries(data.updates).forEach(([symbol, update]: [string, any]) => {
          const stockPrice: StockPrice = {
            symbol,
            price: update.price || 0,
            change: update.change || 0,
            changePercent: update.changePercent || 0,
            volume: update.volume || 0,
            timestamp: update.timestamp || now,
            lastUpdated: updateTime
          }
          
          newPrices[symbol] = stockPrice
          hasNewData = true
          
          // Cache the data
          cacheRef.current[symbol] = {
            data: stockPrice,
            expiry: now + CACHE_DURATION
          }
        })
        
        // Only update if we have new data
        if (hasNewData) {
          setStockPrices(prev => ({ ...prev, ...newPrices }))
          setLastUpdate(updateTime)
        }
      }
    } catch (error) {
      console.warn('Failed to fetch real-time prices:', error)
    }
  }, [])

  // Start polling for subscribed stocks
  const startPolling = useCallback(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current)
    }
    
    pollIntervalRef.current = setInterval(() => {
      if (subscribedStocks.size > 0) {
        fetchRealTimePrices(Array.from(subscribedStocks))
      }
    }, POLL_INTERVAL)
  }, [fetchRealTimePrices]) // Remove subscribedStocks dependency to prevent recreation

  // Public API methods
  const subscribeToStock = useCallback((symbol: string) => {
    setSubscribedStocks(prev => {
      const newSet = new Set(prev)
      newSet.add(symbol.toUpperCase())
      return newSet
    })
  }, [])

  const unsubscribeFromStock = useCallback((symbol: string) => {
    setSubscribedStocks(prev => {
      const newSet = new Set(prev)
      newSet.delete(symbol.toUpperCase())
      return newSet
    })
  }, [])

  const getStockPrice = useCallback((symbol: string): StockPrice | null => {
    return stockPrices[symbol.toUpperCase()] || null
  }, [stockPrices])

  // Fetch real initial prices - PRIORITY stocks first, then secondary
  const fetchInitialPrices = useCallback(async () => {
    const updateTime = new Date().toLocaleTimeString()
    
    // Helper function to fetch a batch of stocks
    const fetchBatch = async (symbols: string[]): Promise<Record<string, StockPrice>> => {
      try {
        const response = await fetch(`${getApiBaseUrl()}/api/watchlist/updates/realtime?symbols=${symbols.join(',')}`)
        
        if (response.ok) {
          const contentType = response.headers.get('content-type')
          if (contentType && contentType.includes('application/json')) {
            const data = await response.json()
            
            if (data.success && data.updates) {
              const prices: Record<string, StockPrice> = {}
              
              Object.entries(data.updates).forEach(([symbol, update]: [string, any]) => {
                if (update.price && update.price > 0) {
                  prices[symbol] = {
                    symbol,
                    price: update.price,
                    change: update.change || 0,
                    changePercent: update.changePercent || 0,
                    volume: update.volume || 0,
                    timestamp: update.timestamp || Date.now(),
                    lastUpdated: updateTime
                  }
                }
              })
              
              return prices
            }
          }
        }
      } catch (error) {
        console.warn(`Failed to fetch batch: ${symbols.slice(0, 3).join(', ')}...`, error)
      }
      return {}
    }
    
    try {
      // STEP 1: Fetch PRIORITY stocks first (first 10 in sidebar)
      console.log('📊 Fetching priority stocks first...')
      const priorityPrices = await fetchBatch(PRIORITY_STOCKS)
      
      if (Object.keys(priorityPrices).length > 0) {
        setStockPrices(prev => ({ ...prev, ...priorityPrices }))
        setIsConnected(true)
        console.log(`✅ Loaded ${Object.keys(priorityPrices).length} priority stocks`)
      }
      
      // STEP 2: Fetch SECONDARY stocks after a small delay to avoid rate limits
      setTimeout(async () => {
        console.log('📊 Fetching secondary stocks...')
        const secondaryPrices = await fetchBatch(SECONDARY_STOCKS)
        
        if (Object.keys(secondaryPrices).length > 0) {
          setStockPrices(prev => ({ ...prev, ...secondaryPrices }))
          console.log(`✅ Loaded ${Object.keys(secondaryPrices).length} secondary stocks`)
        }
      }, 2000) // 2 second delay
      
      return Object.keys(priorityPrices).length > 0
    } catch (error) {
      console.warn('Could not fetch initial real prices:', error)
    }
    
    return false
  }, [])

  // Initialize on mount - immediately start connecting and fetching data
  useEffect(() => {
    // Start with EMPTY data to show loading state - don't use mock data initially
    // setStockPrices({}) - already empty by default
    
    // Auto-subscribe to all tracked stocks immediately
    setSubscribedStocks(new Set(ALL_TRACKED_STOCKS))
    
    // Immediately start trying to connect and fetch real data
    const connectAndFetch = async () => {
      try {
        // Try to fetch real prices first
        const hasRealData = await fetchInitialPrices()
        
        // Check WebSocket status
        const connected = await checkWebSocketStatus()
        
        if (connected) {
          // Subscribe to WebSocket updates
          await subscribeToStocks(ALL_TRACKED_STOCKS)
          console.log('✅ WebSocket connected and subscribed to all stocks')
        } else {
          console.log('⚠️ WebSocket not available, using polling')
        }
        
        // Only use mock data if we couldn't get real data after timeout
        if (!hasRealData) {
          // Wait a bit more before falling back to mock
          setTimeout(() => {
            setStockPrices(prev => {
              // Only initialize mock if we still have no real data
              if (Object.keys(prev).length === 0) {
                console.log('⚠️ Using mock data - real API unavailable')
                initializeMockData()
              }
              return prev
            })
          }, 5000) // Wait 5 seconds before fallback
        }
      } catch (error) {
        console.warn('Error during initial connection:', error)
        // Fallback to mock data after error
        setTimeout(() => {
          setStockPrices(prev => {
            if (Object.keys(prev).length === 0) {
              initializeMockData()
            }
            return prev
          })
        }, 3000)
      }
    }
    
    // Start immediately
    connectAndFetch()
    
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
    }
  }, []) // Empty dependency array to run only once on mount

  // Start/stop polling based on subscriptions
  useEffect(() => {
    if (subscribedStocks.size > 0) {
      startPolling()
    } else {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
        pollIntervalRef.current = null
      }
    }
    
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
      }
    }
  }, [subscribedStocks.size, startPolling]) // Use .size instead of the Set object itself

  // Periodic connection check
  useEffect(() => {
    const connectionCheckInterval = setInterval(() => {
      checkWebSocketStatus()
    }, 30000) // Check every 30 seconds
    
    return () => clearInterval(connectionCheckInterval)
  }, [checkWebSocketStatus])

  // Memoize context value to prevent unnecessary re-renders
  const value = useMemo<WebSocketContextType>(() => ({
    stockPrices,
    isConnected,
    lastUpdate,
    subscribeToStock,
    unsubscribeFromStock,
    getStockPrice
  }), [stockPrices, isConnected, lastUpdate, subscribeToStock, unsubscribeFromStock, getStockPrice])

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  )
}

export function useWebSocket() {
  const context = useContext(WebSocketContext)
  
  // During SSR, return safe defaults instead of throwing
  if (typeof window === 'undefined') {
    return {
      stockPrices: {},
      isConnected: false,
      lastUpdate: '',
      subscribeToStock: () => {},
      unsubscribeFromStock: () => {},
      getStockPrice: () => null,
    }
  }
  
  if (context === undefined) {
    throw new Error('useWebSocket must be used within a WebSocketProvider')
  }
  return context
}

// Hook for getting specific stock data
export function useStockPrice(symbol: string) {
  const { getStockPrice, subscribeToStock, unsubscribeFromStock } = useWebSocket()
  
  useEffect(() => {
    // Only run on client side
    if (typeof window !== 'undefined' && symbol) {
      subscribeToStock(symbol)
      return () => unsubscribeFromStock(symbol)
    }
  }, [symbol, subscribeToStock, unsubscribeFromStock])
  
  return getStockPrice(symbol)
}

// Hook for getting multiple stocks data
export function useStockPrices(symbols: string[]) {
  const { stockPrices, subscribeToStock, unsubscribeFromStock } = useWebSocket()
  
  useEffect(() => {
    // Only run on client side
    if (typeof window !== 'undefined') {
      symbols.forEach(symbol => subscribeToStock(symbol))
      return () => {
        symbols.forEach(symbol => unsubscribeFromStock(symbol))
      }
    }
  }, [symbols, subscribeToStock, unsubscribeFromStock])
  
  return symbols.reduce((acc, symbol) => {
    acc[symbol] = stockPrices[symbol.toUpperCase()] || null
    return acc
  }, {} as Record<string, StockPrice | null>)
} 