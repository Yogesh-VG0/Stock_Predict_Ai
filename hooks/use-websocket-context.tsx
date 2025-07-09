"use client"

import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react'

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

// All stocks to track across the application
const ALL_TRACKED_STOCKS = [
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
  'JPM', 'V', 'JNJ', 'WMT', 'PG', 'UNH', 'HD', 'MA', 'BAC', 'XOM', 
  'LLY', 'ABBV', 'BRK.B', 'AVGO', 'COST', 'ORCL', 'CRM', 'KO'
]

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
  const POLL_INTERVAL = 2000 // Poll every 2 seconds
  const MIN_FETCH_INTERVAL = 1000 // Minimum 1 second between API calls

  // Initialize with realistic mock data based on actual stock price ranges
  const initializeMockData = useCallback(() => {
    // Realistic price ranges for major stocks
    const stockPriceRanges: Record<string, { min: number; max: number }> = {
      'AAPL': { min: 180, max: 220 },
      'MSFT': { min: 350, max: 450 },
      'GOOGL': { min: 140, max: 180 },
      'AMZN': { min: 140, max: 180 },
      'TSLA': { min: 200, max: 300 },
      'NVDA': { min: 800, max: 1200 },
      'META': { min: 450, max: 550 },
      'NFLX': { min: 600, max: 700 },
      'JPM': { min: 180, max: 220 },
      'V': { min: 250, max: 300 },
      'JNJ': { min: 150, max: 180 },
      'WMT': { min: 160, max: 190 },
      'PG': { min: 150, max: 180 },
      'UNH': { min: 500, max: 600 },
      'HD': { min: 350, max: 400 },
      'MA': { min: 450, max: 550 },
      'BAC': { min: 35, max: 45 },
      'XOM': { min: 100, max: 130 },
      'LLY': { min: 700, max: 900 },
      'ABBV': { min: 150, max: 180 },
      'BRK.B': { min: 400, max: 500 },
      'AVGO': { min: 1200, max: 1600 },
      'COST': { min: 850, max: 950 },
      'ORCL': { min: 100, max: 140 },
      'CRM': { min: 250, max: 350 },
      'KO': { min: 60, max: 70 }
    }

    const mockPrices: Record<string, StockPrice> = {}
    ALL_TRACKED_STOCKS.forEach(symbol => {
      const range = stockPriceRanges[symbol] || { min: 50, max: 200 }
      const basePrice = range.min + Math.random() * (range.max - range.min)
      const changePercent = (Math.random() - 0.5) * 6 // Random change ±3%
      mockPrices[symbol] = {
        symbol,
        price: basePrice,
        change: (basePrice * changePercent) / 100,
        changePercent,
        volume: Math.floor(Math.random() * 1000000),
        timestamp: Date.now(),
        lastUpdated: new Date().toLocaleTimeString()
      }
    })
    setStockPrices(mockPrices)
  }, [])

  // Check WebSocket backend status
  const checkWebSocketStatus = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:5000/api/watchlist/status/websocket')
      
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
      const response = await fetch('http://localhost:5000/api/watchlist/subscribe', {
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
      const response = await fetch(`http://localhost:5000/api/watchlist/updates/realtime?symbols=${symbolsParam}`)
      
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

  // Fetch real initial prices
  const fetchInitialPrices = useCallback(async () => {
    try {
      // Try to fetch real prices from the API
      const response = await fetch(`http://localhost:5000/api/watchlist/updates/realtime?symbols=${ALL_TRACKED_STOCKS.join(',')}`)
      
      if (response.ok) {
        const contentType = response.headers.get('content-type')
        if (contentType && contentType.includes('application/json')) {
          const data = await response.json()
          
          if (data.success && data.updates) {
            const realPrices: Record<string, StockPrice> = {}
            const updateTime = new Date().toLocaleTimeString()
            
            Object.entries(data.updates).forEach(([symbol, update]: [string, any]) => {
              realPrices[symbol] = {
                symbol,
                price: update.price || 0,
                change: update.change || 0,
                changePercent: update.changePercent || 0,
                volume: update.volume || 0,
                timestamp: update.timestamp || Date.now(),
                lastUpdated: updateTime
              }
            })
            
            // Only update stocks that have real data
            if (Object.keys(realPrices).length > 0) {
              setStockPrices(prev => ({ ...prev, ...realPrices }))
              setIsConnected(true)
              console.log(`✅ Loaded real prices for ${Object.keys(realPrices).length} stocks`)
              return true
            }
          }
        }
      }
    } catch (error) {
      console.warn('Could not fetch initial real prices, using mock data:', error)
    }
    
    return false
  }, [])

  // Initialize on mount - immediately start connecting and fetching data
  useEffect(() => {
    // Start with realistic mock data immediately
    initializeMockData()
    
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
          console.log('⚠️ WebSocket not available, using polling with mock data')
        }
      } catch (error) {
        console.warn('Error during initial connection:', error)
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

  const value: WebSocketContextType = {
    stockPrices,
    isConnected,
    lastUpdate,
    subscribeToStock,
    unsubscribeFromStock,
    getStockPrice
  }

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