"use client"

import { useState, useEffect, useCallback } from "react"
import { motion } from "framer-motion"
import { Link } from "react-router-dom"
import { Star, TrendingUp, TrendingDown, Plus, MoreHorizontal, Bell, Trash2, MoveVertical, Wifi, WifiOff, AlertTriangle, CheckCircle, Clock } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import Sparkline from "@/components/ui/sparkline"
import { cn } from "@/lib/utils"
import { 
  getWatchlist, 
  addToWatchlist, 
  removeFromWatchlist, 
  type WatchlistItem,
  type WatchlistData
} from "@/lib/api"
import { useWebSocket, useStockPrices } from "@/hooks/use-websocket-context"

interface Alert {
  id: string;
  symbol: string;
  type: 'price_above' | 'price_below' | 'earnings' | 'volume_spike';
  value: number | string;
  active: boolean;
  triggered: boolean;
  createdAt: Date;
}

// In-memory cache for the watchlist
let cachedWatchlist: WatchlistItem[] | null = null;

export default function WatchlistPage() {
  const [isLoading, setIsLoading] = useState(true)
  const [watchlist, setWatchlist] = useState<WatchlistItem[]>([])
  const [error, setError] = useState<string | null>(null)
  
  // Use centralized WebSocket service
  const { stockPrices, isConnected: isWebSocketConnected } = useWebSocket()
  const [showAddStock, setShowAddStock] = useState(false)
  const [newSymbol, setNewSymbol] = useState("")
  const [showAddAlert, setShowAddAlert] = useState(false)
  const [alertForm, setAlertForm] = useState({
    symbol: '',
    type: 'price_above' as Alert['type'],
    value: ''
  })
  const [alerts, setAlerts] = useState<Alert[]>([
    {
      id: '1',
      symbol: 'AAPL',
      type: 'price_above',
      value: 190,
      active: true,
      triggered: false,
      createdAt: new Date()
    },
    {
      id: '2',
      symbol: 'TSLA',
      type: 'price_below',
      value: 240,
      active: true,
      triggered: false,
      createdAt: new Date()
    },
    {
      id: '3',
      symbol: 'NVDA',
      type: 'earnings',
      value: '2024-01-25',
      active: true,
      triggered: false,
      createdAt: new Date()
    }
  ])

  // Fetch watchlist data
  const fetchWatchlist = useCallback(async (forceRefresh = false) => {
    if (cachedWatchlist && !forceRefresh) {
      setWatchlist(cachedWatchlist);
      setIsLoading(false);
      return;
    }

    try {
      setIsLoading(true)
      setError(null)
      
      const response = await fetch('/api/watchlist/default')
      const data: WatchlistData = await response.json()
      
      if (data.success) {
        cachedWatchlist = data.watchlist; // Cache the data
        setWatchlist(data.watchlist)
      } else {
        setError('Failed to fetch watchlist')
      }
    } catch (error) {
      console.error('Error fetching watchlist:', error)
      setError('Failed to connect to server')
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Update watchlist with real-time prices from centralized WebSocket service
  useEffect(() => {
    if (watchlist.length > 0) {
      setWatchlist(prev => {
        let hasChanges = false
        const newWatchlist = prev.map(item => {
          const realTimeData = stockPrices[item.symbol]
          if (realTimeData && realTimeData.price !== item.price) {
            hasChanges = true
            return {
              ...item,
              price: realTimeData.price,
              change: realTimeData.change,
              changePercent: realTimeData.changePercent,
              timestamp: realTimeData.timestamp
            }
          }
          return item
        })
        
        // Only update if there are actual changes to prevent infinite loops
        return hasChanges ? newWatchlist : prev
      })
    }
  }, [stockPrices])

  // Subscribe to real-time updates
  const subscribeToUpdates = useCallback(async (symbols: string[]) => {
    try {
      const response = await fetch('/api/watchlist/subscribe', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ symbols })
      })
      
      const data = await response.json()
      if (data.success) {
        console.log('Subscribed to real-time updates for:', symbols)
      }
    } catch (error) {
      console.error('Error subscribing to updates:', error)
    }
  }, [])

  // Add stock to watchlist
  const addToWatchlist = async (symbol: string) => {
    try {
      const response = await fetch('/api/watchlist/default/add', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ symbol: symbol.toUpperCase() })
      })
      
      const data = await response.json()
      
      if (data.success) {
        // Refresh watchlist and force a cache update
        await fetchWatchlist(true)
        setNewSymbol("")
        setShowAddStock(false)
      } else {
        setError(data.error || 'Failed to add stock')
      }
    } catch (error) {
      console.error('Error adding to watchlist:', error)
      setError('Failed to add stock')
    }
  }

  // Remove stock from watchlist
  const removeFromWatchlist = async (symbol: string) => {
    try {
      const response = await fetch(`/api/watchlist/default/${symbol}`, {
        method: 'DELETE'
      })
      
      const data = await response.json()
      
      if (data.success) {
        // Refresh watchlist and force a cache update
        await fetchWatchlist(true)
      } else {
        setError(data.error || 'Failed to remove stock')
      }
    } catch (error) {
      console.error('Error removing from watchlist:', error)
      setError('Failed to remove stock')
    }
  }

  // Initial load
  useEffect(() => {
    fetchWatchlist()
  }, [fetchWatchlist])

  // Subscribe to updates when watchlist changes
  useEffect(() => {
    if (watchlist.length > 0) {
      const symbols = watchlist.map(item => item.symbol)
      subscribeToUpdates(symbols)
    }
  }, [watchlist, subscribeToUpdates])

  // Calculate performance data for chart
  const getPerformanceData = () => {
    if (watchlist.length === 0) return [];
    
    const performanceData = watchlist.map(stock => ({
      symbol: stock.symbol,
      changePercent: stock.changePercent,
      volume: stock.volume,
      price: stock.price
    }));
    
    return performanceData.sort((a, b) => b.changePercent - a.changePercent);
  };

  // Check alerts
  const checkAlerts = useCallback(() => {
    setAlerts(prevAlerts => {
      return prevAlerts.map(alert => {
        const stock = watchlist.find(s => s.symbol === alert.symbol);
        if (!stock) return alert;

        let triggered = false;
        
        switch (alert.type) {
          case 'price_above':
            triggered = stock.price > (alert.value as number);
            break;
          case 'price_below':
            triggered = stock.price < (alert.value as number);
            break;
          case 'volume_spike':
            // Trigger if volume is 50% higher than average
            const avgVolume = watchlist.reduce((sum, s) => sum + s.volume, 0) / watchlist.length;
            triggered = stock.volume > avgVolume * 1.5;
            break;
          case 'earnings':
            // Simple earnings alert (in real app, would check actual earnings dates)
            const today = new Date();
            const earningsDate = new Date(alert.value as string);
            const daysDiff = Math.ceil((earningsDate.getTime() - today.getTime()) / (1000 * 60 * 60 * 24));
            triggered = daysDiff <= 1 && daysDiff >= 0;
            break;
        }

        return { ...alert, triggered };
      });
    });
  }, [watchlist]);

  // Check alerts when watchlist updates
  useEffect(() => {
    if (watchlist.length > 0 && alerts.length > 0) {
      checkAlerts();
    }
  }, [watchlist, alerts.length, checkAlerts]);

  // Add new alert
  const addAlert = () => {
    if (!alertForm.symbol || !alertForm.value) return;
    
    const newAlert: Alert = {
      id: Date.now().toString(),
      symbol: alertForm.symbol.toUpperCase(),
      type: alertForm.type,
      value: alertForm.type === 'earnings' ? alertForm.value : parseFloat(alertForm.value),
      active: true,
      triggered: false,
      createdAt: new Date()
    };
    
    setAlerts(prev => [...prev, newAlert]);
    setAlertForm({ symbol: '', type: 'price_above', value: '' });
    setShowAddAlert(false);
  };

  // Remove alert
  const removeAlert = (alertId: string) => {
    setAlerts(prev => prev.filter(alert => alert.id !== alertId));
  };

  return (
    <div className="space-y-6">
      {/* Header - stack on mobile, row on larger screens */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <motion.h1
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-2xl font-bold flex items-center gap-2"
        >
          <Star className="h-6 w-6 text-amber-400" />
          My Watchlist
          <div className="flex items-center gap-2 ml-4">
            {isWebSocketConnected ? (
              <div className="flex items-center gap-1 text-emerald-500 text-sm">
                <Wifi className="h-4 w-4" />
                <span>Live</span>
              </div>
            ) : (
              <div className="flex items-center gap-1 text-amber-500 text-sm">
                <Clock className="h-4 w-4 animate-pulse" />
                <span>Syncing</span>
              </div>
            )}
          </div>
        </motion.h1>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="flex items-center gap-2"
        >
          <button 
            onClick={() => setShowAddStock(!showAddStock)}
            className="bg-emerald-500 hover:bg-emerald-600 text-black rounded-md px-3 py-2 text-sm font-medium transition-colors flex items-center gap-1.5"
          >
            <Plus className="h-4 w-4" />
            <span>Add Stock</span>
          </button>
        </motion.div>
      </div>

      {/* Add Stock Form */}
      {showAddStock && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="bg-zinc-900 rounded-lg p-4 border border-zinc-800"
        >
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
            <input
              type="text"
              placeholder="Enter stock symbol (e.g., AAPL)"
              value={newSymbol}
              onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
              className="flex-1 bg-zinc-800 border border-zinc-700 rounded-md px-3 py-2 text-white placeholder-zinc-400 focus:outline-none focus:ring-2 focus:ring-emerald-500"
              onKeyDown={(e) => e.key === 'Enter' && addToWatchlist(newSymbol)}
            />
            <button
              onClick={() => addToWatchlist(newSymbol)}
              disabled={!newSymbol}
              className="bg-emerald-500 hover:bg-emerald-600 disabled:bg-zinc-700 disabled:cursor-not-allowed text-black rounded-md px-4 py-2 text-sm font-medium transition-colors"
            >
              Add
            </button>
            <button
              onClick={() => setShowAddStock(false)}
              className="bg-zinc-700 hover:bg-zinc-600 text-white rounded-md px-4 py-2 text-sm font-medium transition-colors"
            >
              Cancel
            </button>
          </div>
        </motion.div>
      )}

      {/* Error Message */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 text-red-400"
        >
          {error}
          <button
            onClick={() => setError(null)}
            className="ml-2 text-red-300 hover:text-red-100"
          >
            ×
          </button>
        </motion.div>
      )}

      {/* Watchlist Summary */}
      {!isLoading && watchlist.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-4"
        >
          <Card className="bg-gradient-to-br from-emerald-500/10 to-emerald-600/5 border-emerald-500/20">
            <CardContent className="p-4">
              <div className="text-sm text-emerald-400 mb-1">Top Performer</div>
              <div className="text-lg font-bold text-white">
                {watchlist.reduce((top, stock) => 
                  stock.changePercent > top.changePercent ? stock : top
                ).symbol}
              </div>
              <div className="text-xs text-emerald-400">
                +{watchlist.reduce((top, stock) => 
                  stock.changePercent > top.changePercent ? stock : top
                ).changePercent.toFixed(2)}%
              </div>
            </CardContent>
          </Card>
          
          <Card className="bg-gradient-to-br from-red-500/10 to-red-600/5 border-red-500/20">
            <CardContent className="p-4">
              <div className="text-sm text-red-400 mb-1">Biggest Loser</div>
              <div className="text-lg font-bold text-white">
                {watchlist.reduce((bottom, stock) => 
                  stock.changePercent < bottom.changePercent ? stock : bottom
                ).symbol}
              </div>
              <div className="text-xs text-red-400">
                {watchlist.reduce((bottom, stock) => 
                  stock.changePercent < bottom.changePercent ? stock : bottom
                ).changePercent.toFixed(2)}%
              </div>
            </CardContent>
          </Card>
          
          <Card className="bg-gradient-to-br from-blue-500/10 to-blue-600/5 border-blue-500/20">
            <CardContent className="p-4">
              <div className="text-sm text-blue-400 mb-1">Total Volume</div>
              <div className="text-lg font-bold text-white">
                {watchlist.reduce((sum, stock) => sum + stock.volume, 0).toLocaleString()}
              </div>
              <div className="text-xs text-blue-400">
                {watchlist.filter(stock => stock.volume > 0).length} active
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle>Stocks</CardTitle>
            <div className="text-xs text-zinc-400">
              {watchlist.length} stocks • {isWebSocketConnected ? 'Live updates' : 'Manual refresh'}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="h-16 bg-zinc-900 animate-pulse rounded-md"></div>
              ))}
            </div>
          ) : watchlist.length === 0 ? (
            <div className="text-center py-8">
              <Star className="h-12 w-12 text-zinc-600 mx-auto mb-4" />
              <p className="text-zinc-400 mb-4">Your watchlist is empty</p>
              <button
                onClick={() => setShowAddStock(true)}
                className="bg-emerald-500 hover:bg-emerald-600 text-black rounded-md px-4 py-2 text-sm font-medium transition-colors"
              >
                Add Your First Stock
              </button>
            </div>
          ) : (
            <div className="space-y-2">
              {watchlist.map((stock, index) => (
                <motion.div
                  key={stock.symbol}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="bg-zinc-900 rounded-md border border-zinc-800 p-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between group hover:border-zinc-700 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div className="cursor-move opacity-0 group-hover:opacity-100 transition-opacity">
                      <MoveVertical className="h-4 w-4 text-zinc-500" />
                    </div>

                    <div className="flex flex-col">
                      <Link
                        to={`/stocks/${stock.symbol}`}
                        className="font-medium hover:text-emerald-500 transition-colors"
                      >
                        {stock.symbol}
                      </Link>
                      <span className="text-xs text-zinc-400">{stock.name}</span>
                      <span className="text-xs text-zinc-500">
                        Vol: {stock.volume > 0 ? stock.volume.toLocaleString() : 'N/A'}
                      </span>
                    </div>
                  </div>

                  <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-end sm:gap-6 w-full sm:w-auto">
                    <Sparkline
                      data={stock.sparklineData}
                      height={30}
                      width={80}
                      color={stock.changePercent >= 0 ? "#10b981" : "#ef4444"}
                    />

                    <div className="flex flex-col items-end">
                      {stock.price > 0 ? (
                        <>
                          <span className="font-medium">${stock.price.toFixed(2)}</span>
                          <span
                            className={cn(
                              "text-xs flex items-center gap-1",
                              stock.changePercent >= 0 ? "text-emerald-500" : "text-red-500",
                            )}
                          >
                            {stock.changePercent >= 0 ? (
                              <TrendingUp className="h-3 w-3" />
                            ) : (
                              <TrendingDown className="h-3 w-3" />
                            )}
                            {stock.changePercent >= 0 ? "+" : ""}
                            {stock.changePercent.toFixed(2)}%
                          </span>
                        </>
                      ) : (
                        <>
                          <div className="h-4 w-16 bg-zinc-800 rounded animate-pulse"></div>
                          <div className="h-3 w-12 bg-zinc-800 rounded animate-pulse mt-1"></div>
                        </>
                      )}
                    </div>

                    <div className="flex items-center gap-1">
                      <button className="p-1.5 rounded-md hover:bg-zinc-800 transition-colors">
                        <Bell className="h-4 w-4 text-zinc-400 hover:text-white" />
                      </button>
                      <button 
                        onClick={() => removeFromWatchlist(stock.symbol)}
                        className="p-1.5 rounded-md hover:bg-zinc-800 transition-colors"
                      >
                        <Trash2 className="h-4 w-4 text-zinc-400 hover:text-red-500" />
                      </button>
                      <button className="p-1.5 rounded-md hover:bg-zinc-800 transition-colors">
                        <MoreHorizontal className="h-4 w-4 text-zinc-400 hover:text-white" />
                      </button>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-emerald-500" />
              Performance Overview
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Performance Chart */}
              <div className="h-48 bg-zinc-900 rounded-lg border border-zinc-800 p-4">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-sm font-medium text-zinc-300">Today's Performance</h4>
                  <div className="text-xs text-zinc-400">
                    {watchlist.filter(s => s.changePercent > 0).length} up, {watchlist.filter(s => s.changePercent < 0).length} down
                  </div>
                </div>
                
                <div className="space-y-2">
                  {getPerformanceData().slice(0, 5).map((stock, index) => (
                    <div key={stock.symbol} className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-zinc-400 w-4">{index + 1}</span>
                        <span className="text-sm font-medium">{stock.symbol}</span>
                      </div>
                      <div className="flex items-center gap-4">
                        <span className="text-xs text-zinc-400">${stock.price.toFixed(2)}</span>
                        <span className={cn(
                          "text-xs font-medium",
                          stock.changePercent >= 0 ? "text-emerald-500" : "text-red-500"
                        )}>
                          {stock.changePercent >= 0 ? "+" : ""}{stock.changePercent.toFixed(2)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Volume Leaders */}
              <div className="h-32 bg-zinc-900 rounded-lg border border-zinc-800 p-4">
                <h4 className="text-sm font-medium text-zinc-300 mb-3">Volume Leaders</h4>
                <div className="space-y-2">
                  {watchlist
                    .filter(stock => stock.volume > 0)
                    .sort((a, b) => b.volume - a.volume)
                    .slice(0, 3)
                    .map((stock, index) => (
                      <div key={stock.symbol} className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-zinc-400 w-4">{index + 1}</span>
                          <span className="text-sm font-medium">{stock.symbol}</span>
                        </div>
                        <span className="text-xs text-zinc-400">
                          {stock.volume.toLocaleString()}
                        </span>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bell className="h-5 w-5 text-amber-500" />
              Active Alerts
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {alerts.map((alert) => (
                <div key={alert.id} className="bg-zinc-900 rounded-md p-3 border border-zinc-800">
                  <div className="flex justify-between items-center mb-1">
                    <div className="font-medium">{alert.symbol}</div>
                    <div className="flex items-center gap-2">
                      <div className={cn(
                        "text-xs px-2 py-1 rounded-full",
                        alert.triggered 
                          ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                          : "bg-red-500/20 text-red-400 border border-red-500/30"
                      )}>
                        {alert.triggered ? "Triggered" : "Active"}
                      </div>
                      <button
                        onClick={() => removeAlert(alert.id)}
                        className="text-zinc-400 hover:text-red-500 transition-colors"
                      >
                        <Trash2 className="h-3 w-3" />
                      </button>
                    </div>
                  </div>
                  <div className="text-xs text-zinc-400">
                    {alert.type === 'price_above' && `Alert when price goes above $${alert.value}`}
                    {alert.type === 'price_below' && `Alert when price goes below $${alert.value}`}
                    {alert.type === 'volume_spike' && `Alert when volume spikes above average`}
                    {alert.type === 'earnings' && `Alert for earnings on ${alert.value}`}
                  </div>
                </div>
              ))}
              <button
                onClick={() => setShowAddAlert(true)}
                className="w-full bg-zinc-800 hover:bg-zinc-700 text-white rounded-md px-3 py-2 text-sm transition-colors flex items-center justify-center gap-1.5 mt-2"
              >
                <Plus className="h-4 w-4" />
                <span>Add New Alert</span>
              </button>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Add Alert Form */}
      {showAddAlert && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="bg-zinc-900 rounded-lg p-6 border border-zinc-800"
        >
          <h3 className="text-lg font-semibold mb-4">Create New Alert</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-zinc-300 mb-2">Stock Symbol</label>
              <input
                type="text"
                placeholder="e.g., AAPL, MSFT, TSLA"
                value={alertForm.symbol}
                onChange={(e) => setAlertForm({ ...alertForm, symbol: e.target.value.toUpperCase() })}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-md px-3 py-2 text-white placeholder-zinc-400 focus:outline-none focus:ring-2 focus:ring-emerald-500"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-zinc-300 mb-2">Alert Type</label>
              <select
                value={alertForm.type}
                onChange={(e) => setAlertForm({ ...alertForm, type: e.target.value as Alert['type'] })}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-md px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
              >
                <option value="price_above">Price Above</option>
                <option value="price_below">Price Below</option>
                <option value="volume_spike">Volume Spike</option>
                <option value="earnings">Earnings Date</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-zinc-300 mb-2">
                {alertForm.type === 'earnings' ? 'Earnings Date' : 'Value'}
              </label>
              <input
                type={alertForm.type === 'earnings' ? 'date' : 'number'}
                placeholder={alertForm.type === 'earnings' ? 'Select date' : 'Enter value'}
                value={alertForm.value}
                onChange={(e) => setAlertForm({ ...alertForm, value: e.target.value })}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-md px-3 py-2 text-white placeholder-zinc-400 focus:outline-none focus:ring-2 focus:ring-emerald-500"
              />
            </div>
            
            <div className="flex gap-3 pt-2">
              <button
                onClick={addAlert}
                disabled={!alertForm.symbol || !alertForm.value}
                className="flex-1 bg-emerald-500 hover:bg-emerald-600 disabled:bg-zinc-700 disabled:cursor-not-allowed text-black rounded-md px-4 py-2 text-sm font-medium transition-colors"
              >
                Create Alert
              </button>
              <button
                onClick={() => {
                  setShowAddAlert(false);
                  setAlertForm({ symbol: '', type: 'price_above', value: '' });
                }}
                className="flex-1 bg-zinc-700 hover:bg-zinc-600 text-white rounded-md px-4 py-2 text-sm font-medium transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}
