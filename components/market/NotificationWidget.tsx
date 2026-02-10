"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import { Bell, X, AlertTriangle, TrendingUp, TrendingDown, Info, RefreshCw } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

interface Notification {
  id: string
  type: "alert" | "news" | "gain" | "loss" | "info" | "market"
  title: string
  message: string
  timestamp: Date
  symbol?: string
  read: boolean
}

interface ApiNotification {
  id: string
  type: string
  title: string
  message: string
  timestamp: string
  symbol?: string
}

export default function NotificationWidget() {
  const [notifications, setNotifications] = useState<Notification[]>([])
  const [showNotifications, setShowNotifications] = useState(false)
  const [unreadCount, setUnreadCount] = useState(0)
  const [isLoading, setIsLoading] = useState(false)
  const [lastFetch, setLastFetch] = useState<string | null>(null)
  const readNotificationIds = useRef<Set<string>>(new Set())

  // Fetch notifications from API
  const fetchNotifications = useCallback(async (isInitial = false) => {
    try {
      setIsLoading(true)
      const params = new URLSearchParams({ limit: '20' })
      if (lastFetch && !isInitial) {
        params.append('since', lastFetch)
      }
      
      const response = await fetch(`/api/notifications?${params}`)
      if (!response.ok) throw new Error('Failed to fetch notifications')
      
      const data = await response.json()
      
      if (data.success && data.notifications) {
        const fetchedNotifications: Notification[] = data.notifications.map((n: ApiNotification) => ({
          id: n.id,
          type: n.type as Notification['type'],
          title: n.title,
          message: n.message,
          timestamp: new Date(n.timestamp),
          symbol: n.symbol,
          read: readNotificationIds.current.has(n.id)
        }))

        setNotifications(prev => {
          let merged: Notification[]

          if (isInitial) {
            // On initial load just use fetched notifications
            merged = fetchedNotifications
          } else {
            // Merge new notifications with existing, avoiding duplicates
            const existingIds = new Set(prev.map(n => n.id))
            const uniqueNew = fetchedNotifications.filter(n => !existingIds.has(n.id))
            merged = [...uniqueNew, ...prev].slice(0, 20) // Keep max 20
          }

          // Recalculate unread count from the merged list so it can't drift
          const unread = merged.filter(n => !n.read).length
          setUnreadCount(unread)

          return merged
        })

        setLastFetch(new Date().toISOString())
      }
    } catch (error) {
      console.error('Error fetching notifications:', error)
    } finally {
      setIsLoading(false)
    }
  }, [lastFetch])

  useEffect(() => {
    // Load initial notifications
    fetchNotifications(true)

    // Poll for new notifications every 30 seconds
    const interval = setInterval(() => {
      fetchNotifications(false)
    }, 30000)

    return () => clearInterval(interval)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const getNotificationIcon = (type: Notification["type"]) => {
    switch (type) {
      case "alert":
      case "market":
        return <AlertTriangle className="h-4 w-4 text-amber-500" />
      case "gain":
        return <TrendingUp className="h-4 w-4 text-emerald-500" />
      case "loss":
        return <TrendingDown className="h-4 w-4 text-red-500" />
      case "news":
        return <Info className="h-4 w-4 text-blue-500" />
      default:
        return <Info className="h-4 w-4 text-zinc-400" />
    }
  }

  const markAsRead = (id: string) => {
    readNotificationIds.current.add(id)
    setNotifications(prev => 
      prev.map(n => n.id === id ? { ...n, read: true } : n)
    )
    setUnreadCount(prev => Math.max(0, prev - 1))
  }

  const markAllAsRead = () => {
    notifications.forEach(n => readNotificationIds.current.add(n.id))
    setNotifications(prev => prev.map(n => ({ ...n, read: true })))
    setUnreadCount(0)
  }

  const handleRefresh = () => {
    fetchNotifications(true)
  }

  const formatTimestamp = (timestamp: Date) => {
    const now = new Date()
    const diffMs = now.getTime() - timestamp.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    
    if (diffMins < 1) return "Just now"
    if (diffMins < 60) return `${diffMins}m ago`
    const diffHours = Math.floor(diffMins / 60)
    if (diffHours < 24) return `${diffHours}h ago`
    return timestamp.toLocaleDateString()
  }

  return (
    <div className="relative">
      <button
        onClick={() => setShowNotifications(!showNotifications)}
        className="relative p-2 rounded-lg bg-zinc-900 border border-zinc-700 hover:border-zinc-600 transition-colors"
      >
        <Bell className="h-5 w-5 text-zinc-400" />
        {unreadCount > 0 && (
          <Badge className="absolute -top-1 -right-1 h-5 w-5 p-0 text-xs bg-red-500 text-white border-0 flex items-center justify-center">
            {unreadCount > 9 ? "9+" : unreadCount}
          </Badge>
        )}
      </button>

      {showNotifications && (
        <Card className="absolute top-full right-0 mt-2 w-80 max-h-96 overflow-y-auto z-50 bg-zinc-900 border-zinc-700">
          <CardContent className="p-0">
            <div className="p-4 border-b border-zinc-700 flex items-center justify-between">
              <h3 className="font-semibold">Notifications</h3>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleRefresh}
                  className="text-zinc-400 hover:text-white p-1"
                  disabled={isLoading}
                >
                  <RefreshCw className={cn("h-3.5 w-3.5", isLoading && "animate-spin")} />
                </button>
                {unreadCount > 0 && (
                  <button
                    onClick={markAllAsRead}
                    className="text-xs text-blue-500 hover:text-blue-400"
                  >
                    Mark all read
                  </button>
                )}
                <button
                  onClick={() => setShowNotifications(false)}
                  className="text-zinc-400 hover:text-white"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>
            
            <div className="divide-y divide-zinc-700">
              {notifications.length > 0 ? (
                notifications.map((notification) => (
                  <div
                    key={notification.id}
                    onClick={() => markAsRead(notification.id)}
                    className={cn(
                      "p-4 hover:bg-zinc-800 cursor-pointer transition-colors",
                      !notification.read && "bg-zinc-850"
                    )}
                  >
                    <div className="flex items-start gap-3">
                      <div className="flex-shrink-0 mt-0.5">
                        {getNotificationIcon(notification.type)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <p className="font-medium text-sm text-white truncate">
                            {notification.title}
                          </p>
                          {notification.symbol && (
                            <Badge variant="outline" className="text-xs">
                              {notification.symbol}
                            </Badge>
                          )}
                          {!notification.read && (
                            <div className="w-2 h-2 bg-blue-500 rounded-full flex-shrink-0" />
                          )}
                        </div>
                        <p className="text-xs text-zinc-400 mb-1">
                          {notification.message}
                        </p>
                        <p className="text-xs text-zinc-500">
                          {formatTimestamp(notification.timestamp)}
                        </p>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="p-4 text-center text-zinc-400">
                  {isLoading ? (
                    <div className="flex items-center justify-center gap-2">
                      <RefreshCw className="h-4 w-4 animate-spin" />
                      <span>Loading...</span>
                    </div>
                  ) : (
                    <div>
                      <p>No notifications yet</p>
                      <p className="text-xs mt-1 text-zinc-500">
                        You&apos;ll see market alerts, price movements & more
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
} 