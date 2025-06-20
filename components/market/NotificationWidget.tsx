"use client"

import { useState, useEffect } from "react"
import { Bell, X, AlertTriangle, TrendingUp, TrendingDown, Info } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

interface Notification {
  id: string
  type: "alert" | "news" | "gain" | "loss" | "info"
  title: string
  message: string
  timestamp: Date
  symbol?: string
  read: boolean
}

export default function NotificationWidget() {
  const [notifications, setNotifications] = useState<Notification[]>([])
  const [showNotifications, setShowNotifications] = useState(false)
  const [unreadCount, setUnreadCount] = useState(0)

  // Mock function to generate notifications - replace with real API
  const generateMockNotifications = (): Notification[] => {
    const now = new Date()
    return [
      {
        id: "1",
        type: "gain",
        title: "AAPL Breaking Out",
        message: "Apple Inc. has broken above $195 resistance level with high volume",
        timestamp: new Date(now.getTime() - 5 * 60000), // 5 minutes ago
        symbol: "AAPL",
        read: false
      },
      {
        id: "2",
        type: "alert",
        title: "Market Open Alert",
        message: "US markets are now open for trading",
        timestamp: new Date(now.getTime() - 30 * 60000), // 30 minutes ago
        read: false
      },
      {
        id: "3",
        type: "news",
        title: "Fed Meeting Today",
        message: "FOMC meeting scheduled at 2:00 PM EST with rate decision expected",
        timestamp: new Date(now.getTime() - 60 * 60000), // 1 hour ago
        read: true
      },
      {
        id: "4",
        type: "loss",
        title: "TSLA Declining",
        message: "Tesla stock down 3.2% on production concerns",
        timestamp: new Date(now.getTime() - 90 * 60000), // 1.5 hours ago
        symbol: "TSLA",
        read: false
      },
      {
        id: "5",
        type: "info",
        title: "Market Sentiment",
        message: "Fear & Greed Index shows 'Neutral' market sentiment",
        timestamp: new Date(now.getTime() - 2 * 60 * 60000), // 2 hours ago
        read: true
      }
    ]
  }

  useEffect(() => {
    // Load initial notifications
    const mockNotifications = generateMockNotifications()
    setNotifications(mockNotifications)
    setUnreadCount(mockNotifications.filter(n => !n.read).length)

    // Simulate real-time notifications
    const interval = setInterval(() => {
      const newNotification: Notification = {
        id: Math.random().toString(),
        type: Math.random() > 0.5 ? "info" : "alert",
        title: "Market Update",
        message: `New market activity detected at ${new Date().toLocaleTimeString()}`,
        timestamp: new Date(),
        read: false
      }
      
      setNotifications(prev => [newNotification, ...prev.slice(0, 9)]) // Keep last 10
      setUnreadCount(prev => prev + 1)
    }, 60000) // Every minute

    return () => clearInterval(interval)
  }, [])

  const getNotificationIcon = (type: Notification["type"]) => {
    switch (type) {
      case "alert":
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
    setNotifications(prev => 
      prev.map(n => n.id === id ? { ...n, read: true } : n)
    )
    setUnreadCount(prev => Math.max(0, prev - 1))
  }

  const markAllAsRead = () => {
    setNotifications(prev => prev.map(n => ({ ...n, read: true })))
    setUnreadCount(0)
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
                  No notifications
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
} 