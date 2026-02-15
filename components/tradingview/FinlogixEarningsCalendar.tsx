"use client"

import { useEffect, useRef, memo, useState } from "react"

function FinlogixEarningsCalendar() {
  const containerRef = useRef<HTMLDivElement>(null)
  const [isClient, setIsClient] = useState(false)
  const initializedRef = useRef(false)

  useEffect(() => {
    setIsClient(true)
  }, [])

  useEffect(() => {
    if (!isClient) return
    
    const container = containerRef.current
    if (!container || initializedRef.current) return

    // Check if widget already exists
    if (container.querySelector('iframe')) {
      return
    }

    initializedRef.current = true

    function initializeWidget() {
      if (!window.Widget) {
        console.warn('Finlogix Widget not loaded')
        initializedRef.current = false
        return
      }

      try {
        // Match the original widget config exactly - NO container property
        window.Widget.init({
          widgetId: "caffc210-a0bd-4502-943f-5078aa5ea13c",
          type: "EarningCalendar",
          language: "en",
          importanceOptions: ["low", "medium", "high"],
          dateRangeOptions: ["recentAndNext", "today", "tomorrow", "thisWeek", "nextWeek", "thisMonth"],
          isAdaptive: true
        })
      } catch (error) {
        console.error('Error initializing Finlogix widget:', error)
        initializedRef.current = false
      }
    }

    // Check if script already loaded
    const existingScript = document.querySelector('script[src="https://widget.finlogix.com/Widget.js"]')
    
    if (window.Widget) {
      initializeWidget()
    } else if (!existingScript) {
      const script = document.createElement("script")
      script.src = "https://widget.finlogix.com/Widget.js"
      script.type = "text/javascript"
      script.onload = initializeWidget
      document.head.appendChild(script)
    } else {
      // Script is loading, wait for it
      const checkWidget = setInterval(() => {
        if (window.Widget) {
          clearInterval(checkWidget)
          initializeWidget()
        }
      }, 100)
      setTimeout(() => clearInterval(checkWidget), 10000)
    }

    return () => {
      initializedRef.current = false
    }
  }, [isClient])

  if (!isClient) {
    return (
      <div className="rounded-lg bg-zinc-900 border border-zinc-800 shadow-lg overflow-hidden" style={{ height: 550 }}>
        <div 
          className="w-full h-full flex items-center justify-center"
        >
          <div className="text-zinc-500 text-sm">Loading earnings calendar...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="rounded-lg bg-zinc-900 border border-zinc-800 shadow-lg overflow-hidden" style={{ height: 550 }}>
      {/* Use the exact class name the widget expects */}
      <div 
        ref={containerRef}
        className="finlogix-container w-full h-full" 
      />
    </div>
  )
}

export default memo(FinlogixEarningsCalendar)

declare global {
  interface Window {
    Widget: {
      init: (config: {
        widgetId: string
        type: string
        language: string
        importanceOptions: string[]
        dateRangeOptions: string[]
        isAdaptive: boolean
      }) => void
    }
  }
}
