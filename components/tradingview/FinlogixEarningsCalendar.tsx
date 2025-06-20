"use client"

import { useEffect, useRef } from "react"

export default function FinlogixEarningsCalendar() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    // Clear any existing content and widgets
    container.innerHTML = ""
    
    // Remove any existing finlogix containers globally to prevent duplicates
    const existingContainers = document.querySelectorAll('.finlogix-container')
    existingContainers.forEach(elem => {
      if (elem !== container) {
        elem.remove()
      }
    })

    // Create a unique finlogix container div
    const finlogixDiv = document.createElement("div")
    finlogixDiv.className = "finlogix-container"
    finlogixDiv.id = `finlogix-${Date.now()}` // Unique ID to prevent conflicts
    container.appendChild(finlogixDiv)

    // Check if Widget script is already loaded
    if (window.Widget) {
      // Script already loaded, initialize immediately
      initializeWidget()
    } else {
      // Load the Finlogix widget script
      const script = document.createElement("script")
      script.src = "https://widget.finlogix.com/Widget.js"
      script.type = "text/javascript"
      script.onload = initializeWidget
      document.head.appendChild(script)
    }

    function initializeWidget() {
      if (window.Widget && finlogixDiv) {
        try {
          window.Widget.init({
            widgetId: "caffc210-a0bd-4502-943f-5078aa5ea13c",
            type: "EarningCalendar",
            language: "en",
            showBrand: false, // Hide brand to reduce clutter
            isShowTradeButton: false, // Hide trade button for cleaner look
            isShowBeneathLink: false, // Hide beneath link
            isShowDataFromACYInfo: false, // Hide ACY info
            importanceOptions: [
              "low",
              "medium", 
              "high"
            ],
            dateRangeOptions: [
              "recentAndNext",
              "today",
              "tomorrow",
              "thisWeek",
              "nextWeek"
            ],
            isAdaptive: true,
            colorScheme: "dark",
            darkMode: true,
            container: finlogixDiv.id,
            width: "100%", // Full width of container
            height: "500", // Fixed height for sidebar
            compact: true, // Enable compact mode for better sidebar fit
            fontSize: "12px" // Smaller font for sidebar space
          })
        } catch (error) {
          console.error('Error initializing Finlogix widget:', error)
        }
      }
    }

    // Cleanup function
    return () => {
      if (container) {
        container.innerHTML = ""
      }
    }
  }, [])

  return (
    <div className="rounded-lg bg-zinc-900 border border-zinc-800 shadow-lg overflow-hidden">
      <div ref={containerRef} className="min-h-[500px] w-full finlogix-dark-theme" />
      <style jsx>{`
        .finlogix-dark-theme {
          background-color: #18181b !important;
          color: #ffffff !important;
          width: 100% !important;
          height: 500px !important;
          overflow: hidden !important;
        }
        .finlogix-dark-theme * {
          background-color: #18181b !important;
          color: #ffffff !important;
          border-color: #27272a !important;
        }
        .finlogix-container {
          background-color: #18181b !important;
          color: #ffffff !important;
          width: 100% !important;
          height: 500px !important;
          overflow: visible !important;
        }
        .finlogix-container table {
          background-color: #18181b !important;
          color: #ffffff !important;
          width: 100% !important;
          font-size: 12px !important;
        }
        .finlogix-container td, .finlogix-container th {
          background-color: #18181b !important;
          color: #ffffff !important;
          border-color: #27272a !important;
          padding: 4px 6px !important;
          font-size: 12px !important;
        }
        .finlogix-container iframe {
          width: 100% !important;
          height: 500px !important;
          overflow: hidden !important;
        }
        /* Hide any outer container scrollbars */
        .finlogix-dark-theme::-webkit-scrollbar {
          display: none !important;
        }
        .finlogix-dark-theme {
          -ms-overflow-style: none !important;
          scrollbar-width: none !important;
        }
        /* Responsive adjustments for smaller screens */
        @media (max-width: 1024px) {
          .finlogix-container {
            height: 400px !important;
          }
          .finlogix-dark-theme {
            height: 400px !important;
          }
          .finlogix-container iframe {
            height: 400px !important;
          }
        }
      `}</style>
    </div>
  )
}

// Extend the Window interface to include Widget
declare global {
  interface Window {
    Widget: {
      init: (config: any) => void
    }
  }
} 