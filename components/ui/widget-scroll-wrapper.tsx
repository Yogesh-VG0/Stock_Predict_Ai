"use client"

import { useState, useRef, useCallback, useEffect } from "react"

interface WidgetScrollWrapperProps {
  children: React.ReactNode
  className?: string
  style?: React.CSSProperties
}

/**
 * WidgetScrollWrapper — prevents embedded iframes (TradingView, Finlogix, etc.)
 * from hijacking page scroll on both mobile and desktop.
 *
 * Approach: CSS `pointer-events: none !important` on child iframes when the
 * wrapper is inactive. This means wheel / touch events pass straight through
 * the iframe to the parent DOM → native smooth scrolling with zero JS in the
 * scroll path (no onWheel, no scrollBy, no jitter).
 *
 *  Desktop  — click anywhere on widget area to activate; mouse-leave deactivates.
 *  Mobile   — long-press (500 ms hold) to activate; auto-deactivates after 6 s.
 *  Keyboard — Enter / Space activates.
 */
export default function WidgetScrollWrapper({
  children,
  className = "",
  style,
}: WidgetScrollWrapperProps) {
  const [isActive, setIsActive] = useState(false)
  const [showHint, setShowHint] = useState(false)
  const [hasShownHint, setHasShownHint] = useState(false)
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const hintTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const longPressRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const lastTouchRef = useRef(0)

  // Detect touch device once on mount
  const isTouchDevice = useRef(false)
  useEffect(() => {
    isTouchDevice.current =
      "ontouchstart" in window || navigator.maxTouchPoints > 0
  }, [])

  // ── Activate widget interaction ──
  const activate = useCallback(() => {
    setIsActive(true)
    setShowHint(false)
    if (hintTimeoutRef.current) clearTimeout(hintTimeoutRef.current)
    if (timeoutRef.current) clearTimeout(timeoutRef.current)

    // Desktop: deactivates on mouseLeave only (no timer)
    // Touch:   auto-deactivate after 6 s
    if (isTouchDevice.current) {
      timeoutRef.current = setTimeout(() => setIsActive(false), 6000)
    }
  }, [])

  // ── Deactivate ──
  const deactivate = useCallback(() => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current)
    if (longPressRef.current) clearTimeout(longPressRef.current)
    setIsActive(false)
    setShowHint(false)
  }, [])

  // ── Desktop: show hint once on first hover ──
  const handleMouseEnter = useCallback(() => {
    if (!isActive && !hasShownHint) {
      setShowHint(true)
      setHasShownHint(true)
      hintTimeoutRef.current = setTimeout(() => setShowHint(false), 1500)
    }
  }, [isActive, hasShownHint])

  const handleMouseLeave = useCallback(() => {
    deactivate()
  }, [deactivate])

  // ── Desktop click activates (ignore touch-emulated clicks) ──
  const handleClick = useCallback(() => {
    if (Date.now() - lastTouchRef.current < 500) return
    activate()
  }, [activate])

  // ── Mobile: long-press (500 ms) to activate ──
  const handleTouchStart = useCallback(
    (e: React.TouchEvent) => {
      lastTouchRef.current = Date.now()

      // Show hint on first touch
      if (!hasShownHint) {
        setShowHint(true)
        setHasShownHint(true)
        hintTimeoutRef.current = setTimeout(() => setShowHint(false), 1500)
      }

      // Start long-press timer
      longPressRef.current = setTimeout(() => {
        activate()
        if (navigator.vibrate) navigator.vibrate(30)
      }, 500)
    },
    [activate, hasShownHint],
  )

  const handleTouchMove = useCallback(() => {
    // Finger moved → it's a scroll gesture, cancel long-press
    if (longPressRef.current) {
      clearTimeout(longPressRef.current)
      longPressRef.current = null
    }
  }, [])

  const handleTouchEnd = useCallback(() => {
    if (longPressRef.current) {
      clearTimeout(longPressRef.current)
      longPressRef.current = null
    }
  }, [])

  // ── Keyboard: Enter / Space activates ──
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault()
        activate()
      }
    },
    [activate],
  )

  // Cleanup all timers
  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
      if (hintTimeoutRef.current) clearTimeout(hintTimeoutRef.current)
      if (longPressRef.current) clearTimeout(longPressRef.current)
    }
  }, [])

  return (
    <div
      className={`widget-scroll-wrapper ${isActive ? "widget-active" : ""} relative ${className}`}
      style={style}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onClick={handleClick}
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
      onKeyDown={handleKeyDown}
      role="group"
      tabIndex={0}
      aria-label="Interactive widget — click or hold to activate"
    >
      {children}

      {/* Hint badge — pointer-events-none so it never blocks anything */}
      {showHint && !isActive && (
        <div className="absolute inset-0 z-50 flex items-center justify-center pointer-events-none">
          <span className="text-xs text-white/70 bg-black/60 px-3 py-1.5 rounded-full backdrop-blur-sm border border-white/10 select-none animate-pulse">
            {isTouchDevice.current ? "Hold to interact" : "Click to interact"}
          </span>
        </div>
      )}
    </div>
  )
}
