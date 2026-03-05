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
 * Behaviour:
 *  - Transparent overlay sits on top of widget by default.
 *  - Wheel events on the overlay explicitly scroll the page's main container
 *    (the `<main>` with `overflow-y-auto`) so trackpad / mouse-wheel never
 *    feels "stuck".
 *  - Click (desktop) or tap (mobile) hides the overlay → widget interactive.
 *  - Desktop: overlay returns on mouse-leave only (no timer).
 *  - Mobile (touch device): overlay auto-restores after 5 s idle.
 *  - Keyboard accessible: Enter / Space also activates.
 */
export default function WidgetScrollWrapper({
  children,
  className = "",
  style,
}: WidgetScrollWrapperProps) {
  const [isInteracting, setIsInteracting] = useState(false)
  const [showHint, setShowHint] = useState(false)
  const [hasShownHint, setHasShownHint] = useState(false)
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const hintTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const touchStartRef = useRef<{ x: number; y: number } | null>(null)

  // Detect touch device once on mount
  const isTouchDevice = useRef(false)
  useEffect(() => {
    isTouchDevice.current =
      "ontouchstart" in window || navigator.maxTouchPoints > 0
  }, [])

  // ── Activate: hide overlay ──
  const activateWidget = useCallback(() => {
    setIsInteracting(true)
    setShowHint(false)
    if (hintTimeoutRef.current) clearTimeout(hintTimeoutRef.current)
    if (timeoutRef.current) clearTimeout(timeoutRef.current)

    // Only auto-restore on touch devices; desktop restores on mouseLeave
    if (isTouchDevice.current) {
      timeoutRef.current = setTimeout(() => setIsInteracting(false), 5000)
    }
  }, [])

  // ── Deactivate: restore overlay immediately ──
  const deactivateWidget = useCallback(() => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current)
    setIsInteracting(false)
    setShowHint(false)
  }, [])

  // ── Desktop hover: show hint once per mount ──
  const handleMouseEnter = useCallback(() => {
    if (!isInteracting && !hasShownHint) {
      setShowHint(true)
      setHasShownHint(true)
      hintTimeoutRef.current = setTimeout(() => setShowHint(false), 1200)
    }
  }, [isInteracting, hasShownHint])

  const handleMouseLeave = useCallback(() => {
    deactivateWidget()
  }, [deactivateWidget])

  // ── Wheel: explicitly scroll main container so trackpads feel smooth ──
  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      if (isInteracting) return
      // The main scroll container is the closest <main> ancestor
      const main = (e.currentTarget as HTMLElement).closest("main")
      if (main) {
        main.scrollBy({ top: e.deltaY, left: e.deltaX, behavior: "auto" })
      }
    },
    [isInteracting],
  )

  // ── Mobile: detect tap vs swipe on overlay ──
  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    const touch = e.touches[0]
    touchStartRef.current = { x: touch.clientX, y: touch.clientY }
  }, [])

  const handleTouchEnd = useCallback(
    (e: React.TouchEvent) => {
      if (!touchStartRef.current) return
      const touch = e.changedTouches[0]
      const dx = Math.abs(touch.clientX - touchStartRef.current.x)
      const dy = Math.abs(touch.clientY - touchStartRef.current.y)
      // Treat as tap only if finger moved < 10 px
      if (dx < 10 && dy < 10) {
        activateWidget()
      }
      touchStartRef.current = null
    },
    [activateWidget],
  )

  // ── Keyboard: Enter / Space activates ──
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault()
        activateWidget()
      }
    },
    [activateWidget],
  )

  // Cleanup timers
  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
      if (hintTimeoutRef.current) clearTimeout(hintTimeoutRef.current)
    }
  }, [])

  return (
    <div
      className={`widget-scroll-wrapper relative ${className}`}
      style={style}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {children}

      {/* Transparent overlay — blocks iframe events when not interacting */}
      {!isInteracting && (
        <div
          className="widget-scroll-overlay absolute inset-0 z-50 cursor-pointer"
          role="button"
          tabIndex={0}
          aria-label="Activate widget interaction"
          onClick={activateWidget}
          onWheel={handleWheel}
          onKeyDown={handleKeyDown}
          onTouchStart={handleTouchStart}
          onTouchEnd={handleTouchEnd}
        >
          {showHint && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/10 transition-opacity duration-200">
              <span className="text-xs text-white/70 bg-black/50 px-3 py-1.5 rounded-full backdrop-blur-sm border border-white/10 select-none">
                Click to interact
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
