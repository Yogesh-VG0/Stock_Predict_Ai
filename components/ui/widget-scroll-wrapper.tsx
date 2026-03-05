"use client"

import { useState, useRef, useCallback, useEffect } from "react"

type WidgetMode = "always" | "hover"

interface WidgetScrollWrapperProps {
  children: React.ReactNode
  className?: string
  style?: React.CSSProperties
  /**
   * `"always"` — iframe is always interactive (for click-only widgets like
   *              Gainers/Losers that have no internal scroll).
   * `"hover"`  — Desktop: hover to activate, mouse-leave to deactivate.
   *              Mobile: long-press (500 ms) to activate, auto-deactivates after 6 s.
   *              Default mode.
   */
  mode?: WidgetMode
}

/**
 * WidgetScrollWrapper
 *
 * Prevents embedded iframes from hijacking page scroll while letting users
 * interact with widgets when they intend to.
 *
 * Uses CSS `pointer-events: none` on child iframes (via `.widget-scroll-wrapper`
 * class rules in globals.css) when the wrapper is not active. Events fall through
 * to the parent DOM → native smooth scrolling, zero JS in the scroll path.
 */
export default function WidgetScrollWrapper({
  children,
  className = "",
  style,
  mode = "hover",
}: WidgetScrollWrapperProps) {
  // "always" mode: render with active class permanently, no event handlers
  if (mode === "always") {
    return (
      <div
        className={`widget-scroll-wrapper widget-active relative ${className}`}
        style={style}
      >
        {children}
      </div>
    )
  }

  return <HoverWrapper className={className} style={style}>{children}</HoverWrapper>
}

// ── Hover mode component (separated so hooks are unconditional) ──
function HoverWrapper({
  children,
  className = "",
  style,
}: {
  children: React.ReactNode
  className?: string
  style?: React.CSSProperties
}) {
  const [isActive, setIsActive] = useState(false)
  const [showHint, setShowHint] = useState(false)
  const [hasShownHint, setHasShownHint] = useState(false)
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const hintTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const longPressRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const lastTouchRef = useRef(0)
  const canHover = useRef(true)

  useEffect(() => {
    canHover.current = window.matchMedia(
      "(hover: hover) and (pointer: fine)",
    ).matches
  }, [])

  // ── Activate ──
  const activate = useCallback(() => {
    setIsActive(true)
    setShowHint(false)
    if (hintTimeoutRef.current) clearTimeout(hintTimeoutRef.current)
    if (timeoutRef.current) clearTimeout(timeoutRef.current)

    // Only auto-deactivate on non-hover (touch) devices
    if (!canHover.current) {
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

  // ── Desktop: hover activates (no click needed) ──
  const handleMouseEnter = useCallback(() => {
    if (!canHover.current) return
    activate()
    if (!hasShownHint) {
      setShowHint(true)
      setHasShownHint(true)
      hintTimeoutRef.current = setTimeout(() => setShowHint(false), 1000)
    }
  }, [activate, hasShownHint])

  const handleMouseLeave = useCallback(() => {
    if (!canHover.current) return
    deactivate()
  }, [deactivate])

  // ── Mobile: long-press (500 ms) to activate ──
  const handleTouchStart = useCallback(
    (e: React.TouchEvent) => {
      lastTouchRef.current = Date.now()

      if (!hasShownHint) {
        setShowHint(true)
        setHasShownHint(true)
        hintTimeoutRef.current = setTimeout(() => setShowHint(false), 1500)
      }

      longPressRef.current = setTimeout(() => {
        activate()
        if (navigator.vibrate) navigator.vibrate(30)
      }, 500)
    },
    [activate, hasShownHint],
  )

  const handleTouchMove = useCallback(() => {
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

  // ── Keyboard: Enter / Space ──
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault()
        activate()
      }
    },
    [activate],
  )

  // Cleanup timers
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
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
      onKeyDown={handleKeyDown}
      role="group"
      tabIndex={0}
      aria-label="Interactive widget — hover or hold to activate"
    >
      {children}

      {/* Hint badge — pointer-events-none so it never blocks anything */}
      {showHint && !isActive && (
        <div className="absolute inset-0 z-50 flex items-center justify-center pointer-events-none">
          <span className="text-xs text-white/70 bg-black/60 px-3 py-1.5 rounded-full backdrop-blur-sm border border-white/10 select-none animate-pulse">
            Hold to interact
          </span>
        </div>
      )}
    </div>
  )
}
