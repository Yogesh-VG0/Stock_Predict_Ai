"use client"

import React, { useState, useEffect, useRef, useCallback, useMemo } from "react"
import { X, Building2 } from "lucide-react"
import { AnimatePresence, motion } from "framer-motion"

// ── Colors MUST match TradingHoursBar.tsx exactly ──
const COLORS = {
  preMarket:  "#ff4fd8", // Neon pink
  regular:    "#39ff14", // Neon green
  afterHours: "#ffb347", // Neon orange
  overnight:  "#7c3aed", // Neon purple
} as const

// Sessions defined in ET minutes-since-midnight
const ET_SESSIONS = [
  { name: "PRE-MARKET",    startH: 4,  startM: 0,  endH: 9,  endM: 30, color: COLORS.preMarket,  label: "Pre-Market" },
  { name: "REGULAR HOURS", startH: 9,  startM: 30, endH: 16, endM: 0,  color: COLORS.regular,    label: "Regular Hours" },
  { name: "AFTER HOURS",   startH: 16, startM: 0,  endH: 20, endM: 0,  color: COLORS.afterHours, label: "After Hours" },
  { name: "OVERNIGHT",     startH: 20, startM: 0,  endH: 4,  endM: 0,  color: COLORS.overnight,  label: "Overnight" },
] as const

const DAYS = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"] as const

function getETOffset(): number {
  const now = new Date()
  const utcStr = now.toLocaleString("en-US", { timeZone: "UTC" })
  const etStr = now.toLocaleString("en-US", { timeZone: "America/New_York" })
  return (new Date(etStr).getTime() - new Date(utcStr).getTime()) / 60000
}

function isDST(): boolean {
  return getETOffset() === -240
}

function getNYTime(): Date {
  return new Date(new Date().toLocaleString("en-US", { timeZone: "America/New_York" }))
}

function etToLocalMinutes(hour: number, minute: number): number {
  const now = new Date()
  const etStr = now.toLocaleDateString("en-US", { timeZone: "America/New_York" })
  const [month, day, year] = etStr.split("/").map(Number)
  const etDate = new Date(`${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}T${String(hour).padStart(2, "0")}:${String(minute).padStart(2, "0")}:00`)
  const etOffset = getETOffset()
  const utcMs = etDate.getTime() - etOffset * 60000
  const localDate = new Date(utcMs + now.getTimezoneOffset() * -60000)
  return localDate.getHours() * 60 + localDate.getMinutes()
}

function formatTime(minutes: number): string {
  const clamped = ((minutes % 1440) + 1440) % 1440
  const h = Math.floor(clamped / 60)
  const m = clamped % 60
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`
}

function getLocalSessionTimes() {
  return ET_SESSIONS.map((s) => ({
    ...s,
    startLocal: etToLocalMinutes(s.startH, s.startM),
    endLocal: etToLocalMinutes(s.endH, s.endM),
    startDisplay: formatTime(etToLocalMinutes(s.startH, s.startM)),
    endDisplay: formatTime(etToLocalMinutes(s.endH, s.endM)),
  }))
}

// Compute colored segment positions as % of 24h
function getSegments() {
  const totalMin = 1440
  const sessions = getLocalSessionTimes()
  const segments: Array<{ name: string; color: string; startPct: number; widthPct: number }> = []

  for (const s of sessions) {
    const start = s.startLocal
    const end = s.endLocal
    if (start < end) {
      segments.push({ name: s.name, color: s.color, startPct: (start / totalMin) * 100, widthPct: ((end - start) / totalMin) * 100 })
    } else {
      segments.push({ name: s.name, color: s.color, startPct: (start / totalMin) * 100, widthPct: ((totalMin - start) / totalMin) * 100 })
      segments.push({ name: s.name, color: s.color, startPct: 0, widthPct: (end / totalMin) * 100 })
    }
  }
  return segments
}

function getCurrentTimePct(): number {
  const now = new Date()
  return ((now.getHours() * 60 + now.getMinutes()) / 1440) * 100
}

function getDayOfWeekET(): number {
  return getNYTime().getDay()
}

function isTradingDay(dayIdx: number): boolean {
  return dayIdx >= 1 && dayIdx <= 5
}

interface TradingHoursPopupProps {
  open: boolean
  onClose: () => void
}

export default function TradingHoursPopup({ open, onClose }: TradingHoursPopupProps) {
  const [dragMinutes, setDragMinutes] = useState<number | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isMobile, setIsMobile] = useState(false)
  const [now, setNow] = useState(new Date())
  const timelineRef = useRef<HTMLDivElement>(null)
  const dragStateRef = useRef<{ isDragging: boolean; rect: DOMRect | null }>({ isDragging: false, rect: null })

  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth < 768)
    check()
    window.addEventListener("resize", check)
    return () => window.removeEventListener("resize", check)
  }, [])

  useEffect(() => {
    if (!open) return
    const timer = setInterval(() => setNow(new Date()), 30000)
    return () => clearInterval(timer)
  }, [open])

  const segments = useMemo(getSegments, [now])
  const localSessions = useMemo(getLocalSessionTimes, [now])
  const currentTimePct = getCurrentTimePct()
  const todayET = getDayOfWeekET()

  // Constants for layout calculations
  const BAR_LEFT_OFFSET = 64
  const BAR_RIGHT_PAD = 16

  // Handle mouse down on timeline to start dragging
  const handleTimelineMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    
    // Prevent multiple simultaneous drags
    if (dragStateRef.current.isDragging) return
    
    try {
      const rect = timelineRef.current?.getBoundingClientRect()
      if (!rect) return
      
      const startX = e.clientX
      const startPct = Math.max(0, Math.min(1, (startX - rect.left) / rect.width))
      const startMinutes = Math.round(startPct * 1440)
      
      setDragMinutes(startMinutes)
      setIsDragging(true)
      dragStateRef.current = { isDragging: true, rect }
      
      const handleMouseMove = (moveEvent: MouseEvent) => {
        const currentRect = dragStateRef.current.rect
        if (!currentRect) return
        
        const currentPct = Math.max(0, Math.min(1, (moveEvent.clientX - currentRect.left) / currentRect.width))
        const currentMinutes = Math.round(currentPct * 1440)
        setDragMinutes(currentMinutes)
      }
      
      const handleMouseUp = () => {
        setIsDragging(false)
        setDragMinutes(null)
        dragStateRef.current = { isDragging: false, rect: null }
        document.removeEventListener("mousemove", handleMouseMove)
        document.removeEventListener("mouseup", handleMouseUp)
      }
      
      document.addEventListener("mousemove", handleMouseMove)
      document.addEventListener("mouseup", handleMouseUp)
    } catch (error) {
      console.error('Error starting mouse drag:', error)
    }
  }, [])

  // Handle touch events for mobile
  const handleTimelineTouchStart = useCallback((e: React.TouchEvent) => {
    e.preventDefault()
    
    // Prevent multiple simultaneous drags
    if (dragStateRef.current.isDragging) return
    
    try {
      const rect = timelineRef.current?.getBoundingClientRect()
      if (!rect) return
      
      const touch = e.touches[0]
      const startPct = Math.max(0, Math.min(1, (touch.clientX - rect.left) / rect.width))
      const startMinutes = Math.round(startPct * 1440)
      
      setDragMinutes(startMinutes)
      setIsDragging(true)
      dragStateRef.current = { isDragging: true, rect }
      
      const handleTouchMove = (moveEvent: TouchEvent) => {
        const currentRect = dragStateRef.current.rect
        if (!currentRect) return
        
        const touch = moveEvent.touches[0]
        const currentPct = Math.max(0, Math.min(1, (touch.clientX - currentRect.left) / currentRect.width))
        const currentMinutes = Math.round(currentPct * 1440)
        setDragMinutes(currentMinutes)
      }
      
      const handleTouchEnd = () => {
        setIsDragging(false)
        setDragMinutes(null)
        dragStateRef.current = { isDragging: false, rect: null }
        document.removeEventListener("touchmove", handleTouchMove)
        document.removeEventListener("touchend", handleTouchEnd)
      }
      
      document.addEventListener("touchmove", handleTouchMove, { passive: false })
      document.addEventListener("touchend", handleTouchEnd)
    } catch (error) {
      console.error('Error starting touch drag:', error)
    }
  }, [])

  // Escape key
  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose() }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [open, onClose])

  // Cleanup event listeners on unmount
  useEffect(() => {
    return () => {
      // Clean up any remaining event listeners
      const events = ['mousemove', 'mouseup', 'touchmove', 'touchend']
      events.forEach(eventType => {
        // Remove all listeners for this event type
        const listeners = (document as any).__tradingHoursListeners || []
        listeners.forEach((listener: EventListener) => {
          document.removeEventListener(eventType, listener)
        })
      })
      ;(document as any).__tradingHoursListeners = []
      dragStateRef.current = { isDragging: false, rect: null }
    }
  }, [])

  // Handle keyboard navigation for the time selector
  const handleKeyDown = useCallback((e: React.KeyboardEvent, currentIsToday: boolean) => {
    if (!currentIsToday) return
    
    const currentMinutes = dragMinutes !== null ? dragMinutes : Math.round((currentTimePct / 100) * 1440)
    let newMinutes = currentMinutes
    
    switch (e.key) {
      case 'ArrowLeft':
        newMinutes = Math.max(0, currentMinutes - 15) // 15 minute steps
        break
      case 'ArrowRight':
        newMinutes = Math.min(1440, currentMinutes + 15)
        break
      case 'Home':
        newMinutes = 0
        break
      case 'End':
        newMinutes = 1440
        break
      default:
        return
    }
    
    e.preventDefault()
    setDragMinutes(newMinutes)
  }, [dragMinutes, currentTimePct])

  // The % position of the marker line — either dragged or current time
  const markerPct = dragMinutes !== null ? (dragMinutes / 1440) * 100 : currentTimePct

  // The time string displayed below today's bar
  const displayTimeStr = dragMinutes !== null
    ? formatTime(dragMinutes)
    : formatTime(now.getHours() * 60 + now.getMinutes())

  if (!open) return null

  // The left offset for the vertical line accounts for the day label width + gap + padding
  // Day label area = w-10 (40px) + gap-3 (12px) + px-3 (12px left) = 64px
  const barLeftOffset = BAR_LEFT_OFFSET
  // Right padding = px-3 (12px) = 12px
  const barRightPad = BAR_RIGHT_PAD

  const popupContent = (
    <div className="w-full max-w-lg mx-auto select-none">
      {/* Header */}
      <div className="flex items-center justify-between px-5 pt-4 pb-3">
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg hover:bg-zinc-800 transition-colors text-zinc-400 hover:text-white"
        >
          <X className="h-5 w-5" />
        </button>
        <h2 className="text-lg font-bold text-white">Trading hours</h2>
        <div className="w-8" />
      </div>

      {/* Weekly schedule */}
      <div className="px-3 pb-2 space-y-1 relative">
        
        {DAYS.map((day, idx) => {
          const isToday = idx === todayET
          const trading = isTradingDay(idx)

          return (
            <div key={day}>
              <div
                className={`flex items-center gap-3 rounded-lg px-3 transition-colors ${
                  isToday ? "bg-teal-900/40 border border-teal-700/40 py-2.5" : "py-1.5"
                }`}
              >
                <span className={`text-xs font-bold w-10 flex-shrink-0 ${isToday ? "text-white" : "text-zinc-500"}`}>
                  {day}
                </span>

                {/* Timeline bar */}
                <div
                  ref={isToday ? timelineRef : undefined}
                  className={`flex-1 rounded-full relative overflow-hidden bg-zinc-800 ${
                    isToday ? "h-3 cursor-grab active:cursor-grabbing" : "h-2"
                  }`}
                  onMouseDown={isToday ? handleTimelineMouseDown : undefined}
                  onTouchStart={isToday ? handleTimelineTouchStart : undefined}
                >
                  {trading ? (
                    <>
                      {segments.map((seg, i) => (
                        <div
                          key={`${day}-${i}`}
                          className="absolute top-0 h-full"
                          style={{
                            left: `${seg.startPct}%`,
                            width: `${seg.widthPct}%`,
                            backgroundColor: seg.color,
                            opacity: 0.85,
                          }}
                        />
                      ))}
                      {/* Draggable indicator on today's bar */}
                      {isToday && (
                        <div
                          className="absolute top-0 bottom-0 w-1 bg-white cursor-grab active:cursor-grabbing"
                          style={{
                            left: `${markerPct}%`,
                            transform: 'translateX(-50%)',
                            boxShadow: '0 0 4px rgba(0,0,0,0.5)'
                          }}
                          role="slider"
                          aria-label="Time selector"
                          aria-valuemin={0}
                          aria-valuemax={1440}
                          aria-valuenow={dragMinutes !== null ? dragMinutes : Math.round((currentTimePct / 100) * 1440)}
                          tabIndex={0}
                          onKeyDown={(e) => handleKeyDown(e, isToday)}
                        />
                      )}
                    </>
                  ) : (
                    <div className="absolute inset-0 bg-zinc-700/30 rounded-full" />
                  )}
                </div>

                              </div>

              {/* Triangle marker + time underneath today's bar */}
              {isToday && (
                <div className="relative" style={{ height: 28, marginLeft: barLeftOffset, marginRight: barRightPad }}>
                  {/* Small white downward triangle */}
                  <div
                    className="absolute -top-0.5 pointer-events-none"
                    style={{
                      left: `${markerPct}%`,
                      transform: "translateX(-50%)",
                    }}
                  >
                    <div
                      style={{
                        width: 0,
                        height: 0,
                        borderLeft: "6px solid transparent",
                        borderRight: "6px solid transparent",
                        borderTop: "8px solid white",
                      }}
                    />
                  </div>

                  {/* Time below the triangle */}
                  <div
                    className="absolute top-2 pointer-events-none"
                    style={{
                      left: `${markerPct}%`,
                      transform: "translateX(-50%)",
                    }}
                  >
                    <span className="text-[11px] font-mono text-zinc-300 font-medium">
                      {displayTimeStr}
                    </span>
                  </div>
                </div>
              )}
            </div>
          )
        })}

              </div>

      {/* Legend */}
      <div className="px-5 pb-5">
        <div className="flex items-center justify-between text-[11px]">
          {localSessions.map((s) => (
            <div key={s.name} className="flex flex-col items-center gap-1">
              <span className="font-semibold tracking-wide" style={{ color: s.color }}>
                {s.label.toUpperCase()}
              </span>
              <div className="h-1 w-6 rounded-full" style={{ backgroundColor: s.color }} />
            </div>
          ))}
        </div>

        {/* Session times detail */}
        <div className="mt-4 space-y-2">
          {localSessions.map((s) => (
            <div key={s.name} className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-2">
                <Building2 className="h-4 w-4" style={{ color: s.color }} />
                <span className="font-medium" style={{ color: s.color }}>{s.label}</span>
              </div>
              <span className="text-zinc-400 font-mono text-xs">
                {s.startDisplay} – {s.endDisplay}
              </span>
            </div>
          ))}
        </div>

        {/* DST notice */}
        <div className="mt-3 text-[10px] text-zinc-600 text-center">
          Times shown in your local timezone • {isDST() ? "EDT (Daylight Saving)" : "EST (Standard)"}
        </div>
      </div>
    </div>
  )

  // Mobile: bottom sheet (no swipe-to-close — only X button)
  if (isMobile) {
    return (
      <AnimatePresence>
        {open && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50"
              onClick={onClose}
            />
            <motion.div
              initial={{ y: "100%" }}
              animate={{ y: 0 }}
              exit={{ y: "100%" }}
              transition={{ type: "spring", damping: 30, stiffness: 300 }}
              className="fixed bottom-0 left-0 right-0 z-50 bg-zinc-900 border-t border-zinc-700 rounded-t-2xl max-h-[85vh] overflow-y-auto"
            >
              {popupContent}
            </motion.div>
          </>
        )}
      </AnimatePresence>
    )
  }

  // Desktop: centered modal
  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50"
            onClick={onClose}
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            onClick={(e) => e.target === e.currentTarget && onClose()}
          >
            <div className="bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-lg overflow-hidden">
              {popupContent}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
