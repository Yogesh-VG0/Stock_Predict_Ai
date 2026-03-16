"use client"

import React, { useState, useEffect, useRef, useCallback, useMemo } from "react"
import { X, Building2, Sun, Moon, Sunrise } from "lucide-react"
import { AnimatePresence, motion } from "framer-motion"

// ── US market sessions in ET (Eastern Time) ──
// These are the canonical ET times. The component auto-adjusts for DST
// by using Intl.DateTimeFormat with America/New_York timezone.
const ET_SESSIONS = [
  { name: "PRE-MARKET",    startET: [4, 0],  endET: [9, 30],  color: "#ff9500", icon: Sunrise,   label: "Pre-Market" },
  { name: "REGULAR HOURS", startET: [9, 30],  endET: [16, 0],  color: "#39ff14", icon: Sun,       label: "Regular Hours" },
  { name: "AFTER HOURS",   startET: [16, 0],  endET: [20, 0],  color: "#ff4fd8", icon: Building2, label: "After Hours" },
  { name: "OVERNIGHT",     startET: [20, 0],  endET: [4, 0],   color: "#5b8dee", icon: Moon,      label: "Overnight" },
] as const

const DAYS = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"] as const

// Convert ET time to user's local minutes-since-midnight
function etToLocalMinutes(hour: number, minute: number): number {
  // Create a date in ET, then read back in local time
  const now = new Date()
  const etStr = now.toLocaleDateString("en-US", { timeZone: "America/New_York" })
  const [month, day, year] = etStr.split("/").map(Number)
  const etDate = new Date(`${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}T${String(hour).padStart(2, "0")}:${String(minute).padStart(2, "0")}:00`)
  // Get ET offset
  const etOffset = getETOffset()
  const utcMs = etDate.getTime() - etOffset * 60000
  const localDate = new Date(utcMs + now.getTimezoneOffset() * -60000)
  return localDate.getHours() * 60 + localDate.getMinutes()
}

function getETOffset(): number {
  // Returns ET offset from UTC in minutes (e.g., -300 for EST, -240 for EDT)
  const now = new Date()
  const utcStr = now.toLocaleString("en-US", { timeZone: "UTC" })
  const etStr = now.toLocaleString("en-US", { timeZone: "America/New_York" })
  const utcDate = new Date(utcStr)
  const etDate = new Date(etStr)
  return (etDate.getTime() - utcDate.getTime()) / 60000
}

function isDST(): boolean {
  return getETOffset() === -240
}

function getNYTime(): Date {
  return new Date(new Date().toLocaleString("en-US", { timeZone: "America/New_York" }))
}

function formatTime(minutes: number): string {
  const h = Math.floor(((minutes % 1440) + 1440) % 1440 / 60)
  const m = ((minutes % 1440) + 1440) % 1440 % 60
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`
}

// Get session times in user-local timezone for display
function getLocalSessionTimes() {
  return ET_SESSIONS.map((s) => {
    const startLocal = etToLocalMinutes(s.startET[0], s.startET[1])
    const endLocal = etToLocalMinutes(s.endET[0], s.endET[1])
    return {
      ...s,
      startLocal,
      endLocal,
      startDisplay: formatTime(startLocal),
      endDisplay: formatTime(endLocal),
    }
  })
}

// For the week bar: compute segment positions as % of 24h
function getSegments() {
  const totalMin = 24 * 60
  // Convert ET session times to local
  const sessions = getLocalSessionTimes()
  const segments: Array<{
    name: string
    color: string
    startPct: number
    widthPct: number
  }> = []

  for (const s of sessions) {
    let start = s.startLocal
    let end = s.endLocal

    if (start < end) {
      segments.push({
        name: s.name,
        color: s.color,
        startPct: (start / totalMin) * 100,
        widthPct: ((end - start) / totalMin) * 100,
      })
    } else {
      // Wraps midnight
      segments.push({
        name: s.name,
        color: s.color,
        startPct: (start / totalMin) * 100,
        widthPct: ((totalMin - start) / totalMin) * 100,
      })
      segments.push({
        name: s.name,
        color: s.color,
        startPct: 0,
        widthPct: (end / totalMin) * 100,
      })
    }
  }

  return segments
}

function getCurrentTimePct(): number {
  const now = new Date()
  const mins = now.getHours() * 60 + now.getMinutes()
  return (mins / (24 * 60)) * 100
}

function getDayOfWeekET(): number {
  return getNYTime().getDay()
}

// Check if a given day has trading (Mon-Fri)
function isTradingDay(dayIdx: number): boolean {
  return dayIdx >= 1 && dayIdx <= 5
}

interface TradingHoursPopupProps {
  open: boolean
  onClose: () => void
}

export default function TradingHoursPopup({ open, onClose }: TradingHoursPopupProps) {
  const [dragMinutes, setDragMinutes] = useState<number | null>(null)
  const [isMobile, setIsMobile] = useState(false)
  const [now, setNow] = useState(new Date())
  const [touchStartY, setTouchStartY] = useState<number | null>(null)
  const [translateY, setTranslateY] = useState(0)
  const containerRef = useRef<HTMLDivElement>(null)
  const timelineRef = useRef<HTMLDivElement>(null)

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

  // Reset translate when opening
  useEffect(() => {
    if (open) setTranslateY(0)
  }, [open])

  const segments = useMemo(getSegments, [now])
  const localSessions = useMemo(getLocalSessionTimes, [now])
  const currentTimePct = getCurrentTimePct()
  const todayET = getDayOfWeekET()

  // Drag handler for the building icon
  const handleDrag = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    if (!timelineRef.current) return
    const rect = timelineRef.current.getBoundingClientRect()
    const clientX = "touches" in e ? e.touches[0].clientX : e.clientX
    const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
    setDragMinutes(Math.round(pct * 24 * 60))
  }, [])

  const handleDragEnd = useCallback(() => {
    setDragMinutes(null)
  }, [])

  // Mobile swipe-to-close
  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    setTouchStartY(e.touches[0].clientY)
  }, [])

  const handleTouchMove = useCallback((e: React.TouchEvent) => {
    if (touchStartY === null) return
    const diff = e.touches[0].clientY - touchStartY
    if (diff > 0) setTranslateY(diff)
  }, [touchStartY])

  const handleTouchEnd = useCallback(() => {
    if (translateY > 100) {
      onClose()
    }
    setTranslateY(0)
    setTouchStartY(null)
  }, [translateY, onClose])

  // Escape key
  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose()
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [open, onClose])

  const displayTimePct = dragMinutes !== null
    ? (dragMinutes / (24 * 60)) * 100
    : currentTimePct

  const displayTimeStr = dragMinutes !== null
    ? formatTime(dragMinutes)
    : formatTime(now.getHours() * 60 + now.getMinutes())

  if (!open) return null

  const popupContent = (
    <div className="w-full max-w-lg mx-auto">
      {/* Mobile drag handle */}
      {isMobile && (
        <div className="flex justify-center pt-3 pb-1">
          <div className="w-10 h-1 rounded-full bg-zinc-600" />
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between px-5 pt-4 pb-3">
        <h2 className="text-lg font-bold text-white">Trading hours</h2>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg hover:bg-zinc-800 transition-colors text-zinc-400 hover:text-white"
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      {/* Weekly schedule */}
      <div className="px-5 pb-4 space-y-2 relative">
        {/* Vertical time indicator line */}
        <div
          className="absolute top-0 bottom-16 pointer-events-none z-10"
          style={{ left: `calc(${displayTimePct}% + 20px - (${displayTimePct} * 0.4px))` }}
        >
          <div className="w-px h-full bg-zinc-400/60 mx-auto" />
        </div>

        {DAYS.map((day, idx) => {
          const isToday = idx === todayET
          const trading = isTradingDay(idx)

          return (
            <div
              key={day}
              className={`flex items-center gap-3 rounded-lg py-2 px-2 transition-colors ${
                isToday ? "bg-teal-900/40 border border-teal-700/40" : ""
              }`}
            >
              <span className={`text-xs font-bold w-8 ${isToday ? "text-white" : "text-zinc-500"}`}>
                {day}
              </span>
              <div
                ref={isToday ? timelineRef : undefined}
                className="flex-1 h-2.5 rounded-full relative overflow-hidden bg-zinc-800"
                onMouseDown={isToday ? (e) => { handleDrag(e); document.addEventListener("mousemove", handleDrag as any); document.addEventListener("mouseup", () => { handleDragEnd(); document.removeEventListener("mousemove", handleDrag as any) }, { once: true }) } : undefined}
                onTouchStart={isToday ? (e) => handleDrag(e) : undefined}
                onTouchMove={isToday ? (e) => handleDrag(e) : undefined}
                onTouchEnd={isToday ? handleDragEnd : undefined}
              >
                {trading ? (
                  segments.map((seg, i) => (
                    <div
                      key={`${day}-${i}`}
                      className="absolute top-0 h-full rounded-sm"
                      style={{
                        left: `${seg.startPct}%`,
                        width: `${seg.widthPct}%`,
                        backgroundColor: seg.color,
                        opacity: 0.85,
                      }}
                    />
                  ))
                ) : (
                  <div className="absolute inset-0 bg-zinc-700/40 rounded-full" />
                )}
              </div>

              {/* Show time on today's row */}
              {isToday && (
                <span className="text-xs font-mono text-white min-w-[40px] text-right">
                  {displayTimeStr}
                </span>
              )}
            </div>
          )
        })}

        {/* Draggable building icon */}
        <div className="relative h-8 flex items-center justify-center">
          <div
            className="absolute cursor-grab active:cursor-grabbing z-20 transition-[left] duration-100"
            style={{ left: `calc(${displayTimePct}% + 8px)` }}
          >
            <div className="flex flex-col items-center -mt-1">
              <div className="w-px h-3 bg-zinc-400/60" />
              <Building2 className="h-5 w-5 text-cyan-400 drop-shadow-[0_0_6px_rgba(34,211,238,0.5)]" />
            </div>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="px-5 pb-5">
        <div className="flex items-center justify-between text-[11px]">
          {localSessions.map((s) => {
            const Icon = s.icon
            return (
              <div key={s.name} className="flex flex-col items-center gap-1">
                <span className="font-semibold tracking-wide" style={{ color: s.color }}>
                  {s.label.toUpperCase()}
                </span>
                <div className="h-1 w-6 rounded-full" style={{ backgroundColor: s.color }} />
              </div>
            )
          })}
        </div>

        {/* Session times detail */}
        <div className="mt-4 space-y-2">
          {localSessions.map((s) => {
            const Icon = s.icon
            return (
              <div key={s.name} className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <Icon className="h-4 w-4" style={{ color: s.color }} />
                  <span className="font-medium" style={{ color: s.color }}>{s.label}</span>
                </div>
                <span className="text-zinc-400 font-mono text-xs">
                  {s.startDisplay} – {s.endDisplay}
                </span>
              </div>
            )
          })}
        </div>

        {/* DST notice */}
        <div className="mt-3 text-[10px] text-zinc-600 text-center">
          Times shown in your local timezone • {isDST() ? "EDT (Daylight Saving)" : "EST (Standard)"}
        </div>
      </div>
    </div>
  )

  // Mobile: bottom sheet
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
              ref={containerRef}
              initial={{ y: "100%" }}
              animate={{ y: translateY }}
              exit={{ y: "100%" }}
              transition={{ type: "spring", damping: 30, stiffness: 300 }}
              onTouchStart={handleTouchStart}
              onTouchMove={handleTouchMove}
              onTouchEnd={handleTouchEnd}
              className="fixed bottom-0 left-0 right-0 z-50 bg-zinc-900 border-t border-zinc-700 rounded-t-2xl max-h-[85vh] overflow-y-auto"
              style={{ transform: `translateY(${translateY}px)` }}
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
