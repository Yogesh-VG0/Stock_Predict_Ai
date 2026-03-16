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

  // Convert clientX to minutes using the timeline bar's bounding rect
  const clientXToMinutes = useCallback((clientX: number): number => {
    if (!timelineRef.current) return 0
    const rect = timelineRef.current.getBoundingClientRect()
    const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
    return Math.round(pct * 1440)
  }, [])

  // Native mouse/touch drag on the entire popup area (once dragging starts)
  useEffect(() => {
    if (!isDragging) return

    const onMouseMove = (e: MouseEvent) => {
      setDragMinutes(clientXToMinutes(e.clientX))
    }
    const onMouseUp = () => {
      setIsDragging(false)
      setDragMinutes(null)
    }
    const onTouchMove = (e: TouchEvent) => {
      if (e.touches.length > 0) {
        setDragMinutes(clientXToMinutes(e.touches[0].clientX))
      }
    }
    const onTouchEnd = () => {
      setIsDragging(false)
      setDragMinutes(null)
    }

    document.addEventListener("mousemove", onMouseMove)
    document.addEventListener("mouseup", onMouseUp)
    document.addEventListener("touchmove", onTouchMove, { passive: true })
    document.addEventListener("touchend", onTouchEnd)
    return () => {
      document.removeEventListener("mousemove", onMouseMove)
      document.removeEventListener("mouseup", onMouseUp)
      document.removeEventListener("touchmove", onTouchMove)
      document.removeEventListener("touchend", onTouchEnd)
    }
  }, [isDragging, clientXToMinutes])

  const startDrag = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
    const clientX = "touches" in e ? e.touches[0].clientX : e.clientX
    setDragMinutes(clientXToMinutes(clientX))
  }, [clientXToMinutes])

  // Escape key
  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose() }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [open, onClose])

  // The % position of the marker line — either dragged or current time
  const markerPct = dragMinutes !== null ? (dragMinutes / 1440) * 100 : currentTimePct

  // The time string displayed below today's bar
  const displayTimeStr = dragMinutes !== null
    ? formatTime(dragMinutes)
    : formatTime(now.getHours() * 60 + now.getMinutes())

  if (!open) return null

  // The left offset for the vertical line accounts for the day label width + gap + padding
  // Day label area = w-10 (40px) + gap-3 (12px) + px-3 (12px left) = 64px
  const barLeftOffset = 64
  // Right padding = px-3 (12px) = 12px
  const barRightPad = 16

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
        {/* Vertical time indicator line — spans all rows */}
        <div
          className="absolute pointer-events-none z-10"
          style={{
            left: `calc(${barLeftOffset}px + (100% - ${barLeftOffset + barRightPad}px) * ${markerPct / 100})`,
            top: 0,
            bottom: 60,
          }}
        >
          <div className="w-px h-full bg-zinc-500/50 mx-auto" />
        </div>

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
                  onMouseDown={isToday ? startDrag : undefined}
                  onTouchStart={isToday ? startDrag : undefined}
                >
                  {trading ? (
                    segments.map((seg, i) => (
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
                    ))
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

        {/* Draggable building icon below all bars */}
        <div className="relative" style={{ height: 32, marginLeft: barLeftOffset, marginRight: barRightPad }}>
          <div
            className="absolute pointer-events-none z-20"
            style={{
              left: `${markerPct}%`,
              transform: "translateX(-50%)",
              top: 0,
            }}
          >
            <div className="flex flex-col items-center">
              <div className="w-px h-3 bg-zinc-500/50" />
              <Building2 className="h-5 w-5 text-cyan-400 drop-shadow-[0_0_6px_rgba(34,211,238,0.5)]" />
            </div>
          </div>
        </div>
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
