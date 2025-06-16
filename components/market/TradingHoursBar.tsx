import React, { useMemo } from "react";
import { useEffect, useState } from "react";
import { Clock } from "lucide-react";

// Configurable session times and colors
const SESSIONS = [
  {
    name: "Overnight",
    start: 20 * 60, // 8:00 PM
    end: 4 * 60, // 4:00 AM (next day)
    color: "#7c3aed", // Brighter neon purple for visibility
    neon: "0 0 16px 4px #7c3aed"
  },
  {
    name: "Pre-market",
    start: 4 * 60, // 4:00 AM in minutes
    end: 9 * 60 + 30, // 9:30 AM
    color: "#ff4fd8", // Neon pink
    neon: "0 0 16px 4px #ff4fd8"
  },
  {
    name: "Regular",
    start: 9 * 60 + 30, // 9:30 AM
    end: 16 * 60, // 4:00 PM
    color: "#39ff14", // Neon green
    neon: "0 0 16px 4px #39ff14"
  },
  {
    name: "After-hours",
    start: 16 * 60, // 4:00 PM
    end: 20 * 60, // 8:00 PM
    color: "#ffb347", // Neon orange
    neon: "0 0 16px 4px #ffb347"
  },
];
const CLOSED_COLOR = "#22223b";
const CLOSED_NEON = "0 0 10px 2px #444";

function getMinutesSinceMidnightNY() {
  // Always use America/New_York timezone
  const now = new Date();
  const nyNow = new Date(now.toLocaleString("en-US", { timeZone: "America/New_York" }));
  return nyNow.getHours() * 60 + nyNow.getMinutes();
}

function getCurrentSession(mins: number) {
  for (const session of SESSIONS) {
    if (session.start < session.end) {
      if (mins >= session.start && mins < session.end) return session;
    } else {
      // Overnight session (wraps past midnight)
      if (mins >= session.start || mins < session.end) return session;
    }
  }
  return null;
}

interface SessionSegment {
  name: string;
  start: number;
  end: number;
  color: string;
  neon: string;
  widthPercent: number;
  label?: boolean;
}

function getSessionSegments(): SessionSegment[] {
  const totalDay = 24 * 60;
  const segments: SessionSegment[] = [];
  let lastEnd = 0;
  for (const session of SESSIONS) {
    let segStart = session.start;
    let segEnd = session.end;
    if (session.start < session.end) {
      // Normal session
      // Only add a closed segment if it's a small gap (<10 minutes)
      if (segStart > lastEnd && segStart - lastEnd < 10) {
        segments.push({
          name: "Closed",
          start: lastEnd,
          end: segStart,
          color: CLOSED_COLOR,
          neon: CLOSED_NEON,
          widthPercent: ((segStart - lastEnd) / totalDay) * 100,
        });
      }
      segments.push({
        ...session,
        start: segStart,
        end: segEnd,
        widthPercent: ((segEnd - segStart) / totalDay) * 100,
      });
      lastEnd = segEnd;
    } else {
      // Overnight session: split into two segments, but only label once
      // 8:00 PM to midnight
      if (segStart > lastEnd && segStart - lastEnd < 10) {
        segments.push({
          name: "Closed",
          start: lastEnd,
          end: segStart,
          color: CLOSED_COLOR,
          neon: CLOSED_NEON,
          widthPercent: ((segStart - lastEnd) / totalDay) * 100,
        });
      }
      segments.push({
        ...session,
        start: segStart,
        end: totalDay,
        widthPercent: ((totalDay - segStart) / totalDay) * 100,
        label: true,
      });
      // Midnight to 4:00 AM
      segments.push({
        ...session,
        start: 0,
        end: segEnd,
        widthPercent: ((segEnd) / totalDay) * 100,
        label: false, // Only label the first segment
      });
      lastEnd = segEnd;
    }
  }
  // Do not add a final closed segment; fill the bar with session colors only
  return segments;
}

interface TradingHoursBarProps {
  compact?: boolean;
  marketClosed?: boolean;
  holidayName?: string;
}

/**
 * TradingHoursBar
 * Props:
 *   - compact: boolean (smaller bar for inline use)
 *   - marketClosed: boolean (if true, show full gray bar and hide arrow)
 *   - holidayName: string (optional, name of the holiday to display)
 */
export function TradingHoursBar({ compact = false, marketClosed = false, holidayName }: TradingHoursBarProps) {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 1000 * 30); // update every 30s
    return () => clearInterval(timer);
  }, []);

  // Always use NY time for session calculation
  const mins = getMinutesSinceMidnightNY();
  const segments = useMemo(getSessionSegments, []);
  const currentSession = getCurrentSession(mins);
  const nyNow = new Date(now.toLocaleString("en-US", { timeZone: "America/New_York" }));

  // Find the current segment and its index
  let markerLeftPercent = 0;
  let found = false;
  let prevWidth = 0;
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    // Handle overnight wrap: mins can be < seg.end for overnight
    const inSegment = seg.start < seg.end
      ? mins >= seg.start && mins < seg.end
      : mins >= seg.start || mins < seg.end;
    if (!found && inSegment) {
      // How far into this segment are we?
      let elapsed = mins - seg.start;
      if (elapsed < 0) elapsed += 24 * 60; // handle overnight wrap
      const segLength = (seg.end > seg.start ? seg.end - seg.start : (24 * 60 - seg.start) + seg.end);
      const percentInSeg = Math.max(0, Math.min(1, elapsed / segLength));
      markerLeftPercent = prevWidth + percentInSeg * seg.widthPercent;
      found = true;
    }
    if (!found) prevWidth += seg.widthPercent;
  }

  // If market is closed for a holiday, show gray bar, hide arrow, and show message
  if (marketClosed) {
    return (
      <div className={compact ? "w-full flex flex-col items-center mb-0 relative" : "w-full flex flex-col items-center mb-4 relative"} style={{overflow: 'visible'}}>
        <div className="mb-1 text-center w-full">
          <span className="text-xs font-bold text-zinc-300 bg-zinc-800 px-2 py-1 rounded shadow">
            Market Closed{holidayName ? ` (${holidayName})` : ' (Holiday)'}
          </span>
        </div>
        <div className={compact ? "relative w-full h-3 rounded-full overflow-visible flex shadow border border-zinc-800 bg-[#181825]" : "relative w-full max-w-2xl h-6 rounded-full overflow-visible flex shadow-lg border border-zinc-800 bg-[#181825]"}>
          <div className="h-full w-full" style={{ background: CLOSED_COLOR, boxShadow: `0 0 8px 2px ${CLOSED_COLOR}` }} />
        </div>
        <div className={compact ? "flex w-full mt-1 px-1 text-[10px] font-semibold tracking-wide" : "flex w-full max-w-2xl mt-2 px-1 text-xs font-semibold tracking-wide"}>
          <div style={{ width: '100%', textAlign: 'center', color: CLOSED_COLOR, textShadow: `0 0 6px ${CLOSED_COLOR}` }}>Closed</div>
        </div>
        {!compact && (
          <div className="mt-1 text-sm text-zinc-300 flex items-center gap-2 justify-center">
            <Clock className="h-4 w-4 text-emerald-400" />
            <span className="font-bold" style={{ color: CLOSED_COLOR, textShadow: `0 0 6px ${CLOSED_COLOR}` }}>
              Market Closed
            </span>
            <span className="text-xs text-zinc-400">{nyNow.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} NY</span>
          </div>
        )}
      </div>
    );
  }

  // Normal bar rendering
  return (
    <div className={compact ? "w-full flex flex-col items-center mb-0 relative" : "w-full flex flex-col items-center mb-4 relative"} style={{overflow: 'visible'}}>
      {/* Arrow Marker - outside and above the bar */}
      <div
        className="absolute pointer-events-none"
        style={{
          left: `calc(${markerLeftPercent}% - ${compact ? '7px' : '10px'})`,
          top: compact ? '-12px' : '-18px',
          transition: 'left 0.3s cubic-bezier(0.4,0,0.2,1)',
          zIndex: 10
        }}
      >
        {/* Small, solid white downward-pointing triangle */}
        <div style={{
          width: 0,
          height: 0,
          borderLeft: compact ? '7px solid transparent' : '10px solid transparent',
          borderRight: compact ? '7px solid transparent' : '10px solid transparent',
          borderTop: compact ? '12px solid white' : '16px solid white',
        }} />
      </div>
      {/* Bar */}
      <div className={compact ? "relative w-full h-3 rounded-full overflow-visible flex shadow border border-zinc-800 bg-[#181825]" : "relative w-full max-w-2xl h-6 rounded-full overflow-visible flex shadow-lg border border-zinc-800 bg-[#181825]"} style={{ borderRadius: compact ? 12 : 18 }}>
        {segments.map((seg, i) => (
          <div
            key={i}
            className="h-full"
            style={{
              width: `${seg.widthPercent}%`,
              background: seg.color,
              transition: "background 0.2s",
              borderTopLeftRadius: i === 0 ? (compact ? 12 : 18) : 0,
              borderBottomLeftRadius: i === 0 ? (compact ? 12 : 18) : 0,
              borderTopRightRadius: i === segments.length - 1 ? (compact ? 12 : 18) : 0,
              borderBottomRightRadius: i === segments.length - 1 ? (compact ? 12 : 18) : 0,
            }}
          />
        ))}
      </div>
      {/* Session labels, aligned with segments */}
      <div className={compact ? "flex w-full mt-1 px-1 text-[11px] font-bold tracking-wide" : "flex w-full max-w-2xl mt-2 px-1 text-sm font-bold tracking-wide"}>
        {segments.map((seg, i) => (
          <div
            key={i}
            style={{
              width: `${seg.widthPercent}%`,
              textAlign: 'center',
              color: seg.name === 'Overnight' ? seg.color : seg.color,
              overflow: 'visible',
              whiteSpace: 'nowrap',
            }}
          >
            {seg.name !== "Closed" && (seg.label === undefined || seg.label) ? seg.name : null}
          </div>
        ))}
      </div>
      {/* Current session and NY time */}
      {!compact && (
        <div className="mt-1 text-sm text-zinc-300 flex items-center gap-2">
          <Clock className="h-4 w-4 text-emerald-400" />
          <span className="font-bold" style={{ color: currentSession ? currentSession.color : CLOSED_COLOR, textShadow: `0 0 6px ${currentSession ? currentSession.color : CLOSED_COLOR}` }}>
            {currentSession ? currentSession.name : "Closed"}
          </span>
          <span className="text-xs text-zinc-400">{nyNow.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} NY</span>
        </div>
      )}
    </div>
  );
} 