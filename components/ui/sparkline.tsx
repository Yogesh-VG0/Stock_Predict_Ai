"use client"

import { useEffect, useRef } from "react"
import { motion } from "framer-motion"

interface SparklineProps {
  data: number[]
  width?: number
  height?: number
  color?: string
  strokeWidth?: number
  className?: string
  fillOpacity?: number
  animated?: boolean
}

export default function Sparkline({
  data,
  width = 100,
  height = 30,
  color = "#10b981",
  strokeWidth = 1.5,
  className = "",
  fillOpacity = 0.1,
  animated = true,
}: SparklineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !data.length) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Set canvas dimensions with device pixel ratio for sharpness
    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr
    canvas.height = height * dpr
    ctx.scale(dpr, dpr)

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Find min and max values for scaling
    const min = Math.min(...data)
    const max = Math.max(...data)
    const range = max - min || 1 // Avoid division by zero

    // Calculate x and y coordinates
    const points = data.map((value, index) => ({
      x: (index / (data.length - 1)) * width,
      y: height - ((value - min) / range) * (height * 0.8) - height * 0.1,
    }))

    // Draw the line
    ctx.beginPath()
    ctx.moveTo(0, height)
    points.forEach((point, i) => {
      if (i === 0) {
        ctx.moveTo(point.x, point.y)
      } else {
        ctx.lineTo(point.x, point.y)
      }
    })
    ctx.lineTo(width, height)
    ctx.lineTo(0, height)

    // Fill area under the line
    ctx.fillStyle = `${color}${Math.round(fillOpacity * 255)
      .toString(16)
      .padStart(2, "0")}`
    ctx.fill()

    // Draw the line
    ctx.beginPath()
    points.forEach((point, i) => {
      if (i === 0) {
        ctx.moveTo(point.x, point.y)
      } else {
        ctx.lineTo(point.x, point.y)
      }
    })
    ctx.strokeStyle = color
    ctx.lineWidth = strokeWidth
    ctx.stroke()
  }, [data, width, height, color, strokeWidth, fillOpacity])

  return animated ? (
    <motion.canvas
      ref={canvasRef}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      width={width}
      height={height}
      className={className}
      style={{ width, height }}
    />
  ) : (
    <canvas ref={canvasRef} width={width} height={height} className={className} style={{ width, height }} />
  )
}
