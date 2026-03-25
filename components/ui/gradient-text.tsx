"use client"

import { motion, type Transition } from "framer-motion"
import type * as React from "react"
import { cn } from "@/lib/utils"

type GradientTextProps = React.ComponentProps<"span"> & {
  text: string
  gradient?: string
  neon?: boolean
  transition?: Transition
}

function GradientText({
  text,
  className,
  gradient = "linear-gradient(90deg, #10b981 0%, #34d399 20%, #22d3ee 50%, #34d399 80%, #10b981 100%)",
  neon = false,
  transition = { duration: 3, repeat: Infinity, ease: "linear" },
  ...props
}: GradientTextProps) {
  const baseStyle: React.CSSProperties = {
    backgroundImage: gradient,
  }

  return (
    <span
      className={cn("relative inline-block", className)}
      data-slot="gradient-text"
      {...(props as any)}
    >
      <motion.span
        animate={{ backgroundPositionX: ["0%", "200%"] }}
        className="m-0 text-transparent bg-clip-text bg-[length:200%_100%]"
        style={baseStyle}
        transition={transition}
      >
        {text}
      </motion.span>

      {neon && (
        <motion.span
          animate={{ backgroundPositionX: ["0%", "200%"] }}
          className="m-0 absolute top-0 left-0 text-transparent bg-clip-text blur-[8px] mix-blend-plus-lighter bg-[length:200%_100%]"
          style={baseStyle}
          transition={transition}
        >
          {text}
        </motion.span>
      )}
    </span>
  )
}

export { GradientText, type GradientTextProps }
export default GradientText
