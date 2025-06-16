"use client"

import type React from "react"
import { useState, useEffect } from "react"
import { usePathname } from "next/navigation"
import { motion, AnimatePresence } from "framer-motion"
import Sidebar from "./sidebar"
import Navbar from "./navbar"
import TradingViewScriptLoader from "../tradingview/trading-view-script-loader"
import TickerTapeWidget from "../tradingview/ticker-tape-widget"
import { SidebarProvider } from "@/components/ui/sidebar"

interface LayoutProps {
  children: React.ReactNode
}

export default function Layout({ children }: LayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [isMobile, setIsMobile] = useState(false)
  const pathname = usePathname()

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768)
      if (window.innerWidth < 768) {
        setSidebarOpen(false)
      } else {
        setSidebarOpen(true)
      }
    }

    checkMobile()
    window.addEventListener("resize", checkMobile)
    return () => window.removeEventListener("resize", checkMobile)
  }, [])

  return (
    <SidebarProvider defaultOpen={!isMobile}>
      <div className="flex h-screen w-full bg-black text-white overflow-hidden">
        <TradingViewScriptLoader />

        <AnimatePresence>
          {sidebarOpen && (
            <motion.div
              initial={{ x: -300 }}
              animate={{ x: 0 }}
              exit={{ x: -300 }}
              transition={{ duration: 0.3, ease: "easeInOut" }}
              className="fixed md:relative z-30"
            >
              <Sidebar onClose={() => setSidebarOpen(false)} />
            </motion.div>
          )}
        </AnimatePresence>

        <div className="flex flex-col flex-1 h-screen overflow-hidden">
          <div className="sticky top-0 z-20">
            <TickerTapeWidget />
            <Navbar sidebarOpen={sidebarOpen} toggleSidebar={() => setSidebarOpen(!sidebarOpen)} />
          </div>

          <motion.main
            key={pathname}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="flex-1 overflow-y-auto p-4 md:p-6"
          >
            {children}
          </motion.main>
        </div>
      </div>
    </SidebarProvider>
  )
}
