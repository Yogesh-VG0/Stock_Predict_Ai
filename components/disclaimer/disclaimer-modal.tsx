"use client"

import { useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { X, AlertTriangle } from "lucide-react"

const DISCLAIMER_STORAGE_KEY = "stockpredict_disclaimer_accepted"

export default function DisclaimerModal() {
  const [isOpen, setIsOpen] = useState(false)

  useEffect(() => {
    // Check if user has already accepted the disclaimer
    const hasAccepted = localStorage.getItem(DISCLAIMER_STORAGE_KEY)
    if (!hasAccepted) {
      // Small delay to prevent flash of content
      const timer = setTimeout(() => setIsOpen(true), 300)
      return () => clearTimeout(timer)
    }
  }, [])

  const handleAccept = () => {
    localStorage.setItem(DISCLAIMER_STORAGE_KEY, "true")
    setIsOpen(false)
  }

  const handleViewDisclaimer = (e?: React.MouseEvent) => {
    if (e) {
      e.stopPropagation()
    }
    // Open disclaimer page in new tab so user doesn't lose their place
    window.open("/disclaimer", "_blank", "noopener,noreferrer")
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={handleAccept}
            className="fixed inset-0 z-[9999] bg-black/80 backdrop-blur-sm"
            aria-hidden="true"
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
            className="fixed inset-0 z-[10000] flex items-center justify-center p-4"
            role="dialog"
            aria-modal="true"
            aria-labelledby="disclaimer-title"
          >
            <div className="relative w-full max-w-lg bg-black border border-zinc-800 rounded-2xl shadow-2xl overflow-hidden">
              {/* Header */}
              <div className="flex items-start justify-between p-6 pb-4 border-b border-zinc-800">
                <div className="flex items-start gap-3 flex-1">
                  <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-red-500/10 border border-red-500/20 flex items-center justify-center">
                    <AlertTriangle className="h-5 w-5 text-red-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h2
                      id="disclaimer-title"
                      className="text-xl font-bold text-white mb-1"
                    >
                      Important Disclaimer
                    </h2>
                  </div>
                </div>
                <button
                  onClick={handleAccept}
                  className="flex-shrink-0 w-8 h-8 rounded-lg hover:bg-zinc-800 flex items-center justify-center transition-colors text-zinc-400 hover:text-white"
                  aria-label="Close disclaimer"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

              {/* Content */}
              <div className="p-6 pt-4">
                <p className="text-sm sm:text-base text-zinc-300 leading-relaxed mb-4">
                  StockPredict AI is for <strong className="text-white">educational purposes only</strong> and does not provide investment advice. Predictions may be wrong; markets involve risk and you can lose money.
                </p>
                <p className="text-sm sm:text-base text-zinc-300 leading-relaxed mb-6">
                  By continuing, you agree you're responsible for your decisions and you've read the{" "}
                  <button
                    onClick={handleViewDisclaimer}
                    className="text-emerald-400 hover:text-emerald-300 underline transition-colors text-left"
                  >
                    full Disclaimer
                  </button>
                  .
                </p>

                {/* Buttons */}
                <div className="flex flex-col sm:flex-row gap-3">
                  <button
                    onClick={handleViewDisclaimer}
                    className="flex-1 px-4 py-2.5 rounded-lg border border-zinc-800 bg-zinc-900/40 hover:bg-zinc-900/60 text-white text-sm font-medium transition-colors"
                  >
                    View Disclaimer
                  </button>
                  <button
                    onClick={handleAccept}
                    className="flex-1 px-4 py-2.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-black text-sm font-semibold transition-colors"
                  >
                    Accept & Continue
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
