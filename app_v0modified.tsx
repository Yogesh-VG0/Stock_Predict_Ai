"use client"

import { BrowserRouter as Router, Routes, Route } from "react-router-dom"
import Layout from "@/components/layout/layout"

// Direct imports - no lazy loading to avoid chunk errors
import HomePage from "@/pages/home"
import StockDetail from "@/pages/stock-detail"
import NewsPage from "@/pages/news"
import WatchlistPage from "@/pages/watchlist"
import PredictionsPage from "@/pages/predictions"

export default function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/stocks/:symbol" element={<StockDetail />} />
          <Route path="/news" element={<NewsPage />} />
          <Route path="/watchlist" element={<WatchlistPage />} />
          <Route path="/predictions" element={<PredictionsPage />} />
        </Routes>
      </Layout>
    </Router>
  )
}
