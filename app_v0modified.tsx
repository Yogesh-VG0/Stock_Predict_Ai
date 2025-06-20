"use client"

import { BrowserRouter as Router, Routes, Route } from "react-router-dom"
import Layout from "./components/layout/layout"
import HomePage from "./pages/home"
import PredictionsPage from "./pages/predictions"
import WatchlistPage from "./pages/watchlist"
import StockDetail from "./pages/stock-detail"
import NewsPage from "./pages/news"

export default function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/predictions" element={<PredictionsPage />} />
          <Route path="/watchlist" element={<WatchlistPage />} />
          <Route path="/stocks/:symbol" element={<StockDetail />} />
          <Route path="/news" element={<NewsPage />} />
        </Routes>
      </Layout>
    </Router>
  )
}
