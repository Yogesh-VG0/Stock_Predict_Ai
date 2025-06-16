"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Newspaper, Search, CheckCircle, XCircle, Clock, ExternalLink, Globe2, X, Loader2 } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

const INDUSTRIES = ["Technology", "Finance", "Healthcare", "Energy", "Automotive"];
const TOP_TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "PLTR", "GOOGL", "META", "NFLX", "AMD", "INTC"];
const SENTIMENTS = ["positive", "neutral", "negative"];
const PAGE_SIZE = 20;

interface NewsArticle {
  provider: string;
  uuid?: string;
  id?: string;
  url: string;
  title: string;
  snippet?: string;
  description?: string;
  published_at: string;
  source: string;
  sentiment?: string;
  tickers?: string[];
  industry?: string;
  image_url?: string;
}

function getSentimentLabel(score: number) {
  if (score > 0.01) return "positive";
  if (score < -0.01) return "negative";
  return "neutral";
}

function providerBadge(provider: string) {
  if (provider === "marketaux") return <span className="bg-blue-900 text-blue-300 text-xs px-2 py-0.5 rounded ml-2">Marketaux</span>;
  if (provider === "finnhub") return <span className="bg-yellow-900 text-yellow-300 text-xs px-2 py-0.5 rounded ml-2">Finnhub</span>;
  if (provider === "webz") return <span className="bg-purple-900 text-purple-300 text-xs px-2 py-0.5 rounded ml-2">Webz.io</span>;
  if (provider === "tickertick") return <span className="bg-cyan-900 text-cyan-300 text-xs px-2 py-0.5 rounded ml-2">TickerTick</span>;
  return null;
}

function truncate(text: string, maxLength = 200) {
  if (!text) return "";
  return text.length > maxLength ? text.slice(0, maxLength) + "..." : text;
}

export default function NewsPage() {
  const [isLoading, setIsLoading] = useState(true)
  const [news, setNews] = useState<NewsArticle[]>([])
  const [activeIndustries, setActiveIndustries] = useState<string[]>([])
  const [activeTickers, setActiveTickers] = useState<string[]>([])
  const [activeSentiment, setActiveSentiment] = useState<string>("")
  const [searchTerm, setSearchTerm] = useState("")
  const [tempSearchTerm, setTempSearchTerm] = useState("")
  const [firstLoad, setFirstLoad] = useState(true)
  const [page, setPage] = useState(1)
  const [hasMore, setHasMore] = useState(true)
  const [isLoadingMore, setIsLoadingMore] = useState(false)
  const [meta, setMeta] = useState<any>(null)
  const [rssNews, setRssNews] = useState<any[]>([]);

  // Fetch news from backend API
  const fetchNews = async (opts: { append?: boolean, pageOverride?: number } = {}) => {
    setIsLoading(true); // Always set loading at the very start
    const params = new URLSearchParams()
    if (activeIndustries.length) params.set("industries", activeIndustries.join(","))
    if (activeTickers.length) params.set("symbols", activeTickers.join(","))
    if (activeSentiment) params.set("sentiment", activeSentiment)
    if (searchTerm) params.set("search", searchTerm)
    params.set("limit", PAGE_SIZE.toString())
    params.set("page", (opts.pageOverride || page).toString())
    
    // Fetch RSS news if ticker filter is active
    if (activeTickers.length === 1) {
      try {
        const rssRes = await fetch(`/api/news/rss?symbol=${activeTickers[0]}`);
        const rssData = await rssRes.json();
        // Apply sentiment filter to RSS news
        let filteredRss = rssData.data || [];
        if (activeSentiment) {
          filteredRss = filteredRss.filter((item: any) => 
            item.sentiment?.toLowerCase() === activeSentiment.toLowerCase()
          );
        }
        setRssNews(filteredRss);
      } catch (error) {
        console.error('Error fetching RSS news:', error);
        setRssNews([]);
      }
    } else {
      setRssNews([]);
    }

    try {
      const res = await fetch(`/api/news/unified?${params.toString()}`)
      const data = await res.json()
      
      // Only update news if we have results
      if (data.data && data.data.length > 0) {
        if (opts.append) {
          setNews((prev) => [...prev, ...data.data])
        } else {
          setNews(data.data)
        }
        setMeta(data.meta)
        // Check if there are more articles to load
        const hasMoreArticles = data.meta.total > (opts.pageOverride || page) * PAGE_SIZE;
        setHasMore(hasMoreArticles);
        console.log('Has more articles:', hasMoreArticles, 'Total:', data.meta.total, 'Current page:', opts.pageOverride || page);
      } else {
        // If no results, clear the news
        setNews([])
        setMeta(null)
        setHasMore(false)
      }
    } catch (error) {
      console.error('Error fetching news:', error);
      setNews([]);
      setMeta(null);
      setHasMore(false);
    } finally {
      setIsLoading(false);
    }
  }

  // On first load, show top tickers by default
  useEffect(() => {
    fetchNews({ append: false, pageOverride: 1 })
    setFirstLoad(false)
    setPage(1)
    // eslint-disable-next-line
  }, [])

  // Handle search input change
  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setTempSearchTerm(e.target.value);
  };

  // Handle search on Enter key press
  const handleSearchKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      setSearchTerm(tempSearchTerm);
    }
  };

  // Clear search
  const handleClearSearch = () => {
    setTempSearchTerm("");
    setSearchTerm("");
  };

  // Refetch when filters/search change (but not on first load)
  useEffect(() => {
    if (!firstLoad) {
      setPage(1)
      setNews([]) // Clear existing news when filters change
      fetchNews({ append: false, pageOverride: 1 })
    }
    // eslint-disable-next-line
  }, [activeIndustries, activeTickers, activeSentiment, searchTerm])

  const loadMoreNews = async () => {
    if (isLoadingMore || !hasMore) return;
    
    setIsLoadingMore(true);
    const nextPage = page + 1;
    
    try {
      const params = new URLSearchParams({
        page: nextPage.toString(),
        limit: PAGE_SIZE.toString()
      });

      if (searchTerm) params.append('search', searchTerm);
      if (activeTickers.length > 0) params.append('symbols', activeTickers.join(','));
      if (activeIndustries.length > 0) params.append('industries', activeIndustries.join(','));
      if (activeSentiment) params.append('sentiment', activeSentiment);

      const response = await fetch(`/api/news/unified?${params.toString()}`);
      const data = await response.json();

      if (data.data && data.data.length > 0) {
        // Filter out any duplicates before adding new articles
        const newArticles = data.data.filter(
          (newArticle: NewsArticle) => !news.some(existingArticle => 
            existingArticle.url === newArticle.url || 
            existingArticle.title === newArticle.title
          )
        );
        
        setNews(prevNews => [...prevNews, ...newArticles]);
        setPage(nextPage);
        setHasMore(data.meta.hasMore);
    } else {
        setHasMore(false);
      }
    } catch (error) {
      console.error('Error loading more news:', error);
    } finally {
      setIsLoadingMore(false);
    }
  };

  const toggleIndustry = (industry: string) => {
    setActiveIndustries((prev) =>
      prev.includes(industry) ? [] : [industry]
    );
  }
  const toggleTicker = (ticker: string) => {
    setActiveTickers((prev) =>
      prev.includes(ticker) ? [] : [ticker]
    )
    }
  const setSentiment = (sentiment: string) => {
    setActiveSentiment((prev) => (prev === sentiment ? "" : sentiment));
    setNews([]); // Clear existing news when sentiment changes
  }

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case "positive":
        return <CheckCircle className="h-4 w-4 text-emerald-500" />
      case "negative":
        return <XCircle className="h-4 w-4 text-red-500" />
      default:
        return <Clock className="h-4 w-4 text-amber-500" />
    }
  }

  return (
    <div className="space-y-6">
      <motion.h1
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-2xl font-bold flex items-center gap-2"
      >
        <Newspaper className="h-6 w-6 text-blue-500" />
        Market News
      </motion.h1>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.1 }}
        className="bg-zinc-900 rounded-lg p-4 border border-zinc-800"
      >
        <div className="flex flex-col md:flex-row gap-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-2.5 h-4 w-4 text-zinc-400" />
            <input
              type="text"
              value={tempSearchTerm}
              onChange={handleSearchChange}
              onKeyPress={handleSearchKeyPress}
              placeholder="Search news by keyword, ticker, or company... (Press Enter to search)"
              className="w-full bg-zinc-800 border border-zinc-700 rounded-md py-2 pl-9 pr-12 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
            />
            {tempSearchTerm && (
              <button
                onClick={handleClearSearch}
                className="absolute right-3 top-2.5 text-zinc-400 hover:text-zinc-200"
                title="Clear search"
              >
                <X className="h-4 w-4" />
              </button>
            )}
          </div>

          <div className="flex items-center gap-2">
            <div className="bg-zinc-800 rounded-md p-1 flex items-center">
              {SENTIMENTS.map((sentiment) => (
              <button
                  key={sentiment}
                  onClick={() => setSentiment(sentiment)}
                className={`flex items-center gap-1 px-2 py-1 rounded-md text-xs ${
                    activeSentiment === sentiment
                      ? sentiment === "positive"
                    ? "bg-emerald-500/20 text-emerald-500"
                        : sentiment === "negative"
                        ? "bg-red-500/20 text-red-500"
                        : "bg-amber-500/20 text-amber-500"
                    : "text-zinc-400 hover:text-white"
                }`}
              >
                  {getSentimentIcon(sentiment)}
                  {sentiment.charAt(0).toUpperCase() + sentiment.slice(1)}
              </button>
              ))}
            </div>
          </div>
        </div>

        <div className="flex flex-wrap gap-2 mt-3">
          <div className="text-xs text-zinc-400 flex items-center">Sectors:</div>
          {INDUSTRIES.map((industry) => (
            <button
              key={industry}
              onClick={() => toggleIndustry(industry)}
              className={`text-xs px-2 py-1 rounded-md ${
                activeIndustries.includes(industry)
                  ? "bg-blue-500/20 text-blue-400"
                  : "bg-zinc-800 text-zinc-400 hover:text-white"
              }`}
            >
              {industry}
            </button>
          ))}

          <div className="text-xs text-zinc-400 flex items-center ml-2">Top Tickers:</div>
          {TOP_TICKERS.map((ticker) => (
            <button
              key={ticker}
              onClick={() => toggleTicker(ticker)}
              className={`text-xs px-2 py-1 rounded-md ${
                activeTickers.includes(ticker)
                  ? "bg-purple-500/20 text-purple-400"
                  : "bg-zinc-800 text-zinc-400 hover:text-white"
              }`}
            >
              {ticker}
            </button>
          ))}
        </div>
        {meta && (
          <div className="flex flex-wrap gap-4 mt-4 text-xs text-zinc-400 items-center">
            <span><Globe2 className="inline h-4 w-4 mr-1" />Total: <b>{meta.total}</b></span>
            <span>Marketaux: <b>{meta.sources?.marketaux}</b></span>
            <span>Finnhub: <b>{meta.sources?.finnhub}</b></span>
            <span>TickerTick: <b>{meta.sources?.tickertick}</b></span>
            <span>NewsAPI: <b>{meta.sources?.newsapi}</b></span>
          </div>
        )}
      </motion.div>

      {isLoading && page === 1 ? (
        <div className="space-y-4">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-32 bg-zinc-900 animate-pulse rounded-lg"></div>
          ))}
        </div>
      ) : (
        <div className="space-y-4">
          {rssNews.length > 0 && (
            <>
              {rssNews.map((item: any, index: number) => {
                const sentiment = item.sentiment || "neutral";
                return (
            <motion.div
                    key={item.uuid}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
            >
              <Card className="hover:border-zinc-700 transition-colors">
                <CardContent className="p-4">
                  <div className="flex flex-col md:flex-row md:items-start gap-4">
                          <div className="w-full md:w-48 flex-shrink-0 mb-2 md:mb-0">
                            <img
                              src={item.image_url && item.image_url.trim() !== ""
                                ? item.image_url
                                : "/news-placeholder.jpg"}
                              alt={item.title}
                              className="rounded-lg w-full h-32 object-cover bg-zinc-800"
                              style={{ objectFit: "cover" }}
                            />
                          </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <div className="text-xs text-zinc-400">
                                {item.published_at ? new Date(item.published_at).toLocaleDateString("en-US", { month: "short", day: "numeric" }) : ""}
                        </div>
                        <div className="text-xs text-zinc-400">•</div>
                        <div className="text-xs text-zinc-400">{item.source}</div>
                              {providerBadge(item.provider)}
                        <div className="flex items-center gap-1 ml-auto">
                                {getSentimentIcon(sentiment)}
                                <span className="text-xs capitalize">{sentiment}</span>
                        </div>
                      </div>
                      <h2 className="text-lg font-medium mb-2">{item.title}</h2>
                            <p className="text-sm text-zinc-300 mb-3">{item.snippet}</p>
                    </div>
                    <div className="flex items-center md:items-start">
                      <a
                        href={item.url}
                              target="_blank"
                              rel="noopener noreferrer"
                        className="flex items-center gap-1 text-xs text-emerald-500 hover:text-emerald-400 transition-colors"
                      >
                        <span>Read Full Article</span>
                        <ExternalLink className="h-3 w-3" />
                      </a>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
                );
              })}
            </>
          )}
          {news.length === 0 && rssNews.length === 0 && (
            <div className="text-center text-zinc-400">No news found for the selected filters.</div>
          )}
          {(() => {
            let tickerTickShown = 0;
            return news.map((item: any, index: number) => {
              const sentiment = item.sentiment || "neutral";
              // Only show the first 5 TickerTick news items
              if (item.provider === "tickertick") {
                if (tickerTickShown >= 5) return null;
                tickerTickShown++;
              }
              // Create a unique key by combining provider and uuid/id
              const uniqueKey = `${item.provider}-${item.uuid || item.id || item.url}-${index}`;
              
              return (
                <motion.div
                  key={uniqueKey}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <Card className="hover:border-zinc-700 transition-colors">
                    <CardContent className="p-4">
                      <div className="flex flex-col md:flex-row md:items-start gap-4">
                        {item.image_url && item.image_url.trim() !== "" && (
                          <div className="w-full md:w-48 flex-shrink-0 mb-2 md:mb-0">
                            <img
                              src={item.image_url}
                              alt={item.title}
                              className="rounded-lg w-full h-32 object-cover bg-zinc-800"
                              style={{ objectFit: "cover" }}
                              onError={(e) => {
                                e.currentTarget.src = "/news-placeholder.jpg";
                              }}
                            />
                          </div>
                        )}
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <div className="text-xs text-zinc-400">
                              {new Date(item.published_at).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                            </div>
                            <div className="text-xs text-zinc-400">•</div>
                            <div className="text-xs text-zinc-400">{item.source}</div>
                            {providerBadge(item.provider)}
                            <div className="flex items-center gap-1 ml-auto">
                              {getSentimentIcon(sentiment)}
                              <span className="text-xs capitalize">{sentiment}</span>
                            </div>
                          </div>
                          <h2 className="text-lg font-medium mb-2">{item.title}</h2>
                          <p className="text-sm text-zinc-300 mb-3">
                            {item.provider === "tickertick" ? truncate(item.snippet) : item.snippet}
                          </p>
                          <div className="flex flex-wrap gap-2">
                            {item.tickers && item.tickers.map((ticker: string) => (
                              <span key={ticker} className="text-xs px-2 py-0.5 bg-zinc-800 rounded-md text-zinc-300">
                                ${ticker}
                              </span>
                            ))}
                            {item.industry && (
                              <span className="text-xs px-2 py-0.5 bg-zinc-800 rounded-md text-zinc-300">
                                {item.industry}
                              </span>
                            )}
                          </div>
                          <div className="flex items-center md:items-start mt-3">
                            <a
                              href={item.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="flex items-center gap-1 text-xs text-emerald-500 hover:text-emerald-400 transition-colors"
                            >
                              <span>Read Full Article</span>
                              <ExternalLink className="h-3 w-3" />
                            </a>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              );
            });
          })()}
          {hasMore && !isLoadingMore && (
          <div className="flex justify-center mt-6">
              <Button
                onClick={loadMoreNews}
                disabled={isLoadingMore}
                className="bg-emerald-600 hover:bg-emerald-700 text-white"
              >
                {isLoadingMore ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Loading...
                  </>
                ) : (
                  'Load More News'
                )}
              </Button>
          </div>
          )}
        </div>
      )}
    </div>
  )
}
