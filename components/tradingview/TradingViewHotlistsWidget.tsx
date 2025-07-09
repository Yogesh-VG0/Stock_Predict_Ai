import { useEffect, useRef, useState } from "react";

export default function TradingViewHotlistsWidget() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  useEffect(() => {
    if (!isMounted) return;
    
    const container = containerRef.current;
    if (!container) return;
    
    // Clear any existing content
    container.innerHTML = "";
    
    // Add a small delay to ensure DOM is fully ready
    const timeoutId = setTimeout(() => {
      try {
        const script = document.createElement("script");
        script.src = "https://s3.tradingview.com/external-embedding/embed-widget-hotlists.js";
        script.type = "text/javascript";
        script.async = true;
        script.onerror = () => {
          console.warn("Failed to load TradingView Hotlists widget");
        };
        script.innerHTML = JSON.stringify({
          colorTheme: "dark",
          exchange: "US",
          showChart: true,
          locale: "en",
          isTransparent: false,
          showSymbolLogo: false,
          showFloatingTooltip: false,
          width: "100%",
          height: "550",
          plotLineColorGrowing: "#3b82f6",
          plotLineColorFalling: "#ef4444",
          gridLineColor: "rgba(42, 46, 57, 0)",
          scaleFontColor: "#e5e7eb",
          belowLineFillColorGrowing: "#3b82f622",
          belowLineFillColorFalling: "#ef444422",
          symbolActiveColor: "#3b82f622"
        });
        container.appendChild(script);
      } catch (error) {
        console.warn("Error loading TradingView Hotlists widget:", error);
      }
    }, 100);
    
    return () => {
      clearTimeout(timeoutId);
      if (container) container.innerHTML = "";
    };
  }, [isMounted]);

  if (!isMounted) {
    return (
      <div className="rounded-lg bg-zinc-900 border border-zinc-800 shadow-lg overflow-hidden p-2">
        <div className="tradingview-widget-container" style={{ minHeight: 550 }}>
          <div className="flex items-center justify-center h-full">
            <div className="text-zinc-400 text-sm">Loading market data...</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg bg-zinc-900 border border-zinc-800 shadow-lg overflow-hidden p-2">
      <div className="tradingview-widget-container" style={{ minHeight: 550 }}>
        <div ref={containerRef} className="tradingview-widget-container__widget" />
        <div className="tradingview-widget-copyright">
          <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
          </a>
        </div>
      </div>
    </div>
  );
} 