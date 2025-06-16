import { useEffect, useRef } from "react";

export default function TradingViewHotlistsWidget() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    container.innerHTML = "";
    const script = document.createElement("script");
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-hotlists.js";
    script.type = "text/javascript";
    script.async = true;
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
    return () => {
      if (container) container.innerHTML = "";
    };
  }, []);

  return (
    <div className="rounded-lg bg-zinc-900 border border-zinc-800 shadow-lg overflow-hidden p-2">
      <div className="tradingview-widget-container" style={{ minHeight: 550 }}>
        <div ref={containerRef} className="tradingview-widget-container__widget" />
        <div className="tradingview-widget-copyright">
          <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
            <span className="blue-text">Track all markets on TradingView</span>
          </a>
        </div>
      </div>
    </div>
  );
} 