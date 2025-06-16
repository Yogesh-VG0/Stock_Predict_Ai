const Parser = require('rss-parser');
const vader = require('vader-sentiment');
const parser = new Parser();

function analyzeSentiment(text) {
  if (!text) return { sentiment: 'neutral', score: 0 };
  const intensity = vader.SentimentIntensityAnalyzer.polarity_scores(text);
  let sentiment = 'neutral';
  if (intensity.compound > 0.05) sentiment = 'positive';
  else if (intensity.compound < -0.05) sentiment = 'negative';
  return { sentiment, score: intensity.compound };
}

async function fetchRssNews(symbol) {
  // Remove any leading $ from the symbol
  const cleanSymbol = symbol.replace(/^\$/, '');
  const yahooUrl = `https://feeds.finance.yahoo.com/rss/2.0/headline?s=${cleanSymbol}&region=US&lang=en-US`;
  const saUrl = `https://seekingalpha.com/api/sa/combined/${cleanSymbol}.xml`;
  const feeds = await Promise.all([
    parser.parseURL(yahooUrl).catch(() => ({ items: [] })),
    parser.parseURL(saUrl).catch(() => ({ items: [] })),
  ]);
  // Normalize and merge
  return feeds.flatMap(feed => (feed.items || []).map(item => {
    const textToAnalyze = [item.title, item.contentSnippet || item.content || ''].filter(Boolean).join('. ');
    const sentimentResult = analyzeSentiment(textToAnalyze);
    return {
      uuid: `rss_${item.guid || item.link}`,
      title: item.title,
      url: item.link,
      published_at: item.pubDate,
      source: item.link && item.link.includes('yahoo') ? 'Yahoo Finance' : 'Seeking Alpha',
      snippet: item.contentSnippet || item.content || '',
      provider: 'rss',
      tickers: [cleanSymbol],
      image_url: '', // RSS rarely has images
      sentiment: sentimentResult.sentiment,
      sentiment_score: sentimentResult.score,
    };
  }));
}

module.exports = { fetchRssNews }; 