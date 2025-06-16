const finnhub = require('finnhub');

const apiKey = process.env.FINNHUB_API_KEY || '';
export const finnhubClient = new finnhub.DefaultApi({
  apiKey,
  isJsonMime: (mime: string) => mime === 'application/json'
}); 