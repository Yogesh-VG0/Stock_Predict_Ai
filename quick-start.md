# 🚀 Quick Start - StockPredict AI (Simple Setup)

**Get your project running in 10 minutes!**

## ⚡ Super Quick Local Setup

```bash
# 1. Copy environment template
cp env.simple.template .env

# 2. Edit .env file with your API keys (see below)

# 3. Install dependencies
npm install
cd backend && npm install && cd ..

# 4. Start both frontend and backend
npm run dev:all
```

**That's it!** Your app will be running at http://localhost:3000

## 🔑 Required API Keys (2 minutes to get)

### 1. Finnhub API Key (Required)
- Go to: https://finnhub.io/register
- Sign up → Get API key
- Paste in `.env` file: `FINNHUB_API_KEY=your_key_here`

### 2. MongoDB Atlas (Required)
- Go to: https://cloud.mongodb.com
- Create free account → Create cluster
- Get connection string → Paste in `.env` file

### 3. JWT Secret (Required)
- Generate random string: https://randomkeygen.com/
- Paste in `.env` file: `JWT_SECRET=your_random_string`

## 📦 With Docker (Alternative)

```bash
# Copy environment file
cp env.simple.template .env
# Edit .env with your API keys

# Start with Docker
npm run docker:simple
```

## 🌐 Deploy to Railway (15 minutes)

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Deploy**: 
   - Go to [railway.app](https://railway.app)
   - Connect GitHub repo
   - Add environment variables
   - Deploy!

**Detailed deployment guide**: [SIMPLE_DEPLOYMENT.md](./SIMPLE_DEPLOYMENT.md)

## ✅ What You'll Have

- 📊 Professional financial dashboard
- 📈 Real-time stock prices and charts
- 🔍 Stock search and watchlist
- 📱 Mobile-responsive design
- 🎨 Beautiful UI with animations
- 📊 Enhanced mock predictions (looks realistic!)

Perfect for showing to recruiters! 🎯

## 🆘 Need Help?

**App not loading?**
- Check MongoDB connection string
- Verify Finnhub API key
- Ensure both frontend and backend are running

**Still stuck?**
- Check [SIMPLE_DEPLOYMENT.md](./SIMPLE_DEPLOYMENT.md) for detailed troubleshooting
- Look at console logs for error messages 