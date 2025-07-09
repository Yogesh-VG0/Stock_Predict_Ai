# 🚀 StockPredict AI - Deployment Guide

This guide provides multiple deployment options for showcasing your StockPredict AI project to recruiters.

## 📋 Quick Start for Recruiters

**Live Demo URLs** (Replace with your actual deployments):
- 🌐 **Frontend**: https://stockpredict-ai.railway.app
- 🤖 **ML API Docs**: https://stockpredict-ml.railway.app/docs
- 📊 **Backend API**: https://stockpredict-backend.railway.app

## 🎯 Deployment Options (Ranked by Ease)

### ⭐ **Option 1: Railway (Recommended)**

**Best for**: Full-stack applications with multiple services
**Free Tier**: $5/month credit, sleeps after inactivity
**Time to Deploy**: ~30 minutes

#### Steps:

1. **Sign up for Railway** at [railway.app](https://railway.app)

2. **Deploy from GitHub**:
   ```bash
   # Connect your GitHub repository
   # Railway will auto-detect your services
   ```

3. **Add Environment Variables**:
   ```env
   # Frontend Service
   NEXT_PUBLIC_API_URL=https://your-backend.railway.app
   NEXT_PUBLIC_NODE_BACKEND_URL=https://your-backend.railway.app
   
   # Backend Service  
   MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/stockpredict
   ML_API_URL=https://your-ml-backend.railway.app
   PORT=5000
   
   # ML Backend Service
   MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/stockpredict
   GOOGLE_API_KEY=your_gemini_api_key
   FINNHUB_API_KEY=your_finnhub_key
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
   FMP_API_KEY=your_fmp_key
   ```

4. **Deploy Services**:
   - Create 3 separate services: Frontend, Backend, ML Backend
   - Each service will get its own URL
   - Update environment variables with actual URLs

#### Railway Deployment Commands:
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and link project
railway login
railway link

# Deploy frontend
cd /
railway up

# Deploy backend
cd backend
railway up

# Deploy ML backend  
cd ml_backend
railway up
```

---

### ⭐ **Option 2: Vercel + Render (Hybrid)**

**Best for**: Optimized frontend performance
**Free Tier**: Vercel (hobby), Render (750 hours/month)
**Time to Deploy**: ~45 minutes

#### Frontend on Vercel:

1. **Connect GitHub to Vercel**
2. **Set Environment Variables**:
   ```env
   NEXT_PUBLIC_API_URL=https://your-backend.onrender.com
   NEXT_PUBLIC_NODE_BACKEND_URL=https://your-backend.onrender.com
   ```
3. **Deploy automatically on git push**

#### Backend Services on Render:

1. **Create Web Service for Node.js Backend**:
   ```yaml
   # render.yaml for backend
   services:
     - type: web
       name: stockpredict-backend
       env: node
       buildCommand: npm install
       startCommand: npm start
       envVars:
         - key: PORT
           value: 10000
         - key: MONGODB_URI
           sync: false
   ```

2. **Create Web Service for ML Backend**:
   ```yaml
   # render.yaml for ML backend
   services:
     - type: web
       name: stockpredict-ml
       env: python
       buildCommand: pip install -r requirements.txt
       startCommand: uvicorn api.main:app --host 0.0.0.0 --port 10000
   ```

---

### ⭐ **Option 3: DigitalOcean App Platform**

**Best for**: Production-grade deployment
**Cost**: ~$12/month for basic setup
**Time to Deploy**: ~1 hour

```yaml
# .do/app.yaml
name: stockpredict-ai
services:
- name: frontend
  source_dir: /
  github:
    repo: your-username/stockpredict-ai
    branch: main
  run_command: npm start
  environment_slug: node-js
  instance_count: 1
  instance_size_slug: basic-xxs
  
- name: backend
  source_dir: /backend
  run_command: npm start
  environment_slug: node-js
  instance_count: 1
  instance_size_slug: basic-xxs
  
- name: ml-backend
  source_dir: /ml_backend
  run_command: uvicorn api.main:app --host 0.0.0.0 --port 8080
  environment_slug: python
  instance_count: 1
  instance_size_slug: basic-xxs

databases:
- name: mongodb
  engine: MONGODB
  version: "5"
```

---

### ⭐ **Option 4: Docker + Any Platform**

**Best for**: Consistent deployment across platforms
**Platforms**: AWS, GCP, DigitalOcean, etc.

#### Docker Compose for Local Testing:

```yaml
# docker-compose.yml
version: '3.8'
services:
  frontend:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:5000
    depends_on:
      - backend
      
  backend:
    build: ./backend
    ports:
      - "5000:5000"
    environment:
      - MONGODB_URI=mongodb://mongo:27017/stockpredict
      - ML_API_URL=http://ml-backend:8000
    depends_on:
      - mongo
      - ml-backend
      
  ml-backend:
    build: ./ml_backend
    ports:
      - "8000:8000"
    environment:
      - MONGODB_URI=mongodb://mongo:27017/stockpredict
    depends_on:
      - mongo
      
  mongo:
    image: mongo:5.0
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db

volumes:
  mongo_data:
```

#### Deploy to AWS ECS/Fargate:
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com

docker build -t stockpredict-frontend .
docker tag stockpredict-frontend:latest your-account.dkr.ecr.us-east-1.amazonaws.com/stockpredict-frontend:latest
docker push your-account.dkr.ecr.us-east-1.amazonaws.com/stockpredict-frontend:latest
```

---

## 🗄️ Database Setup (Required for All Options)

### MongoDB Atlas (Free Tier):

1. **Create Account** at [mongodb.com/cloud/atlas](https://mongodb.com/cloud/atlas)
2. **Create Free Cluster** (512MB, shared)
3. **Create Database User**
4. **Whitelist IP Addresses** (0.0.0.0/0 for demo)
5. **Get Connection String**:
   ```
   mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/stockpredict?retryWrites=true&w=majority
   ```

---

## 🔑 API Keys Setup

### Required API Keys (Free Tiers Available):

```bash
# Financial Data APIs
FINNHUB_API_KEY=your_key_here          # Free: 60 calls/min
ALPHA_VANTAGE_API_KEY=your_key_here    # Free: 5 calls/min
FMP_API_KEY=your_key_here              # Free: 250 calls/day

# AI/ML APIs  
GOOGLE_API_KEY=your_gemini_key_here    # Free: Generous limits

# Social Media APIs (Optional)
REDDIT_CLIENT_ID=your_reddit_id        # Free: 60 requests/min
REDDIT_CLIENT_SECRET=your_reddit_secret

# Other
JWT_SECRET=your_random_secret_string
```

### How to Get API Keys:

1. **Finnhub**: [finnhub.io](https://finnhub.io) → Register → Get free API key
2. **Alpha Vantage**: [alphavantage.co](https://www.alphavantage.co/support/#api-key) → Get free API key
3. **FMP**: [financialmodelingprep.com](https://financialmodelingprep.com/developer/docs) → Register
4. **Google Gemini**: [ai.google.dev](https://ai.google.dev) → Get API key
5. **Reddit**: [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) → Create app

---

## 🎨 Demo Optimization for Recruiters

### 1. **Create a Demo Dataset**:
```bash
# Run this to populate demo data
cd ml_backend
python scripts/create_demo_data.py
```

### 2. **Add Resume Integration**:
```typescript
// Add to your frontend
const DEMO_CONFIG = {
  showDemoNotice: true,
  resumeLink: "https://your-resume-link.com",
  githubRepo: "https://github.com/yourusername/stockpredict-ai",
  linkedinProfile: "https://linkedin.com/in/your-profile"
}
```

### 3. **Performance Optimizations**:
```javascript
// Enable caching for demo
export const DEMO_CACHE_CONFIG = {
  predictions: 5 * 60 * 1000,    // 5 minutes
  stockPrices: 30 * 1000,        // 30 seconds  
  sentiment: 10 * 60 * 1000      // 10 minutes
}
```

---

## 🔧 Troubleshooting Common Deployment Issues

### Frontend Issues:
```bash
# Build fails
npm run build --verbose

# Environment variables not loading
echo $NEXT_PUBLIC_API_URL

# Production build too large
npm run analyze
```

### Backend Issues:
```bash
# MongoDB connection fails
ping cluster0.xxxxx.mongodb.net

# API keys not working
curl "https://api.finnhub.io/api/v1/quote?symbol=AAPL&token=YOUR_KEY"

# Memory issues
node --max-old-space-size=4096 src/server.js
```

### ML Backend Issues:
```bash
# Python dependencies fail
pip install --no-cache-dir -r requirements.txt

# TensorFlow issues on ARM
pip install tensorflow-cpu

# Model loading fails
ls -la models/
```

---

## 🌟 **Recommended Final Setup for Resume**

```markdown
## StockPredict AI - Live Demo

🔗 **Live Application**: https://stockpredict-ai.railway.app
📊 **API Documentation**: https://stockpredict-ml.railway.app/docs
💻 **Source Code**: https://github.com/yourusername/stockpredict-ai

### Tech Stack:
- Frontend: Next.js, TypeScript, Tailwind CSS
- Backend: Node.js, Express, MongoDB
- ML: Python, FastAPI, TensorFlow, XGBoost
- Deployment: Railway, MongoDB Atlas

### Features:
- Real-time stock predictions for S&P 100
- Multi-source sentiment analysis
- Interactive data visualizations
- RESTful APIs with comprehensive documentation
```

---

## 💡 Quick Tips for Recruiters

1. **Include Demo Credentials** if needed
2. **Add Loading States** for better UX during API calls
3. **Create a Help/Tour** feature explaining key functionalities
4. **Optimize for Mobile** viewing
5. **Add Error Boundaries** for graceful failure handling
6. **Include Performance Metrics** in your demo

---

## 🚀 One-Click Deploy Buttons

Add these to your README.md for easy deployment:

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template/your-template)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/yourusername/stockpredict-ai)

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/yourusername/stockpredict-ai)

---

This setup will give recruiters immediate access to see your full-stack AI application in action! 🎯 