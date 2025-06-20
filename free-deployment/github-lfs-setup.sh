#!/bin/bash

# GitHub LFS Setup Script for Free Model Storage
# This script sets up Git LFS to store your ML models for free on GitHub

echo "🚀 Setting up GitHub LFS for FREE model storage..."

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "❌ Git LFS is not installed. Please install it first:"
    echo "   - Windows: Download from https://git-lfs.github.io/"
    echo "   - macOS: brew install git-lfs"
    echo "   - Ubuntu: sudo apt install git-lfs"
    exit 1
fi

# Initialize Git LFS
echo "📦 Initializing Git LFS..."
git lfs install

# Track model files
echo "🎯 Setting up LFS tracking for model files..."
git lfs track "models/**/*.h5"          # Keras models
git lfs track "models/**/*.joblib"      # Scikit-learn models
git lfs track "models/**/*.pkl"         # Pickle files
git lfs track "models/**/*.json"        # Model metadata
git lfs track "models/**/*.txt"         # Model info files

# Check current LFS status
echo "📊 Current LFS tracking patterns:"
git lfs track

# Add .gitattributes to git
git add .gitattributes

# Check model files size
echo "📏 Checking model files size..."
if [ -d "models" ]; then
    total_size=$(du -sh models/ | cut -f1)
    echo "   Total models directory size: $total_size"
    
    # GitHub LFS free tier is 1GB storage + 1GB bandwidth
    echo "   GitHub LFS Free Tier: 1GB storage + 1GB bandwidth per month"
    echo "   If you exceed this, consider:"
    echo "   - Compress models using model quantization"
    echo "   - Use model distillation for smaller models"
    echo "   - Rotate old models periodically"
fi

# Add model files to LFS
echo "📤 Adding model files to LFS..."
if [ -d "models" ]; then
    git add models/
    echo "   ✅ Model files staged for LFS"
else
    echo "   ⚠️  No models/ directory found. Create it and add your .h5/.joblib files"
fi

# Show LFS status
echo "📋 LFS Status:"
git lfs ls-files

# Instructions for committing
echo ""
echo "🎉 GitHub LFS setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Commit the changes:"
echo "      git commit -m 'Add models to Git LFS'"
echo ""
echo "   2. Push to GitHub:"
echo "      git push origin main"
echo ""
echo "   3. Your models will be stored in GitHub LFS (FREE up to 1GB)"
echo ""
echo "📥 To download models in deployment:"
echo "   - Vercel/Netlify: Models auto-download during build"
echo "   - Manual: git lfs pull"
echo ""
echo "🔗 Model URLs will be:"
echo "   https://github.com/YOUR_USERNAME/YOUR_REPO/raw/main/models/TICKER/model_file.h5"
echo ""
echo "💡 Pro Tips:"
echo "   - Monitor LFS usage: git lfs ls-files --size"
echo "   - Check LFS quota on GitHub repo settings"
echo "   - Use model compression to reduce file sizes"
echo ""

# Create example deployment files
echo "📄 Creating example deployment configuration..."

# Create vercel.json for model download
cat > vercel.json << 'EOF'
{
  "functions": {
    "api/predict/[ticker].js": {
      "maxDuration": 30
    }
  },
  "env": {
    "MONGODB_URI": "@mongodb-uri",
    "GITHUB_REPO": "your-username/your-repo-name",
    "UPSTASH_REDIS_URL": "@upstash-redis-url"
  },
  "build": {
    "env": {
      "GITHUB_LFS_SKIP_SMUDGE": "1"
    }
  }
}
EOF

# Create netlify.toml
cat > netlify.toml << 'EOF'
[build]
  functions = "netlify/functions"
  environment = { GITHUB_LFS_SKIP_SMUDGE = "1" }

[functions]
  node_bundler = "esbuild"

[[functions]]
  name = "sentiment-finviz"
  timeout = 30

[env]
  MONGODB_URI = "your-mongodb-connection-string"
EOF

# Create railway.json
cat > railway.json << 'EOF'
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "nixpacks"
  },
  "deploy": {
    "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT"
  }
}
EOF

echo "   ✅ Created deployment configuration files"
echo ""
echo "🎊 Setup complete! Your models are now ready for FREE cloud deployment."
echo "   Total cost: $0/month 🆓" 