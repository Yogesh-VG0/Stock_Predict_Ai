#!/bin/bash

# StockPredict AI Deployment Setup Script
# This script helps prepare your project for deployment

set -e

echo "🚀 StockPredict AI Deployment Setup"
echo "===================================="

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_requirements() {
    print_status "Checking requirements..."
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js 18+ from https://nodejs.org/"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.9+ from https://python.org/"
        exit 1
    fi
    
    # Check Docker (optional)
    if command -v docker &> /dev/null; then
        print_status "Docker is available for containerized deployment"
    else
        print_warning "Docker not found. Install Docker for containerized deployment"
    fi
    
    # Check Git
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed. Please install Git from https://git-scm.com/"
        exit 1
    fi
    
    print_status "✅ All required tools are available"
}

# Setup environment files
setup_environment() {
    print_status "Setting up environment files..."
    
    if [ ! -f ".env" ]; then
        if [ -f "env.template" ]; then
            cp env.template .env
            print_status "Created .env file from template"
            print_warning "Please edit .env file and add your API keys!"
        else
            print_error "env.template not found. Please ensure you're in the project root directory."
            exit 1
        fi
    else
        print_warning ".env file already exists. Skipping creation."
    fi
    
    # Create backend .env if it doesn't exist
    if [ ! -f "backend/.env" ]; then
        cp .env backend/.env
        print_status "Created backend/.env file"
    fi
    
    # Create ml_backend .env if it doesn't exist
    if [ ! -f "ml_backend/.env" ]; then
        cp .env ml_backend/.env
        print_status "Created ml_backend/.env file"
    fi
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    # Install frontend dependencies
    print_status "Installing frontend dependencies..."
    npm install
    
    # Install backend dependencies
    print_status "Installing backend dependencies..."
    cd backend && npm install && cd ..
    
    # Install ML backend dependencies
    print_status "Installing ML backend dependencies..."
    cd ml_backend && pip install -r requirements.txt && cd ..
    
    print_status "✅ All dependencies installed"
}

# Build the project
build_project() {
    print_status "Building the project..."
    
    # Build frontend
    print_status "Building frontend..."
    npm run build
    
    print_status "✅ Project built successfully"
}

# Check API keys
check_api_keys() {
    print_status "Checking API keys in .env file..."
    
    if [ ! -f ".env" ]; then
        print_error ".env file not found. Please run the setup first."
        return 1
    fi
    
    # Check for placeholder values
    if grep -q "your_.*_here" .env; then
        print_warning "Found placeholder values in .env file. Please update with real API keys:"
        grep "your_.*_here" .env
        print_warning "Get free API keys from:"
        echo "  • Finnhub: https://finnhub.io/register"
        echo "  • Alpha Vantage: https://www.alphavantage.co/support/#api-key"
        echo "  • FMP: https://financialmodelingprep.com/developer/docs"
        echo "  • Google Gemini: https://ai.google.dev"
        return 1
    else
        print_status "✅ API keys appear to be configured"
    fi
}

# Test local deployment
test_local() {
    print_status "Testing local deployment with Docker Compose..."
    
    if ! command -v docker-compose &> /dev/null && ! command -v docker &> /dev/null; then
        print_error "Docker is required for local testing. Please install Docker."
        return 1
    fi
    
    # Start services
    docker-compose up -d
    
    # Wait for services to start
    sleep 10
    
    # Test endpoints
    print_status "Testing endpoints..."
    
    # Test frontend
    if curl -s http://localhost:3000 > /dev/null; then
        print_status "✅ Frontend is responding"
    else
        print_error "❌ Frontend is not responding"
    fi
    
    # Test backend
    if curl -s http://localhost:5000/health > /dev/null; then
        print_status "✅ Backend is responding"
    else
        print_error "❌ Backend is not responding"
    fi
    
    # Test ML backend
    if curl -s http://localhost:8000/health > /dev/null; then
        print_status "✅ ML Backend is responding"
    else
        print_error "❌ ML Backend is not responding"
    fi
    
    print_status "Local testing complete. Access your app at http://localhost:3000"
    print_status "To stop services: docker-compose down"
}

# Generate deployment guide
generate_deployment_guide() {
    print_status "Generating personalized deployment guide..."
    
    cat > DEPLOYMENT_INSTRUCTIONS.md << EOF
# Your StockPredict AI Deployment Instructions

## Quick Deploy to Railway (Recommended)

1. **Push to GitHub** (if not already done):
   \`\`\`bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   \`\`\`

2. **Sign up for Railway**: https://railway.app

3. **Deploy from GitHub**:
   - Connect your GitHub account
   - Select this repository
   - Railway will auto-detect your services

4. **Add Environment Variables** (in Railway dashboard):
   \`\`\`env
   $(cat .env | grep -v '^#' | grep -v '^$')
   \`\`\`

5. **Update URLs** in Railway after deployment:
   - Update NEXT_PUBLIC_API_URL with your backend URL
   - Update ML_API_URL with your ML backend URL

## Your Project URLs (update after deployment):
- 🌐 Frontend: https://your-frontend.railway.app
- 🔧 Backend: https://your-backend.railway.app  
- 🤖 ML API: https://your-ml-backend.railway.app/docs

## Add to Your Resume:
\`\`\`markdown
**StockPredict AI - Live Demo**
🔗 Live App: https://your-frontend.railway.app
📊 API Docs: https://your-ml-backend.railway.app/docs
💻 Source: https://github.com/$(git config user.name || echo "yourusername")/stockpredict-ai
\`\`\`

## Alternative: Vercel + Render
1. Deploy frontend to Vercel: Connect GitHub → Auto-deploy
2. Deploy backends to Render: Create web services from GitHub

## Need Help?
- Check DEPLOYMENT_GUIDE.md for detailed instructions
- Ensure all API keys are set in environment variables
- Test locally first with: docker-compose up
EOF

    print_status "✅ Deployment instructions created: DEPLOYMENT_INSTRUCTIONS.md"
}

# Main menu
show_menu() {
    echo ""
    echo "Select an option:"
    echo "1) Full setup (recommended for first time)"
    echo "2) Check requirements only"
    echo "3) Setup environment files"
    echo "4) Install dependencies"
    echo "5) Build project"
    echo "6) Check API keys"
    echo "7) Test local deployment"
    echo "8) Generate deployment guide"
    echo "9) Exit"
    echo ""
    read -p "Enter your choice (1-9): " choice
}

# Main execution
main() {
    echo ""
    check_requirements
    
    while true; do
        show_menu
        case $choice in
            1)
                setup_environment
                install_dependencies
                check_api_keys
                build_project
                generate_deployment_guide
                print_status "🎉 Setup complete! Check DEPLOYMENT_INSTRUCTIONS.md for next steps."
                break
                ;;
            2)
                check_requirements
                ;;
            3)
                setup_environment
                ;;
            4)
                install_dependencies
                ;;
            5)
                build_project
                ;;
            6)
                check_api_keys
                ;;
            7)
                test_local
                ;;
            8)
                generate_deployment_guide
                ;;
            9)
                print_status "Goodbye! 👋"
                exit 0
                ;;
            *)
                print_error "Invalid option. Please choose 1-9."
                ;;
        esac
    done
}

# Run main function
main 