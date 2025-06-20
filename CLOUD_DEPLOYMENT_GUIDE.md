# Cloud Deployment Guide for Stock Prediction AI

This guide provides a comprehensive strategy for deploying your stock prediction system to the cloud with 24/7 availability, automated sentiment pipeline, and model management.

## Overview

Your deployment includes:
- **24/7 ML Backend API** for real-time predictions
- **Automated Sentiment Pipeline** (every 4 hours)
- **Model Training & Upload** (weekly)
- **Daily Data Ingestion**
- **Cloud Model Storage** (AWS S3 + Google Cloud Storage)

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   ML Backend     │    │  Cloud Storage  │
│   (Vercel)      │◄──►│   (Render)       │◄──►│  (S3/GCS)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                         │
                              ▼                         ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │    MongoDB       │    │  Model Files    │
                       │   (Atlas)        │    │  (.h5, .joblib) │
                       └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  Cron Jobs       │
                       │  - Sentiment     │
                       │  - Training      │
                       │  - Data Ingest   │
                       └──────────────────┘
```

## Step-by-Step Deployment

### 1. Cloud Storage Setup

#### Option A: AWS S3 (Recommended)
```bash
# Create S3 bucket for models
aws s3 mb s3://your-stockpredict-models
aws s3api put-bucket-versioning --bucket your-stockpredict-models --versioning-configuration Status=Enabled
```

#### Option B: Google Cloud Storage
```bash
# Create GCS bucket
gsutil mb gs://your-stockpredict-models
gsutil versioning set on gs://your-stockpredict-models
```

### 2. Environment Variables Setup

Add these to your Render environment:

```bash
# Database
MONGODB_URI=your_mongodb_connection_string

# API Keys
ALPHAVANTAGE_API_KEY=your_alphavantage_key
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_reddit_user_agent
FRED_API_KEY=your_fred_api_key
FINVIZ_API_KEY=your_finviz_key
JWT_SECRET_KEY=your_jwt_secret

# Cloud Storage (choose one or both for redundancy)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_S3_BUCKET_MODELS=your-stockpredict-models

# OR/AND Google Cloud Storage
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GOOGLE_CLOUD_STORAGE_BUCKET=your-stockpredict-models
```

### 3. Initial Model Upload

Before deploying, upload your existing models:

```python
# Run this locally to upload existing models
python scripts/train_and_upload_models.py
```

### 4. Deploy to Render

The `render.yaml` file is already configured with:

- **ML Backend API**: Runs 24/7 on `stock-predictor-ml-api` service
- **Sentiment Cron**: Runs every 4 hours
- **Training Cron**: Runs weekly (Sunday 2 AM UTC)
- **Data Ingestion Cron**: Runs daily (6 AM UTC)

Deploy using:
```bash
# Push to GitHub and connect to Render
git add .
git commit -m "Add cloud deployment configuration"
git push origin main
```

### 5. Verify Deployment

Check your deployment:

1. **API Health**: `https://your-app.onrender.com/health`
2. **Models Loaded**: `https://your-app.onrender.com/models`
3. **Predictions**: `https://your-app.onrender.com/api/v1/predictions/AAPL`

## Automated Workflows

### Sentiment Pipeline (Every 4 Hours)
```
0 */4 * * * → sentiment_cron.py
├── Fetch FinViz sentiment
├── Fetch Reddit sentiment  
├── Fetch RSS news sentiment
├── Fetch SeekingAlpha sentiment
└── Store in MongoDB
```

### Model Training (Weekly)
```
0 2 * * 0 → train_and_upload_models.py
├── Load latest data from MongoDB
├── Train models for top 20 tickers
├── Save models locally
├── Upload to S3/GCS
└── Update model registry
```

### Data Ingestion (Daily)
```
0 6 * * * → daily_data_ingestion.py
├── Fetch market data for all tickers
├── Store in MongoDB
└── Log ingestion metadata
```

## Model Management

### Automatic Model Downloads
When the API starts, it automatically:
1. Checks for local models
2. Downloads from cloud storage if needed
3. Creates model registry
4. Loads models into memory

### Model Versioning
- Models are stored with timestamps
- Old versions are kept for rollback
- Training metadata tracks model performance

## Monitoring & Maintenance

### Health Checks
- API health endpoint: `/health`
- Model availability: `/models`
- System metrics via Render dashboard

### Logs Monitoring
```bash
# View real-time logs
render logs -s stock-predictor-ml-api --tail

# Check cron job logs
render logs -s sentiment-pipeline-cron
render logs -s model-training-cron
```

### Performance Tuning
- Render Standard plan for sufficient memory
- Redis for caching (optional)
- Database indexing for faster queries

## Cost Optimization

### Render Pricing
- **ML Backend**: ~$25/month (Standard plan)
- **Cron Jobs**: $7/month each × 3 = $21/month
- **Total Render**: ~$46/month

### Storage Costs
- **AWS S3**: ~$1-5/month (model files)
- **MongoDB Atlas**: Free tier or ~$9/month

### Total Monthly Cost: ~$50-60

## Scaling Considerations

### High Availability
- Use multiple Render regions
- Database replication
- Load balancing for heavy traffic

### Performance Scaling
- Add Redis for caching
- Implement model caching
- Use CDN for static assets

## Troubleshooting

### Common Issues

1. **Models not loading**
   ```bash
   # Check if models exist in cloud storage
   aws s3 ls s3://your-bucket/models/ --recursive
   
   # Manual download
   python scripts/download_models.py
   ```

2. **Sentiment pipeline failing**
   ```bash
   # Check API keys
   curl "https://your-app.onrender.com/api/v1/sentiment/AAPL"
   ```

3. **Out of memory errors**
   - Upgrade to Render Pro plan
   - Reduce model batch sizes
   - Implement model lazy loading

### Recovery Procedures

1. **Rollback models**: Keep previous model versions in cloud storage
2. **Data recovery**: MongoDB backups and point-in-time recovery
3. **Service restart**: Use Render dashboard or API

## Security Best Practices

- Use environment variables for all secrets
- Rotate API keys regularly
- Implement rate limiting
- Use HTTPS for all endpoints
- Regular dependency updates

## Next Steps

1. **Deploy basic setup** using the provided configuration
2. **Monitor performance** for first week
3. **Optimize based on usage** patterns
4. **Add monitoring dashboards** (optional)
5. **Implement alerts** for critical failures

## Support

For issues with this deployment:
1. Check Render logs first
2. Verify environment variables
3. Test individual components
4. Monitor cloud storage costs
5. Review MongoDB connection limits

This setup provides a robust, scalable, and cost-effective solution for running your stock prediction AI system 24/7 in the cloud. 