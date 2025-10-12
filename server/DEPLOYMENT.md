# Deployment Guide

## Pre-Deployment Checklist

### ✅ Required Files
- [x] `server.py` - Main Flask application
- [x] `requirements.txt` - Python dependencies
- [x] `Procfile` - Deployment command
- [x] `runtime.txt` - Python version
- [x] `models/digit_model.pth` - Trained model file
- [x] `README.md` - Documentation
- [x] `.gitignore` - Ignore unnecessary files

### ⚙️ Configuration Check

1. **Model File Location**
   ```
   server/
   └── models/
       └── digit_model.pth  ✅
   ```

2. **Debug Mode**
   - For production: Set `debug=False` in `server.py`
   - Current: `debug=True` (for development)

3. **CORS Settings**
   - Currently allows all origins (`*`)
   - For production, consider restricting to your frontend domain

## Deployment Steps

### Option 1: Railway (Recommended)

1. **Push to GitHub**
   ```bash
   cd server
   git add .
   git commit -m "Add digit recognizer API"
   git push origin main
   ```

2. **Deploy on Railway**
   - Go to [Railway](https://railway.app)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway auto-detects configuration
   - Click "Deploy"

3. **Verify Deployment**
   - Wait for build to complete
   - Get the public URL from Railway dashboard
   - Test: `curl https://your-app.railway.app/health`

### Option 2: Heroku

1. **Install Heroku CLI**
   ```bash
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Login and Create App**
   ```bash
   heroku login
   cd server
   heroku create your-app-name
   ```

3. **Deploy**
   ```bash
   git add .
   git commit -m "Deploy digit recognizer"
   git push heroku main
   ```

4. **Verify**
   ```bash
   heroku open /health
   ```

### Option 3: Docker (Any Platform)

1. **Create Dockerfile** (in `server/` directory)
   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   EXPOSE 5000

   CMD ["gunicorn", "server:app", "--bind", "0.0.0.0:5000"]
   ```

2. **Build and Run**
   ```bash
   docker build -t digit-recognizer .
   docker run -p 5000:5000 digit-recognizer
   ```

3. **Deploy to any Docker-compatible platform**
   - Google Cloud Run
   - AWS ECS
   - Azure Container Instances
   - DigitalOcean App Platform

## Post-Deployment

### 1. Test Endpoints

```bash
# Health check
curl https://your-app.railway.app/health

# Test with image (replace with your URL)
curl -X POST https://your-app.railway.app/predict-image \
  -F "image=@test_digit.png"
```

### 2. Monitor Logs

**Railway:**
- View logs in Railway dashboard
- Or use CLI: `railway logs`

**Heroku:**
```bash
heroku logs --tail
```

### 3. Update Frontend

Update your frontend to use the deployed API URL:
```javascript
const API_URL = 'https://your-app.railway.app';
```

## Troubleshooting

### Issue: Model file not found
**Solution:** Ensure `models/digit_model.pth` is committed to git and deployed

### Issue: Out of memory
**Solution:** Upgrade to a plan with more RAM (model needs ~100MB)

### Issue: CORS errors
**Solution:** Check CORS settings in `server.py`, ensure frontend URL is allowed

### Issue: Slow cold starts
**Solution:** 
- Use a paid plan to prevent sleeping
- Or implement health check pings

## Security Recommendations

1. **Rate Limiting**
   ```bash
   pip install flask-limiter
   ```
   Add to `server.py`:
   ```python
   from flask_limiter import Limiter
   limiter = Limiter(app, default_limits=["100 per hour"])
   ```

2. **API Key (Optional)**
   - Add API key authentication for production
   - Use environment variables for keys

3. **CORS**
   - Restrict to your frontend domain:
   ```python
   CORS(app, origins=['https://yourfrontend.com'])
   ```

## Monitoring

Recommended tools:
- **Uptime Monitoring**: UptimeRobot, Pingdom
- **Error Tracking**: Sentry
- **Analytics**: Google Analytics, Mixpanel

## Cost Estimates

- **Railway**: Free tier (500 hours/month), then $5-20/month
- **Heroku**: $7/month (Eco plan)
- **Render**: Free tier available, $7/month for paid

## Support

For issues, check:
1. Railway/Heroku logs
2. GitHub repository issues
3. README.md for API documentation

