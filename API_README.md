# Digit Recognizer API - Deployment

## 🚀 Quick Deploy to Railway

### Step 1: Copy Model File
```bash
# Copy model from server/models/ to root models/
mkdir models
copy server\models\digit_model.pth models\digit_model.pth
```

Or manually:
1. Create `models/` folder in root directory
2. Copy `server/models/digit_model.pth` to `models/digit_model.pth`

### Step 2: Verify Files Structure
```
Kaggle_Competitions/  (root)
├── app.py                 ✅
├── requirements.txt       ✅
├── Procfile              ✅
├── runtime.txt           ✅
├── models/
│   └── digit_model.pth   ✅ IMPORTANT!
└── ...
```

### Step 3: Test Locally (Optional)
```bash
pip install -r requirements.txt
python app.py
```

Visit: `http://localhost:5000/health`

### Step 4: Push to GitHub
```bash
git add app.py requirements.txt Procfile runtime.txt models/
git commit -m "Add digit recognizer API for deployment"
git push
```

### Step 5: Deploy on Railway

1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your repository
5. Railway will auto-detect all files
6. Wait for deployment to complete
7. Get your public URL

### Step 6: Test Deployed API

```bash
# Test health
curl https://your-app.railway.app/health

# Test with an image
curl -X POST https://your-app.railway.app/predict-image \
  -F "image=@digit.png"
```

## API Endpoints

### GET /
Home page with API information

### GET /health
Health check - returns `{"status": "healthy"}`

### POST /predict-image
Upload image file for prediction

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `image` (file)

**Response:**
```json
{
  "digit": 8,
  "confidence": 0.9876,
  "probabilities": [0.001, 0.002, ..., 0.9876, ...]
}
```

### POST /predict-base64
Send base64 encoded image

**Request:**
- Method: POST
- Content-Type: application/json
- Body: `{"image": "base64_string_here"}`

**Response:** Same as /predict-image

## Environment Variables

No environment variables required. The app automatically:
- Uses PORT from Railway (default: 5000)
- Loads model from `models/digit_model.pth`
- Runs in production mode (debug=False)

## Frontend Integration

Update your React app:

```javascript
// Replace localhost with your Railway URL
const API_URL = 'https://your-app.railway.app';

// Example usage
const predictDigit = async (imageFile) => {
  const formData = new FormData();
  formData.append('image', imageFile);
  
  const response = await fetch(`${API_URL}/predict-image`, {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
};
```

## Troubleshooting

### Issue: Model file not found
**Solution:** Make sure `models/digit_model.pth` exists in root directory and is committed to git

```bash
git add models/digit_model.pth -f
git commit -m "Add model file"
git push
```

### Issue: Build failed on Railway
**Solution:** Check Railway logs. Common issues:
- Missing model file
- Wrong Python version
- Dependencies not installed

### Issue: CORS error from frontend
**Solution:** API has CORS enabled for all origins. If still having issues, clear browser cache.

## Tech Stack

- **Framework:** Flask 3.0.0
- **ML:** PyTorch 2.1.0
- **Image Processing:** Pillow 10.1.0
- **Server:** Gunicorn 21.2.0
- **Python:** 3.11

## Model Details

- Architecture: CNN (2 conv layers + 2 FC layers)
- Input: 28x28 grayscale images
- Output: 10 classes (digits 0-9)
- Automatic preprocessing and color inversion

## Support

For detailed deployment guide, see: `server/DEPLOYMENT.md`
For API documentation, see: `server/README.md`

---

Made with ❤️ for digit recognition

