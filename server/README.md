# Digit Recognizer API

A Flask REST API for handwritten digit recognition using PyTorch CNN model.

## Features

- 🔢 Recognizes handwritten digits (0-9)
- 📸 Accepts image file uploads
- 🖼️ Supports base64 encoded images
- 🎯 Returns prediction with confidence scores
- 🌐 CORS enabled for cross-origin requests

## API Endpoints

### 1. Health Check

```
GET /health
```

Returns: `{"status": "healthy"}`

### 2. Predict from Image File

```
POST /predict-image
Content-Type: multipart/form-data
Body: image (file)
```

**Example with curl:**

```bash
curl -X POST http://localhost:5000/predict-image \
  -F "image=@digit.png"
```

**Response:**

```json
{
  "digit": 8,
  "confidence": 0.9876,
  "probabilities": [0.001, 0.002, ..., 0.9876, ...]
}
```

### 3. Predict from Base64

```
POST /predict-base64
Content-Type: application/json
Body: {"image": "base64_string"}
```

**Example:**

```bash
curl -X POST http://localhost:5000/predict-base64 \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/png;base64,iVBORw0KG..."}'
```

## Local Development

### Prerequisites

- Python 3.11+
- pip

### Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure model file exists:

- Place `digit_model.pth` in the `server/` directory
- Or place it in `../models/digit_model.pth`

3. Run the server:

```bash
python server.py
```

Server will start at `http://localhost:5000`

## Deployment

### Railway

1. Create a new project on [Railway](https://railway.app)

2. Connect your GitHub repository

3. Railway will automatically detect:

   - `requirements.txt` for dependencies
   - `Procfile` for start command
   - `runtime.txt` for Python version

4. Ensure `digit_model.pth` is in the repository

5. Deploy!

### Environment Variables

No environment variables required for basic deployment.

### Production Settings

For production, set `debug=False` in `server.py`:

```python
app.run(host='0.0.0.0', port=5000, debug=False)
```

## Model Details

- **Architecture**: Simple CNN
- **Input**: 28x28 grayscale images
- **Output**: 10 classes (digits 0-9)
- **Framework**: PyTorch

### Preprocessing

- Images are converted to grayscale
- Resized to 28x28 pixels
- Automatically inverted if white background is detected
- Normalized using the same transform as training

## Tech Stack

- **Framework**: Flask 3.0.0
- **ML Library**: PyTorch 2.1.0
- **Image Processing**: Pillow 10.1.0
- **Server**: Gunicorn 21.2.0

## License

MIT
