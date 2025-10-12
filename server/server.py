# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)  

# Define the same transform as during training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64*7*7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Get model path (compatible with local and deployment environments)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'models', 'digit_model.pth')

# Fallback paths
if not os.path.exists(model_path):
    model_path = os.path.join(current_dir, 'digit_model.pth')
if not os.path.exists(model_path):
    model_path = os.path.join(os.path.dirname(current_dir), 'models', 'digit_model.pth')

model = SimpleCNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
print(f"✅ Model loaded successfully: {model_path}")

def preprocess_image_from_pil(pil_image):
    """Preprocess image from PIL Image - consistent with training"""
    img = pil_image.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img).astype(np.float32)  # Keep float32, range [0, 255]
    
    # Smart inversion: MNIST is white digit on black background
    # If white background detected, invert it
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    
    # Apply the same transform as training
    # ToTensor (doesn't divide by 255 for float32) + Normalize((0.5,), (0.5,))
    # Result: (x - 0.5) / 0.5, range [-1, 509]
    img_tensor = transform(img_array)
    return img_tensor.unsqueeze(0)

@app.route('/predict-image', methods=['POST'])
def predict_image():
    """Accept image file upload"""
    try:
        # Check if file exists
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded, please use key="image"'}), 400
        
        file = request.files['image']
        
        # Check filename
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Read and preprocess image
        img = Image.open(file.stream)
        image_tensor = preprocess_image_from_pil(img)
        
        # Predict
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted = torch.argmax(output, 1).item()
            confidence = probabilities[0][predicted].item()
        
        return jsonify({
            'digit': int(predicted),
            'confidence': float(confidence),
            'probabilities': probabilities[0].tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-base64', methods=['POST'])
def predict_base64():
    """Accept base64 encoded image"""
    try:
        data = request.json
        
        if 'image' not in data:
            return jsonify({'error': 'Missing image field'}), 400
        
        # Decode base64
        import base64
        img_data = data['image']
        # Remove data:image/png;base64, prefix if exists
        if ',' in img_data:
            img_data = img_data.split(',')[1]
        
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes))
        
        # Preprocess image
        image_tensor = preprocess_image_from_pil(img)
        
        # Predict
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted = torch.argmax(output, 1).item()
            confidence = probabilities[0][predicted].item()
        
        return jsonify({
            'digit': int(predicted),
            'confidence': float(confidence),
            'probabilities': probabilities[0].tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
