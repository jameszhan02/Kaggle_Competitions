# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  

# 定义和训练时完全相同的 transform
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

model = SimpleCNN()
model.load_state_dict(torch.load('../models/digit_model.pth', map_location=torch.device('cpu')))
model.eval()

def preprocess_image_from_pil(pil_image):
    """从 PIL Image 预处理图片 - 和训练时完全一致"""
    img = pil_image.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img).astype(np.float32)  # 保持 float32，范围 [0, 255]
    
    # 智能反转：MNIST 是黑底白字，如果检测到白底黑字则反转
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    
    # 应用和训练时相同的 transform
    # ToTensor (对 float32 不除以 255) + Normalize((0.5,), (0.5,))
    # 结果：(x - 0.5) / 0.5，范围 [-1, 509]
    img_tensor = transform(img_array)
    return img_tensor.unsqueeze(0)

@app.route('/predict-image', methods=['POST'])
def predict_image():
    """接收图片文件上传"""
    try:
        # 检查是否有文件
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图片文件，请使用 key="image"'}), 400
        
        file = request.files['image']
        
        # 检查文件名
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        
        # 读取并预处理图片
        img = Image.open(file.stream)
        image_tensor = preprocess_image_from_pil(img)
        
        # 预测
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
    """接收 base64 编码的图片"""
    try:
        data = request.json
        
        if 'image' not in data:
            return jsonify({'error': '缺少 image 字段'}), 400
        
        # 解码 base64
        import base64
        img_data = data['image']
        # 移除 data:image/png;base64, 前缀（如果有）
        if ',' in img_data:
            img_data = img_data.split(',')[1]
        
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes))
        
        # 预处理图片
        image_tensor = preprocess_image_from_pil(img)
        
        # 预测
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