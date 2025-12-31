# Python Libraries Guide for Image Classification

This document provides a comprehensive overview of all the libraries used in this leaf classification project.

## 📦 Table of Contents

- [Basic Utilities](#basic-utilities)
- [Data Processing](#data-processing)
- [Image Processing](#image-processing)
- [PyTorch Core](#pytorch-core)
- [Data Loading](#data-loading)
- [Image Transformations](#image-transformations)
- [Machine Learning Tools](#machine-learning-tools)
- [Progress Monitoring](#progress-monitoring)
- [Workflow Integration](#workflow-integration)

---

## 🗂️ Basic Utilities

### 1. `os`

```python
import os
```

**Purpose**: Operating system interface for file path operations

**Common Use Cases**:

```python
os.path.join('data', 'images', '0.jpg')  # Join path: data/images/0.jpg
os.path.exists('model.pth')               # Check if file exists
os.listdir('data/images')                 # List all files in directory
```

---

## 📊 Data Processing

### 2. `pandas` (pd)

```python
import pandas as pd
```

**Purpose**: Data analysis and manipulation, primarily for reading CSV files

**In This Project**:

```python
# Read training data CSV
train_df = pd.read_csv('data/train.csv')
# train_df contains: image (path), label (class name)

# Data exploration
train_df.head()                    # View first 5 rows
train_df['label'].value_counts()   # Count samples per class
train_df.shape                     # Check data shape (rows, columns)

# Data manipulation
train_df['label_encoded'] = [0, 1, 2, ...]  # Add new column
```

### 3. `numpy` (np)

```python
import numpy as np
```

**Purpose**: Numerical computing library for high-performance array operations

**In This Project**:

```python
# Set random seed
np.random.seed(42)

# Array operations
predictions = np.array([0, 1, 2, 3])     # Create array
np.mean(losses)                           # Calculate mean
```

**Why Need It**: Many libraries use NumPy arrays internally, and it's essential for reproducibility.

---

## 🖼️ Image Processing

### 4. `PIL.Image`

```python
from PIL import Image
```

**PIL = Python Imaging Library (now called Pillow)**

**Purpose**: Read, process, and save images

**In This Project**:

```python
# Load image
img = Image.open('data/images/0.jpg')
print(img.size)        # (width, height), e.g., (256, 256)
print(img.mode)        # 'RGB' or 'L' (grayscale)

# Basic operations
img_resized = img.resize((224, 224))     # Resize
img_gray = img.convert('L')               # Convert to grayscale
img_rgb = img.convert('RGB')              # Ensure RGB format

# Display image
img.show()
```

### 5. `matplotlib.pyplot` (plt)

```python
import matplotlib.pyplot as plt
```

**Purpose**: Data visualization and plotting

**In This Project**:

```python
# Display images
plt.imshow(image)
plt.title('Tree Leaf')
plt.axis('off')
plt.show()

# Plot training curves
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

---

## 🔥 PyTorch Core

### 6. `torch`

```python
import torch
```

**Purpose**: Core PyTorch library for deep learning

**Main Features**:

#### Tensor Operations

```python
# Tensors - PyTorch's core data structure
x = torch.tensor([1, 2, 3])              # Create tensor
x = torch.randn(32, 3, 224, 224)         # Random tensor (batch, channels, H, W)
```

#### Device Management

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)                         # Move data to GPU
```

#### Gradient Computation

```python
x.requires_grad = True                   # Enable gradient tracking
y = x * 2
y.backward()                             # Backpropagation
print(x.grad)                            # View gradients
```

#### Model Saving/Loading

```python
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
```

### 7. `torch.nn` (nn)

```python
import torch.nn as nn
```

**Purpose**: Neural network modules including layers and loss functions

**In This Project**:

#### Network Layers

```python
nn.Linear(512, 176)              # Fully connected layer (input_dim, output_dim)
nn.Conv2d(3, 64, 3)              # Convolutional layer (in_channels, out_channels, kernel_size)
nn.ReLU()                        # Activation function
nn.Dropout(0.5)                  # Dropout layer
```

#### Loss Functions

```python
criterion = nn.CrossEntropyLoss()  # Multi-class cross-entropy loss
loss = criterion(outputs, labels)   # Compute loss
```

#### Model Modification

```python
model.fc = nn.Linear(512, num_classes)  # Replace ResNet's final layer
```

### 8. `torch.optim` (optim)

```python
import torch.optim as optim
```

**Purpose**: Optimizers for updating network weights

**In This Project**:

#### Define Optimizer

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # SGD
```

#### Training Loop Usage

```python
optimizer.zero_grad()    # Clear gradients
loss.backward()          # Backpropagation to compute gradients
optimizer.step()         # Update weights
```

#### Learning Rate Scheduler

```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
scheduler.step()         # Adjust learning rate after each epoch
```

---

## 📥 Data Loading

### 9. `torch.utils.data.Dataset` & `DataLoader`

```python
from torch.utils.data import Dataset, DataLoader
```

#### `Dataset` - Custom Dataset Class

**Purpose**: Define how to read and process individual samples

```python
class LeafDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        # Return dataset size
        return len(self.df)

    def __getitem__(self, idx):
        # Return the idx-th sample
        img_path = self.df.iloc[idx]['image']
        image = Image.open(img_path)
        label = self.df.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label
```

#### `DataLoader` - Batch Data Loader

**Purpose**: Automatically batch, shuffle, and load data with multiprocessing

```python
train_loader = DataLoader(
    dataset,
    batch_size=32,      # 32 images per batch
    shuffle=True,       # Shuffle data
    num_workers=4       # 4 parallel processes for loading
)

# Usage
for images, labels in train_loader:
    # images: (32, 3, 224, 224) - batch of 32 images
    # labels: (32,) - corresponding 32 labels
    outputs = model(images)
```

---

## 🎨 Image Transformations

### 10. `torchvision.transforms`

```python
import torchvision.transforms as transforms
```

**Purpose**: Image preprocessing and data augmentation

**In This Project**:

#### Training Transforms (with augmentation)

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),           # Resize to 224x224
    transforms.RandomHorizontalFlip(),       # Random horizontal flip
    transforms.RandomRotation(15),           # Random rotation ±15 degrees
    transforms.ColorJitter(                  # Random color adjustment
        brightness=0.2,                      # Brightness
        contrast=0.2                         # Contrast
    ),
    transforms.ToTensor(),                   # PIL Image → Tensor
    transforms.Normalize(                    # Normalization
        [0.485, 0.456, 0.406],              # ImageNet mean
        [0.229, 0.224, 0.225]               # ImageNet std
    )
])

# Apply transform
image = Image.open('0.jpg')          # PIL Image
image_tensor = train_transform(image)  # torch.Tensor (3, 224, 224)
```

#### What `ToTensor` Does:

- PIL Image (H, W, C) → Tensor (C, H, W)
- Pixel values [0, 255] → [0.0, 1.0]

#### What `Normalize` Does:

```python
# For each channel: output = (input - mean) / std
# Purpose: Match the distribution of pre-trained model's training data
```

### 11. `torchvision.models`

```python
import torchvision.models as models
```

**Purpose**: Pre-trained computer vision models

**In This Project**:

#### Load Pre-trained Model

```python
# Load pre-trained ResNet18
model = models.resnet18(pretrained=True)
# pretrained=True: Use ImageNet pre-trained weights
# pretrained=False: Random initialization

# Other available models
models.resnet34(pretrained=True)
models.resnet50(pretrained=True)
models.vgg16(pretrained=True)
models.efficientnet_b0(pretrained=True)

# View model architecture
print(model)

# Modify final layer for your task
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

---

## 🔬 Machine Learning Tools

### 12. `sklearn.model_selection.train_test_split`

```python
from sklearn.model_selection import train_test_split
```

**Purpose**: Split data into training and validation sets

**In This Project**:

```python
train_data, val_data = train_test_split(
    train_df,              # Original data
    test_size=0.2,         # 20% for validation
    random_state=42,       # Random seed
    stratify=train_df['label']  # Stratified sampling to maintain class ratios
)
# train_data: 80% for training
# val_data: 20% for validation
```

#### What `stratify` Does:

```python
# Example with 3 classes: A(100), B(50), C(50)
# Without stratify: Validation might be A(30), B(10), C(0) ❌ Imbalanced
# With stratify: Validation will be A(20), B(10), C(10) ✅ Maintains ratio
```

### 13. `sklearn.preprocessing.LabelEncoder`

```python
from sklearn.preprocessing import LabelEncoder
```

**Purpose**: Convert text labels to numerical values

**In This Project**:

```python
# Original labels are strings
labels = ['maclura_pomifera', 'ulmus_rubra', 'maclura_pomifera', ...]

# Convert to numbers
label_encoder = LabelEncoder()
label_encoder.fit(labels)
encoded = label_encoder.transform(labels)
# Result: [0, 1, 0, ...]

# Inverse transformation
decoded = label_encoder.inverse_transform([0, 1, 0])
# Result: ['maclura_pomifera', 'ulmus_rubra', 'maclura_pomifera']

# Practical usage
train_df['label_encoded'] = label_encoder.fit_transform(train_df['label'])
num_classes = len(label_encoder.classes_)  # Total number of classes
```

---

## 📊 Progress Monitoring

### 14. `tqdm`

```python
from tqdm import tqdm
```

**Purpose**: Display progress bars

**In This Project**:

```python
# Show progress in training loop
for images, labels in tqdm(train_loader, desc='Training'):
    # Training code...
    pass

# Output example:
# Training: 100%|██████████| 573/573 [01:23<00:00, 6.87it/s, loss=0.234, acc=92.5]

# Update progress bar with custom info
pbar = tqdm(train_loader)
for images, labels in pbar:
    loss = train_step(images, labels)
    pbar.set_postfix({'loss': loss})  # Display loss in real-time
```

---

## 🎯 Workflow Integration

Here's how these libraries work together in the complete pipeline:

```python
# 1. Read CSV → pandas
train_df = pd.read_csv('train.csv')

# 2. Encode labels → sklearn
encoder = LabelEncoder()
train_df['label_encoded'] = encoder.fit_transform(train_df['label'])

# 3. Split dataset → sklearn
train_data, val_data = train_test_split(train_df, test_size=0.2)

# 4. Define image transforms → torchvision.transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 5. Create dataset → torch.utils.data.Dataset
class MyDataset(Dataset):
    def __getitem__(self, idx):
        img = Image.open(path)      # PIL reads image
        img = self.transform(img)   # Apply transforms
        return img, label

# 6. Create data loader → DataLoader
train_loader = DataLoader(dataset, batch_size=32)

# 7. Load model → torchvision.models
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, num_classes)  # torch.nn modifies network

# 8. Define optimizer → torch.optim
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 9. Training loop → tqdm shows progress
for images, labels in tqdm(train_loader):
    images = images.to(device)        # torch moves to GPU
    outputs = model(images)            # Forward pass
    loss = criterion(outputs, labels)  # Calculate loss
    loss.backward()                    # Backpropagation
    optimizer.step()                   # Update weights

# 10. Visualization → matplotlib
plt.plot(losses)
plt.show()
```

---

## 📌 Quick Reference Table

| Library                  | Main Purpose        | Core Use                         |
| ------------------------ | ------------------- | -------------------------------- |
| `os`                     | File system         | Path joining, file checking      |
| `pandas`                 | Data processing     | Read CSV, data analysis          |
| `numpy`                  | Numerical computing | Array operations, random numbers |
| `PIL`                    | Image processing    | Read images                      |
| `matplotlib`             | Visualization       | Display images, plot curves      |
| **`torch`**              | Deep learning       | Tensors, GPU, model saving       |
| **`torch.nn`**           | Neural networks     | Network layers, loss functions   |
| **`torch.optim`**        | Optimizers          | Update weights                   |
| **`Dataset/DataLoader`** | Data loading        | Batch loading                    |
| **`transforms`**         | Image transforms    | Preprocessing, augmentation      |
| **`models`**             | Pre-trained models  | ResNet, VGG, etc.                |
| `sklearn`                | Machine learning    | Data splitting, label encoding   |
| `tqdm`                   | Progress bar        | Display training progress        |

---

## 🚀 Getting Started

To install all required libraries:

```bash
pip install torch torchvision pandas numpy Pillow matplotlib scikit-learn tqdm
```

Or with conda:

```bash
conda install pytorch torchvision -c pytorch
conda install pandas numpy pillow matplotlib scikit-learn tqdm
```

---

## 📚 Further Reading

- **PyTorch Official Documentation**: https://pytorch.org/docs/
- **Torchvision Models**: https://pytorch.org/vision/stable/models.html
- **Data Augmentation Guide**: https://pytorch.org/vision/stable/transforms.html
- **Transfer Learning Tutorial**: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

---

_Last updated: December 30, 2025_
