# 1. Classic CNN implementation, for classification applications

A traditional Convolutional Neural Network implementation for image classification on the CIFAR-10 dataset.

## 📊 Results
- **Final Validation Accuracy**: 80.5%
- **Training Epochs**: 50
- **Optimizer**: SGD (lr=0.008)
- **Loss Function**: CrossEntropyLoss

## 🏗️ Architecture
```python
Conv2d(3, 64, kernel_size=4) → ReLU → MaxPool2d(3,2)
Conv2d(64, 192, kernel_size=4) → ReLU → MaxPool2d(3,2)  
Conv2d(192, 384, kernel_size=3) → ReLU
Conv2d(384, 256, kernel_size=3) → ReLU
Conv2d(256, 256, kernel_size=3) → ReLU → MaxPool2d(3,2)
Linear(1024, 512) → ReLU → Dropout(0.5)
Linear(512, 256) → ReLU → Dropout(0.5)
Linear(256, 10)
```

## 📁 Project Structure
```
Classic_CNN/
│
├── classic_cnn.ipynb  # Main training notebook
├── classic_cnn.pt     # Trained model weights
```

## 📸 Screenshots
 - Training Progress: ![](Classic_CNN/media/training_progress.png)
 - Classification Sample: ![](Classic_CNN/media/sample1.png)

## 🚀 Features
- Data preprocessing with normalization (mean=[0.4700, 0.4740, 0.4749], std=[0.2485, 0.2502, 0.2537])
- Batch training with DataLoader
- GPU acceleration support


-------
# 2. Faster R-CNN implementation

