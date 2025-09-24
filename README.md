# Classic CNN for CIFAR-10 Classification

A traditional Convolutional Neural Network implementation for image classification on the CIFAR-10 dataset.

## Dataset
- CIFAR-10: 50,000 training + 10,000 test images (32×32 RGB)
- Data normalization applied based on training set statistics
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

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
classic_cnn.ipynb  # Main training notebook
classic_cnn.pt     # Trained model weights
```

## Training Progress
![Training Progress](results/training_progress.png)
