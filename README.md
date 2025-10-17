# 🖼️ CIFAR-10 Computer Vision Demo with Docker

A comprehensive demonstration of deep learning for computer vision using PyTorch, Docker, and modern MLOps practices. This project trains a Convolutional Neural Network (CNN) on the CIFAR-10 dataset with production-ready training practices.

## 📚 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Training Details](#training-details)
- [Monitoring with TensorBoard](#monitoring-with-tensorboard)
- [Docker Hub Deployment](#docker-hub-deployment)
- [Model Serving with TorchServe](#model-serving-with-torchserve)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Learning Resources](#learning-resources)

## 🎯 Overview

This project demonstrates:
- **Deep Learning**: CNN architecture for image classification
- **PyTorch Best Practices**: Mixed precision training, gradient clipping, learning rate scheduling
- **Containerization**: Docker for reproducible environments
- **Experiment Tracking**: TensorBoard for visualization
- **MLOps**: Automated testing, checkpointing, and model deployment

### Dataset: CIFAR-10

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes:
- ✈️ Airplane
- 🚗 Automobile  
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐕 Dog
- 🐸 Frog
- 🐴 Horse
- 🚢 Ship
- 🚚 Truck

**Split**: 50,000 training images + 10,000 test images

## ✨ Features

### Modern Training Techniques
- ✅ **Mixed Precision Training** (AMP) for faster computation
- ✅ **OneCycleLR Scheduler** for optimal learning rate scheduling
- ✅ **Gradient Clipping** to prevent exploding gradients
- ✅ **Label Smoothing** for better generalization
- ✅ **Early Stopping** with patience to prevent overfitting
- ✅ **Comprehensive Checkpointing** with state recovery
- ✅ **Data Augmentation** (RandomCrop, RandomHorizontalFlip, Normalization)

### Production Features
- 🐳 **Docker Support** for reproducible environments
- 📊 **TensorBoard Integration** for real-time monitoring
- 💾 **Automatic Model Saving** (best model + last checkpoint)
- 🔄 **Docker Compose** for easy orchestration
- 📦 **UV Package Manager** for fast dependency management
- 🚀 **TorchServe Integration** for production model serving
- 🔌 **REST API** for real-time inference

## 📋 Prerequisites

- **Docker** (version 20.10+)
- **Docker Compose** (version 2.0+)
- **Optional**: Docker Hub account (for publishing)

No Python installation required! Everything runs in Docker containers.

## 🚀 Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/semilleroCV/demo-docker.git
   cd demo-docker
   ```

2. **Train the model**
   ```bash
   docker compose up train
   ```

3. **Test the model**
   ```bash
   docker compose up test
   ```

### Option 2: Using Docker directly

```bash
# Build the image
docker build -t cifar10-training .

# Run training
docker run --rm -v $(pwd)/checkpoints:/app/checkpoints \
                -v $(pwd)/runs:/app/runs \
                -v $(pwd)/data:/app/data \
                cifar10-training python train.py

# Run testing
docker run --rm -v $(pwd)/checkpoints:/app/checkpoints \
                -v $(pwd)/data:/app/data \
                cifar10-training python test.py
```

## 📁 Project Structure

```
cv_demo/
├── 📄 train.py              # Training script with best practices
├── 📄 test.py               # Evaluation script
├── 📄 compare_improvements.py  # Compare training approaches
├── 📂 model/                # Model architecture and utilities
│   ├── __init__.py
│   ├── net.py              # CNN architecture definition
│   └── loader.py           # Data loading and preprocessing
├── 📂 checkpoints/          # Saved model checkpoints
│   ├── best_model.pth      # Best model (highest val accuracy)
│   ├── last_model.pth      # Latest checkpoint
│   └── cifar_net.pth       # Legacy format (compatibility)
├── 📂 data/                 # CIFAR-10 dataset (auto-downloaded)
├── 📂 runs/                 # TensorBoard logs
├── � model-store/          # TorchServe model archives (.mar files)
├── 📂 torchserve-config/    # TorchServe configuration
│   └── config.properties   # Server settings
├── �🐳 Dockerfile           # Container definition
├── 🐳 compose.yml          # Docker Compose configuration
├── 📄 pyproject.toml       # Python dependencies
├── 📄 cifar10_handler.py   # TorchServe custom handler
├── 📜 create_model_archive.sh  # Model packaging script
├── 📜 test_torchserve.py   # TorchServe testing utility
└── 📜 publish_docker.sh    # Docker Hub publishing script
```

## 💻 Usage

### Training

The training script implements state-of-the-art practices:

```bash
docker compose up train
```

**Training Configuration** (see `train.py`):
- **Batch Size**: 128
- **Epochs**: 100 (with early stopping)
- **Optimizer**: AdamW (lr=3e-4, weight_decay=0.01)
- **Scheduler**: OneCycleLR with cosine annealing
- **Loss**: CrossEntropyLoss with label smoothing (0.1)
- **Regularization**: Dropout (0.3) + L2 regularization

**Features**:
- Automatic mixed precision training (AMP)
- Gradient clipping (max_norm=1.0)
- Comprehensive checkpointing with full state recovery
- Early stopping (patience=15 epochs)
- Real-time logging to TensorBoard

### Testing

Evaluate the trained model on the test set:

```bash
docker compose up test
```

**Output includes**:
- Overall accuracy
- Per-class accuracy breakdown
- Sample predictions visualization

### Local Development (Without Docker)

If you prefer to run locally:

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run training
uv run python train.py

# Run testing
uv run python test.py
```

## 📈 Training Details

### Model Architecture

Custom CNN with modern design:

```
Conv2d(3, 32, 3x3) → BatchNorm → ReLU → MaxPool(2x2)
Conv2d(32, 64, 3x3) → BatchNorm → ReLU → MaxPool(2x2)
Conv2d(64, 128, 3x3) → BatchNorm → ReLU → MaxPool(2x2)
Flatten
Linear(2048 → 512) → BatchNorm → ReLU → Dropout(0.3)
Linear(512 → 256) → BatchNorm → ReLU → Dropout(0.3)
Linear(256 → 10)
```

**Total Parameters**: ~2.7M trainable parameters

### Data Augmentation

**Training transforms**:
- Random Crop (32x32 with padding=4)
- Random Horizontal Flip (p=0.5)
- Normalization (ImageNet stats)

**Test transforms**:
- Normalization only

### Learning Rate Schedule

OneCycleLR strategy:
1. **Warmup** (30% of training): lr increases from `3e-6` to `3e-3`
2. **Decay** (70% of training): lr decreases to `3e-10` with cosine annealing

This provides:
- Fast initial convergence
- Escape from local minima
- Fine-tuning in later epochs

## 📊 Monitoring with TensorBoard

View training progress in real-time using the integrated TensorBoard service:

### Option 1: Using Docker Compose (Recommended)

```bash
# Start TensorBoard (runs in background)
docker compose --profile monitoring up -d tensorboard

# Now run training (TensorBoard will update in real-time)
docker compose up train

# Stop TensorBoard when done
docker compose --profile monitoring down
```

### Option 2: Standalone TensorBoard

```bash
# Start TensorBoard manually
docker run --rm -p 6006:6006 \
    -v $(pwd)/runs:/app/runs \
    tensorflow/tensorflow:latest \
    tensorboard --logdir=/app/runs --host=0.0.0.0
```

### Accessing TensorBoard

Open http://localhost:6006 in your browser to see:
- **Scalars**: Training/validation loss and accuracy curves
- **Graphs**: Model architecture visualization
- **Distributions**: Weight and gradient histograms
- **Time Series**: Learning rate schedule
- **Images**: Sample predictions (if configured)

**Pro Tip**: Start TensorBoard before training to see metrics update in real-time!

## 🚢 Docker Hub Deployment

Publish your trained model to Docker Hub:

```bash
# Make script executable
chmod +x publish_docker.sh

# Run publishing script
./publish_docker.sh [version]

# Example: publish as version v1.0
./publish_docker.sh v1.0

# Or publish as latest
./publish_docker.sh
```

The script will:
1. ✅ Check Docker authentication
2. ✅ Build the image
3. ✅ Tag appropriately (converts username to lowercase)
4. ✅ Optional: Test locally
5. ✅ Push to Docker Hub

**Pull published image**:
```bash
docker pull <your-username>/cifar10-training:latest
```

## 🚀 Model Serving with TorchServe

Deploy your trained model as a production REST API using TorchServe!

### Step 1: Create Model Archive

After training, package your model for TorchServe:

```bash
# Make script executable
chmod +x create_model_archive.sh

# Create .mar file (model archive)
./create_model_archive.sh
```

This creates `model-store/cifar10_classifier.mar` - a packaged model ready for deployment.

### Step 2: Start TorchServe

```bash
# Start TorchServe server
docker compose --profile serving up -d torchserve

# Check if server is running
curl http://localhost:8080/ping
```

### Step 3: Register Your Model

```bash
# Register the model
curl -X POST "http://localhost:8081/models?url=cifar10_classifier.mar"

# Verify registration
curl http://localhost:8081/models
```

### Step 4: Make Predictions

**Using Python script**:
```bash
# Install requests library
pip install requests

# Make prediction
python test_torchserve.py --image path/to/image.jpg

# Check model status
python test_torchserve.py --status

# List all models
python test_torchserve.py --list
```

**Using curl**:
```bash
# Predict from image file
curl -X POST http://localhost:8080/predictions/cifar10_classifier \
     -T path/to/image.jpg

# Get prediction with detailed response
curl -X POST http://localhost:8080/predictions/cifar10_classifier \
     -T path/to/image.jpg | jq
```

### TorchServe API Endpoints

| Endpoint | Port | Purpose |
|----------|------|---------|
| `/predictions/{model_name}` | 8080 | Inference API |
| `/models` | 8081 | Model management |
| `/metrics` | 8082 | Prometheus metrics |
| `/ping` | 8080 | Health check |

### Example Response

```json
[
  {
    "predicted_class": "airplane",
    "confidence": 0.8523,
    "predictions": [
      {"class": "airplane", "confidence": 0.8523},
      {"class": "ship", "confidence": 0.0892},
      {"class": "bird", "confidence": 0.0341}
    ]
  }
]
```

### Advanced: Scale Workers

```bash
# Scale up workers for better performance
curl -X PUT "http://localhost:8081/models/cifar10_classifier?min_worker=2&max_worker=4"

# Check worker status
curl http://localhost:8081/models/cifar10_classifier
```

### Stop TorchServe

```bash
docker compose --profile serving down
```

## 📊 Results

### Expected Performance

With the implemented best practices:
- **Validation Accuracy**: 75-82%
- **Training Time**: ~30-60 minutes (CPU) / ~10-15 minutes (GPU)
- **Convergence**: Typically in 40-60 epochs

### Sample Output

```
Epoch 45/100
------------------------------------------------------------
Loss: 0.324 | Acc: 88.54% | GradNorm: 0.847
Validation - Loss: 0.512 | Accuracy: 81.23%
✓ Best model saved with accuracy: 81.23%

Per-Class Accuracy:
============================================================
airplane  : 84.2% (842/1000)
automobile: 88.1% (881/1000)
bird      : 73.5% (735/1000)
cat       : 68.9% (689/1000)
deer      : 77.8% (778/1000)
dog       : 75.3% (753/1000)
frog      : 87.6% (876/1000)
horse     : 86.4% (864/1000)
ship      : 89.2% (892/1000)
truck     : 87.3% (873/1000)
============================================================
```

## 🔧 Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
```bash
# Reduce batch size in train.py
BATCH_SIZE = 64  # Instead of 128
```

**2. Docker Build Fails**
```bash
# Clear Docker cache and rebuild
docker compose build --no-cache train
```

**3. Permission Denied on publish_docker.sh**
```bash
chmod +x publish_docker.sh
```

**4. TensorBoard Not Loading**
```bash
# Ensure runs directory exists and has data
ls -la runs/
```

**5. Model Not Found During Testing**
```bash
# Verify checkpoint exists
ls -la checkpoints/
# Train first if no checkpoints exist
docker compose up train
```

## 📚 Learning Resources

### Computer Vision Fundamentals
- [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/)
- [Deep Learning Book - Chapter 9](https://www.deeplearningbook.org/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

### PyTorch
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)

### Docker & MLOps
- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)

### Key Papers
- [Bag of Tricks for Image Classification](https://arxiv.org/abs/1812.01187)
- [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120)
- [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629)

## 🤝 Contributing

This is an educational project. Suggestions and improvements are welcome!

## 📝 License

MIT License - Feel free to use for educational purposes.

## 👥 Authors

Computer Vision Seminar - University Demo Project

---

**Happy Learning! 🎓**

For questions or issues, please open a GitHub issue or contact the course instructor.
