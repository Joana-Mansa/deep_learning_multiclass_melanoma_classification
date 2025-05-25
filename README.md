# deep_learning_multiclass_melanoma_classification
skin cancer (melanoma) classification project using deep learning. The major problem here is the class imbalance and it is solved with two data augmentation strategies.

# 🔬 Melanoma Classification Project

A comprehensive deep learning pipeline for multi-class skin cancer classification using advanced computer vision techniques. This project classifies dermoscopic images into three categories: **Melanoma (MEL)**, **Basal Cell Carcinoma (BCC)**, and **Squamous Cell Carcinoma (SCC)**.

## 🌟 Key Features

- **🏗️ Modular Architecture**: Clean, maintainable code structure with separate modules
- **🚀 Advanced Training**: MixUp, CutMix, early stopping, and learning rate scheduling
- **🤖 Multiple Models**: ResNet-50, Inception V3, Swin Transformer, VGG-16, EfficientNet
- **🎯 Ensemble Methods**: Train and evaluate multiple models together
- **📊 Cross-Validation**: Robust k-fold cross-validation for reliable evaluation
- **🔍 Test-Time Augmentation**: Improved inference accuracy through TTA
- **📈 Comprehensive Analysis**: Detailed predictions with confidence scores and visualizations

## 📁 Project Structure

```
melanoma-classification/
├── config.py              # Configuration and hyperparameters
├── data_utils.py          # Data extraction and preprocessing
├── datasets.py            # Dataset classes and transforms
├── models.py              # Model architectures
├── training.py            # Training functions and utilities
├── evaluation.py          # Testing and evaluation functions
├── utils.py               # Visualization and utility functions
├── main.py                # Main execution script
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/melanoma-classification.git
cd melanoma-classification

# Install required packages
pip install -r requirements.txt
```

### Required Packages

```
torch>=1.12.0
torchvision>=0.13.0
torchmetrics>=0.9.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
Pillow>=8.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.62.0
```

## 🚀 Quick Start

### 1. Basic Usage

```python
# Run the complete pipeline
python main.py

# Choose training mode when prompted:
# - 'single': Train a single model
# - 'ensemble': Train multiple models together
# - 'cv': Perform cross-validation
```

### 2. Advanced Configuration

Edit `config.py` to customize hyperparameters:

```python
# Model parameters
IM_SIZE = 299              # Image size (299 for Inception/Swin, 224 for ResNet)
BATCH_SIZE = 8             # Batch size
EPOCHS = 30                # Training epochs
LEARNING_RATE = 1e-4       # Learning rate

# Data augmentation
CLASSES_TO_AUGMENT = {
    "bcc": 100,            # Augment 100 BCC images
    "scc": 450             # Augment 450 SCC images
}
```

## 📚 Usage Examples

### Single Model Training

```python
from main import train_single_model
from datasets import create_datasets, create_data_loaders

# Setup data
train_ds, val_ds, test_ds = create_datasets()
train_loader, val_loader, test_loader = create_data_loaders(train_ds, val_ds, test_ds)

# Train ResNet-50
train_single_model(train_loader, val_loader, test_loader, model_type='resnet')
```

### Ensemble Training

```python
from main import train_ensemble_models

# Train ensemble of ResNet + Inception + Swin Transformer
train_ensemble_models(train_loader, val_loader, test_loader)
```

### Cross-Validation

```python
from main import perform_cross_validation

# 5-fold cross-validation with ResNet-50
cv_results = perform_cross_validation(train_ds)
```

## 🏗️ Model Architectures

| Model | Input Size | Parameters | Features |
|-------|------------|------------|----------|
| **ResNet-50** | 224×224 | ~25M | Deep residual connections |
| **Inception V3** | 299×299 | ~27M | Multi-scale feature extraction |
| **Swin Transformer** | 299×299 | ~88M | Vision transformer with shifted windows |
| **VGG-16** | 224×224 | ~138M | Classic CNN architecture |
| **EfficientNet** | 224×224 | ~5M | Efficient scaling |
| **Custom CNN** | Any | ~50M | Custom architecture |

## 📊 Data Pipeline

### 1. Data Extraction
```python
# Automatically extracts tar archives
extract_dataset()
```

### 2. Data Augmentation
- **Traditional**: Rotation, flipping, brightness adjustment
- **Advanced**: MixUp, CutMix for training
- **Intelligent Cropping**: Removes black borders from dermoscopic images

### 3. Preprocessing
- Resize to target resolution
- Normalization with custom statistics
- Test-time augmentation for inference

## 🎯 Training Modes

### Single Model Training
- Train one model with advanced techniques
- Early stopping and learning rate scheduling
- MixUp/CutMix augmentation
- Comprehensive metrics tracking

### Ensemble Training
- Train multiple different architectures
- Automatic weight optimization
- Combined predictions for better accuracy

### Cross-Validation
- K-fold validation for robust evaluation
- Statistical significance testing
- Model performance comparison

## 📈 Advanced Features

### Training Enhancements
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate
- **MixUp/CutMix**: Advanced data augmentation
- **Gradient Clipping**: Training stability
- **Label Smoothing**: Reduces overconfidence

### Evaluation Features
- **Test-Time Augmentation**: Multiple predictions per image
- **Confidence Analysis**: Identifies uncertain predictions
- **Detailed Metrics**: Accuracy, Cohen's Kappa, confusion matrix
- **Visualization**: Training curves and prediction analysis

## 📁 Input Data Structure

```
data/
├── train/
│   ├── mel/           # Melanoma images
│   ├── bcc/           # Basal cell carcinoma images
│   └── scc/           # Squamous cell carcinoma images
├── val/
│   ├── mel/
│   ├── bcc/
│   └── scc/
└── test/              # Test images (no labels)
    ├── xxx001.jpg
    ├── xxx002.jpg
    └── ...
```

## 📤 Output Files

### Model Checkpoints
- `best_model.pt` - Best single model
- `best_ensemble.pt` - Best ensemble model
- `cv_fold_*.pt` - Cross-validation models

### Predictions
- `predictions.csv` - Simple format (filename, class)
- `predictions_detailed.csv` - With probabilities and confidence
- `ensemble_predictions.csv` - Ensemble results

### Visualizations
- `training_history.png` - Loss and accuracy curves
- `class_distribution.png` - Dataset statistics
- `confusion_matrix.png` - Performance analysis

## ⚙️ Configuration Options

### Model Selection
```python
AVAILABLE_MODELS = {
    'resnet': 'ResNet-50',
    'inception': 'Inception V3', 
    'swin': 'Swin Transformer V2',
    'vgg': 'VGG-16',
    'efficientnet': 'EfficientNet-B0'
}
```

### Training Parameters
```python
# Hyperparameters
EPOCHS = 30
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
WEIGHT_DECAY = 1e-4

# Early stopping
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.001

# Augmentation
MIXUP_ALPHA = 1.0
CUTMIX_ALPHA = 1.0
```

## 📊 Performance Metrics

The project tracks multiple metrics:

- **Accuracy**: Overall classification accuracy
- **Cohen's Kappa**: Inter-rater agreement
- **Per-class Accuracy**: Individual class performance
- **Confusion Matrix**: Detailed error analysis
- **Confidence Scores**: Prediction certainty

## 🔧 Troubleshooting

### Common Issues

**Out of Memory Errors**
```python
# Reduce batch size in config.py
BATCH_SIZE = 4  # or smaller
```

**CUDA Errors**
```python
# Check GPU availability
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

**Data Loading Issues**
```python
# Verify data paths in config.py
TRAIN_DATASET_PATH = "/path/to/your/train/data"
VAL_DATASET_PATH = "/path/to/your/val/data"
```

### Performance Optimization

1. **Enable Mixed Precision**: Add to training config
2. **Increase Batch Size**: If you have more GPU memory
3. **Use DataLoader Workers**: Set `NUM_WORKERS = 4` or higher
4. **Pin Memory**: Enabled by default for GPU training

## 📋 Results Interpretation

### Confidence Analysis
```python
# Low confidence predictions indicate uncertain cases
# Threshold can be adjusted in config.py
LOW_CONFIDENCE_THRESHOLD = 0.6
```

### Class Imbalance Handling
- Automatic data augmentation for minority classes
    - Selected a random number of files from each minority class
    - applied 3 augmentations to each
    - generated 3 files per each random file selected
- Weighted loss functions available
- Stratified cross-validation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include unit tests for new features
- Update README for significant changes

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Torchvision for pre-trained models
- The dermoscopy community for dataset contributions
- Research papers that inspired the methodologies

## 📧 Contact

- **Author**: Joana Owusu-Appiah
- **Email**: joanaowusu279@gmail.com
- **GitHub**: [@Jo-Mansa](https://github.com/Jo-Mansa)


## 📊 Citation

If you use this project in your research, please cite:

```bibtex
@software{melanoma_classification_2024,
  author = {Joana Owusu-Appiah},
  title = {Advanced Melanoma Classification Pipeline},
  url = {https://github.com/Jo-Mansa/melanoma-classification},
  year = {2024}
}
```

---

## 🚀 Getting Started Checklist

- [ ] Install Python 3.8+
- [ ] Install required packages
- [ ] Prepare dataset in correct structure
- [ ] Configure paths in `config.py`
- [ ] Run `python main.py`
- [ ] Choose training mode
- [ ] Monitor training progress
- [ ] Analyze results and predictions

**Happy Classifying! 🔬✨**
