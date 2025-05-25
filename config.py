"""
Configuration file for melanoma classification project.
Contains all hyperparameters, paths, and constants.
"""

import torch
import os

# ========================================
# DEVICE CONFIGURATION
# ========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================================
# DATA PATHS
# ========================================
# Input data paths (Kaggle)
TRAIN_TAR_PATH = "../input/cad-melanoma-data/train (1).tgz"
VAL_TAR_PATH = "../input/cad-melanoma-data/val (1).tgz"
TEST_SOURCE_PATH = "../input/cad-melanoma-data/testX-20241212T204848Z-001/testX"

# Working directory paths
TRAIN_DATASET_PATH = "/kaggle/working/extracted_images/train/"
VAL_DATASET_PATH = "/kaggle/working/val_images/val"
TEST_DATASET_PATH = "/kaggle/working/working/testX"

# Output paths
OUTPUT_DIR = "/kaggle/working"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")
ENSEMBLE_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_ensemble.pt")
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "predictions.csv")
TRAINING_PLOT_PATH = os.path.join(OUTPUT_DIR, "training_history.png")

# ========================================
# MODEL HYPERPARAMETERS
# ========================================
# Dataset parameters
IM_SIZE = 299  # 299 for Inception/Swin, 224 for ResNet
NUM_CLASSES = 3
BATCH_SIZE = 8
NUM_WORKERS = 2

# Training parameters
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Early stopping parameters
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.001

# Learning rate scheduler parameters
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5

# Augmentation parameters
MIXUP_ALPHA = 1.0
CUTMIX_ALPHA = 1.0
AUGMENTATION_PROBABILITY = 0.5

# Cross-validation parameters
CV_FOLDS = 5
CV_EPOCHS = 20  # Reduced for faster CV

# ========================================
# DATA AUGMENTATION SETTINGS
# ========================================
# Classes to augment and their sample sizes
CLASSES_TO_AUGMENT = {
    "bcc": 100,  # Augment 100 BCC images
    "scc": 450   # Augment 450 SCC images
}

# Image preprocessing parameters
CROPPING_THRESHOLD = 100

# Normalization parameters (computed from dataset)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Custom normalization (if computed from your dataset)
CUSTOM_MEAN = [0.6675, 0.5295, 0.5244]
CUSTOM_STD = [0.2234, 0.2034, 0.2150]

# ========================================
# CLASS INFORMATION
# ========================================
CLASS_NAMES = ['mel', 'bcc', 'scc']
CLASS_TO_IDX = {'mel': 0, 'bcc': 1, 'scc': 2}
IDX_TO_CLASS = {0: 'mel', 1: 'bcc', 2: 'scc'}

# ========================================
# MODEL CONFIGURATIONS
# ========================================
AVAILABLE_MODELS = {
    'resnet': {
        'name': 'ResNet-50',
        'input_size': 224,
        'pretrained': True
    },
    'inception': {
        'name': 'Inception V3',
        'input_size': 299,
        'pretrained': True
    },
    'swin': {
        'name': 'Swin Transformer V2',
        'input_size': 299,
        'pretrained': True
    },
    'vgg': {
        'name': 'VGG-16',
        'input_size': 224,
        'pretrained': True
    }
}

# ========================================
# TRAINING MODES
# ========================================
TRAINING_MODES = ['single', 'ensemble', 'cv']
DEFAULT_TRAINING_MODE = 'single'
DEFAULT_MODEL = 'resnet'

# ========================================
# EVALUATION SETTINGS
# ========================================
# Confidence threshold for low-confidence prediction analysis
LOW_CONFIDENCE_THRESHOLD = 0.6

# Metrics to track
METRICS_TO_TRACK = ['accuracy', 'cohen_kappa', 'loss']

# ========================================
# ENSEMBLE SETTINGS
# ========================================
ENSEMBLE_MODELS = ['resnet', 'inception', 'swin']
ENSEMBLE_PATIENCE = 10

# ========================================
# RANDOM SEEDS
# ========================================
RANDOM_SEED = 42
AUGMENTATION_SEED = 42

# ========================================
# LOGGING SETTINGS
# ========================================
LOG_LEVEL = 'INFO'
PROGRESS_BAR = True

# ========================================
# UTILITY FUNCTIONS
# ========================================
def print_config():
    """Print current configuration"""
    print("=" * 60)
    print("CONFIGURATION SETTINGS")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Image Size: {IM_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"Class Names: {CLASS_NAMES}")
    print("=" * 60)

def get_model_input_size(model_name):
    """Get the required input size for a model"""
    return AVAILABLE_MODELS.get(model_name, {}).get('input_size', IM_SIZE)

def validate_config():
    """Validate configuration settings"""
    assert BATCH_SIZE > 0, "Batch size must be positive"
    assert LEARNING_RATE > 0, "Learning rate must be positive"
    assert EPOCHS > 0, "Number of epochs must be positive"
    assert NUM_CLASSES > 0, "Number of classes must be positive"
    assert IM_SIZE > 0, "Image size must be positive"
    
    print("✓ Configuration validation passed")

if __name__ == "__main__":
    print_config()
    validate_config()