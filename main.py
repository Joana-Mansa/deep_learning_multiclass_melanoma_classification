# ========================================
# MELANOMA/SKIN CANCER CLASSIFICATION
# Multi-class Classification: MEL, BCC, SCC
# ========================================

# ========================================
# 1. IMPORTS AND DEPENDENCIES
# ========================================
import os
import math
import random
import shutil
import tarfile
import pickle
import subprocess
from datetime import datetime
from collections import Counter, OrderedDict
from os.path import join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.color import rgb2gray
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data.sampler import WeightedRandomSampler

import torchvision
import torchvision.transforms as T
from torchvision import models
from torchvision.transforms import functional as F
from torchvision.models.inception import InceptionOutputs
from torchsummary import summary

from torchmetrics import Accuracy, CohenKappa
from sklearn.svm import SVC
from sklearn.metrics import hinge_loss, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold
import xgboost as xgb

# ========================================
# 2. DEVICE SETUP AND CONFIGURATION
# ========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# ========================================
# 3. DATA PATHS AND PARAMETERS
# ========================================
# Dataset parameters
im_size = 299  # 299 for Inception/Swin, 224 for ResNet
num_classes = 3
batch_size = 8
num_workers = 2

# Training parameters
epochs = 30
lr = 1e-4

# Data paths (will be set after extraction)
train_dataset_path = "/kaggle/working/extracted_images/train/"
val_dataset_path = "/kaggle/working/val_images/val"
test_dataset_path = "/kaggle/working/working/testX"

# ========================================
# 4. DATA EXTRACTION FUNCTIONS
# ========================================
def extract_dataset():
    """Extract training and validation datasets from tar archives"""
    
    # Extract training data
    train_tar = "../input/cad-melanoma-data/train (1).tgz"
    with tarfile.open(train_tar, "r:gz") as tar:
        train_dir = './extracted_images'
        tar.extractall(path=train_dir)
    print("Training data extracted successfully")
    
    # Extract validation data
    val_tar = "../input/cad-melanoma-data/val (1).tgz"
    with tarfile.open(val_tar, "r:gz") as tar:
        val_dir = './val_images'
        tar.extractall(path=val_dir)
    print("Validation data extracted successfully")
    
    # Copy test data
    source_dir = "../input/cad-melanoma-data/testX-20241212T204848Z-001/testX"
    test_dir = "./working/testX"
    if os.path.exists(source_dir):
        shutil.copytree(source_dir, test_dir)
        print(f"Test data copied to {test_dir}")
    else:
        print(f"Source directory {source_dir} does not exist.")

# ========================================
# 5. DATA AUGMENTATION FUNCTIONS
# ========================================
def horizontal_flip(image):
    """Apply horizontal flip to image"""
    return cv2.flip(image, 1)

def rotate_image(image, angle):
    """Rotate image by specified angle"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated

def enhance_brightness(image, beta=50):
    """Enhance image brightness"""
    return cv2.convertScaleAbs(image, beta=beta)

def augment_image(image_path, augmentations, save_dir):
    """Apply augmentations to a single image"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    file_name = os.path.basename(image_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    for i, augmentation in enumerate(augmentations):
        augmented_img = augmentation(img)
        augmented_file_name = f'aug_{i}_{file_name}'
        augmented_path = os.path.join(save_dir, augmented_file_name)
        cv2.imwrite(augmented_path, augmented_img)

def augment_images_in_folders(base_folder, classes_to_augment, augmentations):
    """Apply augmentation to multiple classes"""
    random.seed(42)
    for class_name, sample_size in classes_to_augment.items():
        class_path = os.path.join(base_folder, class_name)
        
        # Get list of images in the class folder
        all_images = [os.path.join(class_path, fname)
                      for fname in os.listdir(class_path)
                      if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(all_images) == 0:
            print(f"No images found in {class_path}")
            continue
            
        # Sample and augment images
        sampled_images = random.sample(all_images, min(sample_size, len(all_images)))
        for img_path in sampled_images:
            augment_image(img_path, augmentations, class_path)

# ========================================
# 6. IMAGE PREPROCESSING FUNCTIONS
# ========================================
def crop_img(img, threshold=100):
    """Intelligent cropping to remove black borders"""
    h, w = img.shape[:2]
    cd = math.gcd(h, w)
    
    y_coords = [i for i in range(0, h, h // cd)]
    x_coords = [i for i in range(0, w, w // cd)]
    
    coordinates = {'y1': 0, 'x1': 0, 'y2': h, 'x2': w}
    
    for i in range(2):
        diag_values = []
        if i == 0:
            diag_values = [np.mean(img[y, x]) for y, x in zip(y_coords, x_coords)]
        else:
            diag_values = [np.mean(img[y, w - 1 - x]) for y, x in zip(reversed(y_coords), x_coords)]
        
        for idx, value in enumerate(diag_values):
            if value >= threshold:
                if i == 0:
                    coordinates['y1'] = y_coords[idx]
                    coordinates['x1'] = x_coords[idx]
                else:
                    coordinates['y2'] = h - y_coords[idx]
                    coordinates['x2'] = w - x_coords[idx]
                break
        
        for idx, value in enumerate(reversed(diag_values)):
            if value >= threshold:
                rev_idx = len(diag_values) - 1 - idx
                if i == 0:
                    coordinates['y2'] = y_coords[rev_idx]
                    coordinates['x2'] = x_coords[rev_idx]
                else:
                    coordinates['y1'] = h - y_coords[rev_idx]
                    coordinates['x1'] = w - x_coords[rev_idx]
                break
    
    y1 = max(coordinates['y1'], 0)
    y2 = min(coordinates['y2'], h)
    x1 = max(coordinates['x1'], 0)
    x2 = min(coordinates['x2'], w)
    
    img_cropped = img[y1:y2, x1:x2]
    if img_cropped.shape[0] == 0 or img_cropped.shape[1] == 0:
        img_cropped = img
    
    return img_cropped

def apply_cropping(image, threshold=100):
    """Apply cropping to PIL image"""
    img_array = np.array(image)
    cropped_img = crop_img(img_array, threshold)
    return F.to_pil_image(cropped_img)

class HistogramEqualization:
    """Custom transform for histogram equalization"""
    def __call__(self, img):
        img_array = np.array(img)
        if len(img_array.shape) == 2:  # Grayscale
            img_eq = cv2.equalizeHist(img_array)
        else:  # Color
            img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
            img_eq = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        return F.to_pil_image(img_eq)

# ========================================
# 7. DATASET CLASSES
# ========================================
class Dermoscopic_dataset(data.Dataset):
    """Dataset class for dermoscopic images"""
    def __init__(self, root_path, name_list, transforms=None, cropping_threshold=100, im_size=299):
        self.root_path = root_path
        self.image_filenames = open(join(root_path, name_list)).read().split('\n')[:-1]
        self.transforms = transforms
        self.cropping_threshold = cropping_threshold
        self.im_size = im_size

    def __getitem__(self, index):
        img_path = join(self.root_path, self.image_filenames[index])
        image = Image.open(img_path).convert('RGB')
        cropped_image = apply_cropping(image, threshold=self.cropping_threshold)
        
        if self.transforms:
            cropped_image = self.transforms(cropped_image)

        assert cropped_image.shape == (3, self.im_size, self.im_size), \
               f"Unexpected image shape: {cropped_image.shape}"

        class_name = self.image_filenames[index].split('/')[0]
        target = 0 if class_name == "mel" else 1 if class_name == "bcc" else 2

        return cropped_image, torch.tensor(target, dtype=torch.long)

    def __len__(self):
        return len(self.image_filenames)

class TestDataset(data.Dataset):
    """Dataset class for test images"""
    def __init__(self, root_path, name_list, transforms=None, cropping_threshold=100):
        self.root_path = root_path
        self.image_filenames = open(join(root_path, name_list)).read().split('\n')[:-1]
        self.transforms = transforms
        self.cropping_threshold = cropping_threshold

    def __getitem__(self, index):
        img_path = join(self.root_path, self.image_filenames[index])
        image = Image.open(img_path).convert('RGB')
        cropped_image = apply_cropping(image, threshold=self.cropping_threshold)

        if self.transforms:
            cropped_image = self.transforms(cropped_image)

        file_name = self.image_filenames[index]
        gt = torch.tensor(0, dtype=torch.long)  # Dummy ground truth

        return cropped_image, gt, file_name

    def __len__(self):
        return len(self.image_filenames)

# ========================================
# 8. ADVANCED AUGMENTATIONS
# ========================================
class MixUp:
    """MixUp augmentation for better generalization"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = batch[0].size(0)
        index = torch.randperm(batch_size)

        mixed_x = lam * batch[0] + (1 - lam) * batch[0][index, :]
        y_a, y_b = batch[1], batch[1][index]
        return mixed_x, y_a, y_b, lam

class CutMix:
    """CutMix augmentation for improved robustness"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, batch):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = batch[0].size(0)
        index = torch.randperm(batch_size)

        y_a, y_b = batch[1], batch[1][index]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(batch[0].size(), lam)
        batch[0][:, :, bbx1:bbx2, bby1:bby2] = batch[0][index, :, bbx1:bbx2, bby1:bby2]
        
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch[0].size()[-1] * batch[0].size()[-2]))

        return batch[0], y_a, y_b, lam

# ========================================
# 9. MODEL DEFINITIONS
# ========================================
def modify_vgg16():
    """Create modified VGG-16 for 3-class classification"""
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    # Freeze early layers
    for param in model.features[:16].parameters():
        param.requires_grad = False
    
    flattened_size = 512 * 7 * 7
    model.classifier = nn.Sequential(
        nn.Linear(flattened_size, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(1024, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(256, 3),
    )
    return model

def modify_resnet50():
    """Create modified ResNet-50 for 3-class classification"""
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(512, 3),
    )
    return model

def modify_inception_v3():
    """Create modified Inception V3 for 3-class classification"""
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(512, 3),
    )
    return model

def create_swin_transformer():
    """Create Swin Transformer V2 Base model"""
    swin_v2_b_model = models.swin_v2_b(weights='DEFAULT')
    in_features = swin_v2_b_model.head.in_features
    swin_v2_b_model.head = torch.nn.Linear(in_features=in_features, out_features=3)
    return swin_v2_b_model

# ========================================
# 10. TRAINING FUNCTIONS
# ========================================
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        """Save model when validation loss improves"""
        self.best_weights = model.state_dict().copy()

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_with_advanced_features(net, train_loader, val_loader, conf_train):
    """Advanced training function with early stopping, mixup, and scheduling"""
    device = conf_train['device']
    epochs = conf_train['epochs']
    loss_function = conf_train['loss_function'].to(device)
    train_metric_function = conf_train['metric_function'].to(device)
    cohen_kappa_function = conf_train['cohen_kappa_function'].to(device)
    optimizer = conf_train['optimizer']
    scheduler = conf_train.get('scheduler', None)
    use_mixup = conf_train.get('use_mixup', False)
    use_cutmix = conf_train.get('use_cutmix', False)
    early_stopping = conf_train.get('early_stopping', None)

    # Initialize augmentation strategies
    mixup = MixUp(alpha=1.0) if use_mixup else None
    cutmix = CutMix(alpha=1.0) if use_cutmix else None

    best_val_accuracy = 0.0
    best_epoch = 0
    best_metrics = {}
    training_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    net.to(device)

    with tqdm(range(epochs)) as progress:
        for epoch in progress:
            # Training phase
            net.train()
            batch_train_losses = []
            
            for batch_idx, (im, gt) in enumerate(train_loader):
                im, gt = im.to(device), gt.to(device)

                # Apply MixUp or CutMix
                if use_mixup and np.random.rand() < 0.5:
                    mixed_x, y_a, y_b, lam = mixup((im, gt))
                    pred = net(mixed_x)
                    loss = mixup_criterion(loss_function, pred, y_a, y_b, lam)
                elif use_cutmix and np.random.rand() < 0.5:
                    mixed_x, y_a, y_b, lam = cutmix((im, gt))
                    pred = net(mixed_x)
                    loss = mixup_criterion(loss_function, pred, y_a, y_b, lam)
                else:
                    pred = net(im)
                    loss = loss_function(pred, gt)

                train_metric_function.update(pred, gt)
                cohen_kappa_function.update(pred.argmax(dim=1), gt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_train_losses.append(loss.detach().cpu().item())

            train_acc = train_metric_function.compute().item()
            train_kappa = cohen_kappa_function.compute().item()
            train_metric_function.reset()
            cohen_kappa_function.reset()

            # Validation phase
            net.eval()
            batch_val_losses = []
            val_metric_function = train_metric_function.clone()
            val_cohen_kappa_function = cohen_kappa_function.clone()

            with torch.no_grad():
                for im, gt in val_loader:
                    im, gt = im.to(device), gt.to(device)
                    pred = net(im)
                    loss = loss_function(pred, gt)

                    val_metric_function.update(pred, gt)
                    val_cohen_kappa_function.update(pred.argmax(dim=1), gt)
                    batch_val_losses.append(loss.detach().cpu().item())

            val_acc = val_metric_function.compute().item()
            val_kappa = val_cohen_kappa_function.compute().item()
            val_metric_function.reset()
            val_cohen_kappa_function.reset()

            # Update learning rate
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(np.mean(batch_val_losses))
                else:
                    scheduler.step()

            # Store training history
            training_history['train_loss'].append(np.mean(batch_train_losses))
            training_history['val_loss'].append(np.mean(batch_val_losses))
            training_history['train_acc'].append(train_acc)
            training_history['val_acc'].append(val_acc)

            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            progress.set_postfix({
                'train_loss': f"{np.mean(batch_train_losses):.4f}",
                'train_acc': f"{train_acc:.4f}",
                'val_loss': f"{np.mean(batch_val_losses):.4f}",
                'val_acc': f"{val_acc:.4f}",
                'lr': f"{current_lr:.6f}"
            })

            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_epoch = epoch
                best_metrics = {
                    'train_loss': np.mean(batch_train_losses),
                    'train_acc': train_acc,
                    'train_kappa': train_kappa,
                    'val_loss': np.mean(batch_val_losses),
                    'val_acc': val_acc,
                    'val_kappa': val_kappa
                }
                torch.save(net.state_dict(), '/kaggle/working/best_model.pt')

            # Early stopping
            if early_stopping:
                if early_stopping(np.mean(batch_val_losses), net):
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break

    # Print best metrics
    print(f"\nBest Model Saved at Epoch {best_epoch}")
    print(f"Training Metrics: Loss={best_metrics['train_loss']:.4f}, "
          f"Accuracy={best_metrics['train_acc']:.4f}, Kappa={best_metrics['train_kappa']:.4f}")
    print(f"Validation Metrics: Loss={best_metrics['val_loss']:.4f}, "
          f"Accuracy={best_metrics['val_acc']:.4f}, Kappa={best_metrics['val_kappa']:.4f}")

    return net, training_history

def ensemble_train(models, train_loader, val_loader, conf_train):
    """Train ensemble of models"""
    device = conf_train['device']
    epochs = conf_train['epochs']
    loss_function = conf_train['loss_function'].to(device)
    metric_function = conf_train['metric_function'].to(device)
    optimizers = conf_train['optimizers']
    schedulers = conf_train.get('schedulers', [None] * len(models))
    early_stopping = conf_train.get('early_stopping', None)

    best_val_loss = float('inf')
    best_epoch = 0

    # Move all models to device
    for model in models:
        model.to(device)

    with tqdm(range(epochs)) as progress:
        for epoch in progress:
            # Training phase
            for model in models:
                model.train()
            
            batch_train_losses = []

            for im, gt in train_loader:
                im, gt = im.to(device), gt.to(device)

                # Forward pass through all models
                ensemble_pred = torch.zeros_like(torch.zeros(im.size(0), 3)).to(device)
                individual_losses = []

                for i, (model, optimizer) in enumerate(zip(models, optimizers)):
                    pred = model(im)
                    
                    # Handle Inception V3 outputs
                    if isinstance(pred, InceptionOutputs):
                        pred = pred.logits
                    
                    loss = loss_function(pred, gt)
                    individual_losses.append(loss)
                    ensemble_pred += pred

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Average ensemble prediction
                ensemble_pred /= len(models)
                ensemble_loss = sum(individual_losses) / len(individual_losses)
                
                metric_function.update(ensemble_pred, gt)
                batch_train_losses.append(ensemble_loss.detach().cpu().item())

            train_acc = metric_function.compute().item()
            metric_function.reset()

            # Update schedulers
            for scheduler in schedulers:
                if scheduler:
                    scheduler.step()

            # Validation phase
            for model in models:
                model.eval()
            
            batch_val_losses = []
            val_metric_function = metric_function.clone()

            with torch.no_grad():
                for im, gt in val_loader:
                    im, gt = im.to(device), gt.to(device)

                    # Ensemble prediction
                    ensemble_pred = torch.zeros_like(torch.zeros(im.size(0), 3)).to(device)
                    
                    for model in models:
                        pred = model(im)
                        if isinstance(pred, InceptionOutputs):
                            pred = pred.logits
                        ensemble_pred += pred

                    ensemble_pred /= len(models)
                    loss = loss_function(ensemble_pred, gt)

                    val_metric_function.update(ensemble_pred, gt)
                    batch_val_losses.append(loss.detach().cpu().item())

            val_acc = val_metric_function.compute().item()
            val_metric_function.reset()

            current_val_loss = np.mean(batch_val_losses)
            
            progress.set_postfix({
                'train_loss': f"{np.mean(batch_train_losses):.4f}",
                'train_acc': f"{train_acc:.4f}",
                'val_loss': f"{current_val_loss:.4f}",
                'val_acc': f"{val_acc:.4f}"
            })

            # Save best ensemble
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_epoch = epoch
                
                # Save all models
                ensemble_dict = {}
                for i, model in enumerate(models):
                    ensemble_dict[f'model_{i}'] = model.state_dict()
                torch.save(ensemble_dict, '/kaggle/working/best_ensemble.pt')

            # Early stopping
            if early_stopping:
                if early_stopping(current_val_loss, models[0]):  # Use first model for early stopping
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break

    print(f"\nBest ensemble saved at epoch {best_epoch} with validation loss: {best_val_loss:.4f}")
    return models

def cross_validate(model_fn, dataset, conf_train, k_folds=5):
    """Perform k-fold cross-validation"""
    from sklearn.model_selection import KFold
    
    # Convert dataset to indices for splitting
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    cv_results = {
        'fold_accuracies': [],
        'fold_kappas': [],
        'fold_losses': []
    }
    
    print(f"Starting {k_folds}-fold cross-validation...")
    
    for fold, (train_indices, val_indices) in enumerate(kfold.split(indices)):
        print(f"\nFold {fold + 1}/{k_folds}")
        
        # Create data subsets
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)
        
        # Create data loaders
        train_loader = data.DataLoader(
            train_subset, 
            batch_size=conf_train.get('batch_size', 8),
            shuffle=True,
            num_workers=conf_train.get('num_workers', 2)
        )
        val_loader = data.DataLoader(
            val_subset,
            batch_size=conf_train.get('batch_size', 8),
            shuffle=False,
            num_workers=conf_train.get('num_workers', 2)
        )
        
        # Create fresh model for this fold
        model = model_fn()
        
        # Create fresh optimizer for this fold
        optimizer = optim.Adam(model.parameters(), lr=conf_train.get('lr', 1e-4))
        conf_train_fold = conf_train.copy()
        conf_train_fold['optimizer'] = optimizer
        
        # Train model for this fold
        trained_model, _ = train_with_advanced_features(model, train_loader, val_loader, conf_train_fold)
        
        # Evaluate this fold
        conf_test = {
            'device': conf_train['device'],
            'loss_function': conf_train['loss_function'],
            'metric_function': conf_train['metric_function']
        }
        
        results = test(trained_model, val_loader, conf_test)
        
        cv_results['fold_accuracies'].append(results['test_acc'])
        cv_results['fold_losses'].append(results['test_loss'])
        
        print(f"Fold {fold + 1} - Accuracy: {results['test_acc']:.4f}, Loss: {results['test_loss']:.4f}")
    
    # Calculate cross-validation statistics
    cv_results['mean_accuracy'] = np.mean(cv_results['fold_accuracies'])
    cv_results['std_accuracy'] = np.std(cv_results['fold_accuracies'])
    cv_results['mean_loss'] = np.mean(cv_results['fold_losses'])
    cv_results['std_loss'] = np.std(cv_results['fold_losses'])
    
    print(f"\n=== Cross-Validation Results ===")
    print(f"Mean Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
    print(f"Mean Loss: {cv_results['mean_loss']:.4f} ± {cv_results['std_loss']:.4f}")
    
    return cv_results

def test(net, test_loader, conf_test):
    """Testing function for single model"""
    device = conf_test['device']
    loss_function = conf_test['loss_function'].to(device)
    metric_function = conf_test['metric_function'].to(device)

    net.to(device)
    net.eval()

    batch_test_losses = []
    metric_function.reset()
    all_preds = []
    filenames = []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:  # Test dataset with filenames
                im, gt, file_names = batch
                filenames.extend(file_names)
            else:  # Regular dataset
                im, gt = batch
                
            im, gt = im.to(device), gt.to(device)

            pred = net(im)
            loss = loss_function(pred, gt)

            batch_test_losses.append(loss.detach().cpu().item())
            metric_function.update(pred, gt)
            all_preds.append(pred.cpu())

    test_loss = np.mean(batch_test_losses)
    test_acc = metric_function.compute().item()

    return {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'filenames': filenames,
        'all_preds': torch.cat(all_preds, dim=0) if all_preds else None
    }

def ensemble_test(models, test_loader, conf_test):
    """Test ensemble of models"""
    device = conf_test['device']
    loss_function = conf_test['loss_function'].to(device)
    metric_function = conf_test['metric_function'].to(device)

    # Move all models to device and set to eval mode
    for model in models:
        model.to(device)
        model.eval()

    batch_test_losses = []
    metric_function.reset()
    all_preds = []
    filenames = []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:  # Test dataset with filenames
                im, gt, file_names = batch
                filenames.extend(file_names)
            else:  # Regular dataset
                im, gt = batch
                
            im, gt = im.to(device), gt.to(device)

            # Ensemble prediction
            ensemble_pred = torch.zeros_like(torch.zeros(im.size(0), 3)).to(device)
            
            for model in models:
                pred = model(im)
                if isinstance(pred, InceptionOutputs):
                    pred = pred.logits
                ensemble_pred += pred

            ensemble_pred /= len(models)
            loss = loss_function(ensemble_pred, gt)

            batch_test_losses.append(loss.detach().cpu().item())
            metric_function.update(ensemble_pred, gt)
            all_preds.append(ensemble_pred.cpu())

    test_loss = np.mean(batch_test_losses)
    test_acc = metric_function.compute().item()

    return {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'filenames': filenames,
        'all_preds': torch.cat(all_preds, dim=0) if all_preds else None
    }

# ========================================
# 10. UTILITY FUNCTIONS
# ========================================
def plot_distribution(data, mode='train'):
    """Plot class distribution"""
    names = list(data.keys())
    values = list(data.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(data)), values, tick_label=names)
    plt.title(f'{mode} Class Distribution')
    plt.xticks(rotation=45, ha="right")
    plt.ylabel('Number of Samples')
    plt.show()

def show_distribution(files, mode='Train'):
    """Show distribution of classes in dataset"""
    labels = [f.split('/')[0] for f in files]
    labels_count = Counter(labels)
    stats = OrderedDict(sorted(labels_count.items()))
    plot_distribution(stats, mode=mode)

def create_txt(path, name_list):
    """Create text file with file names"""
    with open(path, 'w') as file:
        for item in name_list:
            file.write(item + "\n")

def delete_path(path):
    """Delete file if exists"""
    if os.path.exists(path):
        os.remove(path)
        print(f"{path} has been deleted.")
    else:
        print(f"{path} does not exist.")

# ========================================
# 11. MAIN EXECUTION
# ========================================
def main():
    """Main execution function"""
    
    # 1. Extract datasets
    print("Extracting datasets...")
    extract_dataset()
    
    # 2. Apply data augmentation
    print("Applying data augmentation...")
    classes_to_augment = {
        "bcc": 100,  # Augment 100 BCC images
        "scc": 450   # Augment 450 SCC images
    }
    
    augmentations = [
        horizontal_flip,
        lambda img: rotate_image(img, 15),
        lambda img: enhance_brightness(img, beta=50)
    ]
    
    train_dir = "/kaggle/working/extracted_images/train"
    augment_images_in_folders(train_dir, classes_to_augment, augmentations)
    print("Data augmentation completed.")
    
    # 3. Create file lists
    print("Creating file lists...")
    import glob
    
    train_files = ['/'.join(x.replace("\\", "/").split("/")[-2:]) 
                   for x in glob.glob(join(train_dataset_path, '*/*.*')) 
                   if x.endswith("jpg")]
    val_files = ['/'.join(x.replace("\\", "/").split("/")[-2:]) 
                 for x in glob.glob(join(val_dataset_path, '*/*.*')) 
                 if x.endswith("jpg")]
    test_files = [x.split('/')[-1] 
                  for x in glob.glob(join(test_dataset_path, '*.*')) 
                  if x.endswith("jpg")]
    
    create_txt(join(train_dataset_path, 'train_list.txt'), train_files)
    create_txt(join(val_dataset_path, 'val_list.txt'), val_files)
    create_txt(join(test_dataset_path, 'test_list.txt'), test_files)
    
    # 4. Show data distribution
    print("Training data distribution:")
    show_distribution(train_files, mode='Train')
    print("Validation data distribution:")
    show_distribution(val_files, mode='Validation')
    
    # 5. Define transforms
    train_transforms = T.Compose([
        T.Resize((im_size, im_size)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.RandomResizedCrop(im_size),
        T.ToTensor(),
        T.Normalize(mean=[0.6675, 0.5295, 0.5244], std=[0.2234, 0.2034, 0.2150]),
    ])
    
    val_transforms = T.Compose([
        T.Resize((im_size, im_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.6675, 0.5295, 0.5244], std=[0.2234, 0.2034, 0.2150]),
    ])
    
    # 6. Create datasets
    train_set = Dermoscopic_dataset(
        root_path=train_dataset_path,
        name_list='train_list.txt',
        transforms=train_transforms,
        cropping_threshold=100
    )
    
    val_set = Dermoscopic_dataset(
        root_path=val_dataset_path,
        name_list='val_list.txt',
        transforms=val_transforms,
        cropping_threshold=100
    )
    
    test_set = TestDataset(
        root_path=test_dataset_path,
        name_list='test_list.txt',
        transforms=val_transforms,
        cropping_threshold=100
    )
    
    # 7. Create data loaders
    train_loader = data.DataLoader(
        dataset=train_set,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = data.DataLoader(
        dataset=val_set,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = data.DataLoader(
        dataset=test_set,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False
    )
    
    # 8. Create model
    print("Creating model...")
    model = modify_resnet50()  # or create_swin_transformer()
    
    # 9. Define loss, optimizer, and metrics
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    metric_function = Accuracy(task='multiclass', num_classes=num_classes, average='macro')
    cohen_kappa_function = CohenKappa(task='multiclass', num_classes=num_classes)
    
    # 10. Training configuration
    conf_train = {
        'device': device,
        'epochs': epochs,
        'loss_function': loss_function,
        'metric_function': metric_function,
        'cohen_kappa_function': cohen_kappa_function,
        'optimizer': optimizer,
    }
    
    # 11. Train model
    print("Starting training...")
    trained_model = train(model, train_loader, val_loader, conf_train)
    
    # 12. Test model
    print("Testing model...")
    conf_test = {
        'device': device,
        'loss_function': loss_function,
        'metric_function': metric_function
    }
    
    results = test(trained_model, test_loader, conf_test)
    
    # 13. Save predictions
    print("Saving predictions...")
    filenames = results['filenames']
    all_preds = results['all_preds']
    
    if all_preds is not None and len(filenames) > 0:
        flattened_preds = all_preds.squeeze().cpu().numpy()
        predicted_classes = flattened_preds.argmax(axis=1)
        
        # Sort filenames numerically
        filenames_sorted = sorted(filenames, 
                                key=lambda x: int(x.split('.')[0].replace('xxx', '')))
        
        # Create predictions DataFrame
        data = {
            'Filename': filenames_sorted,
            'Predicted_Class': predicted_classes
        }
        
        # Add class probabilities
        for i in range(flattened_preds.shape[1]):
            data[f'Class_{i}_Probability'] = flattened_preds[:, i]
        
        predictions_df = pd.DataFrame(data)
        output_csv_path = '/kaggle/working/predictions.csv'
        predictions_df.to_csv(output_csv_path, index=False, header=False)
        
        print(f"Predictions exported to: {output_csv_path}")
    else:
        print("No predictions were made. CSV export skipped.")

if __name__ == "__main__":
    main()