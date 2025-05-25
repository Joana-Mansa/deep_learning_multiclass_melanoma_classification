"""
Data utilities for melanoma classification project.
Handles data extraction, augmentation, and preprocessing.
"""

import os
import math
import random
import shutil
import tarfile
import glob
from os.path import join
from collections import Counter, OrderedDict

import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import functional as F

from config import *

# ========================================
# DATA EXTRACTION FUNCTIONS
# ========================================
def extract_dataset():
    """Extract training and validation datasets from tar archives"""
    
    print("Extracting datasets...")
    
    # Extract training data
    print("  - Extracting training data...")
    if os.path.exists(TRAIN_TAR_PATH):
        with tarfile.open(TRAIN_TAR_PATH, "r:gz") as tar:
            train_dir = './extracted_images'
            tar.extractall(path=train_dir)
        print("    ✓ Training data extracted successfully")
    else:
        print(f"    ✗ Training tar file not found: {TRAIN_TAR_PATH}")
    
    # Extract validation data
    print("  - Extracting validation data...")
    if os.path.exists(VAL_TAR_PATH):
        with tarfile.open(VAL_TAR_PATH, "r:gz") as tar:
            val_dir = './val_images'
            tar.extractall(path=val_dir)
        print("    ✓ Validation data extracted successfully")
    else:
        print(f"    ✗ Validation tar file not found: {VAL_TAR_PATH}")
    
    # Copy test data
    print("  - Copying test data...")
    test_dir = "./working/testX"
    if os.path.exists(TEST_SOURCE_PATH):
        if not os.path.exists(test_dir):
            shutil.copytree(TEST_SOURCE_PATH, test_dir)
        print(f"    ✓ Test data copied to {test_dir}")
    else:
        print(f"    ✗ Test source directory not found: {TEST_SOURCE_PATH}")

# ========================================
# DATA AUGMENTATION FUNCTIONS
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
        try:
            augmented_img = augmentation(img)
            augmented_file_name = f'aug_{i}_{file_name}'
            augmented_path = os.path.join(save_dir, augmented_file_name)
            cv2.imwrite(augmented_path, augmented_img)
        except Exception as e:
            print(f"Error augmenting image {image_path}: {e}")

def augment_images_in_folders(base_folder, classes_to_augment, augmentations):
    """Apply augmentation to multiple classes"""
    random.seed(AUGMENTATION_SEED)
    
    print("Applying data augmentation...")
    
    for class_name, sample_size in classes_to_augment.items():
        print(f"  - Augmenting {class_name} class ({sample_size} samples)...")
        class_path = os.path.join(base_folder, class_name)
        
        if not os.path.exists(class_path):
            print(f"    ✗ Class directory not found: {class_path}")
            continue
        
        # Get list of images in the class folder
        all_images = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            all_images.extend(glob.glob(os.path.join(class_path, f'*{ext}')))
        
        if len(all_images) == 0:
            print(f"    ✗ No images found in {class_path}")
            continue
        
        # Sample and augment images
        sampled_images = random.sample(all_images, min(sample_size, len(all_images)))
        
        augmented_count = 0
        for img_path in sampled_images:
            augment_image(img_path, augmentations, class_path)
            augmented_count += len(augmentations)
        
        print(f"    ✓ Created {augmented_count} augmented images for {class_name}")

def get_default_augmentations():
    """Get default augmentation functions"""
    return [
        horizontal_flip,
        lambda img: rotate_image(img, 15),  # Rotate by 15 degrees
        lambda img: enhance_brightness(img, beta=50)  # Enhance brightness
    ]

# ========================================
# IMAGE PREPROCESSING FUNCTIONS
# ========================================
def crop_img(img, threshold=100):
    """Intelligent cropping to remove black borders"""
    h, w = img.shape[:2]
    
    if h == 0 or w == 0:
        return img
    
    cd = math.gcd(h, w)  # Greatest Common Divisor
    
    # Avoid division by zero
    if cd == 0:
        return img
    
    y_coords = [i for i in range(0, h, max(1, h // cd))]
    x_coords = [i for i in range(0, w, max(1, w // cd))]
    
    coordinates = {'y1': 0, 'x1': 0, 'y2': h, 'x2': w}
    
    for i in range(2):
        diag_values = []
        if i == 0:
            diag_values = [np.mean(img[y, x]) for y, x in zip(y_coords, x_coords) if y < h and x < w]
        else:
            diag_values = [np.mean(img[y, w - 1 - x]) for y, x in zip(reversed(y_coords), x_coords) if y < h and (w - 1 - x) >= 0]
        
        if not diag_values:
            continue
            
        for idx, value in enumerate(diag_values):
            if value >= threshold:
                if i == 0:
                    coordinates['y1'] = y_coords[idx] if idx < len(y_coords) else 0
                    coordinates['x1'] = x_coords[idx] if idx < len(x_coords) else 0
                else:
                    coordinates['y2'] = h - y_coords[idx] if idx < len(y_coords) else h
                    coordinates['x2'] = w - x_coords[idx] if idx < len(x_coords) else w
                break
        
        for idx, value in enumerate(reversed(diag_values)):
            if value >= threshold:
                rev_idx = len(diag_values) - 1 - idx
                if i == 0:
                    coordinates['y2'] = y_coords[rev_idx] if rev_idx < len(y_coords) else h
                    coordinates['x2'] = x_coords[rev_idx] if rev_idx < len(x_coords) else w
                else:
                    coordinates['y1'] = h - y_coords[rev_idx] if rev_idx < len(y_coords) else 0
                    coordinates['x1'] = w - x_coords[rev_idx] if rev_idx < len(x_coords) else 0
                break
    
    y1 = max(coordinates['y1'], 0)
    y2 = min(coordinates['y2'], h)
    x1 = max(coordinates['x1'], 0)
    x2 = min(coordinates['x2'], w)
    
    # Ensure valid crop dimensions
    if y2 <= y1 or x2 <= x1:
        return img
    
    img_cropped = img[y1:y2, x1:x2]
    
    if img_cropped.shape[0] == 0 or img_cropped.shape[1] == 0:
        img_cropped = img
    
    return img_cropped

def apply_cropping(image, threshold=100):
    """Apply cropping to PIL image"""
    try:
        img_array = np.array(image)
        cropped_img = crop_img(img_array, threshold)
        return F.to_pil_image(cropped_img)
    except Exception as e:
        print(f"Error in cropping: {e}")
        return image

# ========================================
# FILE OPERATIONS
# ========================================
def create_file_lists():
    """Create file lists for train, validation, and test sets"""
    print("Creating file lists...")
    
    # Get file paths
    train_files = []
    val_files = []
    test_files = []
    
    try:
        train_files = ['/'.join(x.replace("\\", "/").split("/")[-2:]) 
                      for x in glob.glob(join(TRAIN_DATASET_PATH, '*/*.*')) 
                      if x.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        val_files = ['/'.join(x.replace("\\", "/").split("/")[-2:]) 
                    for x in glob.glob(join(VAL_DATASET_PATH, '*/*.*')) 
                    if x.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        test_files = [x.split('/')[-1] 
                     for x in glob.glob(join(TEST_DATASET_PATH, '*.*')) 
                     if x.lower().endswith(('.jpg', '.jpeg', '.png'))]
    except Exception as e:
        print(f"Error creating file lists: {e}")
        return [], [], []
    
    # Create text files
    create_txt(join(TRAIN_DATASET_PATH, 'train_list.txt'), train_files)
    create_txt(join(VAL_DATASET_PATH, 'val_list.txt'), val_files)
    create_txt(join(TEST_DATASET_PATH, 'test_list.txt'), test_files)
    
    print(f"  ✓ Train files: {len(train_files)}")
    print(f"  ✓ Validation files: {len(val_files)}")
    print(f"  ✓ Test files: {len(test_files)}")
    
    return train_files, val_files, test_files

def create_txt(path, name_list):
    """Create text file with file names"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            for item in name_list:
                file.write(item + "\n")
        print(f"  ✓ Created file list: {os.path.basename(path)} with {len(name_list)} entries")
    except Exception as e:
        print(f"  ✗ Error creating {path}: {e}")

def delete_path(path):
    """Delete file if exists"""
    if os.path.exists(path):
        try:
            os.remove(path)
            print(f"✓ {path} has been deleted.")
        except Exception as e:
            print(f"✗ Error deleting {path}: {e}")
    else:
        print(f"Path {path} does not exist.")

# ========================================
# DATA ANALYSIS FUNCTIONS
# ========================================
def analyze_dataset_distribution(files, mode='Dataset'):
    """Analyze and return distribution statistics"""
    if not files:
        print(f"No files found in {mode} dataset")
        return {}
    
    labels = [f.split('/')[0] for f in files]
    labels_count = Counter(labels)
    stats = OrderedDict(sorted(labels_count.items()))
    
    print(f"\n=== {mode} Dataset Statistics ===")
    total = sum(stats.values())
    
    if total == 0:
        print("No samples found")
        return stats
    
    for class_name, count in stats.items():
        percentage = (count / total) * 100
        print(f"{class_name.upper()}: {count:4d} samples ({percentage:5.1f}%)")
    print(f"TOTAL: {total:4d} samples")
    
    # Calculate imbalance ratio
    if len(stats) > 1:
        max_count = max(stats.values())
        min_count = min(stats.values())
        if min_count > 0:
            imbalance_ratio = max_count / min_count
            print(f"Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    return stats

def check_data_integrity():
    """Check data integrity and paths"""
    print("Checking data integrity...")
    
    paths_to_check = [
        TRAIN_DATASET_PATH,
        VAL_DATASET_PATH, 
        TEST_DATASET_PATH
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"  ✓ {path} exists")
        else:
            print(f"  ✗ {path} does not exist")
    
    return all(os.path.exists(path) for path in paths_to_check)

# ========================================
# MAIN EXECUTION FUNCTIONS
# ========================================
def setup_data_pipeline():
    """Setup complete data pipeline"""
    print("Setting up data pipeline...")
    
    # 1. Extract datasets
    extract_dataset()
    
    # 2. Check data integrity
    if not check_data_integrity():
        print("⚠️  Warning: Some data paths are missing")
    
    # 3. Apply data augmentation
    augmentations = get_default_augmentations()
    augment_images_in_folders(TRAIN_DATASET_PATH, CLASSES_TO_AUGMENT, augmentations)
    
    # 4. Create file lists
    train_files, val_files, test_files = create_file_lists()
    
    # 5. Analyze distributions
    train_stats = analyze_dataset_distribution(train_files, 'Training')
    val_stats = analyze_dataset_distribution(val_files, 'Validation')
    
    print("✓ Data pipeline setup completed")
    
    return train_files, val_files, test_files, train_stats, val_stats

if __name__ == "__main__":
    # Test the data pipeline
    setup_data_pipeline()