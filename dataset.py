"""
Dataset classes and custom transforms for melanoma classification project.
"""

import os
import numpy as np
import cv2
from os.path import join
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as T
from torchvision.transforms import functional as F

from config import *
from data_utils import apply_cropping

# ========================================
# CUSTOM TRANSFORMS
# ========================================
class HistogramEqualization:
    """Custom transform for histogram equalization"""
    
    def __call__(self, img):
        """Apply histogram equalization to PIL image"""
        try:
            img_array = np.array(img)
            
            if len(img_array.shape) == 2:  # Grayscale
                img_eq = cv2.equalizeHist(img_array)
            else:  # Color
                # Convert to LAB color space
                img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                # Apply CLAHE to the L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
                # Convert back to RGB
                img_eq = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
            
            return F.to_pil_image(img_eq)
        except Exception as e:
            print(f"Error in histogram equalization: {e}")
            return img

class GaussianNoise:
    """Add Gaussian noise to image"""
    
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        """Add Gaussian noise to tensor"""
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class RandomErasing:
    """Randomly erase rectangular regions in image"""
    
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):
        """Apply random erasing to tensor image"""
        if np.random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = np.random.uniform(self.sl, self.sh) * area
            aspect_ratio = np.random.uniform(self.r1, 1/self.r1)

            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = np.random.randint(0, img.size()[1] - h)
                y1 = np.random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = np.random.uniform(0, 1)
                    img[1, x1:x1+h, y1:y1+w] = np.random.uniform(0, 1)
                    img[2, x1:x1+h, y1:y1+w] = np.random.uniform(0, 1)
                else:
                    img[0, x1:x1+h, y1:y1+w] = np.random.uniform(0, 1)
                return img

        return img

# ========================================
# TRANSFORM FACTORIES
# ========================================
def get_train_transforms(image_size=IM_SIZE, use_advanced=True):
    """Get training transforms"""
    
    base_transforms = [
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        T.Normalize(mean=CUSTOM_MEAN, std=CUSTOM_STD),
    ]
    
    if use_advanced:
        # Insert advanced augmentations before ToTensor
        advanced_transforms = [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(degrees=20),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            # HistogramEqualization(),  # Uncomment if needed
            T.ToTensor(),
            T.Normalize(mean=CUSTOM_MEAN, std=CUSTOM_STD),
            # RandomErasing(probability=0.3),  # Uncomment if needed
        ]
        return T.Compose(advanced_transforms)
    
    return T.Compose(base_transforms)

def get_val_transforms(image_size=IM_SIZE):
    """Get validation/test transforms"""
    
    transforms = [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=CUSTOM_MEAN, std=CUSTOM_STD),
    ]
    
    return T.Compose(transforms)

def get_tta_transforms(image_size=IM_SIZE):
    """Get Test-Time Augmentation transforms"""
    
    tta_transforms = []
    
    # Original
    tta_transforms.append(T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=CUSTOM_MEAN, std=CUSTOM_STD),
    ]))
    
    # Horizontal flip
    tta_transforms.append(T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(mean=CUSTOM_MEAN, std=CUSTOM_STD),
    ]))
    
    # Rotation variants
    for angle in [90, 180, 270]:
        tta_transforms.append(T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomRotation(degrees=(angle, angle)),
            T.ToTensor(),
            T.Normalize(mean=CUSTOM_MEAN, std=CUSTOM_STD),
        ]))
    
    return tta_transforms

# ========================================
# DATASET CLASSES
# ========================================
class DermoscopicDataset(data.Dataset):
    """Dataset class for dermoscopic images with cropping and transforms"""
    
    def __init__(self, root_path, name_list, transforms=None, 
                 cropping_threshold=CROPPING_THRESHOLD, image_size=IM_SIZE):
        """
        Args:
            root_path (str): Path to the dataset directory
            name_list (str): Path to the file containing image file names
            transforms (callable, optional): Transformations to apply after cropping
            cropping_threshold (int): Threshold for the cropping function
            image_size (int): Target image size
        """
        self.root_path = root_path
        self.transforms = transforms
        self.cropping_threshold = cropping_threshold
        self.image_size = image_size
        
        # Load image filenames
        list_path = join(root_path, name_list)
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"File list not found: {list_path}")
            
        with open(list_path, 'r') as f:
            self.image_filenames = [line.strip() for line in f.readlines() if line.strip()]
        
        if not self.image_filenames:
            raise ValueError(f"No image filenames found in {list_path}")
        
        print(f"Loaded {len(self.image_filenames)} images from {name_list}")

    def __getitem__(self, index):
        """Get a single data point"""
        try:
            # Load image
            img_path = join(self.root_path, self.image_filenames[index])
            
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            
            image = Image.open(img_path).convert('RGB')
            
            # Apply cropping
            cropped_image = apply_cropping(image, threshold=self.cropping_threshold)
            
            # Apply transforms
            if self.transforms:
                cropped_image = self.transforms(cropped_image)
            
            # Validate image shape
            if hasattr(cropped_image, 'shape'):
                expected_shape = (3, self.image_size, self.image_size)
                if cropped_image.shape != expected_shape:
                    print(f"Warning: Unexpected image shape {cropped_image.shape}, expected {expected_shape}")
            
            # Extract class from filename
            class_name = self.image_filenames[index].split('/')[0].lower()
            
            # Map class name to index
            if class_name in CLASS_TO_IDX:
                target = CLASS_TO_IDX[class_name]
            else:
                print(f"Warning: Unknown class '{class_name}', using class 0")
                target = 0
            
            return cropped_image, torch.tensor(target, dtype=torch.long)
            
        except Exception as e:
            print(f"Error loading image at index {index}: {e}")
            # Return a dummy sample
            dummy_image = torch.zeros(3, self.image_size, self.image_size)
            return dummy_image, torch.tensor(0, dtype=torch.long)

    def __len__(self):
        return len(self.image_filenames)
    
    def get_class_distribution(self):
        """Get distribution of classes in dataset"""
        class_counts = {class_name: 0 for class_name in CLASS_NAMES}
        
        for filename in self.image_filenames:
            class_name = filename.split('/')[0].lower()
            if class_name in class_counts:
                class_counts[class_name] += 1
        
        return class_counts

class TestDataset(data.Dataset):
    """Dataset class for test images without ground truth labels"""
    
    def __init__(self, root_path, name_list, transforms=None, 
                 cropping_threshold=CROPPING_THRESHOLD):
        """
        Args:
            root_path (str): Path to the dataset directory
            name_list (str): Path to the file containing image file names
            transforms (callable, optional): Transformations to apply after cropping
            cropping_threshold (int): Threshold for the cropping function
        """
        self.root_path = root_path
        self.transforms = transforms
        self.cropping_threshold = cropping_threshold
        
        # Load image filenames
        list_path = join(root_path, name_list)
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"File list not found: {list_path}")
            
        with open(list_path, 'r') as f:
            self.image_filenames = [line.strip() for line in f.readlines() if line.strip()]
        
        if not self.image_filenames:
            raise ValueError(f"No image filenames found in {list_path}")
            
        print(f"Loaded {len(self.image_filenames)} test images from {name_list}")

    def __getitem__(self, index):
        """Get a single data point"""
        try:
            # Load image
            img_path = join(self.root_path, self.image_filenames[index])
            
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            
            image = Image.open(img_path).convert('RGB')
            
            # Apply cropping
            cropped_image = apply_cropping(image, threshold=self.cropping_threshold)
            
            # Apply transforms
            if self.transforms:
                cropped_image = self.transforms(cropped_image)
            
            # Get filename for identification
            file_name = self.image_filenames[index]
            
            # Dummy ground truth for compatibility
            gt = torch.tensor(0, dtype=torch.long)
            
            return cropped_image, gt, file_name
            
        except Exception as e:
            print(f"Error loading test image at index {index}: {e}")
            # Return a dummy sample
            dummy_image = torch.zeros(3, IM_SIZE, IM_SIZE)
            dummy_filename = f"error_{index}.jpg"
            return dummy_image, torch.tensor(0, dtype=torch.long), dummy_filename

    def __len__(self):
        return len(self.image_filenames)

class TTADataset(data.Dataset):
    """Dataset class for Test-Time Augmentation"""
    
    def __init__(self, base_dataset, tta_transforms):
        """
        Args:
            base_dataset: Base dataset (TestDataset or DermoscopicDataset)
            tta_transforms: List of transform compositions for TTA
        """
        self.base_dataset = base_dataset
        self.tta_transforms = tta_transforms
        self.n_tta = len(tta_transforms)
    
    def __getitem__(self, index):
        """Get TTA variants of a single image"""
        base_index = index // self.n_tta
        tta_index = index % self.n_tta
        
        # Get original data (without transforms)
        if hasattr(self.base_dataset, 'transforms'):
            original_transforms = self.base_dataset.transforms
            self.base_dataset.transforms = None
            
        item = self.base_dataset[base_index]
        
        if hasattr(self.base_dataset, 'transforms'):
            self.base_dataset.transforms = original_transforms
        
        # Apply TTA transform
        if len(item) == 3:  # Test dataset with filename
            image, gt, filename = item
            transformed_image = self.tta_transforms[tta_index](image)
            return transformed_image, gt, filename
        else:  # Regular dataset
            image, gt = item
            transformed_image = self.tta_transforms[tta_index](image)
            return transformed_image, gt
    
    def __len__(self):
        return len(self.base_dataset) * self.n_tta

# ========================================
# DATASET FACTORY FUNCTIONS
# ========================================
def create_datasets(use_advanced_transforms=True):
    """Create train, validation, and test datasets"""
    
    print("Creating datasets...")
    
    # Get appropriate image size
    image_size = IM_SIZE
    
    # Create transforms
    train_transforms = get_train_transforms(image_size, use_advanced_transforms)
    val_transforms = get_val_transforms(image_size)
    
    try:
        # Create datasets
        train_dataset = DermoscopicDataset(
            root_path=TRAIN_DATASET_PATH,
            name_list='train_list.txt',
            transforms=train_transforms,
            cropping_threshold=CROPPING_THRESHOLD,
            image_size=image_size
        )
        
        val_dataset = DermoscopicDataset(
            root_path=VAL_DATASET_PATH,
            name_list='val_list.txt',
            transforms=val_transforms,
            cropping_threshold=CROPPING_THRESHOLD,
            image_size=image_size
        )
        
        test_dataset = TestDataset(
            root_path=TEST_DATASET_PATH,
            name_list='test_list.txt',
            transforms=val_transforms,
            cropping_threshold=CROPPING_THRESHOLD
        )
        
        print(f"✓ Created datasets:")
        print(f"  - Train: {len(train_dataset)} samples")
        print(f"  - Validation: {len(val_dataset)} samples")
        print(f"  - Test: {len(test_dataset)} samples")
        
        return train_dataset, val_dataset, test_dataset
        
    except Exception as e:
        print(f"✗ Error creating datasets: {e}")
        raise

def create_data_loaders(train_dataset, val_dataset, test_dataset, 
                       batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """Create data loaders from datasets"""
    
    print("Creating data loaders...")
    
    try:
        train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        test_loader = data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"✓ Created data loaders:")
        print(f"  - Train: {len(train_loader)} batches")
        print(f"  - Validation: {len(val_loader)} batches")
        print(f"  - Test: {len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"✗ Error creating data loaders: {e}")
        raise

if __name__ == "__main__":
    # Test dataset creation
    from data_utils import setup_data_pipeline
    
    # Setup data first
    setup_data_pipeline()
    
    # Create datasets
    train_ds, val_ds, test_ds = create_datasets()
    
    # Test loading a sample
    print("\nTesting dataset loading...")
    train_sample = train_ds[0]
    print(f"Train sample shape: {train_sample[0].shape}, label: {train_sample[1]}")
    
    test_sample = test_ds[0]
    print(f"Test sample shape: {test_sample[0].shape}, filename: {test_sample[2]}")
    
    # Show class distribution
    print("\nClass distribution in training set:")
    train_dist = train_ds.get_class_distribution()
    for class_name, count in train_dist.items():
        print(f"  {class_name}: {count}")