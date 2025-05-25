"""
Model architectures for melanoma classification project.
Contains definitions for various CNN architectures.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.inception import InceptionOutputs

from config import *

# ========================================
# MODEL ARCHITECTURES
# ========================================
class ModifiedVGG16(nn.Module):
    """Modified VGG-16 for 3-class classification"""
    
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.6, freeze_layers=16):
        super(ModifiedVGG16, self).__init__()
        
        # Load pretrained VGG16
        self.backbone = models.vgg16(pretrained=True)
        
        # Freeze early layers
        if freeze_layers > 0:
            for i, param in enumerate(self.backbone.features.parameters()):
                if i < freeze_layers:
                    param.requires_grad = False
        
        # Modify classifier
        flattened_size = 512 * 7 * 7
        self.backbone.classifier = nn.Sequential(
            nn.Linear(flattened_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        return self.backbone(x)

class ModifiedResNet50(nn.Module):
    """Modified ResNet-50 for 3-class classification"""
    
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.6, pretrained=True):
        super(ModifiedResNet50, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Get number of features from the last layer
        num_features = self.backbone.fc.in_features
        
        # Replace the final layer
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        return self.backbone(x)

class ModifiedInceptionV3(nn.Module):
    """Modified Inception V3 for 3-class classification"""
    
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.6, pretrained=True):
        super(ModifiedInceptionV3, self).__init__()
        
        # Load pretrained Inception V3
        self.backbone = models.inception_v3(pretrained=pretrained)
        
        # Get number of features from the last layer
        num_features = self.backbone.fc.in_features
        
        # Replace the final layer
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        return self.backbone(x)

class ModifiedSwinTransformer(nn.Module):
    """Modified Swin Transformer V2 for 3-class classification"""
    
    def __init__(self, num_classes=NUM_CLASSES, model_size='base', pretrained=True):
        super(ModifiedSwinTransformer, self).__init__()
        
        # Load appropriate Swin Transformer model
        if model_size == 'tiny':
            self.backbone = models.swin_v2_t(weights='DEFAULT' if pretrained else None)
        elif model_size == 'small':
            self.backbone = models.swin_v2_s(weights='DEFAULT' if pretrained else None)
        elif model_size == 'base':
            self.backbone = models.swin_v2_b(weights='DEFAULT' if pretrained else None)
        else:
            raise ValueError(f"Unsupported model size: {model_size}")
        
        # Get number of features from the head
        in_features = self.backbone.head.in_features
        
        # Replace the head
        self.backbone.head = nn.Linear(in_features=in_features, out_features=num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class ModifiedEfficientNet(nn.Module):
    """Modified EfficientNet for 3-class classification"""
    
    def __init__(self, num_classes=NUM_CLASSES, model_variant='b0', dropout=0.2, pretrained=True):
        super(ModifiedEfficientNet, self).__init__()
        
        # Load appropriate EfficientNet model
        if model_variant == 'b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
        elif model_variant == 'b1':
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
        elif model_variant == 'b2':
            self.backbone = models.efficientnet_b2(pretrained=pretrained)
        elif model_variant == 'b3':
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {model_variant}")
        
        # Get number of features from classifier
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(num_features, num_classes),
        )
    
    def forward(self, x):
        return self.backbone(x)

class CustomCNN(nn.Module):
    """Custom CNN architecture for melanoma classification"""
    
    def __init__(self, num_classes=NUM_CLASSES, input_channels=3, dropout=0.5):
        super(CustomCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ========================================
# MODEL FACTORY FUNCTIONS
# ========================================
def create_model(model_name, num_classes=NUM_CLASSES, pretrained=True, **kwargs):
    """Factory function to create models"""
    
    model_name = model_name.lower()
    
    if model_name == 'vgg16':
        return ModifiedVGG16(num_classes=num_classes, **kwargs)
    
    elif model_name == 'resnet50' or model_name == 'resnet':
        return ModifiedResNet50(num_classes=num_classes, pretrained=pretrained, **kwargs)
    
    elif model_name == 'inception' or model_name == 'inceptionv3':
        return ModifiedInceptionV3(num_classes=num_classes, pretrained=pretrained, **kwargs)
    
    elif model_name.startswith('swin'):
        model_size = 'base'  # default
        if 'tiny' in model_name:
            model_size = 'tiny'
        elif 'small' in model_name:
            model_size = 'small'
        return ModifiedSwinTransformer(num_classes=num_classes, model_size=model_size, pretrained=pretrained)
    
    elif model_name.startswith('efficientnet'):
        variant = model_name.replace('efficientnet', '') or 'b0'
        return ModifiedEfficientNet(num_classes=num_classes, model_variant=variant, **kwargs)
    
    elif model_name == 'custom':
        return CustomCNN(num_classes=num_classes, **kwargs)
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def get_model_info(model_name):
    """Get information about a model"""
    
    model_name = model_name.lower()
    
    info = {
        'name': model_name,
        'input_size': IM_SIZE,
        'pretrained': True,
        'parameters': 'Unknown'
    }
    
    if model_name in AVAILABLE_MODELS:
        info.update(AVAILABLE_MODELS[model_name])
    
    return info

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }

def print_model_summary(model, model_name="Model"):
    """Print a summary of the model"""
    
    param_info = count_parameters(model)
    
    print(f"\n{'='*50}")
    print(f"{model_name.upper()} SUMMARY")
    print(f"{'='*50}")
    print(f"Total Parameters: {param_info['total']:,}")
    print(f"Trainable Parameters: {param_info['trainable']:,}")
    print(f"Frozen Parameters: {param_info['frozen']:,}")
    print(f"{'='*50}")

def initialize_weights(model):
    """Initialize model weights"""
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

# ========================================
# ENSEMBLE MODEL CLASS
# ========================================
class EnsembleModel(nn.Module):
    """Ensemble of multiple models"""
    
    def __init__(self, models, weights=None):
        """
        Args:
            models: List of models to ensemble
            weights: List of weights for each model (if None, equal weights)
        """
        super(EnsembleModel, self).__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Number of weights must match number of models"
            # Normalize weights
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
    
    def forward(self, x):
        """Forward pass through all models and combine predictions"""
        
        predictions = []
        
        for model in self.models:
            pred = model(x)
            # Handle Inception V3 auxiliary outputs
            if isinstance(pred, InceptionOutputs):
                pred = pred.logits
            predictions.append(pred)
        
        # Weighted average of predictions
        ensemble_pred = torch.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def set_weights(self, weights):
        """Update ensemble weights"""
        assert len(weights) == len(self.models), "Number of weights must match number of models"
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]

# ========================================
# MODEL TESTING
# ========================================
def test_model_creation():
    """Test model creation and basic functionality"""
    
    print("Testing model creation...")
    
    test_input = torch.randn(1, 3, IM_SIZE, IM_SIZE)
    
    models_to_test = ['resnet', 'inception', 'swin', 'vgg16', 'custom']
    
    for model_name in models_to_test:
        try:
            print(f"\nTesting {model_name}...")
            model = create_model(model_name, num_classes=NUM_CLASSES)
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                output = model(test_input)
            
            print(f"  ✓ Output shape: {output.shape}")
            print(f"  ✓ Expected shape: (1, {NUM_CLASSES})")
            
            # Print parameter count
            param_info = count_parameters(model)
            print(f"  ✓ Parameters: {param_info['trainable']:,}")
            
        except Exception as e:
            print(f"  ✗ Error with {model_name}: {e}")

if __name__ == "__main__":
    test_model_creation()