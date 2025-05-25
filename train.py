"""
Training functions and utilities for melanoma classification project.
Includes advanced training techniques like MixUp, CutMix, and early stopping.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.inception import InceptionOutputs
from tqdm import tqdm

from config import *

# ========================================
# ADVANCED AUGMENTATION TECHNIQUES
# ========================================
class MixUp:
    """MixUp augmentation for better generalization"""
    
    def __init__(self, alpha=MIXUP_ALPHA):
        self.alpha = alpha

    def __call__(self, batch):
        """Apply MixUp to a batch"""
        images, targets = batch
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)

        mixed_images = lam * images + (1 - lam) * images[index, :]
        targets_a, targets_b = targets, targets[index]
        
        return mixed_images, targets_a, targets_b, lam

class CutMix:
    """CutMix augmentation for improved robustness"""
    
    def __init__(self, alpha=CUTMIX_ALPHA):
        self.alpha = alpha

    def rand_bbox(self, size, lam):
        """Generate random bounding box"""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, batch):
        """Apply CutMix to a batch"""
        images, targets = batch
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)

        targets_a, targets_b = targets, targets[index]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
        
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

        return images, targets_a, targets_b, lam

# ========================================
# EARLY STOPPING
# ========================================
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, 
                 min_delta=EARLY_STOPPING_MIN_DELTA, 
                 restore_best_weights=True, mode='min'):
        """
        Args:
            patience: Number of epochs to wait after last time validation loss improved
            min_delta: Minimum change in the monitored quantity to qualify as an improvement
            restore_best_weights: Whether to restore model weights from the best epoch
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
        self.is_better = self._init_is_better()

    def _init_is_better(self):
        """Initialize comparison function based on mode"""
        if self.mode == 'min':
            return lambda score, best: score < best - self.min_delta
        else:  # mode == 'max'
            return lambda score, best: score > best + self.min_delta

    def __call__(self, score, model):
        """Check if early stopping should be triggered"""
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        """Save model when validation score improves"""
        if self.restore_best_weights:
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

# ========================================
# LOSS FUNCTIONS
# ========================================
def mixup_criterion(criterion, pred, targets_a, targets_b, lam):
    """MixUp/CutMix loss calculation"""
    return lam * criterion(pred, targets_a) + (1 - lam) * criterion(pred, targets_b)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = torch.tensor([alpha] * NUM_CLASSES, dtype=torch.float32)

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Loss"""
    
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

# ========================================
# TRAINING FUNCTIONS
# ========================================
def train_single_epoch(model, train_loader, optimizer, criterion, 
                      metric_function, cohen_kappa_function, device,
                      use_mixup=False, use_cutmix=False):
    """Train model for one epoch"""
    
    model.train()
    batch_losses = []
    
    # Initialize augmentation strategies
    mixup = MixUp(alpha=MIXUP_ALPHA) if use_mixup else None
    cutmix = CutMix(alpha=CUTMIX_ALPHA) if use_cutmix else None
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        
        # Apply MixUp or CutMix randomly
        use_aug = np.random.rand() < AUGMENTATION_PROBABILITY
        
        if use_mixup and use_aug and np.random.rand() < 0.5:
            mixed_images, targets_a, targets_b, lam = mixup((images, targets))
            outputs = model(mixed_images)
            
            # Handle Inception V3 outputs
            if isinstance(outputs, InceptionOutputs):
                outputs = outputs.logits
                
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
        elif use_cutmix and use_aug:
            mixed_images, targets_a, targets_b, lam = cutmix((images, targets))
            outputs = model(mixed_images)
            
            # Handle Inception V3 outputs
            if isinstance(outputs, InceptionOutputs):
                outputs = outputs.logits
                
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
        else:
            outputs = model(images)
            
            # Handle Inception V3 outputs
            if isinstance(outputs, InceptionOutputs):
                outputs = outputs.logits
                
            loss = criterion(outputs, targets)
        
        # Update metrics (use original targets for metric calculation)
        metric_function.update(outputs, targets)
        cohen_kappa_function.update(outputs.argmax(dim=1), targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        batch_losses.append(loss.detach().cpu().item())
    
    # Compute metrics
    train_acc = metric_function.compute().item()
    train_kappa = cohen_kappa_function.compute().item()
    
    # Reset metrics
    metric_function.reset()
    cohen_kappa_function.reset()
    
    return np.mean(batch_losses), train_acc, train_kappa

def validate_single_epoch(model, val_loader, criterion, metric_function, 
                         cohen_kappa_function, device):
    """Validate model for one epoch"""
    
    model.eval()
    batch_losses = []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            
            # Handle Inception V3 outputs
            if isinstance(outputs, InceptionOutputs):
                outputs = outputs.logits
                
            loss = criterion(outputs, targets)
            
            batch_losses.append(loss.detach().cpu().item())
            metric_function.update(outputs, targets)
            cohen_kappa_function.update(outputs.argmax(dim=1), targets)
    
    # Compute metrics
    val_acc = metric_function.compute().item()
    val_kappa = cohen_kappa_function.compute().item()
    
    # Reset metrics
    metric_function.reset()
    cohen_kappa_function.reset()
    
    return np.mean(batch_losses), val_acc, val_kappa

def train_model(model, train_loader, val_loader, config):
    """Complete training loop with advanced features"""
    
    # Extract configuration
    device = config['device']
    epochs = config['epochs']
    criterion = config['loss_function'].to(device)
    optimizer = config['optimizer']
    scheduler = config.get('scheduler', None)
    early_stopping = config.get('early_stopping', None)
    metric_function = config['metric_function'].to(device)
    cohen_kappa_function = config['cohen_kappa_function'].to(device)
    use_mixup = config.get('use_mixup', False)
    use_cutmix = config.get('use_cutmix', False)
    save_path = config.get('save_path', MODEL_SAVE_PATH)
    
    # Move model to device
    model.to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_kappa': [],
        'val_kappa': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    best_metrics = {}
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Using MixUp: {use_mixup}, Using CutMix: {use_cutmix}")
    
    with tqdm(range(epochs), desc="Training") as pbar:
        for epoch in pbar:
            # Train one epoch
            train_loss, train_acc, train_kappa = train_single_epoch(
                model, train_loader, optimizer, criterion,
                metric_function.clone(), cohen_kappa_function.clone(), device,
                use_mixup, use_cutmix
            )
            
            # Validate one epoch
            val_loss, val_acc, val_kappa = validate_single_epoch(
                model, val_loader, criterion,
                metric_function.clone(), cohen_kappa_function.clone(), device
            )
            
            # Update learning rate
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Store metrics
            current_lr = optimizer.param_groups[0]['lr']
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_kappa'].append(train_kappa)
            history['val_kappa'].append(val_kappa)
            history['lr'].append(current_lr)
            
            # Update progress bar
            pbar.set_postfix({
                'TrL': f'{train_loss:.4f}',
                'TrA': f'{train_acc:.4f}',
                'VL': f'{val_loss:.4f}',
                'VA': f'{val_acc:.4f}',
                'LR': f'{current_lr:.6f}'
            })
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_metrics = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'train_kappa': train_kappa,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_kappa': val_kappa
                }
                
                # Save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'metrics': best_metrics
                }, save_path)
            
            # Early stopping
            if early_stopping:
                if early_stopping(val_loss, model):
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break
    
    # Print final results
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Validation Kappa: {best_metrics.get('val_kappa', 0):.4f}")
    print(f"Model saved to: {save_path}")
    print(f"{'='*60}")
    
    return model, history, best_metrics

def create_training_config(model, learning_rate=LEARNING_RATE, 
                          weight_decay=WEIGHT_DECAY, use_scheduler=True,
                          use_early_stopping=True, use_mixup=True, use_cutmix=True,
                          loss_type='crossentropy'):
    """Create training configuration"""
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Loss function
    if loss_type == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_type == 'focal':
        criterion = FocalLoss(gamma=2.0)
    elif loss_type == 'label_smoothing':
        criterion = LabelSmoothingLoss(num_classes=NUM_CLASSES, smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Scheduler
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, 
            patience=LR_SCHEDULER_PATIENCE, verbose=True
        )
    
    # Early stopping
    early_stopping = None
    if use_early_stopping:
        early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA,
            restore_best_weights=True,
            mode='min'
        )
    
    # Metrics
    from torchmetrics import Accuracy, CohenKappa
    metric_function = Accuracy(task='multiclass', num_classes=NUM_CLASSES, average='macro')
    cohen_kappa_function = CohenKappa(task='multiclass', num_classes=NUM_CLASSES)
    
    config = {
        'device': device,
        'epochs': EPOCHS,
        'loss_function': criterion,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'early_stopping': early_stopping,
        'metric_function': metric_function,
        'cohen_kappa_function': cohen_kappa_function,
        'use_mixup': use_mixup,
        'use_cutmix': use_cutmix,
        'save_path': MODEL_SAVE_PATH
    }
    
    return config

# ========================================
# MODEL LOADING/SAVING
# ========================================
def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer, filepath):
    """Load training checkpoint"""
    if not os.path.exists(filepath):
        print(f"Checkpoint not found: {filepath}")
        return model, optimizer, 0, {}
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    print(f"Loaded checkpoint from epoch {epoch}")
    return model, optimizer, epoch, metrics

def load_best_model(model, filepath=MODEL_SAVE_PATH):
    """Load the best saved model"""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded best model from {filepath}")
        return model, checkpoint.get('metrics', {})
    else:
        print(f"✗ Model file not found: {filepath}")
        return model, {}

if __name__ == "__main__":
    # Test training components
    print("Testing training components...")
    
    # Test MixUp
    batch_size = 4
    test_images = torch.randn(batch_size, 3, 224, 224)
    test_targets = torch.randint(0, NUM_CLASSES, (batch_size,))
    
    mixup = MixUp()
    mixed_images, targets_a, targets_b, lam = mixup((test_images, test_targets))
    print(f"✓ MixUp test passed - Lambda: {lam:.3f}")
    
    # Test CutMix
    cutmix = CutMix()
    mixed_images, targets_a, targets_b, lam = cutmix((test_images, test_targets))
    print(f"✓ CutMix test passed - Lambda: {lam:.3f}")
    
    # Test Early Stopping
    early_stopping = EarlyStopping(patience=3)
    print("✓ Early stopping initialized")
    
    print("All training components tested successfully!")