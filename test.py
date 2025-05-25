"""
Evaluation functions for melanoma classification project.
Includes testing, cross-validation, and ensemble evaluation.
"""

import os
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.models.inception import InceptionOutputs
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from config import *
from models import EnsembleModel, create_model
from training import train_model, create_training_config

# ========================================
# SINGLE MODEL EVALUATION
# ========================================
def test_model(model, test_loader, config):
    """Test a single model"""
    
    device = config['device']
    criterion = config['loss_function'].to(device)
    metric_function = config['metric_function'].to(device)
    
    model.to(device)
    model.eval()
    
    batch_losses = []
    all_predictions = []
    all_targets = []
    filenames = []
    
    metric_function.reset()
    
    print("Testing model...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if len(batch) == 3:  # Test dataset with filenames
                images, targets, file_names = batch
                filenames.extend(file_names)
            else:  # Regular dataset
                images, targets = batch
            
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Handle Inception V3 outputs
            if isinstance(outputs, InceptionOutputs):
                outputs = outputs.logits
            
            # Calculate loss (only if we have real targets)
            if not (len(set(targets.cpu().numpy())) == 1 and targets[0].item() == 0):
                loss = criterion(outputs, targets)
                batch_losses.append(loss.item())
                metric_function.update(outputs, targets)
                all_targets.extend(targets.cpu().numpy())
            
            all_predictions.append(outputs.cpu())
    
    # Compute metrics
    test_loss = np.mean(batch_losses) if batch_losses else None
    test_acc = metric_function.compute().item() if batch_losses else None
    
    # Combine all predictions
    all_preds = torch.cat(all_predictions, dim=0) if all_predictions else None
    
    results = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'predictions': all_preds,
        'targets': all_targets,
        'filenames': filenames
    }
    
    if test_acc is not None:
        print(f"Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    
    return results

def evaluate_model_detailed(model, test_loader, config, class_names=CLASS_NAMES):
    """Detailed evaluation with classification report and confusion matrix"""
    
    results = test_model(model, test_loader, config)
    
    if results['targets']:  # Only if we have ground truth
        # Get predicted classes
        predictions = results['predictions']
        predicted_classes = predictions.argmax(dim=1).numpy()
        true_classes = np.array(results['targets'])
        
        # Classification report
        print("\n" + "="*60)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*60)
        report = classification_report(
            true_classes, predicted_classes, 
            target_names=class_names, 
            digits=4
        )
        print(report)
        
        # Confusion matrix
        print("\nCONFUSION MATRIX")
        print("-" * 30)
        cm = confusion_matrix(true_classes, predicted_classes)
        print(cm)
        
        # Per-class accuracy
        print("\nPER-CLASS ACCURACY")
        print("-" * 30)
        class_acc = cm.diagonal() / cm.sum(axis=1)
        for i, (class_name, acc) in enumerate(zip(class_names, class_acc)):
            print(f"{class_name}: {acc:.4f}")
        
        results['classification_report'] = report
        results['confusion_matrix'] = cm
        results['per_class_accuracy'] = class_acc
    
    return results

# ========================================
# ENSEMBLE EVALUATION
# ========================================
def test_ensemble(models, test_loader, config, weights=None):
    """Test ensemble of models"""
    
    device = config['device']
    criterion = config['loss_function'].to(device)
    metric_function = config['metric_function'].to(device)
    
    # Move all models to device and set to eval mode
    for model in models:
        model.to(device)
        model.eval()
    
    # Set ensemble weights
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    else:
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
    
    batch_losses = []
    all_predictions = []
    all_targets = []
    filenames = []
    
    metric_function.reset()
    
    print(f"Testing ensemble of {len(models)} models...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing Ensemble"):
            if len(batch) == 3:  # Test dataset with filenames
                images, targets, file_names = batch
                filenames.extend(file_names)
            else:  # Regular dataset
                images, targets = batch
            
            images = images.to(device)
            targets = targets.to(device)
            
            # Get predictions from all models
            ensemble_output = torch.zeros(images.size(0), NUM_CLASSES).to(device)
            
            for model, weight in zip(models, weights):
                output = model(images)
                
                # Handle Inception V3 outputs
                if isinstance(output, InceptionOutputs):
                    output = output.logits
                
                ensemble_output += weight * output
            
            # Calculate loss (only if we have real targets)
            if not (len(set(targets.cpu().numpy())) == 1 and targets[0].item() == 0):
                loss = criterion(ensemble_output, targets)
                batch_losses.append(loss.item())
                metric_function.update(ensemble_output, targets)
                all_targets.extend(targets.cpu().numpy())
            
            all_predictions.append(ensemble_output.cpu())
    
    # Compute metrics
    test_loss = np.mean(batch_losses) if batch_losses else None
    test_acc = metric_function.compute().item() if batch_losses else None
    
    # Combine all predictions
    all_preds = torch.cat(all_predictions, dim=0) if all_predictions else None
    
    results = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'predictions': all_preds,
        'targets': all_targets,
        'filenames': filenames,
        'ensemble_weights': weights
    }
    
    if test_acc is not None:
        print(f"Ensemble Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    
    return results

def train_ensemble(model_names, train_loader, val_loader, config):
    """Train an ensemble of different models"""
    
    print(f"Training ensemble with models: {model_names}")
    
    trained_models = []
    ensemble_histories = []
    
    for i, model_name in enumerate(model_names):
        print(f"\n{'='*60}")
        print(f"Training Model {i+1}/{len(model_names)}: {model_name.upper()}")
        print(f"{'='*60}")
        
        # Create model
        model = create_model(model_name, num_classes=NUM_CLASSES)
        
        # Create individual training config
        individual_config = create_training_config(
            model, 
            learning_rate=config.get('learning_rate', LEARNING_RATE),
            use_mixup=config.get('use_mixup', False),  # Disable for ensemble training
            use_cutmix=config.get('use_cutmix', False)
        )
        
        # Update save path for individual model
        individual_config['save_path'] = os.path.join(
            OUTPUT_DIR, f'ensemble_model_{i}_{model_name}.pt'
        )
        
        # Train model
        trained_model, history, best_metrics = train_model(
            model, train_loader, val_loader, individual_config
        )
        
        trained_models.append(trained_model)
        ensemble_histories.append(history)
        
        print(f"✓ {model_name} training completed")
        print(f"  Best Validation Accuracy: {best_metrics.get('val_acc', 0):.4f}")
    
    # Save ensemble
    ensemble_save_path = os.path.join(OUTPUT_DIR, 'ensemble_complete.pt')
    ensemble_state = {
        'models': [model.state_dict() for model in trained_models],
        'model_names': model_names,
        'histories': ensemble_histories
    }
    torch.save(ensemble_state, ensemble_save_path)
    print(f"\n✓ Ensemble saved to {ensemble_save_path}")
    
    return trained_models, ensemble_histories

# ========================================
# CROSS-VALIDATION
# ========================================
def cross_validate_model(model_fn, dataset, config, k_folds=CV_FOLDS):
    """Perform k-fold cross-validation"""
    
    print(f"Starting {k_folds}-fold cross-validation...")
    
    # Convert dataset to indices for splitting
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_SEED)
    
    cv_results = {
        'fold_accuracies': [],
        'fold_losses': [],
        'fold_kappas': [],
        'fold_histories': []
    }
    
    for fold, (train_indices, val_indices) in enumerate(kfold.split(indices)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold + 1}/{k_folds}")
        print(f"{'='*50}")
        
        # Create data subsets
        train_subset = data.Subset(dataset, train_indices)
        val_subset = data.Subset(dataset, val_indices)
        
        # Create data loaders
        train_loader = data.DataLoader(
            train_subset,
            batch_size=config.get('batch_size', BATCH_SIZE),
            shuffle=True,
            num_workers=config.get('num_workers', NUM_WORKERS),
            drop_last=True
        )
        val_loader = data.DataLoader(
            val_subset,
            batch_size=config.get('batch_size', BATCH_SIZE),
            shuffle=False,
            num_workers=config.get('num_workers', NUM_WORKERS)
        )
        
        print(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}")
        
        # Create fresh model for this fold
        model = model_fn()
        
        # Create training config for this fold
        fold_config = create_training_config(
            model,
            learning_rate=config.get('learning_rate', LEARNING_RATE),
            use_early_stopping=True,
            use_mixup=config.get('use_mixup', False),
            use_cutmix=config.get('use_cutmix', False)
        )
        
        # Reduce epochs for CV
        fold_config['epochs'] = config.get('cv_epochs', CV_EPOCHS)
        fold_config['save_path'] = os.path.join(OUTPUT_DIR, f'cv_fold_{fold}.pt')
        
        # Train model for this fold
        trained_model, history, best_metrics = train_model(
            model, train_loader, val_loader, fold_config
        )
        
        # Store results
        cv_results['fold_accuracies'].append(best_metrics.get('val_acc', 0))
        cv_results['fold_losses'].append(best_metrics.get('val_loss', float('inf')))
        cv_results['fold_kappas'].append(best_metrics.get('val_kappa', 0))
        cv_results['fold_histories'].append(history)
        
        print(f"Fold {fold + 1} Results:")
        print(f"  Accuracy: {best_metrics.get('val_acc', 0):.4f}")
        print(f"  Loss: {best_metrics.get('val_loss', 0):.4f}")
        print(f"  Kappa: {best_metrics.get('val_kappa', 0):.4f}")
    
    # Calculate cross-validation statistics
    cv_results['mean_accuracy'] = np.mean(cv_results['fold_accuracies'])
    cv_results['std_accuracy'] = np.std(cv_results['fold_accuracies'])
    cv_results['mean_loss'] = np.mean(cv_results['fold_losses'])
    cv_results['std_loss'] = np.std(cv_results['fold_losses'])
    cv_results['mean_kappa'] = np.mean(cv_results['fold_kappas'])
    cv_results['std_kappa'] = np.std(cv_results['fold_kappas'])
    
    # Print final CV results
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS ({k_folds}-FOLD)")
    print(f"{'='*60}")
    print(f"Mean Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
    print(f"Mean Loss: {cv_results['mean_loss']:.4f} ± {cv_results['std_loss']:.4f}")
    print(f"Mean Kappa: {cv_results['mean_kappa']:.4f} ± {cv_results['std_kappa']:.4f}")
    print(f"{'='*60}")
    
    return cv_results

# ========================================
# TEST-TIME AUGMENTATION
# ========================================
def test_with_tta(model, test_loader, tta_transforms, config):
    """Test model with Test-Time Augmentation"""
    
    device = config['device']
    model.to(device)
    model.eval()
    
    all_predictions = []
    filenames = []
    
    print(f"Testing with TTA ({len(tta_transforms)} augmentations)...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="TTA Testing"):
            if len(batch) == 3:  # Test dataset with filenames
                images, _, file_names = batch
                filenames.extend(file_names)
            else:  # Regular dataset
                images, _ = batch
            
            batch_predictions = []
            
            # Apply each TTA transform
            for tta_transform in tta_transforms:
                # Apply transform to batch
                tta_images = torch.stack([tta_transform(img) for img in images])
                tta_images = tta_images.to(device)
                
                # Get predictions
                outputs = model(tta_images)
                if isinstance(outputs, InceptionOutputs):
                    outputs = outputs.logits
                
                batch_predictions.append(outputs)
            
            # Average predictions across all TTA transforms
            avg_prediction = torch.stack(batch_predictions).mean(dim=0)
            all_predictions.append(avg_prediction.cpu())
    
    # Combine all predictions
    all_preds = torch.cat(all_predictions, dim=0) if all_predictions else None
    
    results = {
        'predictions': all_preds,
        'filenames': filenames,
        'tta_transforms': len(tta_transforms)
    }
    
    print(f"✓ TTA testing completed with {len(tta_transforms)} augmentations")
    
    return results

# ========================================
# PREDICTION ANALYSIS
# ========================================
def analyze_predictions(predictions, filenames, class_names=CLASS_NAMES, 
                       confidence_threshold=LOW_CONFIDENCE_THRESHOLD):
    """Analyze predictions and provide detailed statistics"""
    
    if predictions is None or len(predictions) == 0:
        print("No predictions to analyze")
        return {}
    
    # Convert to probabilities
    probs = torch.softmax(predictions, dim=1).numpy()
    predicted_classes = np.argmax(probs, axis=1)
    confidence_scores = np.max(probs, axis=1)
    
    # Prediction statistics
    print(f"\n{'='*60}")
    print(f"PREDICTION ANALYSIS")
    print(f"{'='*60}")
    
    # Class distribution
    pred_counts = Counter(predicted_classes)
    total_predictions = len(predicted_classes)
    
    print("Predicted Class Distribution:")
    for i, class_name in enumerate(class_names):
        count = pred_counts.get(i, 0)
        percentage = (count / total_predictions) * 100
        print(f"  {class_name.upper()}: {count:4d} ({percentage:5.1f}%)")
    
    # Confidence statistics
    print(f"\nConfidence Statistics:")
    print(f"  Mean Confidence: {np.mean(confidence_scores):.4f}")
    print(f"  Std Confidence:  {np.std(confidence_scores):.4f}")
    print(f"  Min Confidence:  {np.min(confidence_scores):.4f}")
    print(f"  Max Confidence:  {np.max(confidence_scores):.4f}")
    
    # Low confidence predictions
    low_conf_mask = confidence_scores < confidence_threshold
    low_conf_count = np.sum(low_conf_mask)
    
    if low_conf_count > 0:
        print(f"\nLow Confidence Predictions (< {confidence_threshold}):")
        print(f"  Count: {low_conf_count} ({low_conf_count/total_predictions*100:.1f}%)")
        
        if filenames:
            low_conf_indices = np.where(low_conf_mask)[0]
            print("  Examples:")
            for i in low_conf_indices[:5]:  # Show first 5
                print(f"    {filenames[i]}: {confidence_scores[i]:.3f} -> {class_names[predicted_classes[i]]}")
    
    analysis = {
        'predicted_classes': predicted_classes,
        'probabilities': probs,
        'confidence_scores': confidence_scores,
        'class_distribution': pred_counts,
        'low_confidence_count': low_conf_count,
        'mean_confidence': np.mean(confidence_scores)
    }
    
    return analysis

# ========================================
# RESULTS EXPORT
# ========================================
def export_predictions(predictions, filenames, output_path, 
                      class_names=CLASS_NAMES, include_probabilities=True):
    """Export predictions to CSV file"""
    
    if predictions is None or len(predictions) == 0:
        print("No predictions to export")
        return None
    
    # Analyze predictions
    analysis = analyze_predictions(predictions, filenames, class_names)
    
    # Create DataFrame
    data = {
        'filename': filenames,
        'predicted_class': analysis['predicted_classes'],
        'predicted_label': [class_names[i] for i in analysis['predicted_classes']],
        'confidence': analysis['confidence_scores']
    }
    
    # Add individual class probabilities if requested
    if include_probabilities:
        for i, class_name in enumerate(class_names):
            data[f'{class_name}_probability'] = analysis['probabilities'][:, i]
    
    # Create DataFrame and sort by filename
    df = pd.DataFrame(data)
    
    # Try to sort numerically if filenames follow a pattern
    try:
        df['sort_key'] = df['filename'].apply(
            lambda x: int(x.split('.')[0].replace('xxx', ''))
        )
        df = df.sort_values('sort_key').drop('sort_key', axis=1)
    except:
        # If sorting fails, use alphabetical order
        df = df.sort_values('filename')
    
    # Export detailed version
    detailed_path = output_path.replace('.csv', '_detailed.csv')
    df.to_csv(detailed_path, index=False)
    
    # Export simple version for submission
    simple_df = df[['filename', 'predicted_class']].copy()
    simple_df.to_csv(output_path, index=False, header=False)
    
    print(f"\nPredictions exported:")
    print(f"  Simple format: {output_path}")
    print(f"  Detailed format: {detailed_path}")
    
    return df

if __name__ == "__main__":
    print("Testing evaluation components...")
    
    # Test prediction analysis
    dummy_predictions = torch.randn(100, NUM_CLASSES)
    dummy_filenames = [f"test_{i:03d}.jpg" for i in range(100)]
    
    analysis = analyze_predictions(dummy_predictions, dummy_filenames)
    print("✓ Prediction analysis test passed")
    
    # Test export
    test_output = "/tmp/test_predictions.csv"
    try:
        df = export_predictions(dummy_predictions, dummy_filenames, test_output)
        print("✓ Export test passed")
    except Exception as e:
        print(f"Export test failed: {e}")
    
    print("Evaluation components tested successfully!")