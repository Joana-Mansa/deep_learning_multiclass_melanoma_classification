"""Portable single-model training entry point for the existing project modules."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', type=Path)
    parser.add_argument('--val-dir', type=Path)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--smoke-test', action='store_true', help='One CPU epoch on generated images; no downloads or scientific results')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error('epochs must be positive and batch-size at least 2 (BatchNorm).')
    if not args.smoke_test and (args.train_dir is None or args.val_dir is None):
        parser.error('provide --train-dir and --val-dir, or use --smoke-test')

    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder, FakeData
    from dataset import get_train_transforms, get_val_transforms
    from models import create_model
    from train import create_training_config, train_model
    from config import CLASS_NAMES, CLASS_TO_IDX

    torch.manual_seed(42)
    torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_size = 64 if args.smoke_test else 224
    if args.smoke_test:
        train_ds = FakeData(size=12, image_size=(3, image_size, image_size), num_classes=3,
                            transform=get_val_transforms(image_size), random_offset=0)
        val_ds = FakeData(size=12, image_size=(3, image_size, image_size), num_classes=3,
                          transform=get_val_transforms(image_size), random_offset=100)
    else:
        def load_folder(path, transform):
            ds = ImageFolder(str(path), transform=transform)
            if set(ds.classes) != set(CLASS_NAMES):
                raise ValueError(f'{path} must contain mel, bcc and scc subfolders; found {ds.classes}')
            # Preserve the project label convention rather than ImageFolder's alphabetical order.
            index_map = {i: CLASS_TO_IDX[name] for name, i in ds.class_to_idx.items()}
            ds.samples = [(f, index_map[label]) for f, label in ds.samples]
            ds.imgs = ds.samples
            ds.targets = [label for _, label in ds.samples]
            ds.class_to_idx = dict(CLASS_TO_IDX)
            ds.classes = list(CLASS_NAMES)
            return ds
        train_ds = load_folder(args.train_dir, get_train_transforms(image_size, use_advanced=False))
        val_ds = load_folder(args.val_dir, get_val_transforms(image_size))
        if args.train_dir.resolve() == args.val_dir.resolve():
            raise ValueError('Training and validation directories must differ.')
    if len(train_ds) < args.batch_size:
        raise ValueError('Training set must contain at least one full batch.')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    model = create_model('resnet', pretrained=not (args.no_pretrained or args.smoke_test))
    cfg = create_training_config(model, use_mixup=False, use_cutmix=False)
    cfg.update(epochs=1 if args.smoke_test else args.epochs,
               save_path=str(args.output_dir / 'best_model.pt'))
    if args.smoke_test:
        cfg['device'] = torch.device('cpu')
    model, history, metrics = train_model(model, train_loader, val_loader, cfg)
    checkpoint = torch.load(cfg['save_path'], map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    result = {'data': 'generated smoke-test images' if args.smoke_test else 'user-provided train/validation folders',
              'class_to_index': CLASS_TO_IDX, 'seed': 42,
              'train_samples': len(train_ds), 'validation_samples': len(val_ds),
              'best_validation': metrics, 'history': history}
    (args.output_dir / 'training_summary.json').write_text(json.dumps(result, indent=2))
    print('Checkpoint reloaded successfully; summary written to', args.output_dir)


if __name__ == '__main__':
    main()
