"""
Train YOLOv8 on SAM 3-Generated Annotations

Fine-tunes YOLOv8 on traffic-specific annotations created by SAM 3.

Usage:
    # Train with default settings
    python train_yolo_with_sam3.py --data data/annotations/yolo_format/dataset.yaml

    # Custom epochs and batch size
    python train_yolo_with_sam3.py --data dataset.yaml --epochs 100 --batch 16

    # Use larger YOLO model
    python train_yolo_with_sam3.py --data dataset.yaml --model yolov8m.pt
"""

import argparse
import sys
from pathlib import Path
import logging
from ultralytics import YOLO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_yolo(
    data_yaml: Path,
    model_size: str = 's',
    epochs: int = 100,
    batch: int = 16,
    img_size: int = 640,
    project: str = 'runs/train',
    name: str = 'traffic_yolo',
    pretrained: bool = True,
    resume: bool = False
):
    """
    Train YOLOv8 on SAM 3-annotated traffic data.

    Args:
        data_yaml: Path to dataset.yaml
        model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        batch: Batch size
        img_size: Input image size
        project: Project directory for outputs
        name: Experiment name
        pretrained: Start from pretrained COCO weights
        resume: Resume from last checkpoint
    """
    logger.info("="*80)
    logger.info("YOLO Training on SAM 3 Annotations")
    logger.info("="*80)
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"Model: YOLOv8{model_size}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch}")
    logger.info(f"Image size: {img_size}")
    logger.info("="*80)

    # Validate dataset
    if not data_yaml.exists():
        logger.error(f"Dataset YAML not found: {data_yaml}")
        logger.error("\nMake sure you've run annotate_with_sam3.py first!")
        return

    # Load model
    if pretrained:
        model_path = f'yolov8{model_size}.pt'
        logger.info(f"Loading pretrained model: {model_path}")
    else:
        model_path = f'yolov8{model_size}.yaml'
        logger.info(f"Training from scratch: {model_path}")

    model = YOLO(model_path)

    # Training parameters
    train_args = {
        'data': str(data_yaml),
        'epochs': epochs,
        'batch': batch,
        'imgsz': img_size,
        'project': project,
        'name': name,
        'pretrained': pretrained,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'auto_augment': 'randaugment',
        'device': None,  # Auto-detect GPU
        'save': True,
        'save_period': -1,
        'cache': False,
        'resume': resume,
        'amp': True,  # Mixed precision training
        'fraction': 1.0,
        'patience': 50,
        'workers': 8,
        'verbose': True
    }

    # Train
    logger.info("\nStarting training...")
    try:
        results = model.train(**train_args)

        logger.info("\n" + "="*80)
        logger.info("Training Complete!")
        logger.info("="*80)

        # Show results
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            logger.info(f"Final mAP@0.5: {metrics.get('metrics/mAP50(B)', 'N/A')}")
            logger.info(f"Final mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")

        # Best model path
        best_model = Path(project) / name / 'weights' / 'best.pt'
        logger.info(f"\nBest model saved to: {best_model}")

        logger.info("\nNext steps:")
        logger.info(f"1. Validate: model.val(data='{data_yaml}')")
        logger.info(f"2. Test inference: model.predict('test_image.jpg')")
        logger.info(f"3. Export: model.export(format='onnx')")
        logger.info(f"4. Use in pipeline: TrafficYOLODetector(model_path='{best_model}')")

        return best_model

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_model(model_path: Path, data_yaml: Path):
    """
    Validate trained YOLO model.

    Args:
        model_path: Path to trained weights
        data_yaml: Path to dataset YAML
    """
    logger.info("="*80)
    logger.info("YOLO Validation")
    logger.info("="*80)

    model = YOLO(model_path)

    logger.info("Running validation...")
    results = model.val(data=str(data_yaml))

    logger.info("\nValidation Results:")
    logger.info(f"mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    logger.info(f"mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on SAM 3-annotated traffic data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train_yolo_with_sam3.py --data data/annotations/yolo_format/dataset.yaml

  # Longer training with larger model
  python train_yolo_with_sam3.py --data dataset.yaml --model m --epochs 200

  # Resume interrupted training
  python train_yolo_with_sam3.py --data dataset.yaml --resume

  # Validate existing model
  python train_yolo_with_sam3.py --validate runs/train/traffic_yolo/weights/best.pt
        """
    )

    parser.add_argument(
        '--data', '-d',
        type=Path,
        help='Path to dataset.yaml (from SAM 3 annotation)'
    )

    parser.add_argument(
        '--model', '-m',
        choices=['n', 's', 'm', 'l', 'x'],
        default='s',
        help='YOLO model size: n(nano), s(small), m(medium), l(large), x(extra) (default: s)'
    )

    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )

    parser.add_argument(
        '--batch', '-b',
        type=int,
        default=16,
        help='Batch size (default: 16, reduce if OOM)'
    )

    parser.add_argument(
        '--img-size', '-i',
        type=int,
        default=640,
        help='Input image size (default: 640)'
    )

    parser.add_argument(
        '--project', '-p',
        default='runs/train',
        help='Project directory (default: runs/train)'
    )

    parser.add_argument(
        '--name', '-n',
        default='traffic_yolo',
        help='Experiment name (default: traffic_yolo)'
    )

    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        help='Resume from last checkpoint'
    )

    parser.add_argument(
        '--from-scratch',
        action='store_true',
        help='Train from scratch (no pretrained weights)'
    )

    parser.add_argument(
        '--validate',
        type=Path,
        help='Validate a trained model (provide path to weights)'
    )

    args = parser.parse_args()

    # Validation mode
    if args.validate:
        if not args.data:
            logger.error("Must provide --data for validation")
            return 1

        validate_model(args.validate, args.data)
        return 0

    # Training mode
    if not args.data:
        logger.error("Must provide --data argument")
        logger.error("\nExpected workflow:")
        logger.error("1. Run: python annotate_with_sam3.py --input screenshots/")
        logger.error("2. Then: python train_yolo_with_sam3.py --data data/annotations/yolo_format/dataset.yaml")
        return 1

    # Train
    best_model = train_yolo(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch=args.batch,
        img_size=args.img_size,
        project=args.project,
        name=args.name,
        pretrained=not args.from_scratch,
        resume=args.resume
    )

    return 0 if best_model else 1


if __name__ == '__main__':
    sys.exit(main())
