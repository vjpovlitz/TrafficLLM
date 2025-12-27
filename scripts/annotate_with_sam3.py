"""
SAM 3 Batch Annotation Script

Annotates traffic images using SAM 3 and exports to YOLO format for training.

Usage:
    # Annotate all images in a directory
    python annotate_with_sam3.py --input screenshots/validated/ --output data/annotations/

    # With custom prompts
    python annotate_with_sam3.py --input screenshots/ --prompts "car" "truck" "bus"

    # Export to multiple formats
    python annotate_with_sam3.py --input screenshots/ --formats yolo coco masks
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import logging
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from trafficllm.annotation import SAM3Annotator, AnnotationExporter, AnnotationVerifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_image_dimensions(image_path: Path) -> tuple:
    """Get image dimensions without loading full image"""
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def annotate_dataset(
    input_dir: Path,
    output_dir: Path,
    prompts: List[str],
    conf_threshold: float = 0.25,
    export_formats: List[str] = ['yolo'],
    use_ultralytics: bool = True,
    verify: bool = True,
    split: str = 'train'
):
    """
    Batch annotate images with SAM 3 and export to training formats.

    Args:
        input_dir: Directory containing images to annotate
        output_dir: Directory to save annotations
        prompts: Text prompts for detection (e.g., ["car", "truck"])
        conf_threshold: Confidence threshold for detections
        export_formats: List of export formats ('yolo', 'coco', 'masks')
        use_ultralytics: Use Ultralytics wrapper (easier) vs native SAM 3
        verify: Run quality verification
        split: Dataset split ('train', 'val', 'test')
    """
    logger.info("="*80)
    logger.info("SAM 3 Batch Annotation")
    logger.info("="*80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Prompts: {prompts}")
    logger.info(f"Confidence threshold: {conf_threshold}")
    logger.info(f"Export formats: {export_formats}")
    logger.info("="*80)

    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = [
        p for p in input_dir.rglob('*')
        if p.suffix.lower() in image_extensions
    ]

    if not image_paths:
        logger.error(f"No images found in {input_dir}")
        return

    logger.info(f"Found {len(image_paths)} images")

    # Initialize annotator
    logger.info("Loading SAM 3 model...")
    try:
        annotator = SAM3Annotator(use_ultralytics=use_ultralytics)
    except Exception as e:
        logger.error(f"Failed to load SAM 3: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("1. Install Ultralytics: pip install -U ultralytics")
        logger.error("2. Or install native SAM 3:")
        logger.error("   pip install git+https://github.com/facebookresearch/sam3.git")
        logger.error("   huggingface-cli login")
        return

    # Annotate images
    logger.info(f"Annotating {len(image_paths)} images...")
    annotations_dict = {}
    image_dimensions = {}

    for image_path in tqdm(image_paths, desc="Annotating"):
        try:
            # Get image dimensions
            width, height = get_image_dimensions(image_path)
            image_dimensions[str(image_path)] = (width, height)

            # Annotate
            results = annotator.annotate_image(
                image_path,
                prompts=prompts,
                conf_threshold=conf_threshold
            )

            annotations_dict[str(image_path)] = results

            logger.debug(f"{image_path.name}: {len(results)} detections")

        except Exception as e:
            logger.error(f"Failed to annotate {image_path.name}: {e}")
            annotations_dict[str(image_path)] = []

    # Verification
    if verify:
        logger.info("\nRunning quality verification...")
        verifier = AnnotationVerifier(min_confidence=conf_threshold)
        metrics = verifier.verify_batch(annotations_dict, image_dimensions)

        # Count issues
        total_warnings = sum(len(m.warnings) for m in metrics.values())
        total_errors = sum(len(m.errors) for m in metrics.values())

        logger.info(f"Verification complete:")
        logger.info(f"  Total warnings: {total_warnings}")
        logger.info(f"  Total errors: {total_errors}")

        # Show worst offenders
        if total_warnings > 0:
            logger.info("\nImages with most warnings:")
            sorted_metrics = sorted(
                metrics.items(),
                key=lambda x: len(x[1].warnings),
                reverse=True
            )
            for image_path, m in sorted_metrics[:5]:
                if m.warnings:
                    logger.info(f"  {Path(image_path).name}: {len(m.warnings)} warnings")

    # Export annotations
    logger.info("\nExporting annotations...")
    exporter = AnnotationExporter(output_dir=output_dir)

    if 'yolo' in export_formats:
        yaml_path = exporter.export_yolo(
            image_paths,
            annotations_dict,
            split=split
        )
        logger.info(f"✅ YOLO format exported: {yaml_path}")

    if 'coco' in export_formats:
        coco_path = exporter.export_coco(image_paths, annotations_dict)
        logger.info(f"✅ COCO format exported: {coco_path}")

    if 'masks' in export_formats:
        masks_dir = exporter.export_segmentation_masks(
            image_paths,
            annotations_dict,
            mode='semantic'
        )
        logger.info(f"✅ Segmentation masks exported: {masks_dir}")

    # Summary report
    summary_path = exporter.export_summary_report(annotations_dict)
    logger.info(f"✅ Summary report: {summary_path}")

    # Final statistics
    total_annotations = sum(len(anns) for anns in annotations_dict.values())
    images_with_detections = sum(1 for anns in annotations_dict.values() if len(anns) > 0)

    logger.info("\n" + "="*80)
    logger.info("Annotation Complete!")
    logger.info("="*80)
    logger.info(f"Total images: {len(image_paths)}")
    logger.info(f"Images with detections: {images_with_detections}")
    logger.info(f"Total annotations: {total_annotations}")
    logger.info(f"Average per image: {total_annotations / len(image_paths):.1f}")
    logger.info("="*80)

    logger.info("\nNext steps:")
    logger.info("1. Review annotations in verification GUI (optional)")
    logger.info("2. Split data into train/val/test sets")
    logger.info("3. Train YOLO with: scripts/train_yolo.py")


def main():
    parser = argparse.ArgumentParser(
        description="Batch annotate traffic images with SAM 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Annotate with default vehicle prompts
  python annotate_with_sam3.py --input screenshots/validated/

  # Custom prompts and confidence
  python annotate_with_sam3.py --input data/ --prompts "red car" "truck" --conf 0.3

  # Export multiple formats
  python annotate_with_sam3.py --input screenshots/ --formats yolo coco masks

  # Annotate validation set
  python annotate_with_sam3.py --input screenshots/val/ --split val
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Input directory containing images'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('data/annotations'),
        help='Output directory for annotations (default: data/annotations)'
    )

    parser.add_argument(
        '--prompts', '-p',
        nargs='+',
        default=['car', 'truck', 'bus', 'motorcycle'],
        help='Text prompts for detection (default: car truck bus motorcycle)'
    )

    parser.add_argument(
        '--conf', '-c',
        type=float,
        default=0.25,
        help='Confidence threshold [0-1] (default: 0.25)'
    )

    parser.add_argument(
        '--formats', '-f',
        nargs='+',
        choices=['yolo', 'coco', 'masks'],
        default=['yolo'],
        help='Export formats (default: yolo)'
    )

    parser.add_argument(
        '--split', '-s',
        choices=['train', 'val', 'test'],
        default='train',
        help='Dataset split (default: train)'
    )

    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip quality verification'
    )

    parser.add_argument(
        '--native',
        action='store_true',
        help='Use native SAM 3 instead of Ultralytics wrapper'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input.exists():
        logger.error(f"Input directory does not exist: {args.input}")
        return 1

    # Run annotation
    try:
        annotate_dataset(
            input_dir=args.input,
            output_dir=args.output,
            prompts=args.prompts,
            conf_threshold=args.conf,
            export_formats=args.formats,
            use_ultralytics=not args.native,
            verify=not args.no_verify,
            split=args.split
        )
        return 0
    except Exception as e:
        logger.error(f"Annotation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
