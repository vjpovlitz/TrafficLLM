#!/usr/bin/env python3
"""
Vehicle Annotation Pipeline: YOLO Detection + SAM 2 Segmentation

This script combines:
1. YOLO for fast vehicle detection (bounding boxes)
2. SAM 2 for precise segmentation masks

Optimized for Apple Silicon Macs using MPS backend.

Usage:
    python scripts/annotate_vehicles_sam2.py --input data/image_library
    python scripts/annotate_vehicles_sam2.py --input path/to/images --output annotations
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from PIL import Image
import cv2


@dataclass
class VehicleAnnotation:
    """Single vehicle annotation with bbox and mask."""
    class_name: str
    class_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    mask_rle: Optional[dict] = None  # Run-length encoding for mask

    def to_yolo_format(self, img_width: int, img_height: int) -> str:
        """Convert to YOLO format: class_id x_center y_center width height"""
        x1, y1, x2, y2 = self.bbox
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        return f"{self.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


class VehicleAnnotator:
    """
    Combined YOLO + SAM 2 annotator for traffic images.

    Workflow:
    1. YOLO detects vehicle bounding boxes
    2. SAM 2 generates precise segmentation masks using those boxes as prompts
    """

    # COCO vehicle classes
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck',
    }

    # Class mapping for YOLO training
    CLASS_MAPPING = {
        'car': 0,
        'motorcycle': 1,
        'bus': 2,
        'truck': 3,
    }

    def __init__(
        self,
        yolo_model: str = "yolo11n.pt",
        sam_model: str = "sam2_t.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.25
    ):
        """
        Initialize the annotator.

        Args:
            yolo_model: YOLO model name (yolo11n.pt, yolo11s.pt, etc.)
            sam_model: SAM 2 model name (sam2_t.pt, sam2_s.pt, etc.)
            device: Device for inference ('mps', 'cuda', 'cpu'). Auto-detects.
            conf_threshold: Minimum confidence for detections
        """
        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        self.conf_threshold = conf_threshold

        print(f"Initializing VehicleAnnotator on {device}")

        # Load models
        self._load_models(yolo_model, sam_model)

    def _load_models(self, yolo_model: str, sam_model: str):
        """Load YOLO and SAM 2 models."""
        from ultralytics import YOLO, SAM

        print(f"Loading YOLO: {yolo_model}")
        self.yolo = YOLO(yolo_model)

        print(f"Loading SAM 2: {sam_model}")
        self.sam = SAM(sam_model)

        print("Models loaded successfully")

    def annotate_image(
        self,
        image_path: Path,
        use_sam: bool = True
    ) -> List[VehicleAnnotation]:
        """
        Annotate vehicles in a single image.

        Args:
            image_path: Path to image file
            use_sam: Whether to generate SAM masks (slower but more precise)

        Returns:
            List of VehicleAnnotation objects
        """
        image_path = Path(image_path)

        # Step 1: YOLO detection
        yolo_results = self.yolo(
            str(image_path),
            device=self.device,
            conf=self.conf_threshold,
            verbose=False
        )

        annotations = []

        if not yolo_results or len(yolo_results) == 0:
            return annotations

        result = yolo_results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return annotations

        # Filter for vehicle classes only
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        vehicle_boxes = []
        vehicle_info = []

        for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
            if cls in self.VEHICLE_CLASSES:
                vehicle_boxes.append(box.tolist())
                vehicle_info.append({
                    'class_name': self.VEHICLE_CLASSES[cls],
                    'class_id': self.CLASS_MAPPING[self.VEHICLE_CLASSES[cls]],
                    'confidence': float(conf)
                })

        if not vehicle_boxes:
            return annotations

        # Step 2: SAM 2 segmentation (optional)
        masks = None
        if use_sam and len(vehicle_boxes) > 0:
            try:
                sam_results = self.sam(
                    str(image_path),
                    bboxes=vehicle_boxes,
                    device=self.device,
                    verbose=False
                )
                if sam_results and len(sam_results) > 0 and sam_results[0].masks is not None:
                    masks = sam_results[0].masks.data.cpu().numpy()
            except Exception as e:
                print(f"SAM failed (using boxes only): {e}")

        # Create annotations
        for i, (box, info) in enumerate(zip(vehicle_boxes, vehicle_info)):
            mask_rle = None
            if masks is not None and i < len(masks):
                # Convert mask to RLE for storage efficiency
                mask_rle = self._mask_to_rle(masks[i])

            annotation = VehicleAnnotation(
                class_name=info['class_name'],
                class_id=info['class_id'],
                bbox=box,
                confidence=info['confidence'],
                mask_rle=mask_rle
            )
            annotations.append(annotation)

        return annotations

    def _mask_to_rle(self, mask: np.ndarray) -> dict:
        """Convert binary mask to run-length encoding."""
        # Flatten mask
        flat = mask.flatten()
        # Find runs
        runs = []
        prev = 0
        count = 0
        for val in flat:
            if val == prev:
                count += 1
            else:
                if count > 0:
                    runs.append(count)
                prev = val
                count = 1
        runs.append(count)

        return {
            'counts': runs,
            'size': list(mask.shape)
        }

    def annotate_directory(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        use_sam: bool = True,
        save_visualizations: bool = True
    ) -> dict:
        """
        Annotate all images in a directory.

        Args:
            input_dir: Directory containing images
            output_dir: Directory for output (default: input_dir/annotations)
            use_sam: Generate SAM masks
            save_visualizations: Save annotated images

        Returns:
            Summary statistics
        """
        input_dir = Path(input_dir)
        output_dir = output_dir or input_dir / "annotations"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find images
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            images.extend(input_dir.rglob(ext))

        # Skip already-annotated visualization files
        images = [img for img in images if '_annotated' not in img.stem and '_sam2' not in img.stem]

        print(f"Found {len(images)} images to annotate")

        stats = {
            'total_images': len(images),
            'images_with_vehicles': 0,
            'total_vehicles': 0,
            'class_counts': {name: 0 for name in self.CLASS_MAPPING.keys()},
            'processing_time': 0
        }

        start_time = time.time()

        for i, image_path in enumerate(images):
            print(f"[{i+1}/{len(images)}] Annotating: {image_path.name}", end=" ")

            try:
                annotations = self.annotate_image(image_path, use_sam=use_sam)

                if annotations:
                    stats['images_with_vehicles'] += 1
                    stats['total_vehicles'] += len(annotations)

                    for ann in annotations:
                        stats['class_counts'][ann.class_name] += 1

                    # Save YOLO format labels
                    self._save_yolo_labels(image_path, annotations, output_dir)

                    # Save visualization
                    if save_visualizations:
                        self._save_visualization(image_path, annotations, output_dir)

                print(f"- {len(annotations)} vehicles")

            except Exception as e:
                print(f"- ERROR: {e}")

        stats['processing_time'] = time.time() - start_time

        # Save summary
        summary_path = output_dir / "annotation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\nAnnotation complete!")
        print(f"  Images processed: {stats['total_images']}")
        print(f"  Images with vehicles: {stats['images_with_vehicles']}")
        print(f"  Total vehicles found: {stats['total_vehicles']}")
        print(f"  Processing time: {stats['processing_time']:.1f}s")
        print(f"  Output saved to: {output_dir}")

        return stats

    def _save_yolo_labels(
        self,
        image_path: Path,
        annotations: List[VehicleAnnotation],
        output_dir: Path
    ):
        """Save annotations in YOLO format."""
        # Get image dimensions
        img = Image.open(image_path)
        width, height = img.size

        # Create labels directory
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(exist_ok=True)

        # Write label file
        label_path = labels_dir / f"{image_path.stem}.txt"
        with open(label_path, 'w') as f:
            for ann in annotations:
                yolo_line = ann.to_yolo_format(width, height)
                f.write(yolo_line + '\n')

    def _save_visualization(
        self,
        image_path: Path,
        annotations: List[VehicleAnnotation],
        output_dir: Path
    ):
        """Save annotated image visualization."""
        # Load image
        img = cv2.imread(str(image_path))

        # Colors for each class
        colors = {
            'car': (0, 255, 0),       # Green
            'motorcycle': (255, 0, 0), # Blue
            'bus': (0, 165, 255),      # Orange
            'truck': (0, 0, 255),      # Red
        }

        for ann in annotations:
            x1, y1, x2, y2 = [int(v) for v in ann.bbox]
            color = colors.get(ann.class_name, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{ann.class_name} {ann.confidence:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Save
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        vis_path = vis_dir / f"{image_path.stem}_annotated.jpg"
        cv2.imwrite(str(vis_path), img)


def create_yolo_dataset(annotations_dir: Path, images_dir: Path):
    """Create YOLO dataset.yaml file."""
    dataset_yaml = {
        'path': str(annotations_dir.absolute()),
        'train': str(images_dir.absolute()),
        'val': str(images_dir.absolute()),  # Same for now, split manually if needed
        'names': {
            0: 'car',
            1: 'motorcycle',
            2: 'bus',
            3: 'truck'
        }
    }

    yaml_path = annotations_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        import yaml
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    print(f"Created dataset config: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description="Annotate traffic images with YOLO + SAM 2"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/image_library",
        help="Input directory containing images"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for annotations (default: input/annotations)"
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolo11n.pt",
        help="YOLO model to use (yolo11n.pt, yolo11s.pt, etc.)"
    )
    parser.add_argument(
        "--sam-model",
        type=str,
        default="sam2_t.pt",
        help="SAM 2 model to use (sam2_t.pt, sam2_s.pt, etc.)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--no-sam",
        action="store_true",
        help="Skip SAM segmentation (faster, boxes only)"
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Skip saving visualizations"
    )
    args = parser.parse_args()

    # Initialize annotator
    annotator = VehicleAnnotator(
        yolo_model=args.yolo_model,
        sam_model=args.sam_model,
        conf_threshold=args.conf
    )

    # Run annotation
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else None

    stats = annotator.annotate_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        use_sam=not args.no_sam,
        save_visualizations=not args.no_vis
    )

    # Create YOLO dataset config if we have annotations
    if stats['total_vehicles'] > 0:
        output_dir = output_dir or input_dir / "annotations"
        create_yolo_dataset(output_dir, input_dir)

    print("\nNext steps:")
    print("1. Review visualizations in annotations/visualizations/")
    print("2. Train YOLO: yolo train data=annotations/dataset.yaml model=yolo11n.pt epochs=100")


if __name__ == "__main__":
    main()
